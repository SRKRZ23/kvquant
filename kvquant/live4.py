"""Live int4 KV cache: actually HOLDS 4-bit keys/values in memory during generation.

Scheme (KIVI-style, the one measured on LongBench with the simulated int4):

* keys:   asymmetric 4-bit, **per channel**, in groups of ``group`` tokens along the sequence
          (one fp16 min + scale per channel per group). Keys have outlier channels, so per-channel
          scales are what keep int4 keys accurate.
* values: asymmetric 4-bit, **per token** over ``head_dim`` (one fp16 min + scale per token).
* two 4-bit codes are packed into one uint8.
* the newest ``< group`` tokens stay in the original dtype (a small residual buffer) until a full
  key group can be quantized; values are quantized at once (per token needs no group).

Memory held per token and head, head_dim = 128, group = 128, bf16 baseline 512 B:
keys 64 B + 4 B/128-token group share, values 64 B + 4 B  ->  ~136 B, i.e. ~3.8x smaller than bf16.

How it hooks in: like :mod:`kvquant.live`, each cache layer's class is swapped for a subclass whose
``keys`` / ``values`` are properties. transformers' layer ``update`` does
``self.keys = cat([self.keys, new], -2)``; the setter sees a longer tensor and only quantizes the
new tail (the prefix it receives is its own dequantized output, so it is not re-quantized and no
error accumulates). A shorter or reordered tensor (crop, beam reorder, eviction) triggers a full
re-quantization of what it was given.

Unfused reads dequantize the whole layer (pure PyTorch): the memory HELD is int4, the attention math runs
in the original dtype. ``make_live_int4(cache, fused=True)`` + ``kvquant.fused4.enable(model)`` instead
reads the 4-bit codes inside a Triton decode-attention kernel (see kvquant/fused4.py).
"""
from __future__ import annotations
import torch

_variants: dict = {}


def _pack(q: torch.Tensor) -> torch.Tensor:
    """uint8 codes in [0, 15], last dim even -> packed uint8 with half the last dim."""
    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)


def _unpack(p: torch.Tensor) -> torch.Tensor:
    lo = p & 0x0F
    hi = p >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def _qasym(x: torch.Tensor, dim: int):
    """4-bit asymmetric quantization along ``dim``; returns codes (uint8) and fp16 min, scale."""
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    scale = ((mx - mn).float().clamp_min(1e-6) / 15.0).to(torch.float16)
    mn = mn.to(torch.float16)
    q = ((x.float() - mn.float()) / scale.float()).round_().clamp_(0, 15).to(torch.uint8)
    return q, mn, scale


class _Arena:
    """One uint8 buffer per cache layer holding K codes, K group mins/scales, V codes, V token mins/scales.

    Why one buffer: PyTorch's caching allocator serves allocations below 10 MB from 20 MB segments, so separate
    ~8 MB code buffers waste ~20 % of the GPU (measured on MI300X: 308 users vs ~370 expected). A single
    per-layer block above 10 MB is rounded to 2 MB only. The two stores of a layer get views into it.
    """

    def __init__(self, group: int):
        self.group, self.buf, self.cap, self.key, self.stores = group, None, 0, None, {}

    def ensure(self, lead, D, device, need):
        key = (tuple(lead), D, device)
        if self.buf is not None and key == self.key and need <= self.cap:
            return
        G = self.group
        cap = -(-need // G) * G
        if self.buf is not None and key == self.key:
            cap = max(cap, -(-(self.cap + self.cap // 8) // G) * G)
        bh = 1
        for x in lead:
            bh *= x
        layout = [("k", "codes", cap, D // 2, 1), ("k", "mn", cap // G, D, 2), ("k", "sc", cap // G, D, 2),
                  ("v", "codes", cap, D // 2, 1), ("v", "mn", cap, 1, 2), ("v", "sc", cap, 1, 2)]
        buf = torch.empty(sum(bh * r * c * e for *_, r, c, e in layout), dtype=torch.uint8, device=device)
        views, o = {}, 0
        for kind, name, r, c, e in layout:
            n = bh * r * c * e
            t = buf[o:o + n]
            views[(kind, name)] = (t.view(torch.float16) if e == 2 else t).view(*lead, r, c)
            o += n
        same = self.buf is not None and key == self.key
        for kind, st in self.stores.items():
            nc, nm, ns = views[(kind, "codes")], views[(kind, "mn")], views[(kind, "sc")]
            if same and st.n_q:
                r = st._mrows()
                nc[..., :st.n_q, :] = st.codes[..., :st.n_q, :]
                nm[..., :r, :] = st.mn[..., :r, :]
                ns[..., :r, :] = st.sc[..., :r, :]
            st.codes, st.mn, st.sc, st.cap = nc, nm, ns, cap
        self.buf, self.cap, self.key = buf, cap, key


class _Store:
    """int4 storage for one K or V tensor of one layer, shape [batch, heads, seq, head_dim].

    Contiguous: one preallocated buffer per field (codes, mins, scales) instead of one small tensor per
    128-token group. Thousands of small tensors per long cache fragment the GPU allocator (MI300X, 32k-token
    caches: 341 users vs 371 expected from the byte ratio). Capacity is exact after the first fill and grows
    by 1/8 (rounded to whole groups) when decoding runs past it.
    """

    def __init__(self, kind: str, group: int, arena: "_Arena | None" = None):
        self.kind, self.group = kind, group
        self.arena = arena
        if arena is not None:
            arena.stores[kind] = self
        self.codes = None     # uint8 [B, H, cap, D/2], rows [0, n_q) valid
        self.mn = None        # fp16: keys [B, H, cap/G, D] (per group, per channel); values [B, H, cap, 1]
        self.sc = None
        self.cap = 0
        self.resid = None     # keys only: newest tokens not yet a full group, original dtype
        self.n_q = 0          # tokens held quantized
        self.dtype = None

    # read-only views of the valid part (kept for callers of the list-based version)
    @property
    def packed(self):
        return [] if self.codes is None or not self.n_q else [self.codes[..., :self.n_q, :]]

    @property
    def mins(self):
        return [] if self.mn is None or not self.n_q else [self.mn[..., :self._mrows(), :]]

    @property
    def scales(self):
        return [] if self.sc is None or not self.n_q else [self.sc[..., :self._mrows(), :]]

    def _mrows(self, n=None):
        n = self.n_q if n is None else n
        return n // self.group if self.kind == "k" else n

    def length(self) -> int:
        return self.n_q + (0 if self.resid is None else self.resid.shape[-2])

    def clear(self):
        self.resid = None
        self.n_q = 0
        if self.arena is None:
            self.codes = self.mn = self.sc = None
            self.cap = 0

    def _ensure(self, lead, D, device, need):
        """make room for `need` quantized tokens in total"""
        if self.arena is not None:
            self.arena.ensure(lead, D, device, need)
            return
        if self.codes is not None and need <= self.cap:
            return
        G = self.group
        cap = -(-need // G) * G
        if self.codes is not None:
            cap = max(cap, -(-(self.cap + self.cap // 8) // G) * G)
        codes = torch.empty(*lead, cap, D // 2, dtype=torch.uint8, device=device)
        mrows = cap // G if self.kind == "k" else cap
        mcols = D if self.kind == "k" else 1
        mn = torch.empty(*lead, mrows, mcols, dtype=torch.float16, device=device)
        sc = torch.empty_like(mn)
        if self.codes is not None and self.n_q:
            r = self._mrows()
            codes[..., :self.n_q, :] = self.codes[..., :self.n_q, :]
            mn[..., :r, :] = self.mn[..., :r, :]
            sc[..., :r, :] = self.sc[..., :r, :]
        self.codes, self.mn, self.sc, self.cap = codes, mn, sc, cap

    def _quant_append(self, x: torch.Tensor):
        """quantize x [B, H, t, D] (keys: t a multiple of group) and append it"""
        D, t = x.shape[-1], x.shape[-2]
        if D % 2:
            raise ValueError("head_dim must be even for 4-bit packing")
        lead = tuple(x.shape[:-2])
        self._ensure(lead, D, x.device, self.n_q + t)
        if self.kind == "k":
            g = t // self.group
            xg = x.reshape(*lead, g, self.group, D)
            q, mn, sc = _qasym(xg, dim=-2)                       # per channel over each group's tokens
            q = q.reshape(*lead, t, D); mn = mn.squeeze(-2); sc = sc.squeeze(-2)
            r0 = self.n_q // self.group
            self.mn[..., r0:r0 + g, :] = mn
            self.sc[..., r0:r0 + g, :] = sc
        else:
            q, mn, sc = _qasym(x, dim=-1)                        # per token over head_dim
            self.mn[..., self.n_q:self.n_q + t, :] = mn
            self.sc[..., self.n_q:self.n_q + t, :] = sc
        self.codes[..., self.n_q:self.n_q + t, :] = _pack(q)
        self.n_q += t

    def append(self, new: torch.Tensor):
        self.dtype = new.dtype
        if self.kind == "v":
            self._quant_append(new)
            return
        buf = new if self.resid is None else torch.cat([self.resid, new], dim=-2)
        full = (buf.shape[-2] // self.group) * self.group
        if full:
            self._quant_append(buf[..., :full, :])
        self.resid = buf[..., full:, :] if buf.shape[-2] > full else None

    def set(self, t: torch.Tensor):
        if t is None or t.numel() == 0:
            self.clear()
            self.empty = t               # transformers' lazy init sets an empty 1-D tensor
            return
        n = self.length()
        if n and t.shape[-2] > n and t.shape[:-2] == self._lead():
            self.append(t[..., n:, :])           # usual decode / chunked-prefill path
        elif n and t.shape[-2] == n and t.shape[:-2] == self._lead():
            return                               # unchanged (e.g. set back after a read)
        else:
            self.clear()
            self.append(t)                       # first fill, crop, reorder: re-quantize all

    def _lead(self):
        if self.codes is not None:
            return tuple(self.codes.shape[:-2])
        return tuple(self.resid.shape[:-2]) if self.resid is not None else None

    def dequant(self, a: int = 0, b: int = None, dtype=None):
        """dequantized quantized rows [a, b) (default: all quantized rows), in `dtype` (default: stored dtype)"""
        b = self.n_q if b is None else b
        dtype = dtype or self.dtype
        c = _unpack(self.codes[..., a:b, :]).float()
        if self.kind == "k":
            G = self.group
            assert a % G == 0 and b % G == 0
            lead = c.shape[:-2]
            c = c.reshape(*lead, (b - a) // G, G, c.shape[-1])
            x = c * self.sc[..., a // G:b // G, None, :].float() + self.mn[..., a // G:b // G, None, :].float()
            return x.reshape(*lead, b - a, x.shape[-1]).to(dtype)
        return (c * self.sc[..., a:b, :].float() + self.mn[..., a:b, :].float()).to(dtype)

    def get(self):
        if not self.n_q and self.resid is None:
            return getattr(self, "empty", None)
        parts = [self.dequant()] if self.n_q else []
        if self.resid is not None:
            parts.append(self.resid)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)

    def nbytes(self) -> int:
        """bytes ALLOCATED (capacity, not just the valid rows): what the GPU actually holds"""
        b = 0
        for t in (self.codes, self.mn, self.sc, self.resid):
            if t is not None:
                b += t.element_size() * t.nelement()
        return b

    def tensors(self):
        return [t for t in (self.codes, self.mn, self.sc, self.resid) if t is not None]


def _variant(base):
    if base in _variants:
        return _variants[base]

    class Int4Layer(base):
        _kvquant_int4 = True

        @property
        def keys(self):
            return self._k4.get()

        @keys.setter
        def keys(self, v):
            self._k4.set(v)

        @property
        def values(self):
            return self._v4.get()

        @values.setter
        def values(self, v):
            self._v4.set(v)

        def update(self, key_states, value_states, *args, **kwargs):
            # one dequantization per layer per step (the base class would read keys/values again)
            if not getattr(self, "is_initialized", True):
                self.lazy_initialization(key_states, value_states)
            if getattr(self, "_pending", None) is not None:
                raise RuntimeError("fused int4 cache: the previous token was never consumed; load the model with "
                                   "kvquant.fused4.enable(model) (attn_implementation 'kvq_int4')")
            if getattr(self, "_fused", False) and key_states.shape[-2] == 1 and self._k4.length() > 0:
                # fused decode: the attention function (kvquant.fused4) reads the 4-bit store directly and
                # appends this token afterwards; nothing is dequantized here
                self._pending = (key_states, value_states)
                return key_states, value_states
            # Attention for THIS step uses the stored past (dequantized) plus the new tokens at full precision,
            # as KIVI and serving stacks do: the prompt attends to itself in the original dtype during prefill and
            # only the STORED copy is 4-bit. Returning the quantized new tokens made prefill attend to 4-bit
            # keys/values and compounded the error through the layers (gate 0, 2026-11-25: real < simulated).
            past_k, past_v = self._k4.get(), self._v4.get()
            self._k4.append(key_states)
            self._v4.append(value_states)
            if past_k is None or past_k.numel() == 0:
                return key_states, value_states
            return torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)

        def get_seq_length(self, *args, **kwargs) -> int:
            return self._k4.length()

        def int4_bytes(self) -> int:
            return self._k4.nbytes() + self._v4.nbytes()

    if base.__name__ != "DynamicLayer":
        # sliding-window / static layers keep their own update logic and go through the properties
        del Int4Layer.update
        del Int4Layer.get_seq_length
    _variants[base] = Int4Layer
    return Int4Layer


def make_live_int4(cache, group: int = 128, fused: bool = False):
    """Convert a transformers>=5 cache in place so its layers hold int4. Idempotent.

    fused=True: decode steps go through kvquant.fused4 (the model must be switched with fused4.enable(model))."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("expected a transformers>=5 Cache exposing `.layers`")
    for layer in layers:
        if getattr(type(layer), "_kvquant_int4", False):
            continue
        k = layer.__dict__.get("keys", getattr(layer, "keys", None))
        v = layer.__dict__.get("values", getattr(layer, "values", None))
        layer.__dict__.pop("keys", None)
        layer.__dict__.pop("values", None)
        layer.__class__ = _variant(type(layer))
        arena = _Arena(group)
        layer._k4 = _Store("k", group, arena)
        layer._v4 = _Store("v", group, arena)
        layer.keys = k
        layer.values = v
    for layer in layers:
        layer._fused = fused
        layer._pending = None
    if fused:
        from . import fused4
        fused4.activate(cache)
    return cache


def held_bytes4(cache) -> int:
    """Live byte footprint of the cache's K/V storage (int4 layers counted as stored)."""
    total = 0
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "int4_bytes"):
            total += layer.int4_bytes()
        else:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None:
                    total += t.element_size() * t.nelement()
    return total


def held_tensors(layer):
    """the tensors that actually hold a layer's cache (one arena buffer + key residual for int4 layers)"""
    if hasattr(layer, "_k4"):
        a = layer._k4.arena
        ts = [a.buf] if a is not None and a.buf is not None else layer._k4.tensors() + layer._v4.tensors()
        return ts + ([layer._k4.resid] if layer._k4.resid is not None else [])
    return [t for t in (getattr(layer, "keys", None), getattr(layer, "values", None)) if t is not None]
