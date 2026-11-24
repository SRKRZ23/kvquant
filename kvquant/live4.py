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

Reads dequantize the whole layer (pure PyTorch, no fused kernel): the memory HELD is int4, the
attention math runs in the original dtype. A fused int4 attention kernel is future work; measure
speed with that in mind.
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


class _Store:
    """int4 storage for one K or V tensor of one layer, shape [batch, heads, seq, head_dim]."""

    def __init__(self, kind: str, group: int):
        self.kind, self.group = kind, group
        self.packed = []      # list of packed uint8 chunks [B, H, n_i, D/2]
        self.mins = []        # fp16 [B, H, 1, D] (keys, per group) or [B, H, n_i, 1] (values)
        self.scales = []
        self.resid = None     # keys only: newest tokens not yet a full group, original dtype
        self.n_q = 0          # tokens held quantized
        self.dtype = None

    def length(self) -> int:
        return self.n_q + (0 if self.resid is None else self.resid.shape[-2])

    def clear(self):
        self.packed, self.mins, self.scales, self.resid, self.n_q = [], [], [], None, 0

    def _quant_append(self, x: torch.Tensor):
        if x.shape[-1] % 2:
            raise ValueError("head_dim must be even for 4-bit packing")
        if self.kind == "k":
            q, mn, sc = _qasym(x, dim=-2)        # per channel over the group's tokens
        else:
            q, mn, sc = _qasym(x, dim=-1)        # per token over head_dim
        self.packed.append(_pack(q))
        self.mins.append(mn)
        self.scales.append(sc)
        self.n_q += x.shape[-2]

    def append(self, new: torch.Tensor):
        self.dtype = new.dtype
        if self.kind == "v":
            self._quant_append(new)
            return
        buf = new if self.resid is None else torch.cat([self.resid, new], dim=-2)
        full = (buf.shape[-2] // self.group) * self.group
        for s in range(0, full, self.group):
            self._quant_append(buf[..., s:s + self.group, :])
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
        if self.packed:
            return tuple(self.packed[0].shape[:-2])
        return tuple(self.resid.shape[:-2]) if self.resid is not None else None

    def get(self):
        if not self.packed and self.resid is None:
            return getattr(self, "empty", None)
        parts = []
        for p, mn, sc in zip(self.packed, self.mins, self.scales):
            parts.append((_unpack(p).float() * sc.float() + mn.float()).to(self.dtype))
        if self.resid is not None:
            parts.append(self.resid)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)

    def nbytes(self) -> int:
        b = 0
        for lst in (self.packed, self.mins, self.scales):
            b += sum(t.element_size() * t.nelement() for t in lst)
        if self.resid is not None:
            b += self.resid.element_size() * self.resid.nelement()
        return b


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
            self._k4.append(key_states)
            self._v4.append(value_states)
            return self._k4.get(), self._v4.get()

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


def make_live_int4(cache, group: int = 128):
    """Convert a transformers>=5 cache in place so its layers hold int4. Idempotent."""
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
        layer._k4 = _Store("k", group)
        layer._v4 = _Store("v", group)
        layer.keys = k
        layer.values = v
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
