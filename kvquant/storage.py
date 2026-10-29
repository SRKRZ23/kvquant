"""Real int8 storage for a KV cache — actual bytes saved, not just a quality probe.

``Int8KVStore`` holds a ``[batch, heads, seq, head_dim]`` K or V tensor as int8 plus a
per-token fp16 scale, so its footprint is ~1/4 of fp32 (~1/2 of fp16) with <1% scale
overhead. Per-token scaling (over the ``head_dim`` axis) is append-friendly: every token
is quantized independently, so a growing cache never needs re-quantizing.

Use it to compress a populated cache for offload / serialization / holding an idle
long context. For int8 *during* attention (read-from-int8 every step) pair this scheme
with transformers' ``QuantizedCache``; kvquant's value-add is the tested scheme, the
stress guard, and the measured numbers.
"""
from __future__ import annotations
import torch


def _bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


class Int8KVStore:
    """int8 + per-token fp16 scale storage for one K or V tensor."""

    def __init__(self, fp: torch.Tensor):
        if fp.ndim < 2:
            raise ValueError("expected a [..., seq, head_dim] tensor")
        scale = fp.abs().amax(dim=-1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.q = torch.round(fp / scale).clamp(-127, 127).to(torch.int8)
        self.scale = scale.to(torch.float16)
        self.dtype = fp.dtype
        self.shape = tuple(fp.shape)

    def dequantize(self) -> torch.Tensor:
        return (self.q.float() * self.scale.float()).to(self.dtype)

    @property
    def nbytes(self) -> int:
        return _bytes(self.q) + _bytes(self.scale)


def cache_bytes(cache) -> int:
    """Live byte footprint of a transformers>=5 cache's K/V tensors."""
    total = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if t is not None:
                total += _bytes(t)
    return total


def compress_cache(cache):
    """Return ``[(Int8KVStore, Int8KVStore), ...]`` per layer and the measured savings.

    Returns ``(stores, report)`` where ``report`` has ``fp_bytes``, ``int8_bytes`` and
    ``ratio`` (fp / int8). The original cache is left untouched.
    """
    stores = []
    fp_bytes = int8_bytes = 0
    for layer in getattr(cache, "layers", []):
        k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
        if k is None:
            continue
        sk, sv = Int8KVStore(k), Int8KVStore(v)
        stores.append((sk, sv))
        fp_bytes += _bytes(k) + _bytes(v)
        int8_bytes += sk.nbytes + sv.nbytes
    report = {
        "fp_bytes": fp_bytes,
        "int8_bytes": int8_bytes,
        "ratio": (fp_bytes / int8_bytes) if int8_bytes else 0.0,
    }
    return stores, report
