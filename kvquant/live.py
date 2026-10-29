"""Live int8 KV cache — actually HOLDS int8 in memory during generation.

Unlike :func:`kvquant.quantize_cache` (fake-quant, measures quality) and
:mod:`kvquant.storage` (compress a finished cache), this converts a transformers>=5
cache so each layer *stores* its keys/values as int8 + a per-token fp16 scale and
dequantizes them on read inside attention. Memory held is ~1/4 of fp32 (~1/2 of fp16);
the quality cost is the int8 cost (sub-1% perplexity on the models tested).

Usage: call :func:`make_live_int8` on the cache after each forward in your decode loop
(or once on a prefilled cache). It is idempotent and keeps the layer's own class
behaviour (e.g. sliding-window) by subclassing it, so attention still works.
"""
from __future__ import annotations
import torch

_variants: dict = {}


def _q(fp):
    s = fp.abs().amax(dim=-1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(fp / s).clamp(-127, 127).to(torch.int8), s.to(torch.float16)


def _dq(q, s, dtype):
    return (q.float() * s.float()).to(dtype) if q is not None else None


def _variant(base):
    """A subclass of ``base`` whose keys/values are int8-backed properties."""
    if base in _variants:
        return _variants[base]

    class Int8Layer(base):
        _kvquant_int8 = True

        @property
        def keys(self):
            return _dq(getattr(self, "_qk", None), getattr(self, "_sk", None),
                       getattr(self, "_dt", torch.float32))

        @keys.setter
        def keys(self, v):
            if v is None:
                self._qk = self._sk = None
            else:
                self._dt = v.dtype
                self._qk, self._sk = _q(v)

        @property
        def values(self):
            return _dq(getattr(self, "_qv", None), getattr(self, "_sv", None),
                       getattr(self, "_dt", torch.float32))

        @values.setter
        def values(self, v):
            if v is None:
                self._qv = self._sv = None
            else:
                self._dt = v.dtype
                self._qv, self._sv = _q(v)

        def int8_bytes(self) -> int:
            b = 0
            for a in ("_qk", "_sk", "_qv", "_sv"):
                t = getattr(self, a, None)
                if t is not None:
                    b += t.element_size() * t.nelement()
            return b

    _variants[base] = Int8Layer
    return Int8Layer


def make_live_int8(cache):
    """Convert a transformers>=5 cache in place so its layers hold int8. Idempotent."""
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("expected a transformers>=5 Cache exposing `.layers`")
    for layer in layers:
        if getattr(type(layer), "_kvquant_int8", False):
            continue  # already converted
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        layer.__class__ = _variant(type(layer))
        layer._dt = k.dtype if k is not None else torch.float32
        layer.keys = k
        layer.values = v
    return cache


def held_bytes(cache) -> int:
    """Live byte footprint of the cache's K/V storage (int8 if converted, else fp)."""
    total = 0
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "int8_bytes"):
            total += layer.int8_bytes()
        else:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None:
                    total += t.element_size() * t.nelement()
    return total
