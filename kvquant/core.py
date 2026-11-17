"""Core KV-cache quantization for Hugging Face transformers >= 5.

The one thing people get wrong on transformers 5.x: the real K/V tensors live at
``cache.layers[i].keys`` / ``.values``. The old ``cache.key_cache[i]`` is a
non-persisting compatibility accessor — mutating it is a **silent no-op** (the model
keeps using the original fp tensors). :func:`quantize_cache` mutates the real tensors
and returns the mean absolute change so you can *prove* it actually bit.
"""
from __future__ import annotations
import torch


def quantize_dequantize(x: torch.Tensor, bits: int = 8, axis: int = -2) -> torch.Tensor:
    """Symmetric per-channel fake-quantization of ``x`` to ``bits`` integers.

    Scale is computed per slice along ``axis`` (default: the sequence axis of a
    ``[batch, heads, seq, head_dim]`` KV tensor). Returns a dequantized float tensor
    with the same shape and dtype — this measures the *quality* cost of an int cache.
    (For real memory savings the cache must *store* int8; see the README roadmap.)
    """
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    lv = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=axis, keepdim=True) / lv
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(x / s).clamp(-lv, lv) * s


def quantize_dequantize_outlier(x: torch.Tensor, bits: int = 4, n_outlier: int = 4,
                                axis: int = -2) -> torch.Tensor:
    """Per-channel fake-quant to ``bits``, but keep the ``n_outlier`` highest-magnitude
    *channels* (last axis, e.g. head_dim) in full precision.

    KV keys carry a few persistent high-magnitude channels; naive int4 wastes its whole
    range on them and shreds the rest. Keeping those few channels in fp16 while int4-ing
    the others recovers most of the quality at close to int4 memory (see
    :func:`effective_bits`). Channels are ranked by per-channel ``max|.|`` over ``axis``.
    ``n_outlier <= 0`` reduces to plain :func:`quantize_dequantize`.
    """
    qx = quantize_dequantize(x, bits=bits, axis=axis)
    if n_outlier <= 0:
        return qx
    mag = x.abs().amax(dim=axis)                       # [..., channels]
    k = min(int(n_outlier), x.shape[-1])
    idx = mag.topk(k, dim=-1).indices
    mask = torch.zeros_like(mag, dtype=torch.bool).scatter(-1, idx, True).unsqueeze(axis)
    return torch.where(mask, x, qx)                    # outlier channels stay full precision


def effective_bits(bits: int, n_outlier: int, head_dim: int, fp_bits: int = 16) -> float:
    """Average bits/element when ``n_outlier`` of ``head_dim`` channels stay at ``fp_bits``
    and the rest use ``bits``. (The per-channel scale overhead is tiny and ignored.)

    e.g. ``effective_bits(4, 4, 64)`` -> ``(4*60 + 16*4)/64`` = ``4.75`` bits/element.
    Use this to report the honest cost of the outlier scheme, not a bare "int4".
    """
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    k = min(int(n_outlier), head_dim)
    return (bits * (head_dim - k) + fp_bits * k) / head_dim


def quantize_cache(cache, bits: int = 8) -> float:
    """In-place int-``bits`` quantization of a transformers>=5 ``Cache``.

    Mutates ``cache.layers[i].keys`` / ``.values`` (the tensors attention actually
    reads). Returns the **mean absolute change** across layers. A return value of
    ``0.0`` means the quantization did not bite — treat that as a bug, not a result.
    """
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(
            "expected a transformers>=5 Cache exposing `.layers`; got "
            f"{type(cache).__name__}. On transformers 5.x do NOT mutate `.key_cache` "
            "(a non-persisting compat accessor) — that is a silent no-op."
        )
    total, n = 0.0, 0
    for layer in layers:
        keys = getattr(layer, "keys", None)
        if keys is None:
            continue
        before = keys.clone()
        layer.keys = quantize_dequantize(layer.keys, bits)
        layer.values = quantize_dequantize(layer.values, bits)
        total += (layer.keys - before).abs().mean().item()
        n += 1
    if n == 0:
        raise RuntimeError("cache had no populated layers to quantize")
    return total / n
