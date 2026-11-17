import numpy as np
import pytest
import torch

from kvquant.core import quantize_dequantize, quantize_cache
from kvquant import doesitbite as dib
from kvquant import rankconf as rc


def test_quantize_dequantize_bits():
    x = torch.randn(1, 2, 64, 16)
    q8 = quantize_dequantize(x, 8)
    q2 = quantize_dequantize(x, 2)
    assert q8.shape == x.shape
    # coarser quantization must distort more
    assert (q2 - x).abs().mean() > (q8 - x).abs().mean() > 0


def test_quantize_cache_rejects_wrong_object():
    with pytest.raises(TypeError):
        quantize_cache(object())  # no .layers -> the transformers-5.x gotcha message


def test_doesitbite_catches_noop():
    @dib.bites()
    def identity(x):
        return x.clone()
    with pytest.raises(AssertionError):
        identity(torch.ones(10))

    @dib.bites()
    def real(x):
        return x + 1
    assert real(torch.zeros(5)).sum().item() == 5


def test_rankconf_monotonic_noop():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    err = rng.random(2000) ** 2
    for f in (lambda x: 1 / (1 + np.exp(-x)), lambda x: x ** 3, np.exp):
        assert rc.is_rank_noop(conf, f, err)


def test_rankconf_agreement_beats_self():
    rng = np.random.default_rng(1)
    n = 3000
    err = rng.random(n) ** 2
    self_conf = -0.3 * err + rng.normal(0, 0.3, n)
    agree = -err + rng.normal(0, 0.25, n)
    assert rc.aurc(agree, err) < rc.aurc(self_conf, err)


def test_int8_store_saves_bytes_and_reconstructs():
    from kvquant.storage import Int8KVStore
    x = torch.randn(1, 4, 128, 64)  # [B, H, seq, head_dim]
    s = Int8KVStore(x)
    # ~4x smaller than fp32 (int8 + tiny fp16 per-token scale)
    assert s.nbytes < x.element_size() * x.nelement() / 3.5
    rel = (s.dequantize() - x).abs().mean() / x.abs().mean()
    assert rel < 0.05  # int8 per-token reconstruction stays within a few percent


# ---- outlier-aware int4 (offline, synthetic; the real-model run is examples/serving_cost) ----

def _keys_with_outliers(seq=64, hd=16, n_big=2, scale=20.0, seed=0):
    """A [1,1,seq,hd] key tensor: most channels ~N(0,1), a few channels blown up."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, seq, hd, generator=g)
    x[..., :n_big] *= scale                      # channels 0..n_big-1 are the outliers
    return x


def test_outlier_int4_beats_naive_int4_on_outlier_keys():
    from kvquant.core import quantize_dequantize, quantize_dequantize_outlier
    x = _keys_with_outliers(n_big=2, scale=20.0)
    naive = quantize_dequantize(x, bits=4, axis=-2)
    outl = quantize_dequantize_outlier(x, bits=4, n_outlier=2, axis=-2)
    e_naive = (naive - x).abs().mean().item()
    e_outl = (outl - x).abs().mean().item()
    # keeping the 2 blown-up channels in fp recovers most of what naive int4 shreds
    assert e_outl < e_naive * 0.5, (e_outl, e_naive)


def test_outlier_channels_are_preserved_exactly():
    from kvquant.core import quantize_dequantize_outlier
    x = _keys_with_outliers(n_big=3, scale=15.0)
    out = quantize_dequantize_outlier(x, bits=4, n_outlier=3, axis=-2)
    # the 3 outlier channels must come back bit-exact (they stayed full precision)
    assert torch.equal(out[..., :3], x[..., :3])
    # a non-outlier channel was quantized, so it must have changed
    assert (out[..., 3:] - x[..., 3:]).abs().sum() > 0


def test_effective_bits_accounting():
    from kvquant.core import effective_bits
    assert effective_bits(4, 4, 64) == (4 * 60 + 16 * 4) / 64        # 4.75
    assert effective_bits(4, 0, 64) == 4.0                           # no outliers -> plain int4
    assert effective_bits(8, 2, 16) == (8 * 14 + 16 * 2) / 16        # 9.0


def _rope(x, base=10000.0):
    """Standard HF rotate-half RoPE on [b,h,s,d] (d even), to show quant position matters."""
    b, h, s, d = x.shape
    half = d // 2
    inv = base ** (-torch.arange(0, half, dtype=x.dtype) / half)
    ang = torch.arange(s, dtype=x.dtype)[:, None] * inv[None, :]     # [s, half]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def test_outlier_int4_must_be_validated_post_rope():
    """Where you quantize matters: RoPE spreads a few concentrated key outliers across
    more channels, so a fixed outlier budget recovers PRE-RoPE keys far better than the
    POST-RoPE keys the cache actually stores. Validate on post-RoPE, and budget for it."""
    from kvquant.core import quantize_dequantize_outlier
    x_pre = _keys_with_outliers(n_big=2, scale=20.0)
    x_post = _rope(x_pre)
    err = lambda a, b: (a - b).abs().mean().item()
    e_pre = err(quantize_dequantize_outlier(x_pre, bits=4, n_outlier=2), x_pre)
    e_post = err(quantize_dequantize_outlier(x_post, bits=4, n_outlier=2), x_post)
    # post-RoPE error is meaningfully larger for the SAME outlier budget (the trap)
    assert e_post > e_pre * 1.2, (e_pre, e_post)
    # and spending more of the budget post-RoPE recovers it (outliers spread, not vanished)
    e_post_more = err(quantize_dequantize_outlier(x_post, bits=4, n_outlier=4), x_post)
    assert e_post_more < e_post


def test_live_int8_cache_holds_int8():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvquant.live import make_live_int8, held_bytes
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    m = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.float32).eval()
    ids = tok("live int8 cache test " * 8, return_tensors="pt").input_ids[:, :40]
    with torch.no_grad():
        cache = m(ids, use_cache=True).past_key_values
    fp = held_bytes(cache)
    make_live_int8(cache)
    assert cache.layers[0]._qk.dtype == torch.int8
    assert held_bytes(cache) < fp / 3          # real int8 storage, ~4x vs fp32
