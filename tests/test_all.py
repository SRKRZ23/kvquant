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
