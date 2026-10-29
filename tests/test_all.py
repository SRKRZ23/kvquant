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
