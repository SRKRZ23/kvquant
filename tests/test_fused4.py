import torch
import pytest

from kvquant.live4 import _Store, make_live_int4
from kvquant import fused4


def test_partial_merge_equals_full_softmax():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 16)
    k = torch.randn(1, 2, 50, 16)
    v = torch.randn(1, 2, 50, 16)
    ref = torch.softmax(torch.einsum("bhrd,bhnd->bhrn", q, k) * 0.25, -1) @ v
    parts = [fused4._partial(q, k[:, :, :20], v[:, :, :20], 0.25), fused4._partial(q, k[:, :, 20:], v[:, :, 20:], 0.25)]
    assert torch.allclose(fused4._merge(parts), ref, atol=1e-5)


def test_region_torch_matches_dequant_attention():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 384, 32)
    ks, vs = _Store("k", 128), _Store("v", 128)
    ks.set(x); vs.set(x * 0.5)
    q = torch.randn(1, 2, 4, 32)
    m, l, o = fused4.region_torch(q, ks, vs, 0.2, 384)
    k = ks.dequant(0, 384, torch.float32); v = vs.dequant(0, 384, torch.float32)
    ref = torch.softmax(torch.einsum("bhrd,bhnd->bhrn", q, k) * 0.2, -1) @ v
    assert torch.allclose(o / l[..., None], ref, atol=1e-5)


def _gen(model, cfg, ids, fused, steps=40):
    from transformers import DynamicCache
    cache = make_live_int4(DynamicCache(config=cfg), fused=fused)
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        logits = [out.logits[:, -1]]
        tok = out.logits[:, -1:].argmax(-1)
        for _ in range(steps):
            out = model(tok, past_key_values=cache, use_cache=True)
            logits.append(out.logits[:, -1]); tok = out.logits[:, -1:].argmax(-1)
    return torch.stack(logits), cache


def test_fused_decode_matches_unfused_tiny_qwen2():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    ids = torch.randint(0, 128, (1, 300))
    m_plain = Qwen2ForCausalLM(cfg).eval()
    l_unf, c_unf = _gen(m_plain, cfg, ids, fused=False)
    m_fused = Qwen2ForCausalLM(cfg).eval()
    m_fused.load_state_dict(m_plain.state_dict())
    fused4.enable(m_fused)
    fused4.BACKEND = "torch"
    before = fused4._STATS["fused_steps"]
    l_fus, c_fus = _gen(m_fused, m_fused.config, ids, fused=True)
    assert fused4._STATS["fused_steps"] - before == 40 * cfg.num_hidden_layers   # every decode step went fused
    assert c_fus.get_seq_length() == c_unf.get_seq_length() == 340                 # 300 + 40, crossed a group edge
    assert c_fus.layers[0]._k4.n_q == 256 and c_fus.layers[0]._pending is None
    rel = (l_fus - l_unf).norm() / l_unf.norm()
    assert rel < 1e-4, rel                                                         # same math, fp32, other order
    assert torch.equal(l_fus.argmax(-1), l_unf.argmax(-1))


def test_unconsumed_pending_token_is_an_error():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    m = Qwen2ForCausalLM(cfg).eval()
    m.set_attn_implementation("sdpa")            # fused cache but NOT the fused attention function
    cache = make_live_int4(DynamicCache(config=cfg), fused=True)
    with torch.no_grad():
        m(torch.randint(0, 64, (1, 20)), past_key_values=cache, use_cache=True)
        with pytest.raises(RuntimeError):
            for _ in range(2):
                m(torch.randint(0, 64, (1, 1)), past_key_values=cache, use_cache=True)
