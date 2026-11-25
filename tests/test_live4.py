import torch
import pytest

from kvquant.live4 import _pack, _unpack, _Store, make_live_int4, held_bytes4


def test_pack_roundtrip():
    q = torch.randint(0, 16, (2, 3, 7, 8), dtype=torch.uint8)
    p = _pack(q)
    assert p.shape == (2, 3, 7, 4) and p.dtype == torch.uint8
    assert torch.equal(_unpack(p), q)


@pytest.mark.parametrize("kind", ["k", "v"])
def test_error_bound(kind):
    torch.manual_seed(0)
    x = torch.randn(1, 2, 256, 64)
    st = _Store(kind, group=128)
    st.set(x)
    y = st.get()
    assert y.shape == x.shape
    # asymmetric 4-bit: error <= scale/2 (+ fp16 rounding of min/scale)
    rng = x.amax(-2 if kind == "k" else -1, keepdim=True) - x.amin(-2 if kind == "k" else -1, keepdim=True)
    assert ((y - x).abs() <= rng / 15 * 0.5 * 1.02 + 1e-3).all()


def test_incremental_equals_oneshot_and_tail_untouched():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 300, 64)
    a, b = _Store("k", 128), _Store("k", 128)
    a.set(x)
    for s in range(0, 300, 7):                     # decode-like growth via the setter protocol
        cur = b.get()
        nxt = x[..., : min(s + 7, 300), :]
        if cur is not None and cur.numel():
            nxt = torch.cat([cur, x[..., cur.shape[-2]: min(s + 7, 300), :]], dim=-2)
        b.set(nxt)
    assert torch.equal(a.get(), b.get())
    assert b.n_q == 256 and b.resid.shape[-2] == 44  # 2 full groups quantized, 44 tokens residual
    assert torch.equal(b.get()[..., 256:, :], x[..., 256:, :])


def test_bytes_ratio():
    x = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16)
    k, v = _Store("k", 128), _Store("v", 128)
    k.set(x); v.set(x)
    ratio = 2 * x.element_size() * x.nelement() / (k.nbytes() + v.nbytes())
    assert 3.6 < ratio < 4.0, ratio


def test_generate_tiny_qwen2():
    tr = pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    m = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 300))

    def run(conv):
        cache = DynamicCache(config=cfg)
        if conv:
            make_live_int4(cache)
        with torch.no_grad():
            out = m(ids, past_key_values=cache, use_cache=True)
            logits = [out.logits[:, -1]]
            tok = out.logits[:, -1:].argmax(-1)
            for _ in range(20):
                out = m(tok, past_key_values=cache, use_cache=True)
                logits.append(out.logits[:, -1]); tok = out.logits[:, -1:].argmax(-1)
        return torch.stack(logits), cache

    l_fp, c_fp = run(False)
    l_q, c_q = run(True)
    assert c_q.get_seq_length() == c_fp.get_seq_length() == 320
    assert type(c_q.layers[0]).__name__ == "Int4Layer"
    rel = (l_q - l_fp).norm() / l_fp.norm()
    assert rel < 0.15, rel
    fp_bytes = sum(2 * l.keys.element_size() * l.keys.nelement() for l in c_fp.layers)
    assert fp_bytes / held_bytes4(c_q) > 2.5   # fp32 tiny model, head_dim 32, residual included


def test_prefill_attends_in_full_precision():
    """Regression (gate 0, 2026-11-25): the prompt must attend to itself at full precision; only the stored copy is
    4-bit. Before the fix, update() returned the quantized new tokens, so prefill logits already differed and real
    int4 scored below simulated int4 on LongBench."""
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    m = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 300))
    with torch.no_grad():
        ref = m(ids, past_key_values=DynamicCache(config=cfg), use_cache=True).logits
        c4 = make_live_int4(DynamicCache(config=cfg))
        got = m(ids, past_key_values=c4, use_cache=True).logits
        assert torch.equal(got, ref)                       # prefill identical: no 4-bit inside the prompt
        assert held_bytes4(c4) < sum(2 * l._k4.length() for l in c4.layers) * 128 * 4  # but stored as int4
        nxt = got[:, -1:].argmax(-1)
        step = m(nxt, past_key_values=c4, use_cache=True).logits    # decode reads the 4-bit past
        assert step.shape == (1, 1, 128) and c4.get_seq_length() == 301
