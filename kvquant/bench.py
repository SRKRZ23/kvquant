"""Perplexity benchmark for KV-cache quantization, with a built-in silent-no-op guard.

Scores tokens autoregressively (so the *cache* is actually exercised) and quantizes
the KV cache after every step. Reports perplexity at fp / int8 / int4 / int2. The int2
row MUST blow up and show a non-zero quant-change — if it does not, the quantization
is a silent no-op and the numbers are meaningless, so :func:`kv_ppl_sweep` asserts it.
"""
from __future__ import annotations
import math
import torch
from .core import quantize_cache

DEFAULT_TEXT = (
    "Perplexity measures how well a probability model predicts a sample; lower is "
    "better. Quantizing the key-value cache to eight-bit integers halves its memory "
    "footprint, which lets a server hold a longer context or a larger batch on the "
    "same hardware. We test whether that compression degrades predictions. "
) * 4


@torch.no_grad()
def _score(model, input_ids, bits):
    past = None
    nll = 0.0
    n = 0
    change = 0.0
    T = input_ids.shape[1]
    for t in range(T - 1):
        out = model(input_ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        if bits:
            change += quantize_cache(past, bits)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        nll += -logp[0, input_ids[0, t + 1]].item()
        n += 1
    return math.exp(nll / n), change


def kv_ppl_sweep(model, tokenizer, text: str = DEFAULT_TEXT, bits_list=(None, 8, 4, 2),
                 max_tokens: int = 400, device: str | None = None):
    """Return ``{bits: {"ppl", "delta_pct", "change"}}`` and assert it isn't a no-op."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)[:, :max_tokens]
    out, base = {}, None
    for bits in bits_list:
        ppl, change = _score(model, ids, bits)
        if bits is None:
            base = ppl
        out[bits] = {"ppl": ppl, "delta_pct": (ppl / base - 1) * 100, "change": change}
    # silent-no-op guard: 2-bit must both bite (change>0) and wreck ppl
    two = out.get(2)
    if two is not None and (two["change"] == 0.0 or two["delta_pct"] < 50):
        raise AssertionError(
            "int2 KV did not degrade perplexity (change=%.3f, delta=%.1f%%) — the "
            "quantization is a SILENT NO-OP; you are probably mutating the wrong cache "
            "attribute." % (two["change"], two["delta_pct"]))
    return out


def _main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description="KV-cache quantization perplexity sweep")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--tokens", type=int, default=400)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    res = kv_ppl_sweep(model, tok, max_tokens=args.tokens)
    print(f"model: {args.model}")
    for bits, r in res.items():
        tag = "fp (baseline)" if bits is None else f"int{bits}"
        print(f"  {tag:14s} ppl={r['ppl']:.4f}  dppl={r['delta_pct']:+.3f}%  "
              f"change={r['change']:.3f}")
    print("int8 KV = ~2x less cache memory; the delta above is the quality cost.")


if __name__ == "__main__":
    _main()
