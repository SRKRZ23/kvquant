"""int4 outlier-aware KV on a real GQA model (Qwen2.5-0.5B, MacBook/MPS, $0).
Keys have persistent per-channel outliers; keeping a few channels in fp16 while
int4-ing the rest recovers quality at ~int4 memory. Measured autoregressively
(cache actually reused + requantized each step). Reuses kvquant's per-channel fake-quant."""
import sys, math, torch
from kvquant import quantize_dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TEXT = ("Perplexity measures how well a probability model predicts a sample; lower is better. "
        "Quantizing the key-value cache lets a server hold a longer context or a larger batch on the "
        "same GPU, which at datacenter scale is the binding cost of serving a large language model. "
        "We test, on a real grouped-query-attention model, whether keeping a handful of outlier key "
        "channels in higher precision recovers the quality that naive int4 throws away. ") * 6

def q_keys(k, mode, n_out=4):           # k: [b, n_kv, seq, hd]; per-channel over seq (axis=-2)
    if mode == "fp16": return k
    bits = {"int8": 8, "int4": 4, "int2": 2, "int4o": 4}[mode]
    qk = quantize_dequantize(k, bits=bits, axis=-2)
    if mode != "int4o": return qk
    mag = k.abs().amax(dim=-2)          # [b, n_kv, hd] per-channel magnitude
    idx = mag.topk(min(n_out, k.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(mag, dtype=torch.bool).scatter(-1, idx, True).unsqueeze(-2)
    return torch.where(mask, k, qk)     # outlier channels stay fp16, rest int4

def q_vals(v, mode):                    # values: per-token (axis=-1)
    if mode == "fp16": return v
    bits = {"int8": 8, "int4": 4, "int2": 2, "int4o": 4}[mode]
    return quantize_dequantize(v, bits=bits, axis=-1)

def bits_per_elem(keymode, n_out=4, hd=64):
    base = {"fp16": 16, "int8": 8, "int4": 4, "int2": 2, "int4o": 4}[keymode]
    key_bits = base if keymode != "int4o" else (base + n_out / hd * 16)   # +scale overhead tiny, ignore
    val_bits = {"fp16": 16, "int8": 8, "int4": 4, "int2": 2, "int4o": 4}[keymode]
    return (key_bits + val_bits) / 2

@torch.no_grad()
def ppl(model, tok, keymode, valmode, n_out=4, max_tokens=160):
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :max_tokens].to(DEV)
    past = None; nll = 0.0; n = 0; changed = 0.0
    for t in range(ids.shape[1] - 1):
        out = model(ids[:, t:t+1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        for layer in past.layers:
            k = getattr(layer, "keys", None)
            if k is None: continue
            nk = q_keys(k, keymode, n_out); changed += float((nk - k).abs().mean())
            layer.keys = nk; layer.values = q_vals(layer.values, valmode)
        logp = torch.log_softmax(out.logits[:, -1].float(), -1)
        nll += -logp[0, ids[0, t+1]].item(); n += 1
    return math.exp(nll / n), changed

print(f"loading {REPO} on {DEV} …", flush=True)
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).to(DEV).eval()
print("layers expose .keys:", hasattr(model(tok("hi", return_tensors="pt").input_ids.to(DEV), use_cache=True).past_key_values.layers[0], "keys"), flush=True)

base, _ = ppl(model, tok, "fp16", "fp16")
print(f"\n{'config':16s} {'ppl':>9s} {'Δ vs fp16':>10s} {'~bits/elem':>11s}  {'bit?'}")
print(f"{'fp16':16s} {base:9.3f} {'0.00%':>10s} {16.0:>11.2f}")
for km, vm, lbl in [("int8","int8","int8 K+V"), ("int4","int4","int4 K+V (naive)"),
                    ("int4o","int4","int4 K (outlier-4) + int4 V"), ("int2","int2","int2 (sanity)")]:
    p, ch = ppl(model, tok, km, vm)
    print(f"{lbl:16s} {p:9.3f} {100*(p-base)/base:>+9.2f}% {bits_per_elem(km):>11.2f}  {'yes' if ch>0 else 'NO-OP'}", flush=True)
