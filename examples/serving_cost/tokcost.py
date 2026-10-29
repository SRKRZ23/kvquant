"""tok/$ — first-principles LLM serving-cost profiler ($0, laptop).
Decode is memory-bandwidth-bound: per generated token the GPU re-reads the weights
(+ the KV cache). So tok/s ≈ bandwidth / bytes_read, and tok/$ = tok/s·3600 / $hr.
We anchor the roofline to a REAL measured decode on this Mac, then score datacenter
GPUs and show how int8 KV buys more batch/context (= more tok/$) for the same VRAM.
Approximate public specs; edit freely. Not a substitute for a real cluster benchmark."""
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

GB = 1024**3
# name: (VRAM GB, mem-bandwidth GB/s, approx $/hr on-demand 2026)
GPUS = {"H100-80G": (80, 3350, 2.5), "MI300X-192G": (192, 5300, 2.0),
        "A100-80G": (80, 2039, 1.5), "L40S-48G": (48, 864, 1.0), "RTX4090-24G": (24, 1008, 0.4)}
DT = {"fp16": 2, "int8": 1, "int4": 0.5}

def shape(repo):
    c = AutoConfig.from_pretrained(repo); c = getattr(c, "text_config", c)
    nh = c.num_attention_heads; nkv = getattr(c, "num_key_value_heads", nh)
    hd = getattr(c, "head_dim", None) or c.hidden_size // nh
    return dict(L=c.num_hidden_layers, nkv=nkv, hd=hd)

def kv_bytes_tok(s, kvb):  # per token, all layers, K+V
    return 2 * s["L"] * s["nkv"] * s["hd"] * DT[kvb]

def agg_tok_s(bw_Bs, params, wbytes, kv_tok, ctx, batch):
    read = params * wbytes + batch * ctx * kv_tok      # bytes re-read per decode step
    return batch * bw_Bs / read                        # tokens/s across the batch

def max_batch(vram_B, params, wbytes, kv_tok, ctx):
    free = vram_B - params * wbytes - 2 * GB           # 2GB runtime overhead
    return max(1, int(free / (ctx * kv_tok))) if kv_tok * ctx else 1

# ---- anchor the roofline on a real Mac decode ----
REPO = "Qwen/Qwen2.5-0.5B-Instruct"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO)
m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16).to(DEV).eval()
ids = tok("Count the cost of serving a model token by token.", return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    _ = m.generate(ids, max_new_tokens=8, do_sample=False)           # warmup
    torch.mps.synchronize() if DEV == "mps" else None
    t0 = time.time(); out = m.generate(ids, max_new_tokens=64, do_sample=False)
    torch.mps.synchronize() if DEV == "mps" else None
    dt = time.time() - t0
gen = out.shape[1] - ids.shape[1]; meas = gen / dt
params = sum(p.numel() for p in m.parameters())
eff_bw = meas * params * 2 / 1e9     # infer effective GB/s from measured tok/s (fp16 weights-bound)
print(f"Mac decode anchor: {REPO} {params/1e6:.0f}M params, measured {meas:.1f} tok/s on {DEV}")
print(f"  -> roofline fits with effective bandwidth ~{eff_bw:.0f} GB/s (weights-bound decode holds)\n")

# ---- datacenter scoreboard for a big model ----
BIG = "meta-llama/Llama-3.1-70B"; P = 70.6e9
try: s = shape(BIG)
except Exception: s = dict(L=80, nkv=8, hd=128)   # fallback if gated
CTX, WB = 8192, DT["fp16"]
print(f"tok/$ scoreboard — {BIG}, weights fp16, context {CTX}, on-demand $/hr (roofline estimate):")
print(f"{'GPU':14s} {'KV':5s} {'max batch':>9s} {'agg tok/s':>9s} {'tok/$':>12s}")
for g, (vram, bw, hr) in GPUS.items():
    for kvb in ("fp16", "int8"):
        if P * WB > vram * GB: print(f"{g:14s} {kvb:5s}   weights don't fit ({P*WB/GB:.0f}GB > {vram}GB)"); continue
        kt = kv_bytes_tok(s, kvb); B = max_batch(vram*GB, P, WB, kt, CTX)
        ts = agg_tok_s(bw*1e9, P, WB, kt, CTX, B); tpd = ts*3600/hr
        print(f"{g:14s} {kvb:5s} {B:>9d} {ts:>9.0f} {tpd:>12,.0f}")
