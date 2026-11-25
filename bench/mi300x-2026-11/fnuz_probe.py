"""Does fp8 KV with the default scale (1.0) hurt Qwen2.5 more on MI300X (e4m3fnuz, max 240) than on
NVIDIA (e4m3fn, max 448)?

Runs on CPU (the GPU is busy with vLLM). Captures post-RoPE q, k, v of Qwen2.5-7B-Instruct on real LongBench
prompts through a transformers attention hook, then measures per layer:
  amax of K and V, share of elements above 240 and 448,
  relative error of the attention output for the last 64 queries when K and V are stored as
    fn_s1     e4m3fn,   scale 1.0  (vLLM default on NVIDIA)
    fnuz_s1   e4m3fnuz, scale 1.0  (vLLM default on MI300X)
    fnuz_s2   e4m3fnuz, scale 2.0  (what vLLM does to checkpoint scales on fnuz)
    fnuz_cal  e4m3fnuz, scale amax/240 per layer (calibrated)
Prints RESULT lines and writes fnuz_probe_result.json.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_bench as B
from transformers import AutoTokenizer, AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

torch.set_num_threads(int(os.environ.get("THREADS", "12")))
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
CAP = int(os.environ.get("CAP", "2000"))
NQ = 64
CAPT = {}


def capture(module, query, key, value, attention_mask, **kw):
    CAPT[module.layer_idx] = (query[:, :, -NQ:].float().clone(), key.float().clone(), value.float().clone())
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("capture", capture)

FMT = {"fn": (torch.float8_e4m3fn, 448.0), "fnuz": (torch.float8_e4m3fnuz, 240.0)}


def q8(x, fmt, scale):
    dt, mx = FMT[fmt]
    return ((x / scale).clamp(-mx, mx).to(dt).float() * scale)


def attn(q, k, v):
    g = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, 1); v = v.repeat_interleave(g, 1)
    n = k.shape[2]
    s = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    pos = torch.arange(n - q.shape[2], n)[:, None]
    s = s.masked_fill(torch.arange(n)[None, :] > pos, float("-inf"))
    return torch.softmax(s, -1) @ v


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="capture").eval()
    B.CTX_CAP = CAP
    items = []
    for task in B.TASKS:
        items += [x[0] for x in B.prompts(tok, task, 2)]
    arms = ["fn_s1", "fnuz_s1", "fnuz_s2", "fnuz_cal"]
    agg = {a: [] for a in arms}
    layer_rows = []
    for pi, text in enumerate(items):
        ids = tok(text, return_tensors="pt").input_ids
        CAPT.clear()
        with torch.no_grad():
            model(ids, use_cache=False)
        for li in sorted(CAPT):
            q, k, v = CAPT[li]
            ref = attn(q, k, v)
            ka, va = k.abs().max().item(), v.abs().max().item()
            row = {"prompt": pi, "layer": li, "k_amax": round(ka, 1), "v_amax": round(va, 2),
                   "k_gt240": (k.abs() > 240).float().mean().item(), "k_gt448": (k.abs() > 448).float().mean().item()}
            for a in arms:
                fmt, s = {"fn_s1": ("fn", 1.0), "fnuz_s1": ("fnuz", 1.0), "fnuz_s2": ("fnuz", 2.0),
                          "fnuz_cal": ("fnuz", None)}[a]
                sk = s if s else max(ka / 240.0, 1e-8); sv = s if s else max(va / 240.0, 1e-8)
                out = attn(q, q8(k, fmt, sk), q8(v, fmt, sv))
                e = ((out - ref).norm() / ref.norm()).item()
                row[a] = e; agg[a].append(e)
            layer_rows.append(row)
        print("PROMPT", pi, "tokens", ids.shape[1], "max k_amax", max(r["k_amax"] for r in layer_rows if r["prompt"] == pi),
              {a: round(sum(r[a] for r in layer_rows if r["prompt"] == pi) / len(CAPT), 4) for a in arms}, flush=True)
    res = {"model": MODEL, "prompts": len(items), "cap": CAP,
           "mean_rel_err": {a: sum(agg[a]) / len(agg[a]) for a in arms},
           "max_rel_err": {a: max(agg[a]) for a in arms},
           "layers_k_over_240": sorted({r["layer"] for r in layer_rows if r["k_amax"] > 240}),
           "layers_k_over_448": sorted({r["layer"] for r in layer_rows if r["k_amax"] > 448}),
           "k_amax_by_layer": {li: max(r["k_amax"] for r in layer_rows if r["layer"] == li) for li in sorted(CAPT)},
           "rows": layer_rows}
    for a in arms:
        print("RESULT", a, "mean_rel_err", round(res["mean_rel_err"][a], 5), "max", round(res["max_rel_err"][a], 5), flush=True)
    print("RESULT layers K>240", res["layers_k_over_240"], "K>448", res["layers_k_over_448"], flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fnuz_probe_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
