"""kvquant live int4 vs bf16 at LONG context (up to 30k tokens) on natural QA, same prompts as the vLLM long-context run.
transformers + kvquant (live4, prefill in full precision). Run: CTX_CAP=30000 TASKS=narrativeqa,musique,hotpotqa N=50 python3 -u kvq_long.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import vllm_bench as B
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from kvquant.live4 import make_live_int4

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kvq_long_result.json")

def run(model, tok, ids, arm):
    cache = DynamicCache(config=model.config)
    if arm == "kvquant_int4":
        make_live_int4(cache)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    out_ids = []
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(B.MAXNEW):
            if nxt.item() in eos:
                break
            out_ids.append(nxt.item())
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
    return tok.decode(out_ids, skip_special_tokens=True)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    res = {}
    for task in B.TASKS:
        items = B.prompts(tok, task, B.N)
        for arm in ("full", "kvquant_int4"):
            t0 = time.time(); sc = []
            for text, golds in items:
                ids = tok(text, return_tensors="pt").input_ids.to("cuda")
                sc.append(B.qa_f1(run(model, tok, ids, arm), golds))
            res[f"{task}_{arm}"] = round(sum(sc) / len(sc), 4)
            print("RESULT KVQ", task, arm, res[f"{task}_{arm}"], "n", len(sc), "wall", round(time.time() - t0), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    print("STEP done", flush=True)

if __name__ == "__main__":
    main()
