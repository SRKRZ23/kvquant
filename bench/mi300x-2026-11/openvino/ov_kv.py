# Kill test: does OpenVINO's quantized KV cache on CPU silently hurt Qwen2.5 (large layer-0 key bias), the way vLLM's
# fp8 / int4_per_token_head KV does on MI300X? Free Kaggle CPU.
# Qwen2.5-1.5B-Instruct exported with fp16 weights (no weight quantization, so only the KV precision changes),
# LongBench multifieldqa_en + hotpotqa, N per task, context capped, greedy 64 tokens, token F1.
# Arms differ only in the CPU plugin's KV-cache properties. Each arm is tried; unsupported properties are recorded.
import json, os, re, string, subprocess, sys, time, zipfile, urllib.request
from collections import Counter

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optimum[openvino]", "openvino"], check=False)
# the image ships a diffusers that clashes with the new huggingface_hub; optimum imports it only if present
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "diffusers"], check=False)
W = "/kaggle/working"; os.chdir(W)
LOG = open("results.txt", "w")
def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
N = int(os.environ.get("N", "25")); CAP = int(os.environ.get("CAP", "1500")); MAXNEW = 64
TASKS = ["multifieldqa_en", "hotpotqa"]
INSTR = "Answer the question based only on the text above. Give only the answer, no other words.\nQuestion: "
ARMS = [
    ("kv_f32", {"KV_CACHE_PRECISION": "f32"}),
    ("kv_u4_key_by_channel", {"KEY_CACHE_PRECISION": "u4", "VALUE_CACHE_PRECISION": "u4", "KEY_CACHE_QUANT_MODE": "BY_CHANNEL"}),
    ("kv_u8_key_by_channel", {"KEY_CACHE_PRECISION": "u8", "VALUE_CACHE_PRECISION": "u8", "KEY_CACHE_QUANT_MODE": "BY_CHANNEL"}),
    ("kv_u4", {"KEY_CACHE_PRECISION": "u4", "VALUE_CACHE_PRECISION": "u4"}),
]


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def f1(p, g):
    p, g = normalize(p).split(), normalize(g).split()
    if not p or not g:
        return 0.0
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def longbench(task):
    z = "longbench_data.zip"
    if not os.path.exists(z):
        urllib.request.urlretrieve("https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip", z)
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl")
        return [json.loads(l) for l in zf.read(name).decode().splitlines() if l.strip()]


def main():
    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    log("openvino", ov.__version__)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if not os.path.exists("ov_model/openvino_model.xml"):
        t = time.time()
        m = OVModelForCausalLM.from_pretrained(MODEL, export=True, load_in_8bit=False, compile=False)
        m.save_pretrained("ov_model"); tok.save_pretrained("ov_model"); del m
        log("exported fp16-weight IR in", round(time.time() - t), "s")
    items = {}
    for task in TASKS:
        rows = longbench(task)[:N]; items[task] = []
        for r in rows:
            ids = tok(r["context"]).input_ids; ctx = r["context"]
            if len(ids) > CAP:
                h = CAP // 2; ctx = tok.decode(ids[:h] + ids[-h:], skip_special_tokens=True)
            msgs = [{"role": "user", "content": ctx + "\n\n" + INSTR + r["input"] + "\nAnswer:"}]
            items[task].append((tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), r["answers"]))
    res = {}
    for arm, cfg in ARMS:
        t0 = time.time()
        try:
            m = OVModelForCausalLM.from_pretrained("ov_model", ov_config=dict(cfg, PERFORMANCE_HINT="LATENCY"))
            cm = m.request.get_compiled_model()
            props = {}
            for p in ("KV_CACHE_PRECISION", "KEY_CACHE_PRECISION", "VALUE_CACHE_PRECISION", "KEY_CACHE_GROUP_SIZE",
                      "INFERENCE_PRECISION_HINT"):
                try:
                    props[p] = str(cm.get_property(p))
                except Exception:
                    pass
            out = {"props": props}
            for task in TASKS:
                sc = []
                for prompt, golds in items[task]:
                    enc = tok(prompt, return_tensors="pt")
                    g = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False)
                    txt = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    sc.append(max(f1(txt, x) for x in golds))
                out[f"f1_{task}"] = round(sum(sc) / len(sc), 4)
                log("RESULT", arm, task, out[f"f1_{task}"], props)
            out["wall_s"] = round(time.time() - t0); res[arm] = out
            del m
        except Exception as e:
            res[arm] = {"error": repr(e)[:400]}
            log("ARM_FAILED", arm, repr(e)[:400])
        json.dump({"model": MODEL, "N": N, "cap": CAP, "arms": res}, open("ov_kv_result.json", "w"), indent=1)
    log("SUMMARY", json.dumps({a: {k: v for k, v in r.items() if k.startswith("f1") or k == "error"} for a, r in res.items()}))


if __name__ == "__main__":
    main()
