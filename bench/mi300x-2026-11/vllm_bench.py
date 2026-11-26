"""vLLM KV-cache comparison on one AMD MI300X (AMD Developer Cloud, ROCm vLLM image). Paid run: keep it short.

For each (model, kv_cache_dtype in {auto=bf16, fp8}) a fresh vLLM engine is started in a subprocess and measured:
  CAPACITY  vLLM's own KV cache size (tokens) and max concurrency for 32,768-token requests (parsed from its log)
  SPEED     single-request decode ms/token at 3,500 and 32,000 context tokens (65 vs 1 generated tokens)
  THRU      decode tokens/s with 16 concurrent 8k-token requests
  QUALITY   LongBench multifieldqa_en / hotpotqa / 2wikimqa token F1, N per task (same prompt and 3,500-token cap as
            our kvquant runs)
Usage on the droplet:
  python3 vllm_bench.py                       # 7B then 72B, both dtypes
  MODELS=Qwen/Qwen2.5-7B-Instruct N=100 python3 vllm_bench.py
  DRY=1 python3 vllm_bench.py                 # no vLLM needed: data, prompts, F1, log parser self-test
Writes vllm_bench_result.json after every unit; prints RESULT lines.
"""
import os, sys, json, re, string, subprocess, time, zipfile, urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("OUTFILE", "vllm_bench_result.json"))
MODELS = os.environ.get("MODELS", "Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-72B-Instruct").split(",")
DTYPES = os.environ.get("DTYPES", "auto,fp8").split(",")
TASKS = os.environ.get("TASKS", "multifieldqa_en,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("N", "100"))
CTX_CAP = int(os.environ.get("CTX_CAP", "3500"))
MAXNEW = 64
INSTR = "Answer the question based only on the text above. Give only the answer, no other words.\nQuestion: "


# ------------------------------------------------------------------------------------------- data + metric
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


def qa_f1(pred, golds):
    return max((f1(pred, g) for g in golds), default=0.0)


def longbench(task):
    """rows of THUDM/LongBench from its data.zip (no `datasets` dependency)"""
    z = os.path.join(HERE, "longbench_data.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve("https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip", z)
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl")
        return [json.loads(l) for l in zf.read(name).decode().splitlines() if l.strip()]


def prompts(tok, task, n):
    out = []
    for row in longbench(task)[:n]:
        ids = tok(row["context"]).input_ids
        ctx = row["context"]
        if len(ids) > CTX_CAP:
            h = CTX_CAP // 2
            ctx = tok.decode(ids[:h] + ids[-h:], skip_special_tokens=True)
        msgs = [{"role": "user", "content": ctx + "\n\n" + INSTR + row["input"] + "\nAnswer:"}]
        out.append((tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), row["answers"]))
    return out


# ------------------------------------------------------------------------------------------- vLLM log parsing
def parse_capacity(log):
    r = {}
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
    if m:
        r["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", log)
    if m:
        r["max_concurrency_at"] = int(m.group(1).replace(",", ""))
        r["max_concurrency"] = float(m.group(2))
    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log)
    if m:
        r["kv_cache_gib"] = float(m.group(1))
    return r


# ------------------------------------------------------------------------------------------- one engine (child)
CHILD = r'''
import os, sys, json, time
sys.path.insert(0, os.environ["BENCH_DIR"])
import vllm_bench as B
from vllm import LLM, SamplingParams

def main():
    model, kv = sys.argv[1], sys.argv[2]
    extra, kvd = {}, kv
    if kvd.endswith("_vimpl"):  # force vLLM's native model class instead of the Transformers modeling backend
        extra["model_impl"] = "vllm"; kvd = kvd[:-len("_vimpl")]
    if os.environ.get("QUANT"):  # online weight quantization, e.g. QUANT=fp8
        extra["quantization"] = os.environ["QUANT"]
    if kvd.endswith("_aiter"):  # AMD AITER flash-attention backend instead of the default ROCm attention
        extra["attention_backend"] = "ROCM_AITER_FA"; kvd = kvd[:-len("_aiter")]
    if kvd.endswith("_skip0"):  # keep layer 0 (Qwen2 key bias) in bf16, vLLM's documented remedy
        extra["kv_cache_dtype_skip_layers"] = ["0"]; kvd = kvd[:-len("_skip0")]
    if kvd.endswith("_skipb"):  # keep first 2 and last 2 layers in bf16, the boundary rule TurboQuant applies itself
        from transformers import AutoConfig
        L = AutoConfig.from_pretrained(model).num_hidden_layers
        extra["kv_cache_dtype_skip_layers"] = [str(i) for i in (0, 1, L - 2, L - 1)]; kvd = kvd[:-len("_skipb")]
    if kvd == "fp8_calib":       # fp8 KV with scales computed from the data (default scale 1.0 is uncalibrated)
        extra["calculate_kv_scales"] = True
    llm = LLM(model=model, max_model_len=int(os.environ.get("MAXLEN", "32768")), gpu_memory_utilization=0.90,
              kv_cache_dtype="fp8" if kvd.startswith("fp8") and kvd != "fp8_per_token_head" else kvd,
              enable_prefix_caching=False, trust_remote_code=False, **extra)
    tok = llm.get_tokenizer()
    res = {"model": model, "kv": kv}
    def gen(prompts, n, greedy=True):
        sp = SamplingParams(max_tokens=n, temperature=0, ignore_eos=True)
        t = time.time(); outs = llm.generate(prompts, sp, use_tqdm=False); return time.time() - t, outs
    for L in (() if os.environ.get("QUALITY_ONLY") else (3500, 32000)):
        p = [{"prompt_token_ids": list(range(1000, 1000 + L))}]
        gen(p, 1); a, _ = gen(p, 1); b, _ = gen(p, 65)
        res[f"decode_ms_per_tok_{L}"] = round((b - a) / 64 * 1000, 2)
        print("RESULT SPEED", model, kv, L, res[f"decode_ms_per_tok_{L}"], flush=True)
    if not os.environ.get("QUALITY_ONLY"):
        p = [{"prompt_token_ids": list(range(2000 + i, 2000 + i + 8000))} for i in range(16)]
        gen(p, 1); a, _ = gen(p, 1); b, _ = gen(p, 129)
        res["thru_tok_s_16x8k"] = round(16 * 128 / (b - a), 1)
        print("RESULT THRU", model, kv, res["thru_tok_s_16x8k"], flush=True)
    sp = SamplingParams(max_tokens=B.MAXNEW, temperature=0)
    for task in B.TASKS:
        items = B.prompts(tok, task, B.N)
        outs = llm.generate([x[0] for x in items], sp, use_tqdm=False)
        score = sum(B.qa_f1(o.outputs[0].text.split("<|im_end|>")[0].strip(), x[1]) for o, x in zip(outs, items)) / len(items)
        res[f"f1_{task}"] = round(score, 4)
        print("RESULT F1", model, kv, task, res[f"f1_{task}"], flush=True)
    print("CHILDJSON " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
'''


def run_engine(model, kv):
    child = os.path.join(HERE, "_vllm_child.py")
    open(child, "w").write(CHILD)
    env = dict(os.environ, BENCH_DIR=HERE, PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, child, model, kv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    log, res = [], {}
    for line in p.stdout:
        log.append(line)
        if line.startswith("RESULT") or "KV cache size" in line or "Maximum concurrency" in line or "Error" in line:
            print(line.rstrip()[-240:], flush=True)
        if line.startswith("CHILDJSON "):
            res = json.loads(line[len("CHILDJSON "):])
    p.wait()
    res.update(parse_capacity("".join(log)))
    res["returncode"], res["wall_s"] = p.returncode, round(time.time() - t0)
    open(os.path.join(HERE, f"vllm_{os.environ.get('TAG', '')}{model.split('/')[-1]}_{kv}.log"), "w").write("".join(log))
    return res


# ------------------------------------------------------------------------------------------- main
def dry():
    from transformers import AutoTokenizer
    sample = ("INFO ... Available KV cache memory: 151.20 GiB\nINFO ... GPU KV cache size: 2,831,104 tokens\n"
              "INFO ... Maximum concurrency for 32,768 tokens per request: 86.40x\n")
    cap = parse_capacity(sample)
    assert cap == {"kv_cache_tokens": 2831104, "max_concurrency_at": 32768, "max_concurrency": 86.4,
                   "kv_cache_gib": 151.2}, cap
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for task in TASKS:
        items = prompts(tok, task, 3)
        lens = [len(tok(x[0]).input_ids) for x in items]
        print("DRY", task, "prompts", len(items), "token lengths", lens, "answers", [x[1][:1] for x in items])
    assert abs(qa_f1("The Eiffel Tower", ["eiffel tower"]) - 1.0) < 1e-9
    print("DRY OK")


if __name__ == "__main__":
    if os.environ.get("DRY"):
        dry(); sys.exit(0)
    r = subprocess.run([sys.executable, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.hip)"],
                       capture_output=True, text=True)
    print("STEP env", r.stdout.strip() or r.stderr.strip()[-300:], flush=True)
    if r.returncode != 0:
        sys.exit("vLLM not importable: use the ROCm vLLM image")
    subprocess.run("rocm-smi --showproductname --showmeminfo vram | head -20", shell=True)
    allres = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(x["model"], x["kv"]) for x in allres if x.get("returncode") == 0}
    for model in MODELS:
        for kv in DTYPES:
            if (model, kv) in done:
                print("SKIP done", model, kv); continue
            print(f"STEP engine {model} kv={kv}", flush=True)
            allres.append(run_engine(model, kv))
            json.dump(allres, open(OUT, "w"), indent=1)
            print("RESULT CAPACITY", model, kv, {k: allres[-1].get(k) for k in ("kv_cache_tokens", "max_concurrency", "kv_cache_gib")}, flush=True)
    print("STEP done", flush=True)
