"""servecheck: autonomous serving-invariance check for vLLM (one GPU, no labels, ~5 min per configuration).

For every (model, configuration) it starts a fresh in-process vLLM engine and asks needle questions that the model
answers correctly alone, then asks them again inside crafted schedules (chunked-prefill continuation next to a longer
prompt, prefix-cache hit, decode batches of 3/5/7/13 padded to CUDA-graph sizes, extreme-activation peers, aged
engine). Any changed answer is a batch-dependent bug. Writes one JSON per run, a markdown report, and exits non-zero
if any configuration fails while the bf16 reference of the same model is clean (CI-friendly).

  python3 servecheck.py --models Qwen/Qwen2.5-7B-Instruct --configs bf16,fp8 --outdir out
Configurations are named presets (see PRESETS) or NAME='{"kv_cache_dtype": ...}' JSON.
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = {
    "bf16": {},
    "bf16_eager": {"enforce_eager": True},
    "fp8kv": {"kv_cache_dtype": "fp8"},
    "fp8kv_skip0": {"kv_cache_dtype": "fp8", "kv_cache_dtype_skip_layers": ["0"]},
    "fp8w_online": {"quantization": "fp8"},
    "tq4": {"kv_cache_dtype": "turboquant_4bit_nc"},
    "uq4": {"kv_cache_dtype": "ultraquant_4bit"},
    "triton": {"attention_backend": "TRITON_ATTN"},
}


def run_one(model, name, kw, outdir, probes):
    env = dict(os.environ, FUZZ_MODEL=model, OUTDIR=outdir, PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "servecheck_probe.py"), name, json.dumps(kw), str(probes)],
                       capture_output=True, text=True, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT INVFUZZ")), None)
    err = "" if line else (p.stderr.strip().splitlines() or ["no output"])[-1][:200]
    print(line or f"RESULT INVFUZZ {model} {name} ERROR {err}", f"({round(time.time() - t0)} s)", flush=True)
    open(os.path.join(outdir, f"log_{model.split('/')[-1]}_{name}.txt"), "w").write(p.stdout[-20000:] + p.stderr[-20000:])
    return bool(line)


def report(outdir):
    rows, bad = [], []
    for f in sorted(glob.glob(os.path.join(outdir, "invfuzz_*.json"))):
        d = json.load(open(f))
        fails = {k: v[0] for k, v in d["fails"].items() if k != "solo"}
        tot = sum(fails.values())
        rows.append((d["model"].split("/")[-1], d["config"], d["usable_probes"], fails, tot))
    ref = {m: t for m, c, u, f, t in rows if c == "bf16"}
    lines = ["| model | config | usable probes | failures by schedule | verdict |", "|---|---|---|---|---|"]
    for m, c, u, f, t in rows:
        nz = ", ".join(f"{k} {v}" for k, v in f.items() if v) or "none"
        if c == "bf16":
            verdict = "REFERENCE" + (" (not clean!)" if t else "")
        elif ref.get(m, 1) == 0 and t:
            verdict = "FAIL"; bad.append(f"{m} {c}")
        else:
            verdict = "clean" if not t else "inconclusive (reference not clean)"
        lines.append(f"| {m} | {c} | {u} | {nz} | {verdict} |")
    md = "# servecheck report\n\n" + "\n".join(lines) + "\n"
    open(os.path.join(outdir, "report.md"), "w").write(md)
    print(md, flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--configs", default="bf16,fp8kv_skip0,tq4")
    ap.add_argument("--outdir", default="servecheck_out")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if not a.report_only:
        for model in a.models.split(","):
            for c in a.configs.split(","):
                name, kw = (c.split("=", 1)[0], json.loads(c.split("=", 1)[1])) if "=" in c else (c, PRESETS[c])
                if os.path.exists(os.path.join(a.outdir, f"invfuzz_{model.split('/')[-1]}_{name}.json")):
                    print("SKIP done", model, name, flush=True); continue
                run_one(model, name, kw, a.outdir, a.probes)
    bad = report(a.outdir)
    print("SERVECHECK", "FAIL " + "; ".join(bad) if bad else "PASS", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
