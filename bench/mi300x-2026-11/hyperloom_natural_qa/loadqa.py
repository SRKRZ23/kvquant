"""Load-aware accuracy check for KV-cache formats and attention backends in vLLM.

Why: on MI300X two quantized-KV backends (TurboQuant, UltraQuant) lost up to 25 % F1 only when the server was busy
(a chunked-prefill continuation or prefix-cache hit batched next to a longer new prompt), while GSM8K and a
low-concurrency long-context check passed. Those checks run the code paths that work. This one runs each candidate
under the conditions that exercise the risky paths and compares it PAIRED against the baseline on the same prompts:

  c1     : 1 request at a time                         (the path every simple check already covers)
  c64    : 64 concurrent requests                       (mixed prefill/decode batches)
  chunk  : 64 concurrent, --max-num-batched-tokens 2048 (every long prompt split; continuations share steps)
  pcache : prefix caching on, every prompt sent twice interleaved, 64 concurrent (cache hits next to cold prompts)

A candidate that passes c1 but regresses in another scenario has a load-dependent bug.
Usage:
  python3 loadqa.py run --model Qwen/Qwen2.5-7B-Instruct --set set.jsonl --out loadqa_out \
      --candidates "turboquant_4bit_nc" "fp8|--kv-cache-dtype-skip-layers 0" --scenarios c1 pcache
  python3 loadqa.py report --set set.jsonl --out loadqa_out
Each candidate is "<kv-cache-dtype>[|extra vllm serve args]"; the baseline is "auto" (bf16 KV).
"""
from __future__ import annotations

import argparse, json, os, shlex, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural_qa as nq

SCENARIOS = {
    "c1": {"concurrency": 1, "serve": ["--no-enable-prefix-caching"], "repeat": False},
    "c64": {"concurrency": 64, "serve": ["--no-enable-prefix-caching"], "repeat": False},
    "chunk": {"concurrency": 64, "serve": ["--no-enable-prefix-caching", "--max-num-batched-tokens", "2048"],
              "repeat": False},
    "pcache": {"concurrency": 64, "serve": ["--enable-prefix-caching"], "repeat": True},
}


def repeated_set(set_path: Path, out: Path) -> Path:
    """Each prompt twice; the second copy goes two places after the first, so warm and cold requests overlap."""
    items = [json.loads(l) for l in open(set_path)]
    rows = []
    for i, it in enumerate(items):
        rows.append(dict(it, id=it["id"] + "#a"))
        if i >= 2:
            rows.append(dict(items[i - 2], id=items[i - 2]["id"] + "#b"))
    rows += [dict(it, id=it["id"] + "#b") for it in items[-2:]]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return out


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 1200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server not ready")


def serve_and_answer(model: str, kv_spec: str, scenario: str, set_path: Path, out_dir: Path, port: int) -> Path:
    kv, _, extra = kv_spec.partition("|")
    sc = SCENARIOS[scenario]
    tag = f"{scenario}__{kv_spec.replace('|', '_').replace(' ', '').replace('-', '')}"
    answers = out_dir / f"{tag}.jsonl"
    if answers.exists():
        return answers
    cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
           "--kv-cache-dtype", kv, *sc["serve"], *shlex.split(extra)]
    log = open(out_dir / f"{tag}.server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        wait_ready(url, proc)
        nq.answer(set_path, url, model, answers, concurrency=sc["concurrency"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(5)
    return answers


def run(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = repeated_set(Path(a.set), out / "set_repeated.jsonl")
    for sc in a.scenarios:
        set_path = rep if SCENARIOS[sc]["repeat"] else Path(a.set)
        for kv in ["auto", *a.candidates]:
            p = serve_and_answer(a.model, kv, sc, set_path, out, a.port)
            print("ANSWERED", sc, kv, p.name, flush=True)
    report(a)


def report(a) -> dict:
    out = Path(a.out)
    rows = {}
    for sc in a.scenarios:
        set_path = out / "set_repeated.jsonl" if SCENARIOS[sc]["repeat"] else Path(a.set)
        base = out / f"{sc}__auto.jsonl"
        for f in sorted(out.glob(f"{sc}__*.jsonl")):
            if f == base or f.name.endswith(".server.log"):
                continue
            cand = f.stem.split("__", 1)[1]
            r = nq.compare(set_path, base, f)
            rows.setdefault(cand, {})[sc] = r
    print(f"{'candidate':42s} " + " ".join(f"{sc:>24s}" for sc in a.scenarios))
    verdicts = {}
    for cand, per in rows.items():
        cells, worst = [], 0.0
        for sc in a.scenarios:
            r = per.get(sc)
            if not r:
                cells.append(f"{'-':>24s}"); continue
            d = min(v["diff"] for v in r.values())
            reg = any(v["regression"] for v in r.values())
            worst = min(worst, d)
            cells.append(f"{('REGRESS ' if reg else 'ok ') + f'{d:+.3f}':>24s}")
        c1_ok = "c1" in per and not any(v["regression"] for v in per["c1"].values())
        load_reg = any(any(v["regression"] for v in per[sc].values()) for sc in per if sc != "c1")
        verdicts[cand] = ("LOAD-DEPENDENT REGRESSION" if c1_ok and load_reg else
                          "REGRESSION" if load_reg or not c1_ok else "PASS")
        print(f"{cand:42s} " + " ".join(cells) + f"   {verdicts[cand]}")
    json.dump({"rows": rows, "verdicts": verdicts}, open(out / "loadqa_report.json", "w"), indent=1)
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True); p.add_argument("--out", required=True)
        p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
        p.add_argument("--candidates", nargs="*", default=[])
        p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
