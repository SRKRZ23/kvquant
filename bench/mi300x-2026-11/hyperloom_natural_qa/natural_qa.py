"""Natural-text long-context QA check, proposed for AMD Hyperloom next to its RULER-style long-context set.

Why: on MI300X, vLLM 0.27.1, Qwen2.5-7B, turboquant_4bit_nc passes Hyperloom's GSM8K round and its synthetic long-context
check (it even scores above bf16 at 32k) while losing 12-16 % on LongBench hotpotqa / 2wikimqa. Synthetic retrieval and
tracking tasks do not exercise reasoning over several relevant passages, and 4 prompts per task cannot resolve a 10-15 %
change. This check uses frozen natural QA prompts and reports a PAIRED comparison with a bootstrap interval, so a
candidate is judged against the baseline on the same questions.

Three steps, all offline except `answer`:
  build   : a frozen prompt set from LongBench (multifieldqa_en, hotpotqa, 2wikimqa by default), chat-templated with the
            model's tokenizer, middle-truncated to a token budget, one JSON object per line
  answer  : greedy completions from an OpenAI-compatible server (/v1/completions, temperature 0)
  compare : token-F1 per prompt, mean per task, and the paired difference candidate - baseline with a 95 % bootstrap CI
Usage:
  python3 natural_qa.py build  --model Qwen/Qwen2.5-7B-Instruct --length 3500 --per-task 100 --out set.jsonl
  python3 natural_qa.py answer --set set.jsonl --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --out base.jsonl
  python3 natural_qa.py compare --set set.jsonl --baseline base.jsonl --candidate cand.jsonl
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, random, re, string, sys, urllib.request, zipfile
from collections import Counter
from pathlib import Path

TASKS = ("multifieldqa_en", "hotpotqa", "2wikimqa")
INSTR = "Answer the question based only on the text above. Give only the answer, no other words.\nQuestion: "
LONGBENCH_ZIP = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"


# ------------------------------------------------------------------------------------------------- grading
def normalize(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, refs: list[str]) -> float:
    """Best token F1 over the references (LongBench's QA metric)."""
    return max((token_f1(pred, r) for r in refs), default=0.0)


def paired_ci(base: list[float], cand: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (cand - base) over the same prompts and its 95 % bootstrap interval."""
    assert len(base) == len(cand) and base
    d = [c - b for b, c in zip(base, cand)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return sum(d) / len(d), means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


# ------------------------------------------------------------------------------------------------- build
def build(model: str, length: int, per_task: int, out: Path, tasks=TASKS, zip_path: Path | None = None) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    zp = zip_path or Path("longbench_data.zip")
    if not zp.exists():
        urllib.request.urlretrieve(LONGBENCH_ZIP, zp)
    n = 0
    with zipfile.ZipFile(zp) as zf, open(out, "w") as f:
        for task in tasks:
            name = next(x for x in zf.namelist() if x.endswith(f"/{task}.jsonl") or x == f"{task}.jsonl")
            rows = [json.loads(l) for l in zf.read(name).decode().splitlines() if l.strip()][:per_task]
            for i, row in enumerate(rows):
                ids = tok(row["context"]).input_ids
                ctx = row["context"]
                if len(ids) > length:
                    h = length // 2
                    ctx = tok.decode(ids[:h] + ids[-h:], skip_special_tokens=True)
                msgs = [{"role": "user", "content": ctx + "\n\n" + INSTR + row["input"] + "\nAnswer:"}]
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                f.write(json.dumps({"id": f"{task}-{i:03d}", "task": task, "input": text, "outputs": row["answers"]},
                                   ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
    return n


# ------------------------------------------------------------------------------------------------- answer
def complete(url: str, model: str, prompt: str, max_tokens: int = 64, timeout: float = 600) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_p": 1}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["text"]


def answer(set_path: Path, url: str, model: str, out: Path, concurrency: int = 8) -> int:
    items = [json.loads(l) for l in open(set_path)]
    with cf.ThreadPoolExecutor(concurrency) as ex:
        texts = list(ex.map(lambda it: complete(url, model, it["input"]), items))
    with open(out, "w") as f:
        for it, t in zip(items, texts):
            f.write(json.dumps({"id": it["id"], "text": t}, ensure_ascii=False) + "\n")
    return len(items)


# ------------------------------------------------------------------------------------------------- compare
def per_prompt(set_path: Path, answers: Path) -> dict[str, tuple[str, float]]:
    refs = {it["id"]: (it["task"], it["outputs"]) for it in map(json.loads, open(set_path))}
    got = {a["id"]: a["text"] for a in map(json.loads, open(answers))}
    missing = set(refs) - set(got)
    if missing:
        raise SystemExit(f"{answers}: {len(missing)} prompts unanswered")
    return {i: (t, score(got[i].split("<|im_end|>")[0].strip(), r)) for i, (t, r) in refs.items()}


def compare(set_path: Path, baseline: Path, candidate: Path) -> dict:
    b, c = per_prompt(set_path, baseline), per_prompt(set_path, candidate)
    report = {}
    for task in sorted({t for t, _ in b.values()}):
        ids = sorted(i for i, (t, _) in b.items() if t == task)
        bs, cs = [b[i][1] for i in ids], [c[i][1] for i in ids]
        d, lo, hi = paired_ci(bs, cs)
        report[task] = {"n": len(ids), "baseline_f1": round(sum(bs) / len(bs), 4), "candidate_f1": round(sum(cs) / len(cs), 4),
                        "diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "regression": hi < 0}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--model", required=True); b.add_argument("--length", type=int, default=3500)
    b.add_argument("--per-task", type=int, default=100); b.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("answer"); a.add_argument("--set", type=Path, required=True); a.add_argument("--url", required=True)
    a.add_argument("--model", required=True); a.add_argument("--out", type=Path, required=True); a.add_argument("--concurrency", type=int, default=8)
    c = sub.add_parser("compare"); c.add_argument("--set", type=Path, required=True)
    c.add_argument("--baseline", type=Path, required=True); c.add_argument("--candidate", type=Path, required=True)
    x = ap.parse_args(argv)
    if x.cmd == "build":
        print(build(x.model, x.length, x.per_task, x.out), "prompts")
    elif x.cmd == "answer":
        print(answer(x.set, x.url, x.model, x.out, x.concurrency), "answers")
    else:
        print(json.dumps(compare(x.set, x.baseline, x.candidate), indent=1))


if __name__ == "__main__":
    sys.exit(main())
