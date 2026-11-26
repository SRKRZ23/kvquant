"""Kill test: does AMD Hyperloom's quality gate accept KV-cache settings that lose accuracy on natural long-context QA?

Hyperloom keeps a change if its GSM8K round and its long-context check hold. The long-context check is RULER-style
(niah_single, niah_multikey, variable_tracking, common_words; 4 prompts each; substring grading; /v1/completions, greedy,
128 new tokens). Its generator is vendored below unchanged (MIT, (c) 2026 AMD, Hyperloom src/hyperloom/measure/long_context/
tasks.py, generator "hl-long-context/2", default seed 20261024).

For each KV setting we start a vLLM server and run: the Hyperloom long-context set at 3,500 and 32,640 tokens, and a 5-shot
GSM8K (N=250). Our LongBench F1 for the same settings comes from vllm_bench.py runs on the same image.
Blind spot = a setting that loses >=10 % on LongBench multi-hop but stays within noise on both Hyperloom checks.
Run inside the vLLM ROCm container:  python3 -u hl_gate_test.py
"""
import concurrent.futures as cf, json, os, re, subprocess, sys, time, types, urllib.request
from dataclasses import dataclass

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
ARMS = os.environ.get("ARMS", "auto,turboquant_4bit_nc,int4_per_token_head_skipb,fp8,turboquant_k8v4").split(",")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "3500,32640").split(",")]
SEED, NG = 20261024, int(os.environ.get("NG", "250"))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hl_gate_result.json")

# ---------------------------------------------------------------- vendored Hyperloom generator (shimmed imports)
pm = types.ModuleType("hyperloom.measure.long_context.prompts")
pm.PER_TASK = 4; pm.TASKS = ("niah_single", "niah_multikey", "variable_tracking", "common_words")
class PromptSetError(Exception): pass
@dataclass(frozen=True)
class Prompt:
    id: str; task: str; text: str; references: tuple; tokens: int
pm.Prompt, pm.PromptSetError = Prompt, PromptSetError
for n in ("hyperloom", "hyperloom.measure", "hyperloom.measure.long_context"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["hyperloom.measure.long_context.prompts"] = pm
TASKS_SRC = '# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.\n# SPDX-License-Identifier: MIT\n\n"""The generators of the four tasks, RULER-style, built offline from a seed and a tokenizer.\n\n* ``niah_single``: one needle (a magic number for a key) somewhere in a haystack of repeated filler sentences.\n* ``niah_multikey``: four needles for four keys; the question names one, so the others are distractors.\n* ``variable_tracking``: two chains of variable assignments (``VAR B = VAR A``) spread over the filler; the question asks\n  for the variables that hold the value at the start of one chain.\n* ``common_words``: a numbered list of words, five of which appear far more often than the rest; the question asks for them.\n\nEach prompt is made as long as the tokenizer allows within ``length`` tokens (and within 1% of it), by searching for the\nnumber of filler units; needle depths are spread evenly over the prompts of a task so every depth is sampled. All of the\nrandomness comes from the seed, so the same tokenizer, length and seed give the same prompts.\n"""\n\nfrom __future__ import annotations\n\nimport math\nimport random\nfrom collections.abc import Callable, Sequence\nfrom typing import NamedTuple\n\nfrom hyperloom.measure.long_context.prompts import PER_TASK, TASKS, Prompt, PromptSetError\n\n__all__ = ["build_prompts"]\n\nEncode = Callable[[str], Sequence[int]]\n_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."\n_SYLLABLES = tuple(c + v for c in "bdfgklmnprstvz" for v in "aeiou")\n_SLACK = 0.01\n_FIT_ROUNDS = 8\n_PROBE_UNITS = 16\n_NEEDLE = "One of the special magic numbers for {key} is: {value}."\n_NUMBER_DIGITS = 7\n_CHAIN_HOPS = 4\n#: The share of the way through a prompt its first chain statement is at most, and a chain\'s last one at most ``_CHAIN_END``.\n_CHAIN_START_SHARE = 0.5\n_CHAIN_END = 0.95\n_POOL_MARGIN = 8\n_COMMON_WORDS = 5\n_COMMON_FREQ = 30\n_UNCOMMON_FREQ = 3\n_MULTIKEY_NEEDLES = 4\n\n\nclass _Made(NamedTuple):\n    """One prompt made by a task: its text, the tokens the tokenizer counted for it and the references an answer holds."""\n\n    text: str\n    tokens: int\n    references: list[str]\n\n\nclass _Chain(NamedTuple):\n    """One chain of assignments: its statements with their depths, its variables and the value they come to."""\n\n    statements: list[tuple[float, str]]\n    names: list[str]\n    value: str\n\n\n_MAGIC_QUESTION = "What is the special magic number for {key} mentioned in the provided text?"\n_MAGIC_ANSWER = "The special magic number for {key} mentioned in the provided text is"\n_MAGIC_INTRO = "Some special magic numbers are hidden within the following text. Make sure to memorize them."\n\n\ndef _words(rng: random.Random, count: int, exclude: frozenset[str] = frozenset()) -> list[str]:\n    """``count`` distinct pronounceable made-up words, none of them in ``exclude``."""\n    syllables = max(3, math.ceil(math.log(_POOL_MARGIN * count + 1, len(_SYLLABLES))))\n    found: dict[str, None] = {}\n    while len(found) < count:\n        word = "".join(rng.choice(_SYLLABLES) for _ in range(syllables))\n        if word not in exclude:\n            found[word] = None\n    return list(found)\n\n\ndef _number(rng: random.Random) -> str:\n    return str(rng.randrange(10 ** (_NUMBER_DIGITS - 1), 10**_NUMBER_DIGITS))\n\n\ndef _fit(encode: Encode, render: Callable[[int], str], length: int, what: str) -> tuple[str, int]:\n    """The longest ``render(units)`` of at most ``length`` tokens, and no more than 1% under it; ``render`` grows with ``units``."""\n    base = len(encode(render(_PROBE_UNITS)))\n    per_unit = (len(encode(render(2 * _PROBE_UNITS))) - base) / _PROBE_UNITS\n    if per_unit <= 0:\n        raise PromptSetError(f"{what}: a longer text does not give more tokens")\n    units = max(1, _PROBE_UNITS + math.floor((length - base) / per_unit))\n    for _ in range(_FIT_ROUNDS):\n        text = render(units)\n        tokens = len(encode(text))\n        if length * (1 - _SLACK) <= tokens <= length:\n            return text, tokens\n        units = max(1, units + math.floor((length - tokens) / per_unit))\n    raise PromptSetError(f"{what}: no text of {length} tokens (within 1%) was found for this tokenizer")\n\n\ndef _haystack(units: int, needles: Sequence[tuple[float, str]]) -> str:\n    """``units`` filler sentences with each ``(depth, text)`` of ``needles`` inserted at that share of the way through."""\n    sentences = [_FILLER] * units\n    for depth, text in sorted(needles, key=lambda needle: -needle[0]):\n        sentences.insert(min(units, int(depth * units)), text)\n    return " ".join(sentences)\n\n\ndef _depth(index: int) -> float:\n    """The depth of prompt ``index``: evenly spread over the prompts of a task, never at an end."""\n    return (index + 0.5) / PER_TASK\n\n\ndef _magic(rng: random.Random, encode: Encode, length: int, index: int, others: int) -> _Made:\n    """A needle prompt: the asked-for needle at the prompt\'s own depth and ``others`` distractors at random depths."""\n    keys = [f"{a}-{b}" for a, b in zip(_words(rng, others + 1), _words(rng, others + 1))]\n    values = [_number(rng) for _ in keys]\n    needles = [(_depth(index), _NEEDLE.format(key=keys[0], value=values[0]))]\n    needles += [(rng.random(), _NEEDLE.format(key=k, value=v)) for k, v in zip(keys[1:], values[1:])]\n    question = f"{_MAGIC_QUESTION.format(key=keys[0])}\\nAnswer: {_MAGIC_ANSWER.format(key=keys[0])}"\n    text, tokens = _fit(encode, lambda n: f"{_MAGIC_INTRO}\\n{_haystack(n, needles)}\\n{question}", length, "needle task")\n    return _Made(text, tokens, [values[0]])\n\n\ndef _niah_single(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:\n    return _magic(rng, encode, length, index, 0)\n\n\ndef _niah_multikey(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:\n    return _magic(rng, encode, length, index, _MULTIKEY_NEEDLES - 1)\n\n\ndef _chain(rng: random.Random, start: float) -> _Chain:\n    """The statements of one chain of assignments, spread from depth ``start`` on, its variables and its value."""\n    names = _words(rng, _CHAIN_HOPS)\n    value = _number(rng)\n    spread = (_CHAIN_END - start) / _CHAIN_HOPS\n    statements = [f"VAR {names[0].upper()} = {value}"] + [\n        f"VAR {names[i].upper()} = VAR {names[i - 1].upper()}" for i in range(1, _CHAIN_HOPS)\n    ]\n    return _Chain([(start + i * spread, f"{s}.") for i, s in enumerate(statements)], [n.upper() for n in names], value)\n\n\ndef _variable_tracking(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:\n    target = _chain(rng, _depth(index) * _CHAIN_START_SHARE)\n    distractor = _chain(rng, rng.random() * _CHAIN_START_SHARE)\n    value = target.value\n    intro = "Memorize and track the chains of variable assignment hidden in the following text."\n    question = (\n        f"Question: Find all variables that are assigned the value {value} in the text above.\\n"\n        f"Answer: According to the chains of variable assignments in the text above, the variables that are assigned "\n        f"the value {value} are:"\n    )\n    needles = target.statements + distractor.statements\n    text, tokens = _fit(encode, lambda n: f"{intro}\\n{_haystack(n, needles)}\\n{question}", length, "tracking task")\n    return _Made(text, tokens, target.names)\n\n\ndef _common_words(rng: random.Random, encode: Encode, length: int, index: int) -> _Made:\n    common = _words(rng, _COMMON_WORDS)\n    rare = _words(rng, length // _UNCOMMON_FREQ + _COMMON_FREQ, exclude=frozenset(common))\n    seed = rng.random()\n\n    def render(rare_lines: int) -> str:\n        pool = [w for w in common for _ in range(_COMMON_FREQ)]\n        pool += [w for w in rare for _ in range(_UNCOMMON_FREQ)][:rare_lines]\n        random.Random(seed).shuffle(pool)\n        listing = "\\n".join(f"{i}. {word}" for i, word in enumerate(pool, 1))\n        return (\n            "Below is a numbered list of words. In these words, some appear more often than others. "\n            f"Memorize the ones that appear most often.\\n{listing}\\n"\n            f"Question: What are the {_COMMON_WORDS} most common words in the above list?\\n"\n            f"Answer: The top {_COMMON_WORDS} words that appear most often in the list are:"\n        )\n\n    text, tokens = _fit(encode, render, length, "word-list task")\n    return _Made(text, tokens, common)\n\n\n_TASKS = {\n    "niah_single": _niah_single,\n    "niah_multikey": _niah_multikey,\n    "variable_tracking": _variable_tracking,\n    "common_words": _common_words,\n}\n\n\ndef build_prompts(encode: Encode, *, length: int, seed: int) -> tuple[Prompt, ...]:\n    """The ``PER_TASK`` prompts of each task for a tokenizer\'s ``encode`` (text to ids, no special tokens), each of at most\n    ``length`` tokens, and a seed.\n\n    Raises:\n        PromptSetError: The tokenizer gives no text of that length within 1%, or the length is too short for a task.\n    """\n    prompts = []\n    for task in TASKS:\n        rng = random.Random(f"{seed}:{task}:{length}")\n        for index in range(PER_TASK):\n            made = _TASKS[task](rng, encode, length, index)\n            prompts.append(Prompt(f"{task}-{index:02d}", task, made.text, tuple(made.references), made.tokens))\n    return tuple(prompts)\n'
hl = types.ModuleType("hl_tasks"); exec(compile(TASKS_SRC, "hyperloom_tasks.py", "exec"), hl.__dict__)

def passes(text, refs):
    t = text.lower(); return bool(refs) and all(r.lower() in t for r in refs)

# ---------------------------------------------------------------- server
def start(kv):
    extra, kvd = [], kv
    if kvd.endswith("_skipb"):
        from transformers import AutoConfig
        L = AutoConfig.from_pretrained(MODEL).num_hidden_layers
        extra = ["--kv-cache-dtype-skip-layers", "0", "1", str(L - 2), str(L - 1)]; kvd = kvd[:-6]
    cmd = ["vllm", "serve", MODEL, "--max-model-len", "32768", "--kv-cache-dtype", kvd, "--gpu-memory-utilization", "0.90",
           "--port", "8000", "--no-enable-prefix-caching"] + extra
    log = open(os.path.join(HERE, f"hl_server_{kv}.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        time.sleep(2)
        if p.poll() is not None: raise RuntimeError(f"server exited {p.returncode}")
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2); return p
        except Exception: pass
    p.kill(); raise RuntimeError("server not ready")

def complete(prompt, max_tokens, stop=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_p": 1}
    if stop: body["stop"] = stop
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["text"]

# ---------------------------------------------------------------- GSM8K
def gsm8k():
    base = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
    get = lambda f: [json.loads(l) for l in urllib.request.urlopen(base + f).read().decode().splitlines() if l.strip()]
    return get("train.jsonl")[:5], get("test.jsonl")[:NG]

def num(s):
    m = re.search(r"####\s*([-\d,\.]+)", s) or None
    if m: s2 = m.group(1)
    else:
        ns = re.findall(r"-?\d[\d,]*\.?\d*", s); s2 = ns[-1] if ns else ""
    return s2.replace(",", "").rstrip(".")

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = lambda t: tok(t, add_special_tokens=False).input_ids
    sets = {L: hl.build_prompts(enc, length=L, seed=SEED) for L in LENGTHS}
    print("built", {L: len(s) for L, s in sets.items()}, flush=True)
    shots, test = gsm8k()
    fs = "".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in shots)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for kv in ARMS:
        if kv in res and "error" not in res[kv]: print("SKIP", kv); continue
        t0 = time.time(); r = {}; p = None
        try:
            p = start(kv)
            with cf.ThreadPoolExecutor(8) as ex:
                for L, ps in sets.items():
                    outs = list(ex.map(lambda q: complete(q.text, 128), ps))
                    per = {}
                    for q, o in zip(ps, outs):
                        per.setdefault(q.task, []).append(passes(o, q.references))
                    r[f"hl_{L}"] = {t: sum(v) / len(v) for t, v in per.items()}
                    r[f"hl_{L}_mean"] = sum(sum(v) for v in per.values()) / len(ps)
                    print("RESULT HL", kv, L, r[f"hl_{L}_mean"], r[f"hl_{L}"], flush=True)
                outs = list(ex.map(lambda x: complete(fs + f"Question: {x['question']}\nAnswer:", 320, ["\n\nQuestion:"]), test))
            r["gsm8k"] = sum(num(o) == num(x["answer"]) for o, x in zip(outs, test)) / len(test)
            print("RESULT GSM8K", kv, r["gsm8k"], flush=True)
            p.terminate(); p.wait(60)
        except Exception as e:
            r["error"] = repr(e)[:300]; print("ARM_FAILED", kv, r["error"], flush=True)
            if p is not None and p.poll() is None:
                p.kill(); p.wait(60)
        r["wall_s"] = round(time.time() - t0); res[kv] = r
        json.dump(res, open(OUT, "w"), indent=1); time.sleep(5)
    print("STEP done", flush=True)

if __name__ == "__main__":
    main()
