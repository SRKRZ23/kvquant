"""Serving-invariance fuzzer: a request's answer must not depend on what else the server is doing.

Known vLLM bugs of this class (chunked-prefill continuation next to a longer prompt: PR #67789; CUDA-graph padding
rows poisoning a per-tensor FP8 scale: issue #65734 and ~8 siblings; history-dependent outputs: #57493) pass
single-request evals and break production serving. This harness needs no labels: each probe is a needle question
whose answer the model gives correctly ALONE; the probe is then re-run inside adversarial schedules and must still
answer it. One engine per config, in-process (deterministic submission order), prefix caching on, chunked prefill on.

Schedules (all in one generate() call so they share steps):
  solo        : [p]                                   (also the reference answer)
  cont_longer : [first, p, longer]  p is split; its last `tail` tokens share a step with a longer fresh prompt
  cache_longer: warm p's prefix, then [p', longer]     p' = same prefix (cache hit) + new question tail
  pad_n       : p with n-1 long-running companions, so decode batches of size n get padded to a graph size
  hot_peers   : p with 3 companions of one repeated token (large activations for per-tensor amax coupling)
  aged        : p alone again after all of the above (history dependence)
Usage: python3 invfuzz.py <config_name> '<LLM kwargs json>' [n_probes]
Prints RESULT INVFUZZ lines; writes invfuzz_<config>.json with per-schedule failures and minimal repro specs.
"""
import json, os, random, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = os.environ.get("FUZZ_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUTDIR = os.environ.get("OUTDIR", "/shared-docker")
BUDGET = 4096


def main():
    name, kw = sys.argv[1], json.loads(sys.argv[2])
    n_probes = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True,
              max_num_batched_tokens=BUDGET, **kw)
    tok = llm.get_tokenizer()
    rng = random.Random(1234)
    head, end = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False,
                                        add_generation_prompt=True).split("@@")
    fill_words = ("The harbour was quiet and the ferries ran on time. Market stalls opened late. "
                  "A cold wind came from the north and the lamps were lit early. ")
    filler = tok.encode(fill_words * 400)
    sp = SamplingParams(temperature=0.0, max_tokens=12)

    def build(salt, text, n, question):
        pre = tok.encode(head) + salt + tok.encode(text)
        post = tok.encode(question + end)
        body = filler[: max(0, n - len(pre) - len(post))]
        return pre + body + post

    def salt():  # distinct start -> no accidental prefix-cache hits between schedules
        return tok.encode(f"Record {rng.randint(10**7, 10**8)}. ")

    def gen(prompts, params=sp):
        return [o.outputs[0].text for o in llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                                         params, use_tqdm=False)]

    long_sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    probes = []
    for i in range(n_probes):
        code = str(rng.randint(10000, 99999))
        L = [700, 1300, 1900, 2600, 3400, 1100, 2200, 3000][i % 8]
        q = f" Question: what is the access code for vault {i + 3}? Answer with the number only."
        probes.append({"i": i, "code": code, "L": L, "text": f"The access code for vault {i + 3} is {code}.", "q": q})

    fails = {}
    detail = []

    def check(sched, p, out, spec):
        ok = p["code"] in out
        fails.setdefault(sched, [0, 0]); fails[sched][1] += 1
        if not ok:
            fails[sched][0] += 1
            detail.append({"sched": sched, "probe": p["i"], "L": p["L"], "out": out[:60], "spec": spec})
        return ok

    usable = []
    for p in probes:   # solo reference: keep probes the model answers alone
        ids = build(salt(), p["text"], p["L"], p["q"])
        if check("solo", p, gen([ids])[0], {"lens": [len(ids)]}):
            usable.append(p)
    for p in usable:
        for tail in (16, 300):   # continuation + longer fresh prompt in the same step (PR #67789 trigger)
            ids = build(salt(), p["text"], p["L"], p["q"])
            first = build(salt(), "", max(64, BUDGET - (len(ids) - tail)), " Summarize.")
            longer = build(salt(), "", len(ids) + 64, " Summarize.")
            out = gen([first, ids, longer])[1]
            check(f"cont_longer_t{tail}", p, out, {"lens": [len(first), len(ids), len(longer)], "budget": BUDGET})
        s = salt()                # prefix-cache hit next to a longer fresh prompt
        warm = build(s, p["text"], p["L"], " Say OK.")
        gen([warm])
        hit = build(s, p["text"], p["L"], p["q"])
        longer = build(salt(), "", len(hit) + 64, " Summarize.")
        check("cache_longer", p, gen([hit, longer])[0], {"lens": [len(warm), len(hit), len(longer)], "prefix_shared": True})
        for n in (3, 5, 7, 13):   # decode batch of n -> padded to a captured graph size
            ids = build(salt(), p["text"], p["L"], p["q"])
            comps = [build(salt(), "", 200, " Write a long story.") for _ in range(n - 1)]
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)] + [TokensPrompt(prompt_token_ids=c) for c in comps],
                                [sp] + [long_sp] * (n - 1), use_tqdm=False)
            check(f"pad_{n}", p, outs[0].outputs[0].text, {"lens": [len(ids)] + [len(c) for c in comps]})
        ids = build(salt(), p["text"], p["L"], p["q"])  # companions of one repeated token
        hot = [[tok.encode(" zzzz")[0]] * 1500 for _ in range(3)]
        check("hot_peers", p, gen([ids] + hot)[0], {"lens": [len(ids), 1500, 1500, 1500], "peer": "repeated token"})
    for p in usable:              # history dependence after all the traffic above
        ids = build(salt(), p["text"], p["L"], p["q"])
        check("aged", p, gen([ids])[0], {"lens": [len(ids)]})

    rate = {k: f"{v[0]}/{v[1]}" for k, v in fails.items()}
    print("RESULT INVFUZZ", MODEL, name, len(usable), json.dumps(rate), flush=True)
    json.dump({"config": name, "kw": kw, "model": MODEL, "usable_probes": len(usable), "fails": fails,
               "detail": detail}, open(os.path.join(OUTDIR, f"invfuzz_{MODEL.split('/')[-1]}_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
