"""Tune PR #68834's grouped TurboQuant decode kernel for gfx942 (MI300X, wave64).

Its launch table _MSE_GQA_LAUNCH[(key_mse, mse_bits, value_bits)] = (heads_on_n, block_kv, num_warps) was tuned on H200
(warp 32). We sweep it on MI300X for turboquant_4bit_nc, Qwen2.5-7B shape (28 q heads, 4 kv heads, D 128), check every
variant against vLLM main's output and print the best per case plus the best single setting across cases.
Needs tq_pr_rocm_bench.py next to it (kernel sources and shims). Run: python3 -u tq_gfx942_tune.py
"""
import itertools, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import tq_pr_rocm_bench as T

def main():
    mod = T.MODS["pr68834"]; main_mod = T.MODS["main"]
    Hq, Hk, D = 28, 4, 128
    Pi = T.hadamard(D).to(T.dev); cent = (torch.linspace(-2.5, 2.5, 16) / math.sqrt(D)).to(T.dev)
    cases = ((1, 32768), (8, 16384), (16, 8192), (32, 8192), (64, 4096))
    grid = list(itertools.product((0, 1), (16, 32, 64, 128), (1, 2, 4, 8), (32, 64, 128)))
    default = mod._MSE_GQA_LAUNCH[(True, 4, 4)]
    table = {}
    rows = []
    for (B, L) in cases:
        c, bt, sl, kps = T.make(B, L, Hk, D)
        q = torch.randn(B, Hq, D, device=T.dev, dtype=torch.bfloat16)
        T.plat.force_cuda = False
        ref = T.call(main_mod, q, c, bt, sl, Pi, cent, kps, 32).float()
        base = T.timeit(lambda: T.call(main_mod, q, c, bt, sl, Pi, cent, kps, 32))
        T.plat.force_cuda = True
        mod._MSE_GQA_LAUNCH[(True, 4, 4)] = default
        pr_default = min(T.timeit(lambda: T.call(mod, q, c, bt, sl, Pi, cent, kps, sp)) for sp in (32, 64, 128))
        for (hn, bkv, w, sp) in grid:
            mod._MSE_GQA_LAUNCH[(True, 4, 4)] = (hn, bkv, w)
            try:
                out = T.call(mod, q, c, bt, sl, Pi, cent, kps, sp).float()
                err = ((out - ref).norm() / ref.norm()).item()
                if not (err < 2e-2 and torch.isfinite(out).all().item()):
                    continue
                us = T.timeit(lambda: T.call(mod, q, c, bt, sl, Pi, cent, kps, sp), n=20)
            except Exception:
                continue
            table.setdefault((hn, bkv, w, sp), {})[(B, L)] = us
            rows.append({"B": B, "L": L, "heads_on_n": hn, "block_kv": bkv, "warps": w, "splits": sp, "us": round(us, 1), "err": err})
        best = min((r for r in rows if r["B"] == B and r["L"] == L), key=lambda r: r["us"])
        print(f"RESULT B={B} L={L}: main {base:.0f} us | PR default table {pr_default:.0f} us ({base/pr_default:.2f}x) | "
              f"best {best['us']:.0f} us ({base/best['us']:.2f}x) at heads_on_n={best['heads_on_n']} block_kv={best['block_kv']} "
              f"warps={best['warps']} splits={best['splits']}", flush=True)
        T.plat.force_cuda = False
    # one setting for all cases: minimise the geometric mean of time / best-of-case
    full = {k: v for k, v in table.items() if len(v) == len(cases)}
    bestcase = {cs: min(v[cs] for v in full.values()) for cs in cases}
    score = {k: math.exp(sum(math.log(v[cs] / bestcase[cs]) for cs in cases) / len(cases)) for k, v in full.items()}
    top = sorted(score.items(), key=lambda kv: kv[1])[:5]
    for k, s in top:
        print("RESULT single-setting", dict(zip(("heads_on_n", "block_kv", "warps", "splits"), k)), "slowdown vs per-case best", round(s, 3), flush=True)
    json.dump({"rows": rows, "top_single": [[list(k), s] for k, s in top]}, open("tq_gfx942_tune.json", "w"), indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
