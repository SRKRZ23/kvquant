"""Users with 32k-token contexts per GPU-dollar for Qwen2.5-72B: MI300X vs H100, by weight and KV format.

Evidence tiers (print them with every number):
  MEASURED  vLLM 0.27.1 on one MI300X today: bf16 weights, bf16 KV 3.17 users, fp8 KV 6.34 users (KV pool 31.7 GiB)
  ARITH     everything else: bytes from the model config, vLLM-like memory budget (0.90 utilization minus the
            overhead vLLM actually reserved on MI300X: 172.5 - 135.4 - 31.7 = 5.4 GiB; 4 GiB assumed per H100)
  RATE      $/GPU-hour: MI300X $1.99 (our AMD Developer Cloud invoice); H100 is a parameter, not a claim
"""
import math

GIB = 2 ** 30
L, KVH, HD, CTX = 80, 8, 128, 32768
KV_BF16 = 2 * L * KVH * HD * 2 * CTX / GIB            # 10.0 GiB per 32k user
KV = {"bf16": KV_BF16, "fp8": KV_BF16 / 2, "int4 (per-channel K, 27.3 %)": KV_BF16 * 0.273}
W = {"bf16": 135.4, "fp8": 70.0}                      # GiB; fp8 weights ~ half plus bf16 embeddings/norms (ARITH)
GPU = {"MI300X": (191.7, 5.4, 1.99), "H100": (79.6, 4.0, None)}


def users(gpu, n, w, kv):
    hbm, ovh, _ = GPU[gpu]
    free = n * (hbm * 0.90 - ovh) - W[w]
    return max(0, math.floor(free / KV[kv]))


def best(gpu, w, kv, max_tp=8):
    for n in (1, 2, 4, 8):
        if n <= max_tp and users(gpu, n, w, kv) > 0:
            return n, users(gpu, n, w, kv)
    return None, 0


if __name__ == "__main__":
    print(f"KV per 32k user: " + ", ".join(f"{k} {v:.2f} GiB" for k, v in KV.items()))
    for h100_price in (2.0, 2.5, 3.0):
        print(f"\nH100 at ${h100_price:.2f}/h, MI300X at $1.99/h  (users = 32k-token users that fit; per $ = users / fleet $ per hour)")
        print("| weights | KV | MI300X GPUs | users | users per $/h | H100 GPUs (TP) | users | users per $/h | MI300X advantage |")
        print("|---|---|---|---|---|---|---|---|---|")
        for w in W:
            for kv in KV:
                n1, u1 = best("MI300X", w, kv); n2, u2 = best("H100", w, kv)
                tag = " (MEASURED in vLLM: %s)" % {"bf16": "3.17", "fp8": "6.34"}[kv] if (w == "bf16" and kv in ("bf16", "fp8")) else ""
                p1 = u1 / (n1 * 1.99) if n1 else 0; p2 = u2 / (n2 * h100_price) if n2 else 0
                adv = f"{p1 / p2:.1f}x" if p2 else "n/a"
                print(f"| {w} | {kv}{tag} | {n1} | {u1} | {p1:.1f} | {n2} | {u2} | {p2:.1f} | {adv} |")
