# Qwen2.5-72B, 32k-token users per GPU-dollar: MI300X vs H100 (2026-11-25)

Script: tco_72b.py. MEASURED rows are vLLM 0.27.1 on one MI300X; all other numbers are ARITH (config bytes + vLLM-like memory budget); H100 price is a parameter.

KV per 32k user: bf16 10.00 GiB, fp8 5.00 GiB, int4 (per-channel K, 27.3 %) 2.73 GiB

H100 at $2.00/h, MI300X at $1.99/h  (users = 32k-token users that fit; per $ = users / fleet $ per hour)
| weights | KV | MI300X GPUs | users | users per $/h | H100 GPUs (TP) | users | users per $/h | MI300X advantage |
|---|---|---|---|---|---|---|---|---|
| bf16 | bf16 (MEASURED in vLLM: 3.17) | 1 | 3 | 1.5 | 4 | 13 | 1.6 | 0.9x |
| bf16 | fp8 (MEASURED in vLLM: 6.34) | 1 | 6 | 3.0 | 4 | 27 | 3.4 | 0.9x |
| bf16 | int4 (per-channel K, 27.3 %) | 1 | 11 | 5.5 | 4 | 49 | 6.1 | 0.9x |
| fp8 | bf16 | 1 | 9 | 4.5 | 2 | 6 | 1.5 | 3.0x |
| fp8 | fp8 | 1 | 19 | 9.5 | 2 | 13 | 3.2 | 2.9x |
| fp8 | int4 (per-channel K, 27.3 %) | 1 | 35 | 17.6 | 2 | 23 | 5.8 | 3.1x |

H100 at $2.50/h, MI300X at $1.99/h  (users = 32k-token users that fit; per $ = users / fleet $ per hour)
| weights | KV | MI300X GPUs | users | users per $/h | H100 GPUs (TP) | users | users per $/h | MI300X advantage |
|---|---|---|---|---|---|---|---|---|
| bf16 | bf16 (MEASURED in vLLM: 3.17) | 1 | 3 | 1.5 | 4 | 13 | 1.3 | 1.2x |
| bf16 | fp8 (MEASURED in vLLM: 6.34) | 1 | 6 | 3.0 | 4 | 27 | 2.7 | 1.1x |
| bf16 | int4 (per-channel K, 27.3 %) | 1 | 11 | 5.5 | 4 | 49 | 4.9 | 1.1x |
| fp8 | bf16 | 1 | 9 | 4.5 | 2 | 6 | 1.2 | 3.8x |
| fp8 | fp8 | 1 | 19 | 9.5 | 2 | 13 | 2.6 | 3.7x |
| fp8 | int4 (per-channel K, 27.3 %) | 1 | 35 | 17.6 | 2 | 23 | 4.6 | 3.8x |

H100 at $3.00/h, MI300X at $1.99/h  (users = 32k-token users that fit; per $ = users / fleet $ per hour)
| weights | KV | MI300X GPUs | users | users per $/h | H100 GPUs (TP) | users | users per $/h | MI300X advantage |
|---|---|---|---|---|---|---|---|---|
| bf16 | bf16 (MEASURED in vLLM: 3.17) | 1 | 3 | 1.5 | 4 | 13 | 1.1 | 1.4x |
| bf16 | fp8 (MEASURED in vLLM: 6.34) | 1 | 6 | 3.0 | 4 | 27 | 2.2 | 1.3x |
| bf16 | int4 (per-channel K, 27.3 %) | 1 | 11 | 5.5 | 4 | 49 | 4.1 | 1.4x |
| fp8 | bf16 | 1 | 9 | 4.5 | 2 | 6 | 1.0 | 4.5x |
| fp8 | fp8 | 1 | 19 | 9.5 | 2 | 13 | 2.2 | 4.4x |
| fp8 | int4 (per-channel K, 27.3 %) | 1 | 35 | 17.6 | 2 | 23 | 3.8 | 4.6x |

## Reading it honestly
- With bf16 weights the MI300X advantage per dollar is small (0.9x-1.4x depending on the H100 price): 4 H100 in TP4
  hold as much memory as ~2 MI300X. The old headline "1 MI300X vs 4 H100" counted GPUs, not dollars. Do not use it.
- The real gap opens with fp8 weights (the normal way to serve 72B today): 72B fp8 fits ONE MI300X with ~100 GiB left
  for KV, while H100 needs TP2 and has ~65 GiB left across two cards. MI300X serves 2.9x-4.6x more 32k users per dollar,
  for every KV format. 4-bit KV multiplies the absolute count (19 -> 35 users on one MI300X vs fp8 KV).
- The 19 users measured in transformers used the whole 192 GB with no serving overhead; inside vLLM's budget the same
  bf16-weight + int4-KV setup is ~11 users.
- This counts users that fit (capacity). It says nothing about tokens/s per user: H100 TP4 spends 4 GPUs of compute on
  each token. A throughput-per-dollar comparison needs a measured vLLM run of 72B fp8 on both.
- Needed to promote ARITH -> MEASURED: vLLM 72B with fp8 weights on one MI300X (fp8 KV and 4-bit KV), ~1 h droplet.
