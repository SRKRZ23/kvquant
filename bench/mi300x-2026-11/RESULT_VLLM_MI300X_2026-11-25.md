# vLLM KV-cache dtypes on one AMD MI300X: capacity, speed, LongBench quality (2026-11-25)

Paid run on AMD Developer Cloud (DigitalOcean droplet, 1x MI300X 192 GB, $1.99/h), image
`vllm/vllm-openai-rocm:v0.27.1` (vLLM 0.27.1, torch 2.11 ROCm 7.2). Script `vllm_bench.py` (child per arm).
Raw: `results/vllm_bench_result.json`, logs `results/*.log`.

Per arm: CAPACITY = vLLM's own "Maximum concurrency for 32,768 tokens per request" (gpu_memory_utilization 0.90),
SPEED = single-request decode ms/token at 3.5k and 32k context, THRU = decode tok/s with 16 concurrent 8k requests,
QUALITY = LongBench token F1, N=100 per task, ctx capped at 3,500 tokens, greedy, 64 new tokens (same prompts and cap
as our kvquant runs). Differences under ~0.02 F1 are noise at N=100.

## Qwen2.5-7B-Instruct
| KV cache | users @32k | ms/tok 3.5k | ms/tok 32k | thru 16x8k | multifieldqa | hotpotqa | 2wikimqa |
|---|---|---|---|---|---|---|---|
| bf16 (auto) | 88.1 | 4.40 | 4.76 | 1124 | 0.487 | 0.414 | 0.399 |
| fp8 (default scale 1.0) | 176.2 | 4.40 | 4.53 | 1183 | **0.245** | **0.276** | **0.265** |
| fp8 + calculate_kv_scales | 176.1 | 4.54 | 4.60 | 1240 | **0.199** | **0.150** | **0.223** |
| fp8 + skip layer 0 | 170.1 | 4.37 | 4.61 | 1177 | 0.495 | 0.410 | 0.400 |
| fp8 + skip layers 0,1,26,27 | 154.2 | 4.40 | 4.51 | 1197 | 0.499 | 0.417 | 0.398 |
| fp8 + skip 0,1,26,27 + ROCM_AITER_FA | 154.2 | 4.38 | 4.63 | **2161** | 0.482 | 0.408 | 0.365 |
| int4_per_token_head | 331.6 | 7.70 | 14.22 | 1195 | **0.009** | **0.001** | **0.001** |
| int4_per_token_head + skip 0,1,26,27 | 237.7 | 8.13 | 21.13 | 1042 | 0.492 | 0.393 | 0.388 |
| turboquant_k8v4 (auto-skips 0,1,26,27) | 187.0 | 6.10 | 17.97 | 653 | 0.494 | 0.387 | 0.346 |
| turboquant_4bit_nc (auto-skips 0,1,26,27) | 241.4 | 6.62 | 17.54 | 592 | 0.495 | 0.363 | 0.333 |
| ours: kvquant live4 int4, per-channel K (transformers, MI300X notebook, same prompts) | ~320 paged (arith.) / 357 measured in transformers | - | 30.0 (fused v2, transformers e2e) | - | 0.499 | 0.421 | 0.398 |
(ours: same-harness full bf16 in transformers was 0.487 / 0.399 / 0.395.)

## Qwen2.5-72B-Instruct
| KV cache | users @32k | ms/tok 3.5k | ms/tok 32k | thru 16x8k | multifieldqa | hotpotqa | 2wikimqa |
|---|---|---|---|---|---|---|---|
| bf16 | 3.17 | 37.38 | 39.75 | 159.7 | 0.492 | 0.520 | 0.458 |
| fp8 | 6.34 | 37.06 | 38.38 | 174.6 | 0.490 | 0.513 | 0.452 |
| fp8 + calculate_kv_scales | 6.34 | 37.26 | 39.04 | 168.6 | 0.481 | 0.517 | 0.441 |
| turboquant_4bit_nc | 10.62 | 45.38 | 91.47 | 83.5 | 0.451 | 0.449 | 0.387 |
vLLM reserves 31.7 GiB for KV on 72B (our transformers capacity test left 56 GiB: 4 bf16 / 19 int4 users).

## Findings
1. Qwen2.5-7B with fp8 KV loses about half its LongBench F1 on MI300X. Calibrating scales makes it worse.
   Known on NVIDIA (vLLM PR #63081, #59674: large key bias in layer 0); this is the first ROCm / real-task confirmation.
2. One flag fixes it: `--kv-cache-dtype-skip-layers 0` restores full quality at 1.93x the bf16 users.
3. vLLM's int4_per_token_head collapses to ~0 F1 on Qwen2.5-7B. With boundary layers in bf16 it holds (-3 %) at 2.7x users,
   but decodes 4.4x slower than bf16 at 32k (21.1 vs 4.8 ms).
4. TurboQuant 4-bit (the path AMD's own LMCache blog uses on MI355X): 2.7x users on 7B, 3.4x on 72B, but multi-hop QA drops
   12-16 % (7B) and 8-16 % (72B), and decode is 3.7x (7B) / 2.3x (72B) slower than bf16 at 32k; throughput halves.
5. AITER flash attention with fp8 KV: 1.81x throughput (2161 vs 1197 tok/s), no garbage output (the May REPOMIND
   AITER+fp8 garbage did not reproduce on 0.27.1 with Qwen2.5-7B). 2wikimqa -0.033 (borderline noise).
6. 72B tolerates fp8 at default scales (key bias stats lower).
7. Per-channel key scales (KIVI-style, ours) hold quality on all 28 layers in 4 bits with no boundary skip. The open problem
   for AMD's 4-bit path is decode speed on gfx942, not capacity.

## fnuz probe (fnuz_probe.py, CPU, real post-RoPE q/k/v of Qwen2.5-7B, 6 LongBench prompts)
Attention-output relative error, mean over layers: fp8 e4m3fn scale 1: 0.141 | e4m3fnuz scale 1: 0.136 | fnuz x2: 0.141 |
fnuz calibrated: 0.121. K amax 420 (layer 27), 173 (layer 0). Worst layers: 0 (0.57), 27 (0.52), 17 (0.26).
-> NOT an AMD-format issue (fnuz is no worse than fn). Task-level: skipping layer 0 alone is enough; the layer-27 error
does not hurt F1 (my probe-based prediction that 27 also needed skipping was wrong).

## Cost
Droplet created ~16:41 Tashkent; main + calibration + TurboQuant/int4 queues + nightly rerun + kernel sweep ~1.9 h (16:41-18:35), ~$4 of $200.
One self-inflicted overlap (two engines at once after a re-queue, ~5 min) was killed and the affected arm re-run clean.

## Rerun on vLLM 0.34.1rc1 nightly (`vllm/vllm-openai-rocm:nightly`, torch 2.13 ROCm 10.0), Qwen2.5-7B
| KV cache | users @32k | ms/tok 3.5k | ms/tok 32k | thru 16x8k | multifieldqa | hotpotqa | 2wikimqa |
|---|---|---|---|---|---|---|---|
| bf16 | 87.4 | 4.22 | 4.72 | 1182 | 0.486 | 0.405 | 0.398 |
| fp8 | 174.9 | 4.28 | 4.64 | 767 | **0.244** | **0.254** | **0.261** |
| fp8 + skip layer 0 | 169.4 | **10.14** | **62.84** | **451** | 0.492 | 0.428 | 0.394 |
| int4_per_token_head | 327.9 | 7.15 | 12.40 | 1444 | **0.008** | **0.002** | **0.001** |
| turboquant_4bit_nc | 237.6 | 6.23 | 16.93 | 619 | 0.496 | 0.356 | 0.336 |
- The quality failures are NOT fixed on the nightly (fp8 and int4 still collapse on Qwen2.5-7B).
- NEW REGRESSION: on the nightly the skip-layer remedy decodes 13.6x slower at 32k (62.8 vs 4.6 ms/token; 0.27.1 was
  4.61) and 16x8k throughput drops to 451. Nightly serves Qwen2 through the Transformers modeling backend; the
  mixed layer-0 bf16 / layers 1-27 fp8 setup (both ROCM_ATTN) is what slows down.
- fp8 16x8k throughput on the nightly is 767 vs 1,183 on 0.27.1 (bf16 unchanged, 1,182).

## TurboQuant decode kernel, launch-parameter sweep (tq_decode_tune.py, vLLM 0.27.1, 7B shape Hq 28 / Hk 4 / D 128)
| case | default (BLOCK_KV 4, 1 warp, 32 splits) | best | speedup | effective KV read |
|---|---|---|---|---|
| B=1, 32k | 569 us | 272 us (BLOCK_KV 64, 1 warp, 32 splits) | 2.09x | 31 -> 65 GB/s |
| B=1, 4k | 69 us | 51 us | 1.34x | 32 -> 43 GB/s |
| B=16, 8k | 872 us | 805 us | 1.08x | 81 -> 87 GB/s |
Outputs identical to the default within 1e-4. MI300X HBM is 5.3 TB/s: the kernel reads KV at ~1-2 % of peak.
Structural causes: grid over query heads (each KV byte read 7x under GQA 28/4), 1 warp per program, bit-gather loads.
Launch tuning alone gives 2.1x for one long request but almost nothing for batched decode. A GQA-aware rewrite
(one program per KV head, padded tl.dot over the 7 query heads, as kvquant.fused4 does) is the real fix.
Honest note: kvquant.fused4 is also far from roofline (381 us per layer at 32k, ~47 GB/s); tuned TurboQuant (272 us)
is faster today. Kill-test verdict: do not port our kernel; rewrite the TurboQuant decode kernel GQA-aware.
