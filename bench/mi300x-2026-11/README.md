# kvquant and vLLM KV-cache dtypes on one AMD Instinct MI300X (November 2026)

Everything here was measured on one MI300X (192 GB): the transformers runs on AMD AI Notebooks, the vLLM runs on an
AMD Developer Cloud droplet (vLLM 0.27.1 ROCm image, plus a rerun on the 0.34.1 nightly).

| file | what it measures |
|---|---|
| `notebook/amd_gate0.py` | real int4 (kvquant live4) vs simulated int4 vs bf16, LongBench F1, Qwen2.5-7B |
| `notebook/amd_gate1.py` | inside NVIDIA KVPress: int4 vs eviction at equal physical memory, eviction + int4 combined, capacity until OOM |
| `notebook/amd_gate2.py`, `amd_gate3.py` | fused Triton int4 decode kernel (kvquant/fused4.py): correctness, per-layer and end-to-end speed, 72B capacity |
| `notebook/RESULT_MI300X_2026-11-25.md` | all transformers / KVPress results |
| `vllm_bench.py` | vLLM per KV dtype: max concurrency at 32k, decode ms/token at 3.5k and 32k, 16x8k throughput, LongBench F1 |
| `fnuz_probe.py` | replays real post-RoPE q/k/v of Qwen2.5-7B through fp8 e4m3fn vs e4m3fnuz on CPU |
| `tq_decode_tune.py` | times vLLM's TurboQuant decode kernel over launch parameters |
| `RESULT_VLLM_MI300X_2026-11-25.md` | all vLLM results and findings |
| `tq_pr_rocm_bench.py` | runs vLLM's TurboQuant decode op from main and from PRs #69273 / #68834 on ROCm without installing vLLM (sources embedded, imports shimmed) |
| `tq_decode_pr68834_gfx942.py` | PR #68834's decode op with the grouped path enabled on ROCm (one-line gate change), used for the end-to-end run |
| `tq_gfx942_tune.py`, `b6_perchannel_grouped.py` | gfx942 launch sweep of the grouped kernel; per-channel int4 (kvquant layout) in a grouped split-K decode kernel |
| `hl_gate_test.py` | AMD Hyperloom's long-context check (vendored generator, MIT) + GSM8K next to LongBench on the same KV settings |
| `tco_72b.py`, `TCO_72B_MI300X_vs_H100.md` | users with 32k contexts per GPU-dollar, MI300X vs H100 (arithmetic, labelled) |
| `openvino/` | the same failure on OpenVINO's CPU KV cache (u4 collapses on Qwen2.5-1.5B, u8 holds) |
| `results/` | raw JSON and full vLLM logs; `results/droplet2_gfx942/` = Hyperloom-check and PR #68834 runs |

Headline (Qwen2.5-7B, LongBench, 100 questions per task, contexts capped at 3,500 tokens):

- kvquant's real int4 cache (per-channel keys in 128-token groups, per-token values) matches bf16:
  0.499 / 0.421 / 0.398 vs 0.487 / 0.399 / 0.395, and fits 357 users with 32k-token contexts on one MI300X vs 99 in bf16.
- vLLM fp8 KV at default settings halves the score (0.245 / 0.276 / 0.265); calibrated scales are worse.
  `--kv-cache-dtype-skip-layers 0` restores it (0.495 / 0.410 / 0.400) at 1.93x the bf16 users.
- vLLM int4_per_token_head scores about 0; with boundary layers in bf16 it holds but decodes 4.4x slower than bf16 at 32k.
- TurboQuant 4-bit: 2.7x the users, multi-hop QA -12 to -16 %, decode 3.7x slower than bf16 at 32k.

Reproduce: `python3 vllm_bench.py` inside a vLLM ROCm container (`DRY=1` checks data, prompts and the log parser
without a GPU). The notebook scripts run one mode per 30-minute AMD AI Notebooks session (`MODE=...`).

Later results (second droplet, 25 Nov):
- PR #68834's grouped TurboQuant decode kernel, enabled on gfx942: 5.3x to 7.4x per decode-attention call vs vLLM main
  (Qwen2.5-7B shape, batch 1 to 32, output within 1.4e-4); end to end 2.41x throughput at 16 x 8k (592 -> 1,426 tok/s,
  bf16 KV: 1,124) with LongBench unchanged.
- Hyperloom's quality gate (GSM8K + its RULER-style long-context set) accepts turboquant_4bit_nc, which loses 12-16 % on
  LongBench hotpotqa / 2wikimqa.
