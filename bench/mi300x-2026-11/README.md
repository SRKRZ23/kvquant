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
| `results/` | raw JSON and full vLLM logs |

Headline (Qwen2.5-7B, LongBench, 100 questions per task, contexts capped at 3,500 tokens):

- kvquant's real int4 cache (per-channel keys in 128-token groups, per-token values) matches bf16:
  0.499 / 0.421 / 0.398 vs 0.487 / 0.399 / 0.395, and fits 357 users with 32k-token contexts on one MI300X vs 99 in bf16.
- vLLM fp8 KV at default settings halves the score (0.245 / 0.276 / 0.265); calibrated scales are worse.
  `--kv-cache-dtype-skip-layers 0` restores it (0.495 / 0.410 / 0.400) at 1.93x the bf16 users.
- vLLM int4_per_token_head scores about 0; with boundary layers in bf16 it holds but decodes 4.4x slower than bf16 at 32k.
- TurboQuant 4-bit: 2.7x the users, multi-hop QA -12 to -16 %, decode 3.7x slower than bf16 at 32k.

Reproduce: `python3 vllm_bench.py` inside a vLLM ROCm container (`DRY=1` checks data, prompts and the log parser
without a GPU). The notebook scripts run one mode per 30-minute AMD AI Notebooks session (`MODE=...`).
