# ROCm attention diagnostics (MI300X, 27 Nov 2026)

## Scripts

- `pa_micro.py` times vLLM's ROCm custom paged-attention decode kernel on its own: batch, context, GQA ratio,
  `max_seq_len`, and contiguous vs scattered blocks. Result: about 3.5 TB/s in every case.
- `prof.py` runs `torch.profiler` over `llm.generate` (16 prompts of 30k tokens) to show which GPU kernels dominate
  in `ROCM_ATTN` and in `TRITON_ATTN`.
- `patch_pp_early_exit.py` patches the prefix-prefill kernels (`vllm/v1/attention/ops/prefix_prefill.py`) to return
  early for query tiles past a request's chunk.
- `pp_identity.py` loads the installed kernel and a patched copy as two modules and runs both on the same mixed
  batches. It checks that the outputs are bitwise equal and times each version.

## Results

Kernel results from `pp_identity.py` (`context_attention_fwd`, bf16, mean of 10 calls):

| batch (query_len, cached ctx) | before (ms) | after (ms) | outputs |
|---|---|---|---|
| (16000, 0) + (384, 29000) | 51.621 | 12.831 | identical |
| 16 x (64, 20000) + (4096, 0) | 86.388 | 3.865 | identical |
| ALiBi: (4096, 0) + (256, 8000) | 9.060 | 1.643 | identical |
| batches without empty tiles | unchanged | unchanged | identical |

The upstream `tests/kernels/attention/test_prefix_prefill.py` with the patch: 406 passed, 64 skipped.
The full log is `../results/nightly_0.34.1/session_2026-11-27/run8_prefix_prefill_and_pr66485.log`.
