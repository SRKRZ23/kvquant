# servecheck

Autonomous serving-invariance check for vLLM. It needs no labels and runs on one GPU, about 2 minutes per configuration.

For each model and configuration, servecheck starts a fresh vLLM engine and picks needle questions the model answers
correctly on its own. It then asks them again inside crafted schedules:
- chunked-prefill continuation after a 16- or 300-token cached prefix;
- a prefix-cache hit;
- decode batches of 3, 5, 7 or 13, padded to CUDA-graph sizes;
- busy peers;
- an aged engine.

A configuration fails if answers change where the same model in bf16 does not change. The exit code is non-zero on
failure, so servecheck can run in CI.

```
python3 servecheck.py --models Qwen/Qwen2.5-7B-Instruct,unsloth/Meta-Llama-3.1-8B-Instruct \
    --configs bf16,fp8kv,tq4,uq4,fp8w_online --outdir out [--probes 8]
```

## Results

MI300X, `vllm/vllm-openai-rocm:nightly` (vllm 0.34.1rc1.dev254+gde7cf7f8c), 27 Nov 2026. JSON files are in
`../results/nightly_0.34.1/session_2026-11-27/`.

- **Baseline** (`sc_out`): bf16 and fp8 KV are clean on Qwen2.5-7B, Llama-3.1-8B and Qwen3-30B-A3B. The tool flagged
  7 of 12 configurations:
  - `turboquant_4bit_nc` and `ultraquant_4bit` on all three models, on the continuation and cache-hit probes;
  - online fp8 weights on Llama and the MoE model, on the padding probes.
- **With vLLM PR #67789 applied** (`sc_e2`, `sc_r3`): both 4-bit formats pass on all three models.
- **With vLLM PR #65820 applied** (`sc_r4`): online fp8 passes on both models.
