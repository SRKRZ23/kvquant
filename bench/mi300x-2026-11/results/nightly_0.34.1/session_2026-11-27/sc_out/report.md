# servecheck report

| model | config | usable probes | failures by schedule | verdict |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | bf16 | 8 | none | REFERENCE |
| Meta-Llama-3.1-8B-Instruct | fp8kv | 8 | none | clean |
| Meta-Llama-3.1-8B-Instruct | fp8w_online | 8 | cont_longer_t16 1, cont_longer_t300 8, pad_3 8, pad_5 8, pad_7 8, pad_13 4 | FAIL |
| Meta-Llama-3.1-8B-Instruct | tq4 | 8 | cont_longer_t16 5, cont_longer_t300 8, cache_longer 8 | FAIL |
| Meta-Llama-3.1-8B-Instruct | uq4 | 8 | cont_longer_t16 5, cont_longer_t300 8, cache_longer 8 | FAIL |
| Qwen2.5-7B-Instruct | bf16 | 8 | none | REFERENCE |
| Qwen2.5-7B-Instruct | uq4 | 8 | cont_longer_t16 6, cont_longer_t300 6, cache_longer 8 | FAIL |
| Qwen3-30B-A3B-Instruct-2507 | bf16 | 8 | none | REFERENCE |
| Qwen3-30B-A3B-Instruct-2507 | fp8kv | 8 | none | clean |
| Qwen3-30B-A3B-Instruct-2507 | fp8w_online | 8 | cont_longer_t16 7, cont_longer_t300 8, pad_3 4, pad_5 8, pad_7 6, pad_13 6 | FAIL |
| Qwen3-30B-A3B-Instruct-2507 | tq4 | 8 | cont_longer_t16 8, cont_longer_t300 8, cache_longer 7 | FAIL |
| Qwen3-30B-A3B-Instruct-2507 | uq4 | 8 | cont_longer_t16 8, cont_longer_t300 8, cache_longer 7 | FAIL |
