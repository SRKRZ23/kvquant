# Serving-cost: measure tok/$ and cut it with KV quantization ($0, laptop)

Two reproducible experiments behind the claim "int8 KV cache is ~2× memory for free,
and that ~doubles throughput-per-dollar at scale." Run on an Apple-silicon MacBook, no GPU.

## 1. KV quality on a real GQA model — `outlier_int4_kv.py`
Autoregressive perplexity on **Qwen2.5-0.5B-Instruct** (grouped-query attention), KV cache
fake-quantized and re-quantized every decode step:

| config | ppl | Δ vs fp16 | ~bits/elem |
|---|---|---|---|
| fp16 | 13.608 | 0.00% | 16 |
| **int8 K+V** | 13.664 | **+0.42%** | 8 |
| int4 K+V (naive) | 17.915 | +31.65% | 4 |
| **int4 K (outlier-4) + int4 V** | 14.870 | **+9.28%** | ~4.5 |
| int2 (sanity) | 2663 | +19472% | 2 |

int8 KV is near-lossless at 2× smaller. Naive int4 breaks on the keys' persistent outlier
channels; keeping just 4 of them in fp16 recovers most of the loss at ~int4 memory. (This is
the KVQuant/KIVI insight, re-measured openly on GQA — not a new technique.)

## 2. tok/$ roofline profiler — `tokcost.py`
Decode is memory-bandwidth-bound, so tok/s ≈ bandwidth / bytes-re-read-per-token and
tok/$ = tok/s·3600 / $hr. The model is **anchored to a real measured decode on this Mac**
(Qwen2.5-0.5B, 98 tok/s → effective ~97 GB/s), then scores datacenter GPUs and shows the
int8-KV lever. Example (Llama-3.1-70B, fp16 weights, 8k context, on-demand $/hr):

```
GPU            KV    max batch agg tok/s        tok/$
MI300X-192G    fp16         23       601    1,081,205
MI300X-192G    int8         46      1201    2,162,411   <- int8 KV doubles tok/$
H100-80G       fp16    weights don't fit (132GB > 80GB)
```

Roofline **estimates**, clearly labelled — not a substitute for a real cluster benchmark
(ignores prefill, compute-bound regimes, speculative decoding). Specs/prices are approximate
and editable. The datacenter-scale proof (>10% tok/$ on a real H100/MI300X) needs a GPU and is
the open next step.
