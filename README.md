# kvquant

**Tested KV-cache quantization for Hugging Face `transformers >= 5` — with a stress test so a silent no-op can't fool you.**

Most "quantize the KV cache" snippets you'll find mutate `cache.key_cache[i]`. On
`transformers 5.x` that is a **non-persisting compatibility accessor**: your edit is
thrown away and the model keeps using the original fp tensors. Everything *looks* like
it worked — same code, no error — but perplexity doesn't move because nothing changed.
The real tensors live at `cache.layers[i].keys` / `.values`. kvquant mutates the right
ones and ships a guard that **proves** the quantization actually bit.

## Install

```bash
pip install kvquant            # core
pip install "kvquant[bench]"   # + transformers/numpy for the perplexity benchmark
```

## Quickstart

```python
from kvquant import quantize_cache          # in-place int-N quant of a transformers>=5 Cache
change = quantize_cache(past_key_values, bits=8)   # returns mean |change|; 0.0 == a no-op bug
```

Measure the quality cost yourself:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvquant import kv_ppl_sweep
m = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
t = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
print(kv_ppl_sweep(m, t))            # {None/8/4/2: {ppl, delta_pct, change}}
```

or from the shell:

```bash
kvquant-bench --model google/gemma-3-270m-it
```

## What int8 KV actually costs (measured)

`google/gemma-3-270m-it`, 300 tokens, per-channel symmetric int quant of K and V:

| KV cache | perplexity | Δ vs fp | note |
|---|---|---|---|
| fp (baseline) | 4.5479 | — | |
| **int8** | 4.5437 | **−0.09%** | within noise — effectively lossless |
| int4 | 4.7528 | +4.5% | visible degradation |
| int2 | 982.76 | +21509% | breaks (this is the stress check) |

int8 K/V carries **half** the bytes of fp16 (a quarter of fp32), so an int8 cache is
the obvious lever for longer context / larger batch at near-zero quality cost. Run the
sweep on *your* model — numbers vary by architecture.

### Real memory, measured (not just quality)

`kvquant.compress_cache` stores a populated cache as int8 + a per-token fp16 scale and
reports the actual byte reduction:

```python
from kvquant import compress_cache
stores, report = compress_cache(past_key_values)
print(report)   # {'fp_bytes': 5_824_512, 'int8_bytes': 1_467_504, 'ratio': 3.97}
```

On the gemma cache above (158 tokens, 18 layers): **5,824,512 → 1,467,504 bytes = 3.97×
smaller vs fp32 (~2× vs fp16)**, ~2% per-tensor reconstruction error. Use it to compress
a cache for offload / serialization / holding an idle long context; for int8 *during*
attention, pair this scheme with transformers' `QuantizedCache`.

## The silent-no-op guard

`kv_ppl_sweep` asserts that int2 both changes the tensors (`change > 0`) **and** wrecks
perplexity. If it doesn't, you're mutating the wrong thing and the whole measurement is
meaningless — so it raises instead of printing a comfortable lie. The same idea is
packaged as a standalone decorator:

```python
from kvquant.doesitbite import bites

@bites()                      # raises AssertionError if the transform doesn't change its input
def my_quantizer(x): ...
```

## Also included

- **`kvquant.doesitbite`** — `@bites` / `assert_bites` to catch transforms that silently
  do nothing (mis-targeted in-place edits, dead paths, clamps that never fire).
- **`kvquant.rankconf`** — confidence utilities for rank metrics (AURC/risk-coverage):
  `aurc`, `is_rank_noop` (any strictly-monotonic rescale is a no-op for a rank metric),
  and `agreement_confidence` (a stronger signal than a model's self-reported confidence).

## Honest limitations (v0)

- Two layers, on purpose: `kv_ppl_sweep` / `quantize_cache` **fake-quantize** (quant→dequant)
  to measure the *quality* cost with the correct non-no-op hook; `compress_cache` /
  `Int8KVStore` do **real int8 storage** with measured byte savings (above). What's not
  here yet is reading from int8 *inside attention every step* — for that, pair this scheme
  with transformers' `QuantizedCache`; it's the roadmap item.
- Symmetric quantization, no outlier handling or post-RoPE split yet (those shrink the
  int8 quality gap further).
- Benchmarked on small models on a laptop. Bring your own model and report back.

## License

MIT.
