# kvquant — launch drafts (honest). Push/Sponsors/PyPI are Sardor's account actions.

Repo: SRKRZ23/kvquant (to create). Enable GitHub Sponsors on the account. `pip install kvquant-hf` after PyPI publish.
Honesty: lead with the measured table + the silent-no-op hook; state the v0 scope (fake-quant for quality, real int8 storage for bytes; live-attention int8 pairs with transformers QuantizedCache). No invented numbers; no AI co-author in commits.

---

## GitHub "About" / one-liner
Tested KV-cache quantization for Hugging Face transformers 5.x — with a stress test so a silent no-op can't fool you. int8 KV ≈ lossless; real int8 storage ~2× smaller (fp16).

## LinkedIn

Most "quantize the LLM KV cache" snippets you'll copy off the internet silently do nothing on transformers 5.x.

They mutate `cache.key_cache[i]` — but on transformers 5.x that's a throwaway compatibility accessor. Your edit is discarded, the model keeps the full fp16 cache, and perplexity doesn't move. No error. It just quietly didn't work. I hit exactly this while validating a result, and only caught it with a 2-bit stress test.

So I packaged the fix: **kvquant** (MIT, open source).
🔧 Quantizes the right tensors (`cache.layers[i].keys/.values`)
📏 Ships a stress guard that *proves* the quantization bit (int2 must wreck perplexity, else it raises)
🧪 One-command perplexity sweep so you see the real quality cost on *your* model
💾 Real int8 storage with measured byte savings, not just a quality probe

Measured on a small model: int8 KV cost **−0.09% perplexity** (within noise), and int8 storage was **~2× smaller than fp16 (3.97× vs fp32)**. int4 cost +4.5%; int2 falls apart (that's the test).

Honest scope: v0 measures quality + does real int8 storage; reading from int8 *inside attention every step* pairs with transformers' own QuantizedCache — that's the roadmap.

If it saves you a GPU, a ⭐ or a sponsor keeps it going.
👉 github.com/SRKRZ23/kvquant

#OpenSource #LLM #MachineLearning #Inference #Python #BuildInPublic

## X thread

1/
Most "quantize the LLM KV cache" snippets silently do NOTHING on transformers 5.x.

They edit `cache.key_cache[i]` — a throwaway compat accessor. Your change is discarded, the fp cache stays, perplexity doesn't move. No error. It just didn't work.

I packaged the fix 👇 (MIT)
github.com/SRKRZ23/kvquant

2/
The real tensors are `cache.layers[i].keys/.values`. kvquant mutates those AND ships a stress test: int2 KV must blow up perplexity, or it raises. You can't be fooled by a silent no-op.

3/
Measured, small model:
• int8 KV: −0.09% perplexity (noise)
• int4: +4.5%
• int2: +21509% (the test)
• int8 storage: ~2× smaller than fp16 (3.97× vs fp32)

Run the one-command sweep on your own model.

4/
Honest v0 scope: it measures the quality cost and does real int8 storage (offload / long-context hold). Reading from int8 inside attention every step pairs with transformers' QuantizedCache — roadmap.

⭐ / sponsor if it saves you a GPU. #OpenSource #LLM

## Hacker News (Show HN)
Show HN: kvquant — tested KV-cache quantization for transformers 5.x (with a no-op stress test)

I kept seeing KV-cache quantization snippets that mutate `cache.key_cache[i]`, which is a non-persisting compatibility accessor on transformers 5.x — the edit is discarded and the model keeps the fp cache, so perplexity never moves and nothing errors. I got bitten by it, caught it only with a 2-bit stress test, and packaged a version that mutates the real tensors (`cache.layers[i].keys/.values`) and ships the stress guard so you can't be fooled. It also measures the quality cost (int8 ≈ lossless on a small model, int4 +4.5%, int2 breaks) and does real int8 storage (~2× vs fp16, measured). Honest v0 scope: reading from int8 inside attention every step pairs with transformers' QuantizedCache — that's next. MIT. Feedback welcome, especially from anyone running longer-context local inference.
