"""kvquant vs eviction INSIDE NVIDIA KVPress, same harness, same GPU (AMD AI Notebooks MI300X). One file.
MODE=quality  (default): LongBench F1 for full | REAL int4 (kvquant live4) | ExpectedAttention eviction at keep 0.25,
               at EQUAL PHYSICAL memory (keep = measured int4 bytes / bf16 bytes, ~0.28) and at keep 0.5.
               Env: N (default 50), TASKS, MODEL.
MODE=compose: light ExpectedAttention eviction (keep 0.75 / 0.5) + REAL int4 in one cache vs eviction alone at
               the same physical memory. Env: N, TASKS (default multifieldqa_en,hotpotqa to fit a 30-min session).
MODE=capacity: how many long-context "users" (one cache each, CTX tokens, default 32768) fit on this GPU,
               bf16 cache vs real int4 cache; caches are filled one by one until out of memory. Env: CTX, MODEL, LIMIT_S.
Prints RESULT / CAPACITY lines as it goes and saves amd_gate1_<mode>.json next to this file.
"""
import os, sys, subprocess, json, time, re, string, gc, traceback
from collections import Counter
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True)
import torch
P = torch.cuda.get_device_properties(0); VRAM = P.total_memory / 2**30
print("gpu:", P.name, "| VRAM GB", round(VRAM, 1), "| torch", torch.__version__, flush=True)
MODE = os.environ.get("MODE", "quality")
sh(f"{sys.executable} -m pip install -q --no-deps kvpress peft cachetools fire termcolor 'datasets==2.21.0'")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LIVE4 = '"""Live int4 KV cache: actually HOLDS 4-bit keys/values in memory during generation.\n\nScheme (KIVI-style, the one measured on LongBench with the simulated int4):\n\n* keys:   asymmetric 4-bit, **per channel**, in groups of ``group`` tokens along the sequence\n          (one fp16 min + scale per channel per group). Keys have outlier channels, so per-channel\n          scales are what keep int4 keys accurate.\n* values: asymmetric 4-bit, **per token** over ``head_dim`` (one fp16 min + scale per token).\n* two 4-bit codes are packed into one uint8.\n* the newest ``< group`` tokens stay in the original dtype (a small residual buffer) until a full\n  key group can be quantized; values are quantized at once (per token needs no group).\n\nMemory held per token and head, head_dim = 128, group = 128, bf16 baseline 512 B:\nkeys 64 B + 4 B/128-token group share, values 64 B + 4 B  ->  ~136 B, i.e. ~3.8x smaller than bf16.\n\nHow it hooks in: like :mod:`kvquant.live`, each cache layer\'s class is swapped for a subclass whose\n``keys`` / ``values`` are properties. transformers\' layer ``update`` does\n``self.keys = cat([self.keys, new], -2)``; the setter sees a longer tensor and only quantizes the\nnew tail (the prefix it receives is its own dequantized output, so it is not re-quantized and no\nerror accumulates). A shorter or reordered tensor (crop, beam reorder, eviction) triggers a full\nre-quantization of what it was given.\n\nReads dequantize the whole layer (pure PyTorch, no fused kernel): the memory HELD is int4, the\nattention math runs in the original dtype. A fused int4 attention kernel is future work; measure\nspeed with that in mind.\n"""\nfrom __future__ import annotations\nimport torch\n\n_variants: dict = {}\n\n\ndef _pack(q: torch.Tensor) -> torch.Tensor:\n    """uint8 codes in [0, 15], last dim even -> packed uint8 with half the last dim."""\n    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)\n\n\ndef _unpack(p: torch.Tensor) -> torch.Tensor:\n    lo = p & 0x0F\n    hi = p >> 4\n    return torch.stack((lo, hi), dim=-1).flatten(-2)\n\n\ndef _qasym(x: torch.Tensor, dim: int):\n    """4-bit asymmetric quantization along ``dim``; returns codes (uint8) and fp16 min, scale."""\n    mn = x.amin(dim=dim, keepdim=True)\n    mx = x.amax(dim=dim, keepdim=True)\n    scale = ((mx - mn).float().clamp_min(1e-6) / 15.0).to(torch.float16)\n    mn = mn.to(torch.float16)\n    q = ((x.float() - mn.float()) / scale.float()).round_().clamp_(0, 15).to(torch.uint8)\n    return q, mn, scale\n\n\nclass _Store:\n    """int4 storage for one K or V tensor of one layer, shape [batch, heads, seq, head_dim]."""\n\n    def __init__(self, kind: str, group: int):\n        self.kind, self.group = kind, group\n        self.packed = []      # list of packed uint8 chunks [B, H, n_i, D/2]\n        self.mins = []        # fp16 [B, H, 1, D] (keys, per group) or [B, H, n_i, 1] (values)\n        self.scales = []\n        self.resid = None     # keys only: newest tokens not yet a full group, original dtype\n        self.n_q = 0          # tokens held quantized\n        self.dtype = None\n\n    def length(self) -> int:\n        return self.n_q + (0 if self.resid is None else self.resid.shape[-2])\n\n    def clear(self):\n        self.packed, self.mins, self.scales, self.resid, self.n_q = [], [], [], None, 0\n\n    def _quant_append(self, x: torch.Tensor):\n        if x.shape[-1] % 2:\n            raise ValueError("head_dim must be even for 4-bit packing")\n        if self.kind == "k":\n            q, mn, sc = _qasym(x, dim=-2)        # per channel over the group\'s tokens\n        else:\n            q, mn, sc = _qasym(x, dim=-1)        # per token over head_dim\n        self.packed.append(_pack(q))\n        self.mins.append(mn)\n        self.scales.append(sc)\n        self.n_q += x.shape[-2]\n\n    def append(self, new: torch.Tensor):\n        self.dtype = new.dtype\n        if self.kind == "v":\n            self._quant_append(new)\n            return\n        buf = new if self.resid is None else torch.cat([self.resid, new], dim=-2)\n        full = (buf.shape[-2] // self.group) * self.group\n        for s in range(0, full, self.group):\n            self._quant_append(buf[..., s:s + self.group, :])\n        self.resid = buf[..., full:, :] if buf.shape[-2] > full else None\n\n    def set(self, t: torch.Tensor):\n        if t is None or t.numel() == 0:\n            self.clear()\n            self.empty = t               # transformers\' lazy init sets an empty 1-D tensor\n            return\n        n = self.length()\n        if n and t.shape[-2] > n and t.shape[:-2] == self._lead():\n            self.append(t[..., n:, :])           # usual decode / chunked-prefill path\n        elif n and t.shape[-2] == n and t.shape[:-2] == self._lead():\n            return                               # unchanged (e.g. set back after a read)\n        else:\n            self.clear()\n            self.append(t)                       # first fill, crop, reorder: re-quantize all\n\n    def _lead(self):\n        if self.packed:\n            return tuple(self.packed[0].shape[:-2])\n        return tuple(self.resid.shape[:-2]) if self.resid is not None else None\n\n    def get(self):\n        if not self.packed and self.resid is None:\n            return getattr(self, "empty", None)\n        parts = []\n        for p, mn, sc in zip(self.packed, self.mins, self.scales):\n            parts.append((_unpack(p).float() * sc.float() + mn.float()).to(self.dtype))\n        if self.resid is not None:\n            parts.append(self.resid)\n        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)\n\n    def nbytes(self) -> int:\n        b = 0\n        for lst in (self.packed, self.mins, self.scales):\n            b += sum(t.element_size() * t.nelement() for t in lst)\n        if self.resid is not None:\n            b += self.resid.element_size() * self.resid.nelement()\n        return b\n\n\ndef _variant(base):\n    if base in _variants:\n        return _variants[base]\n\n    class Int4Layer(base):\n        _kvquant_int4 = True\n\n        @property\n        def keys(self):\n            return self._k4.get()\n\n        @keys.setter\n        def keys(self, v):\n            self._k4.set(v)\n\n        @property\n        def values(self):\n            return self._v4.get()\n\n        @values.setter\n        def values(self, v):\n            self._v4.set(v)\n\n        def update(self, key_states, value_states, *args, **kwargs):\n            # one dequantization per layer per step (the base class would read keys/values again)\n            if not getattr(self, "is_initialized", True):\n                self.lazy_initialization(key_states, value_states)\n            # Attention for THIS step uses the stored past (dequantized) plus the new tokens at full precision,\n            # as KIVI and serving stacks do: the prompt attends to itself in the original dtype during prefill and\n            # only the STORED copy is 4-bit. Returning the quantized new tokens made prefill attend to 4-bit\n            # keys/values and compounded the error through the layers (gate 0, 2026-11-25: real < simulated).\n            past_k, past_v = self._k4.get(), self._v4.get()\n            self._k4.append(key_states)\n            self._v4.append(value_states)\n            if past_k is None or past_k.numel() == 0:\n                return key_states, value_states\n            return torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)\n\n        def get_seq_length(self, *args, **kwargs) -> int:\n            return self._k4.length()\n\n        def int4_bytes(self) -> int:\n            return self._k4.nbytes() + self._v4.nbytes()\n\n    if base.__name__ != "DynamicLayer":\n        # sliding-window / static layers keep their own update logic and go through the properties\n        del Int4Layer.update\n        del Int4Layer.get_seq_length\n    _variants[base] = Int4Layer\n    return Int4Layer\n\n\ndef make_live_int4(cache, group: int = 128):\n    """Convert a transformers>=5 cache in place so its layers hold int4. Idempotent."""\n    layers = getattr(cache, "layers", None)\n    if layers is None:\n        raise TypeError("expected a transformers>=5 Cache exposing `.layers`")\n    for layer in layers:\n        if getattr(type(layer), "_kvquant_int4", False):\n            continue\n        k = layer.__dict__.get("keys", getattr(layer, "keys", None))\n        v = layer.__dict__.get("values", getattr(layer, "values", None))\n        layer.__dict__.pop("keys", None)\n        layer.__dict__.pop("values", None)\n        layer.__class__ = _variant(type(layer))\n        layer._k4 = _Store("k", group)\n        layer._v4 = _Store("v", group)\n        layer.keys = k\n        layer.values = v\n    return cache\n\n\ndef held_bytes4(cache) -> int:\n    """Live byte footprint of the cache\'s K/V storage (int4 layers counted as stored)."""\n    total = 0\n    for layer in getattr(cache, "layers", []):\n        if hasattr(layer, "int4_bytes"):\n            total += layer.int4_bytes()\n        else:\n            for name in ("keys", "values"):\n                t = getattr(layer, name, None)\n                if t is not None:\n                    total += t.element_size() * t.nelement()\n    return total\n'
open(os.path.join(HERE, "live4.py"), "w").write(LIVE4)
from live4 import make_live_int4, held_bytes4
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
MODEL = os.environ.get("MODEL") or ("Qwen/Qwen2.5-7B-Instruct" if VRAM >= 40 else "Qwen/Qwen2.5-3B-Instruct")
print("mode:", MODE, "| model:", MODEL, flush=True)

def kv_bytes(cache):
    b = held_bytes4(cache)                     # int4 layers counted as stored, plain layers as tensors
    return b

if MODE == "capacity":
    CTX = int(os.environ.get("CTX", "32768")); LIMIT_S = float(os.environ.get("LIMIT_S", "300"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = model.config
    ids = torch.randint(100, 20000, (1, CTX), device="cuda")
    w_gib = torch.cuda.memory_allocated() / 2**30
    print(f"weights {w_gib:.1f} GiB | context {CTX} tokens per user", flush=True)
    out = {"model": MODEL, "ctx": CTX, "weights_gib": round(w_gib, 2), "vram_gib": round(VRAM, 1)}
    def stored_tensors(cache):
        ts = []
        for L in cache.layers:
            if hasattr(L, "_k4"):
                for st in (L._k4, L._v4):
                    ts += list(st.packed) + list(st.mins) + list(st.scales) + ([st.resid] if st.resid is not None else [])
            else:
                ts += [L.keys, L.values]
        return ts
    for arm in ("bf16", "real4"):
        c0 = DynamicCache(config=cfg)
        if arm == "real4": make_live_int4(c0)
        with torch.no_grad():
            model.model(input_ids=ids, past_key_values=c0)
        base = stored_tensors(c0); per_user = sum(t.element_size() * t.nelement() for t in base)
        users = [base]; t0 = time.time(); why = "time limit"
        try:
            while time.time() - t0 < LIMIT_S:      # one more user = one more stored copy of a CTX-token cache
                users.append([t.clone() for t in base])
                if len(users) % 25 == 0:
                    print(f"  {arm}: {len(users)} users, {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
        except torch.OutOfMemoryError:
            why = "out of memory"; users.pop() if len(users) > 1 else None
        out[arm] = {"users": len(users), "stopped_by": why, "bytes_per_token": round(per_user / CTX, 1),
                    "gib_per_user": round(per_user / 2**30, 3),
                    "gib_used": round(torch.cuda.memory_allocated() / 2**30, 1), "seconds": round(time.time() - t0, 1),
                    "max_single_context_tokens_est": int((VRAM - w_gib - 8) * 2**30 / (per_user / CTX))}
        print("CAPACITY", arm, json.dumps(out[arm]), flush=True)
        del users, base, c0; gc.collect(); torch.cuda.empty_cache()
    if out["bf16"]["users"] and out["real4"]["users"]:
        out["users_ratio"] = round(out["real4"]["users"] / out["bf16"]["users"], 2)
    json.dump(out, open(os.path.join(HERE, "amd_gate1_capacity.json"), "w"), indent=1)
    print("\nCAPACITY SUMMARY", json.dumps(out), flush=True)
    sys.exit(0)

# ---------------- MODE=quality ----------------
TASKS = os.environ.get("TASKS", "multifieldqa_en,hotpotqa" if MODE == "compose" else "multifieldqa_en,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("N", "100")); CTX_CAP = 3500; MAXNEW = 64
def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation); return re.sub(r"\s+", " ", s).strip()
def f1(p, g):
    p, g = normalize(p).split(), normalize(g).split()
    if not p or not g: return 0.0
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g); return 2 * pr * rc / (pr + rc)
def qa_f1(pred, golds): return max((f1(pred, g) for g in golds), default=0.0)
def load_raw(task):
    from datasets import load_dataset
    try:
        return load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    except Exception as e:
        print("  script load failed, parquet:", str(e)[:80], flush=True)
        import pandas as pd
        return pd.read_parquet(f"hf://datasets/THUDM/LongBench@refs/convert/parquet/{task}/test/0000.parquet").to_dict("records")
INSTR = "Answer the question based only on the text above. Give only the answer, no other words.\nQuestion: "
from transformers import pipeline
import kvpress
from kvpress import ExpectedAttentionPress
print("kvpress", getattr(kvpress, "__version__", "?"), flush=True)
# kvpress 0.5.5 targets transformers < 5.3; 5.12 no longer passes `cache_position` to attention layers.
# The hook only uses it to ask "is this the prefill?". Derive it from the cache: after the layer ran, the layer
# holds L tokens, so this step covered positions L-q .. L-1 (prefill <=> L == q). Same test, no other change.
from kvpress.presses import base_press as _bp
_orig_hook = _bp.BasePress.forward_hook
def _hook(self, module, input, kwargs, output):
    if "cache_position" not in kwargs:
        q = kwargs["hidden_states"].shape[1]
        L = kwargs["past_key_values"].get_seq_length(module.layer_idx)
        kwargs = dict(kwargs); kwargs["cache_position"] = torch.arange(L - q, L, device=kwargs["hidden_states"].device)
    return _orig_hook(self, module, input, kwargs, output)
_bp.BasePress.forward_hook = _hook
print("patched kvpress forward_hook for transformers 5.12 (cache_position derived from the cache)", flush=True)
pipe = pipeline("kv-press-text-generation", model=MODEL, device_map="cuda", dtype=torch.bfloat16)
tok = pipe.tokenizer; cfg = pipe.model.config
res = []
for task in TASKS:
    exs = []
    for row in load_raw(task):
        ctx = row["context"]; ids = tok(ctx, return_tensors="pt").input_ids[0]
        if ids.shape[0] > CTX_CAP:
            h = CTX_CAP // 2; ctx = tok.decode(torch.cat([ids[:h], ids[-h:]]), skip_special_tokens=True)
        exs.append((ctx, row["input"], row["answers"]))
        if len(exs) >= N: break
    r = {"model": MODEL, "task": task, "n": len(exs)}
    ctx_len = []
    def score(arm, press=None, int4=False):
        tot = 0.0; err = 0; rel = []; t0 = time.time()
        for i, (ctx, q, ans) in enumerate(exs):
            try:
                c = DynamicCache(config=cfg)
                if int4: make_live_int4(c)
                a = pipe(ctx, question=INSTR + q + "\nAnswer:", press=press, cache=c, max_new_tokens=MAXNEW)["answer"]
                tot += qa_f1(a, ans)
                L = c.get_seq_length()
                if arm == "full": ctx_len.append(L); bf16_tok = kv_bytes(c) / max(1, L); r["bf16_bytes_per_token"] = round(bf16_tok, 1)
                elif i < len(ctx_len): rel.append(kv_bytes(c) / (r["bf16_bytes_per_token"] * ctx_len[i]))
            except Exception as ex:
                err += 1
                if err == 1: print("   err", arm, type(ex).__name__, str(ex)[:300], flush=True); traceback.print_exc()
        out = {"f1": round(tot / len(exs), 4), "errors": err, "wall_s": round(time.time() - t0)}
        if rel: out["rel_kv_bytes"] = round(sum(rel) / len(rel), 4)
        r[arm] = out
        print("RESULT", MODEL.split("/")[-1], task, arm, json.dumps(out), flush=True)
        return out
    if MODE == "compose":
        # light eviction + real int4 in one cache, vs eviction alone at the SAME physical memory.
        # Needs the full arm once for the bf16 byte baseline (cheap); int4 alone was measured in MODE=quality.
        score("full")
        for keep in (0.75, 0.5):
            c = score(f"compose_keep{keep:.2f}+int4", ExpectedAttentionPress(compression_ratio=1 - keep), int4=True)
            m = c.get("rel_kv_bytes", keep * 0.273)
            score(f"evict_equal_mem_keep{m:.3f}", ExpectedAttentionPress(compression_ratio=1 - m))
    else:
        score("full")
        q4 = score("real4", int4=True)
        keep_eq = q4.get("rel_kv_bytes", 0.28)
        score("evict_keep0.25", ExpectedAttentionPress(compression_ratio=0.75))
        score(f"evict_equal_mem_keep{keep_eq:.3f}", ExpectedAttentionPress(compression_ratio=1 - keep_eq))
        score("evict_keep0.50", ExpectedAttentionPress(compression_ratio=0.50))
    res.append(r); json.dump(res, open(os.path.join(HERE, f"amd_gate1_{MODE}.json"), "w"), indent=1)
print("\nSUMMARY (F1, and KV bytes relative to bf16)")
for r in res:
    print(r["task"], {k: (v["f1"], v.get("rel_kv_bytes")) for k, v in r.items() if isinstance(v, dict)})
