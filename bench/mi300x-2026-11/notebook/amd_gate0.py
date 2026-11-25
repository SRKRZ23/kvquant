"""kvquant gate on a REAL AMD GPU (AMD AI Notebooks / any ROCm box). One file, run:  !python amd_gate0.py
Env knobs: N (questions per task, default 30), MODEL (override HF id), TASKS (comma list).
Prints RESULT lines as each (task, arm) finishes and saves amd_gate0_result.json next to this file.
Arms: full bf16 cache | simulated int4 (fake-quant after prefill) | REAL int4 (kvquant live4, prefill fix 2026-11-25).
"""
import os, sys, subprocess, json, time
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True)
print("== environment", flush=True)
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| hip", torch.version.hip, "| gpu", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "no GPU visible to torch"
P = torch.cuda.get_device_properties(0); VRAM = P.total_memory / 2**30
print("gpu:", P.name, "| VRAM GB", round(VRAM, 1), "| count", torch.cuda.device_count(), flush=True)
try:
    import transformers; tv = int(transformers.__version__.split(".")[0])
except Exception:
    tv = 0
if tv < 5:
    sh(f"{sys.executable} -m pip install -q -U 'transformers>=5' accelerate")
sh(f"{sys.executable} -m pip install -q 'datasets==2.21.0' pandas fsspec huggingface_hub")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LIVE4 = '"""Live int4 KV cache: actually HOLDS 4-bit keys/values in memory during generation.\n\nScheme (KIVI-style, the one measured on LongBench with the simulated int4):\n\n* keys:   asymmetric 4-bit, **per channel**, in groups of ``group`` tokens along the sequence\n          (one fp16 min + scale per channel per group). Keys have outlier channels, so per-channel\n          scales are what keep int4 keys accurate.\n* values: asymmetric 4-bit, **per token** over ``head_dim`` (one fp16 min + scale per token).\n* two 4-bit codes are packed into one uint8.\n* the newest ``< group`` tokens stay in the original dtype (a small residual buffer) until a full\n  key group can be quantized; values are quantized at once (per token needs no group).\n\nMemory held per token and head, head_dim = 128, group = 128, bf16 baseline 512 B:\nkeys 64 B + 4 B/128-token group share, values 64 B + 4 B  ->  ~136 B, i.e. ~3.8x smaller than bf16.\n\nHow it hooks in: like :mod:`kvquant.live`, each cache layer\'s class is swapped for a subclass whose\n``keys`` / ``values`` are properties. transformers\' layer ``update`` does\n``self.keys = cat([self.keys, new], -2)``; the setter sees a longer tensor and only quantizes the\nnew tail (the prefix it receives is its own dequantized output, so it is not re-quantized and no\nerror accumulates). A shorter or reordered tensor (crop, beam reorder, eviction) triggers a full\nre-quantization of what it was given.\n\nReads dequantize the whole layer (pure PyTorch, no fused kernel): the memory HELD is int4, the\nattention math runs in the original dtype. A fused int4 attention kernel is future work; measure\nspeed with that in mind.\n"""\nfrom __future__ import annotations\nimport torch\n\n_variants: dict = {}\n\n\ndef _pack(q: torch.Tensor) -> torch.Tensor:\n    """uint8 codes in [0, 15], last dim even -> packed uint8 with half the last dim."""\n    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)\n\n\ndef _unpack(p: torch.Tensor) -> torch.Tensor:\n    lo = p & 0x0F\n    hi = p >> 4\n    return torch.stack((lo, hi), dim=-1).flatten(-2)\n\n\ndef _qasym(x: torch.Tensor, dim: int):\n    """4-bit asymmetric quantization along ``dim``; returns codes (uint8) and fp16 min, scale."""\n    mn = x.amin(dim=dim, keepdim=True)\n    mx = x.amax(dim=dim, keepdim=True)\n    scale = ((mx - mn).float().clamp_min(1e-6) / 15.0).to(torch.float16)\n    mn = mn.to(torch.float16)\n    q = ((x.float() - mn.float()) / scale.float()).round_().clamp_(0, 15).to(torch.uint8)\n    return q, mn, scale\n\n\nclass _Store:\n    """int4 storage for one K or V tensor of one layer, shape [batch, heads, seq, head_dim]."""\n\n    def __init__(self, kind: str, group: int):\n        self.kind, self.group = kind, group\n        self.packed = []      # list of packed uint8 chunks [B, H, n_i, D/2]\n        self.mins = []        # fp16 [B, H, 1, D] (keys, per group) or [B, H, n_i, 1] (values)\n        self.scales = []\n        self.resid = None     # keys only: newest tokens not yet a full group, original dtype\n        self.n_q = 0          # tokens held quantized\n        self.dtype = None\n\n    def length(self) -> int:\n        return self.n_q + (0 if self.resid is None else self.resid.shape[-2])\n\n    def clear(self):\n        self.packed, self.mins, self.scales, self.resid, self.n_q = [], [], [], None, 0\n\n    def _quant_append(self, x: torch.Tensor):\n        if x.shape[-1] % 2:\n            raise ValueError("head_dim must be even for 4-bit packing")\n        if self.kind == "k":\n            q, mn, sc = _qasym(x, dim=-2)        # per channel over the group\'s tokens\n        else:\n            q, mn, sc = _qasym(x, dim=-1)        # per token over head_dim\n        self.packed.append(_pack(q))\n        self.mins.append(mn)\n        self.scales.append(sc)\n        self.n_q += x.shape[-2]\n\n    def append(self, new: torch.Tensor):\n        self.dtype = new.dtype\n        if self.kind == "v":\n            self._quant_append(new)\n            return\n        buf = new if self.resid is None else torch.cat([self.resid, new], dim=-2)\n        full = (buf.shape[-2] // self.group) * self.group\n        for s in range(0, full, self.group):\n            self._quant_append(buf[..., s:s + self.group, :])\n        self.resid = buf[..., full:, :] if buf.shape[-2] > full else None\n\n    def set(self, t: torch.Tensor):\n        if t is None or t.numel() == 0:\n            self.clear()\n            self.empty = t               # transformers\' lazy init sets an empty 1-D tensor\n            return\n        n = self.length()\n        if n and t.shape[-2] > n and t.shape[:-2] == self._lead():\n            self.append(t[..., n:, :])           # usual decode / chunked-prefill path\n        elif n and t.shape[-2] == n and t.shape[:-2] == self._lead():\n            return                               # unchanged (e.g. set back after a read)\n        else:\n            self.clear()\n            self.append(t)                       # first fill, crop, reorder: re-quantize all\n\n    def _lead(self):\n        if self.packed:\n            return tuple(self.packed[0].shape[:-2])\n        return tuple(self.resid.shape[:-2]) if self.resid is not None else None\n\n    def get(self):\n        if not self.packed and self.resid is None:\n            return getattr(self, "empty", None)\n        parts = []\n        for p, mn, sc in zip(self.packed, self.mins, self.scales):\n            parts.append((_unpack(p).float() * sc.float() + mn.float()).to(self.dtype))\n        if self.resid is not None:\n            parts.append(self.resid)\n        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)\n\n    def nbytes(self) -> int:\n        b = 0\n        for lst in (self.packed, self.mins, self.scales):\n            b += sum(t.element_size() * t.nelement() for t in lst)\n        if self.resid is not None:\n            b += self.resid.element_size() * self.resid.nelement()\n        return b\n\n\ndef _variant(base):\n    if base in _variants:\n        return _variants[base]\n\n    class Int4Layer(base):\n        _kvquant_int4 = True\n\n        @property\n        def keys(self):\n            return self._k4.get()\n\n        @keys.setter\n        def keys(self, v):\n            self._k4.set(v)\n\n        @property\n        def values(self):\n            return self._v4.get()\n\n        @values.setter\n        def values(self, v):\n            self._v4.set(v)\n\n        def update(self, key_states, value_states, *args, **kwargs):\n            # one dequantization per layer per step (the base class would read keys/values again)\n            if not getattr(self, "is_initialized", True):\n                self.lazy_initialization(key_states, value_states)\n            # Attention for THIS step uses the stored past (dequantized) plus the new tokens at full precision,\n            # as KIVI and serving stacks do: the prompt attends to itself in the original dtype during prefill and\n            # only the STORED copy is 4-bit. Returning the quantized new tokens made prefill attend to 4-bit\n            # keys/values and compounded the error through the layers (gate 0, 2026-11-25: real < simulated).\n            past_k, past_v = self._k4.get(), self._v4.get()\n            self._k4.append(key_states)\n            self._v4.append(value_states)\n            if past_k is None or past_k.numel() == 0:\n                return key_states, value_states\n            return torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)\n\n        def get_seq_length(self, *args, **kwargs) -> int:\n            return self._k4.length()\n\n        def int4_bytes(self) -> int:\n            return self._k4.nbytes() + self._v4.nbytes()\n\n    if base.__name__ != "DynamicLayer":\n        # sliding-window / static layers keep their own update logic and go through the properties\n        del Int4Layer.update\n        del Int4Layer.get_seq_length\n    _variants[base] = Int4Layer\n    return Int4Layer\n\n\ndef make_live_int4(cache, group: int = 128):\n    """Convert a transformers>=5 cache in place so its layers hold int4. Idempotent."""\n    layers = getattr(cache, "layers", None)\n    if layers is None:\n        raise TypeError("expected a transformers>=5 Cache exposing `.layers`")\n    for layer in layers:\n        if getattr(type(layer), "_kvquant_int4", False):\n            continue\n        k = layer.__dict__.get("keys", getattr(layer, "keys", None))\n        v = layer.__dict__.get("values", getattr(layer, "values", None))\n        layer.__dict__.pop("keys", None)\n        layer.__dict__.pop("values", None)\n        layer.__class__ = _variant(type(layer))\n        layer._k4 = _Store("k", group)\n        layer._v4 = _Store("v", group)\n        layer.keys = k\n        layer.values = v\n    return cache\n\n\ndef held_bytes4(cache) -> int:\n    """Live byte footprint of the cache\'s K/V storage (int4 layers counted as stored)."""\n    total = 0\n    for layer in getattr(cache, "layers", []):\n        if hasattr(layer, "int4_bytes"):\n            total += layer.int4_bytes()\n        else:\n            for name in ("keys", "values"):\n                t = getattr(layer, name, None)\n                if t is not None:\n                    total += t.element_size() * t.nelement()\n    return total\n'
open(os.path.join(HERE, "live4.py"), "w").write(LIVE4)
from live4 import make_live_int4, held_bytes4
# ---- self-test on the GPU: prefill through the int4 cache must equal the plain cache (prefill fix)
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
torch.manual_seed(0)
_cfg = Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, max_position_embeddings=1024)
_m = Qwen2ForCausalLM(_cfg).eval().cuda(); _ids = torch.randint(0, 128, (1, 300), device="cuda")
with torch.no_grad():
    _a = _m(_ids, past_key_values=DynamicCache(config=_cfg), use_cache=True).logits
    _c = make_live_int4(DynamicCache(config=_cfg)); _b = _m(_ids, past_key_values=_c, use_cache=True).logits
print("SELFTEST prefill identical on this GPU:", bool(torch.equal(_a, _b)), "| max abs diff", float((_a - _b).abs().max()), flush=True)
del _m, _c
import re, string, json, time, traceback, os
from collections import Counter
import torch
MODELS = [os.environ.get("MODEL") or ("Qwen/Qwen2.5-7B-Instruct" if VRAM >= 40 else "Qwen/Qwen2.5-3B-Instruct" if VRAM >= 14 else "Qwen/Qwen2.5-1.5B-Instruct")]
print("model:", MODELS[0], flush=True)
TASKS = os.environ.get("TASKS", "multifieldqa_en,hotpotqa,2wikimqa").split(",")
N = int(os.environ.get("N", "30")); CTX_CAP = 3500; MAXNEW = 64
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
def _qsim(x, nbits, dim):
    qmax = (1 << nbits) - 1
    mn = x.amin(dim=dim, keepdim=True); mx = x.amax(dim=dim, keepdim=True)
    scale = (mx - mn).clamp_min(1e-6) / qmax
    return (((x - mn) / scale).round().clamp_(0, qmax)) * scale + mn
def load_raw(task):
    from datasets import load_dataset
    try:
        return load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    except Exception as e:
        print("  script load failed, parquet:", str(e)[:80], flush=True)
        import pandas as pd
        return pd.read_parquet(f"hf://datasets/THUDM/LongBench@refs/convert/parquet/{task}/test/0000.parquet").to_dict("records")
def examples(tok, task):
    exs = []
    for row in load_raw(task):
        ctx = row["context"]; ids = tok(ctx, return_tensors="pt").input_ids[0]
        if ids.shape[0] > CTX_CAP:
            h = CTX_CAP // 2; ctx = tok.decode(torch.cat([ids[:h], ids[-h:]]), skip_special_tokens=True)
        exs.append((ctx, row["input"], row["answers"]))
        if len(exs) >= N: break
    return exs
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
INSTR = "Answer the question based only on the text above. Give only the answer, no other words.\nQuestion: "
def gen(model, tok, ids, arm):
    cache = DynamicCache(config=model.config)
    if arm == "real4": make_live_int4(cache)
    eos = set([tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else []))
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        if arm == "sim4":
            for L in cache.layers:
                L.keys = _qsim(L.keys, 4, 2); L.values = _qsim(L.values, 4, -1)
        S = ids.shape[1]
        if arm == "real4": held = held_bytes4(cache)
        else: held = sum(L.keys.element_size() * L.keys.nelement() + L.values.element_size() * L.values.nelement() for L in cache.layers)
        tok_ids = []; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(MAXNEW):
            if nxt.item() in eos: break
            tok_ids.append(nxt.item())
            out = model(nxt, past_key_values=cache, use_cache=True); nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); dt = time.time() - t
    return tok.decode(tok_ids, skip_special_tokens=True), held, S, dt / max(1, len(tok_ids))
res = []
for name in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto").eval()
        for task in TASKS:
            exs = examples(tok, task); r = {"model": name, "task": task, "n": len(exs)}
            for arm in ("full", "sim4", "real4"):
                tot = 0.0; hb = 0; ntok = 0; ms = []; err = 0; t0 = time.time()
                for ctx, q, ans in exs:
                    try:
                        msgs = [{"role": "user", "content": ctx + "\n\n" + INSTR + q + "\nAnswer:"}]
                        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                        if not torch.is_tensor(ids): ids = ids["input_ids"]
                        ids = ids.to(model.device)
                        a, held, S, mspt = gen(model, tok, ids, arm)
                        tot += qa_f1(a, ans); hb += held; ntok += S; ms.append(mspt)
                    except Exception as ex:
                        err += 1
                        if err == 1: print("   err", arm, type(ex).__name__, str(ex)[:200], flush=True)
                r[arm] = {"f1": round(tot / len(exs), 4), "bytes_per_token": round(hb / max(1, ntok), 1),
                          "ms_per_decode_token": round(1000 * sum(ms) / max(1, len(ms)), 2), "errors": err,
                          "wall_s": round(time.time() - t0)}
                r[arm]["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); torch.cuda.reset_peak_memory_stats()
                print("RESULT", name.split("/")[-1], task, arm, json.dumps(r[arm]), flush=True)
            r["bf16_over_real4_bytes"] = round(r["full"]["bytes_per_token"] / max(1, r["real4"]["bytes_per_token"]), 3)
            res.append(r); json.dump(res, open(os.path.join(HERE, "amd_gate0_result.json"), "w"), indent=1)
        del model; torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
print("\n| model | task | full F1 | sim int4 F1 | REAL int4 F1 | bf16/real4 bytes | ms/tok full vs real4 |")
print("|---|---|---|---|---|---|---|")
for r in res:
    print(f"| {r['model'].split('/')[-1]} | {r['task']} | {r['full']['f1']:.3f} | {r['sim4']['f1']:.3f} | {r['real4']['f1']:.3f} | "
          f"{r['bf16_over_real4_bytes']:.2f}x | {r['full']['ms_per_decode_token']} vs {r['real4']['ms_per_decode_token']} |")
