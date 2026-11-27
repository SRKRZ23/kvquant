"""Which GPU kernels make ROCM_ATTN's long-context batch decode slow inside vLLM? (in-process engine + torch.profiler)
usage: ATTN=ROCM_ATTN|TRITON_ATTN python3 prof.py   -> prints top kernels for a (1 new token) and b (65 new tokens)
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import time, torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams

attn = os.environ.get("ATTN", "ROCM_ATTN")
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=32768, gpu_memory_utilization=0.90,
          enable_prefix_caching=False, attention_backend=attn)
C, BS = 30000, 16
p = [{"prompt_token_ids": list(range(3000 + 7 * i, 3000 + 7 * i + C))} for i in range(BS)]
llm.generate(p, SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)  # warm-up
for n in (1, 65):
    sp = SamplingParams(max_tokens=n, temperature=0, ignore_eos=True)
    torch.cuda.synchronize(); t = time.time()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate(p, sp, use_tqdm=False); torch.cuda.synchronize()
    wall = time.time() - t
    print(f"PROF {attn} new_tokens={n} wall={wall:.2f}s", flush=True)
    rows = sorted(prof.key_averages(), key=lambda e: -getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)))
    for e in rows[:12]:
        tot = getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3
        print(f"PROF {attn} n={n} {tot:10.1f} ms  calls={e.count:6d}  {e.key[:110]}", flush=True)
print("PROF_DONE", flush=True)
