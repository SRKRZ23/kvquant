"""Microbenchmark of vLLM's ROCm custom paged-attention decode kernel alone (no scheduler, no graphs).

Question: is ROCM_ATTN's long-context batch>1 collapse (Qwen2.5-7B, ctx 30k, bs16: 339 ms/step in vLLM) inside the
HIP kernel itself, or in how vLLM calls it? Sweeps batch, context, GQA ratio, max_seq_len (exact vs 32768 as baked
into CUDA graphs) and block placement (contiguous vs randomly scattered over a vLLM-sized pool).
Prints: RESULT PA gqa bs ctx max_seq_len placement ms_per_layer GBps
"""
import os, time, torch
from vllm import _custom_ops as ops

dev = "cuda"
dt = torch.bfloat16
HS, BS, X = 128, 16, 8                    # head size, block size, x = 16 bytes / 2
POOL_TOKENS = int(os.environ.get("POOL_TOKENS", "2873344"))   # what vLLM allocated for Qwen2.5-7B at util 0.90
NB = POOL_TOKENS // BS


def run(gqa, kvh, bs, ctx, max_seq_len, placement, iters=20):
    nh = gqa * kvh
    kc = torch.empty(NB, kvh, HS // X, BS, X, dtype=dt, device=dev).normal_()
    vc = torch.empty(NB, kvh, HS, BS, dtype=dt, device=dev).normal_()
    nblk = (ctx + BS - 1) // BS
    maxblk = (max(max_seq_len, ctx) + BS - 1) // BS
    if placement == "contig":
        ids = torch.arange(bs * nblk, device=dev, dtype=torch.int32)
    else:
        ids = torch.randperm(NB, device=dev, dtype=torch.int64)[: bs * nblk].to(torch.int32)
    bt = torch.zeros(bs, maxblk, dtype=torch.int32, device=dev)
    bt[:, :nblk] = ids.view(bs, nblk)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=dev)
    qsl = torch.arange(bs + 1, dtype=torch.int32, device=dev)
    q = torch.randn(bs, nh, HS, dtype=dt, device=dev)
    out = torch.empty_like(q)
    npart = (max_seq_len + 255) // 256
    tmp = torch.empty(bs, nh, npart, HS, dtype=dt, device=dev)
    es = torch.empty(bs, nh, npart, dtype=torch.float32, device=dev)
    ml = torch.empty_like(es)
    one = torch.ones(1, dtype=torch.float32, device=dev)

    def call():
        ops.paged_attention_rocm(out, es, ml, tmp, q, kc, vc, kvh, HS ** -0.5, bt, seq_lens, qsl, BS, max_seq_len,
                                 None, "auto", one, one, None)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    gb = bs * ctx * kvh * HS * 2 * 2 / 1e9     # K + V bytes read
    print(f"RESULT PA gqa={gqa} kvh={kvh} bs={bs} ctx={ctx} max_seq_len={max_seq_len} {placement} "
          f"{ms:.3f} ms/layer {gb / (ms / 1e3):.0f} GB/s", flush=True)
    del kc, vc; torch.cuda.empty_cache()


for gqa, kvh in ((7, 4), (4, 8)):            # Qwen2.5-7B (mfma16 path), Llama-3.1-8B (mfma4 path)
    for ctx in (8000, 30000):
        for bs in (1, 4, 16, 64):
            for msl, pl in ((ctx, "contig"), (32768, "contig"), (32768, "random")):
                run(gqa, kvh, bs, ctx, msl, pl)
print("PA_DONE")
