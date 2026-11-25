"""Kill test: can vLLM's TurboQuant 4-bit decode kernel be made fast on MI300X by launch-parameter tuning alone?

vLLM 0.27.1 launches _tq_decode_stage1 with grid (B, Hq, 32 splits), num_warps=1, BLOCK_KV=4 (the kernel's own
comment says 16). Each program handles one QUERY head, so with GQA (Qwen2.5-7B: 28 q heads, 4 kv heads) every KV
byte is read 7 times. This script times stage1+stage2 for a grid of (BLOCK_KV, num_warps, splits) on a valid
synthetic turboquant_4bit_nc cache, checks every variant against the default output, and prints a table.

Run inside the vLLM ROCm container:  python3 tq_decode_tune.py
"""
import itertools, json, math, os, sys
import torch
from vllm.v1.attention.ops import triton_turboquant_decode as TQ

dev = "cuda"
D, HQ, HK = 128, 28, 4                    # Qwen2.5-7B attention shape
MSE_BITS, VQB = 4, 4                      # turboquant_4bit_nc
KPS = math.ceil(D * MSE_BITS / 8) + 2     # 66: indices + fp16 norm
VPS = math.ceil(D * VQB / 8) + 4          # 68: codes + fp16 scale + fp16 zero
SLOT = KPS + VPS; SLOT += SLOT % 2        # 134
BS = 16                                   # vLLM ROCm block size


def make_cache(B, L, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    nblk = math.ceil(L / BS)
    cache = torch.randint(0, 256, (B * nblk, BS, HK, SLOT), generator=g, dtype=torch.uint8)
    one = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    small = torch.tensor([0.02], dtype=torch.float16).view(torch.uint8)
    cache[..., KPS - 2:KPS] = one                                  # key norm = 1.0 (valid fp16)
    cache[..., KPS + VPS - 4:KPS + VPS - 2] = small                # value scale / zero: valid fp16
    cache[..., KPS + VPS - 2:KPS + VPS] = small
    bt = torch.arange(B * nblk, dtype=torch.int32).view(B, nblk)
    return cache.to(dev), bt.to(dev), torch.full((B,), L, dtype=torch.int32, device=dev)


def run(q_rot, cache, bt, sl, cent, block_kv, warps, splits, gqa_note=False):
    B = q_rot.shape[0]
    cfg = TQ._get_layout(D, MSE_BITS, VQB, KPS)
    mid = torch.empty(B, HQ, splits, D + 1, dtype=torch.float32, device=dev)
    out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(B, HQ, dtype=torch.float32, device=dev)

    def go():
        TQ._tq_decode_stage1[(B, HQ, splits)](
            q_rot, cache, bt, sl, cent, mid,
            q_rot.stride(0), q_rot.stride(1), cache.stride(0), cache.stride(1), cache.stride(2), bt.stride(0),
            mid.stride(0), mid.stride(1), mid.stride(2),
            NUM_KV_HEADS=HK, HEAD_DIM=D, BLOCK_SIZE=BS, NUM_KV_SPLITS=splits, KV_GROUP_SIZE=HQ // HK,
            MSE_BITS=MSE_BITS, MSE_BYTES=cfg["mse_bytes"], KPS=KPS, VQB=VQB, VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=1 / math.sqrt(D), BLOCK_D=cfg["BLOCK_D"], BLOCK_KV=block_kv, KEY_FP8=0, NORM_CORRECTION=1,
            FP8_E4B15=0, num_warps=warps, num_stages=1)
        TQ._fwd_kernel_stage2[(B, HQ)](
            mid, out, lse, sl, mid.stride(0), mid.stride(1), mid.stride(2), out.stride(0), out.stride(1),
            lse.stride(0), NUM_KV_SPLITS=splits, BLOCK_DV=cfg["BLOCK_D"], Lv=D, OUTPUT_FP16=0,
            num_warps=4, num_stages=2)
        return out

    for _ in range(3):
        go()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n = 30
    s.record()
    for _ in range(n):
        go()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000, out.float().clone()


def main():
    torch.manual_seed(0)
    cent = torch.linspace(-2.5, 2.5, 2 ** MSE_BITS, device=dev) / math.sqrt(D)
    rows = []
    for B, L in ((1, 32768), (1, 4096), (16, 8192)):
        cache, bt, sl = make_cache(B, L)
        q_rot = torch.randn(B, HQ, D, device=dev, dtype=torch.float32)
        bytes_read = B * L * HK * SLOT
        base_us, ref = run(q_rot, cache, bt, sl, cent, 4, 1, 32)
        print(f"B={B} L={L}: default (BLOCK_KV 4, warps 1, splits 32) {base_us:.1f} us, "
              f"KV bytes {bytes_read/1e6:.1f} MB -> {bytes_read/base_us/1e3:.0f} GB/s effective", flush=True)
        best = (base_us, (4, 1, 32))
        for bkv, w, sp in itertools.product((4, 8, 16, 32, 64), (1, 2, 4, 8), (16, 32, 64, 128)):
            try:
                us, o = run(q_rot, cache, bt, sl, cent, bkv, w, sp)
            except Exception as ex:          # compile failures (register pressure etc.)
                continue
            err = ((o - ref).norm() / ref.norm()).item()
            ok = err < 2e-2 and torch.isfinite(o).all().item()
            rows.append({"B": B, "L": L, "block_kv": bkv, "warps": w, "splits": sp, "us": round(us, 1),
                         "speedup": round(base_us / us, 2), "rel_err_vs_default": err, "ok": ok})
            if ok and us < best[0]:
                best = (us, (bkv, w, sp))
        print(f"RESULT B={B} L={L} default {base_us:.1f} us | best {best[0]:.1f} us at (BLOCK_KV, warps, splits)="
              f"{best[1]} -> {base_us/best[0]:.2f}x, {bytes_read/best[0]/1e3:.0f} GB/s", flush=True)
        top = sorted([r for r in rows if r['B'] == B and r['L'] == L and r['ok']], key=lambda r: r['us'])[:5]
        for r in top:
            print("   ", r, flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tq_decode_tune_result.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
