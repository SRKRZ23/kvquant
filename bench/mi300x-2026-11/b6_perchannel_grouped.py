"""B6 kill-test: kvquant's per-channel int4 KV (lossless on LongBench) in a GQA-grouped split-K decode kernel.

Layout (kvquant live4): keys 4-bit per CHANNEL in 128-token groups (fp16 min + scale per group and channel), values 4-bit
per TOKEN (fp16 min + scale per token), two codes per byte (low nibble = even channel). One Triton program per
(batch, kv head, split) handles all query heads of the group with padded tl.dot, like vLLM PR #68834's grouped kernel.
Algebra used so the codes never get dequantized element by element:
  q.k = sum_d (q_d sc_d) code_d + sum_d q_d mn_d          (scale folded into q once per 128-token group)
  p.v = (p * vsc) @ vcode + sum_t p_t vmn_t               (per-token value scale folded into p)
A bf16 variant of the same kernel is the apples-to-apples speed reference. Compares against PR #68834's TurboQuant
grouped kernel (via tq_pr_rocm_bench.py, gfx942-tuned table if tq_gfx942_tune.json exists).
Prints RESULT lines; writes b6_result.json. Kill bar: per-channel int4 at least as fast as tuned TurboQuant-4bit AND
within 1.5x of the bf16 kernel at long context, with output error vs its own dequantized reference < 1e-2.
"""
import json, math, os, sys
import torch, triton
import triton.language as tl

dev = "cuda"


@triton.jit
def _stage1(Q, Kc, Kmn, Ksc, Vc, Vmn, Vsc, Kb, Vb, MidO, MidL, T,
            sQb, sQh, sKb, sKh, sGb, sGh, sVsb, sVsh, sMb, sMh, sMs, sLb, sLh,
            GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
            NSPLIT: tl.constexpr, BF16: tl.constexpr, SCALE: tl.constexpr):
    b = tl.program_id(0); h = tl.program_id(1); s = tl.program_id(2)
    D2: tl.constexpr = D // 2
    offs_h = tl.arange(0, BLOCK_H); hmask = offs_h < GROUP; hq = h * GROUP + offs_h
    offs_d2 = tl.arange(0, D2)
    qp = Q + b * sQb + hq[:, None] * sQh
    q_e = tl.load(qp + 2 * offs_d2[None, :], mask=hmask[:, None], other=0.0).to(tl.float32)
    q_o = tl.load(qp + 2 * offs_d2[None, :] + 1, mask=hmask[:, None], other=0.0).to(tl.float32)
    split_len = tl.cdiv(tl.cdiv(T, NSPLIT), 128) * 128
    start = s * split_len
    end = tl.minimum(start + split_len, T)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_e = tl.zeros([BLOCK_H, D2], tl.float32)
    acc_o = tl.zeros([BLOCK_H, D2], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        if BF16:
            kp = Kb + b * sKb + h * sKh + offs_n[:, None] * D
            k_e = tl.load(kp + 2 * offs_d2[None, :], mask=nmask[:, None], other=0.0)
            k_o = tl.load(kp + 2 * offs_d2[None, :] + 1, mask=nmask[:, None], other=0.0)
            sc = tl.dot(q_e.to(tl.bfloat16), tl.trans(k_e)) + tl.dot(q_o.to(tl.bfloat16), tl.trans(k_o))
        else:
            kc = tl.load(Kc + b * sKb + h * sKh + offs_n[:, None] * D2 + offs_d2[None, :], mask=nmask[:, None], other=0)
            lo = (kc & 15).to(tl.float16)
            hi = (kc >> 4).to(tl.float16)
            g = n0 // 128
            gp = b * sGb + h * sGh + g * D
            sc_e = tl.load(Ksc + gp + 2 * offs_d2).to(tl.float32)
            sc_o = tl.load(Ksc + gp + 2 * offs_d2 + 1).to(tl.float32)
            mn_e = tl.load(Kmn + gp + 2 * offs_d2).to(tl.float32)
            mn_o = tl.load(Kmn + gp + 2 * offs_d2 + 1).to(tl.float32)
            qm = tl.sum(q_e * mn_e[None, :], 1) + tl.sum(q_o * mn_o[None, :], 1)
            sc = tl.dot((q_e * sc_e[None, :]).to(tl.float16), tl.trans(lo)) + \
                tl.dot((q_o * sc_o[None, :]).to(tl.float16), tl.trans(hi)) + qm[:, None]
        sc = sc * SCALE
        sc = tl.where(nmask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc_e = acc_e * alpha[:, None]
        acc_o = acc_o * alpha[:, None]
        if BF16:
            vp = Vb + b * sKb + h * sKh + offs_n[:, None] * D
            v_e = tl.load(vp + 2 * offs_d2[None, :], mask=nmask[:, None], other=0.0)
            v_o = tl.load(vp + 2 * offs_d2[None, :] + 1, mask=nmask[:, None], other=0.0)
            pb = p.to(tl.bfloat16)
            acc_e += tl.dot(pb, v_e)
            acc_o += tl.dot(pb, v_o)
        else:
            vc = tl.load(Vc + b * sKb + h * sKh + offs_n[:, None] * D2 + offs_d2[None, :], mask=nmask[:, None], other=0)
            vlo = (vc & 15).to(tl.float16)
            vhi = (vc >> 4).to(tl.float16)
            vsc = tl.load(Vsc + b * sVsb + h * sVsh + offs_n, mask=nmask, other=0.0).to(tl.float32)
            vmn = tl.load(Vmn + b * sVsb + h * sVsh + offs_n, mask=nmask, other=0.0).to(tl.float32)
            pv = (p * vsc[None, :]).to(tl.float16)
            pm = tl.sum(p * vmn[None, :], 1)
            acc_e += tl.dot(pv, vlo) + pm[:, None]
            acc_o += tl.dot(pv, vhi) + pm[:, None]
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    op = MidO + b * sMb + hq[:, None] * sMh + s * sMs
    tl.store(op + 2 * offs_d2[None, :], acc_e / l_safe[:, None], mask=hmask[:, None])
    tl.store(op + 2 * offs_d2[None, :] + 1, acc_o / l_safe[:, None], mask=hmask[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    tl.store(MidL + b * sLb + hq * sLh + s, lse, mask=hmask)


@triton.jit
def _stage2(MidO, MidL, Out, sMb, sMh, sMs, sLb, sLh, sOb, sOh, NSPLIT: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0); hq = tl.program_id(1)
    offs_s = tl.arange(0, BS); smask = offs_s < NSPLIT
    lse = tl.load(MidL + b * sLb + hq * sLh + offs_s, mask=smask, other=float("-inf"))
    m = tl.max(lse, 0)
    w = tl.where(smask, tl.exp(lse - m), 0.0)
    w = w / tl.sum(w, 0)
    offs_d = tl.arange(0, D)
    o = tl.load(MidO + b * sMb + hq * sMh + offs_s[:, None] * sMs + offs_d[None, :], mask=smask[:, None], other=0.0)
    tl.store(Out + b * sOb + hq * sOh + offs_d, tl.sum(o * w[:, None], 0).to(Out.dtype.element_ty))


def quantize(K, V):
    """kvquant live4 scheme on [B, Hk, T, D] fp32 tensors (T multiple of 128)."""
    B, Hk, T, D = K.shape
    kg = K.view(B, Hk, T // 128, 128, D)
    kmn = kg.amin(3); ksc = ((kg.amax(3) - kmn).clamp_min(1e-6) / 15)
    kmn16, ksc16 = kmn.half(), ksc.half()
    kq = ((kg - kmn16.float()[:, :, :, None]) / ksc16.float()[:, :, :, None]).round().clamp(0, 15).to(torch.uint8).view(B, Hk, T, D)
    vmn = V.amin(-1); vsc = ((V.amax(-1) - vmn).clamp_min(1e-6) / 15)
    vmn16, vsc16 = vmn.half(), vsc.half()
    vq = ((V - vmn16.float()[..., None]) / vsc16.float()[..., None]).round().clamp(0, 15).to(torch.uint8)
    pack = lambda q: (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    kdq = (kq.view(B, Hk, T // 128, 128, D).float() * ksc16.float()[:, :, :, None] + kmn16.float()[:, :, :, None]).view(B, Hk, T, D)
    vdq = vq.float() * vsc16.float()[..., None] + vmn16.float()[..., None]
    return pack(kq), kmn16.contiguous(), ksc16.contiguous(), pack(vq), vmn16.contiguous(), vsc16.contiguous(), kdq, vdq


def run(q, T, nsplit, bn, warps, int4=None, bf=None, Hk=4):
    B, Hq, D = q.shape; G = Hq // Hk
    mid = torch.empty(B, Hq, nsplit, D, device=dev, dtype=torch.float32)
    midl = torch.empty(B, Hq, nsplit, device=dev, dtype=torch.float32)
    out = torch.empty(B, Hq, D, device=dev, dtype=torch.bfloat16)
    if int4 is not None:
        kc, kmn, ksc, vc, vmn, vsc = int4; kb = vb = kc
        sKb, sKh = kc.stride(0), kc.stride(1); sGb, sGh = ksc.stride(0), ksc.stride(1); sVb, sVh = vsc.stride(0), vsc.stride(1)
    else:
        kb, vb = bf; kc = kmn = ksc = vc = vmn = vsc = kb
        sKb, sKh = kb.stride(0), kb.stride(1); sGb = sGh = sVb = sVh = 0
    _stage1[(B, Hk, nsplit)](q, kc, kmn, ksc, vc, vmn, vsc, kb, vb, mid, midl, T,
                             q.stride(0), q.stride(1), sKb, sKh, sGb, sGh, sVb, sVh,
                             mid.stride(0), mid.stride(1), mid.stride(2), midl.stride(0), midl.stride(1),
                             GROUP=G, BLOCK_H=max(16, triton.next_power_of_2(G)), D=D, BLOCK_N=bn, NSPLIT=nsplit,
                             BF16=int4 is None, SCALE=1 / math.sqrt(D), num_warps=warps, num_stages=1)
    _stage2[(B, Hq)](mid, midl, out, mid.stride(0), mid.stride(1), mid.stride(2), midl.stride(0), midl.stride(1),
                     out.stride(0), out.stride(1), NSPLIT=nsplit, D=D, BS=triton.next_power_of_2(nsplit), num_warps=4)
    return out


def ref_attn(q, K, V):
    B, Hq, D = q.shape; Hk = K.shape[1]; G = Hq // Hk
    k = K.repeat_interleave(G, 1); v = V.repeat_interleave(G, 1)
    s = torch.einsum("bhd,bhtd->bht", q.float(), k) / math.sqrt(D)
    return torch.einsum("bht,bhtd->bhd", torch.softmax(s, -1), v)


def timeit(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1000


def tq_best(B, L):
    """PR #68834 grouped TurboQuant on the same shape (paged layout), gfx942-tuned table if available."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import tq_pr_rocm_bench as TB
        mod = TB.MODS["pr68834"]
        if os.path.exists("tq_gfx942_tune.json"):
            hn, bkv, w, sp = json.load(open("tq_gfx942_tune.json"))["top_single"][0][0]
            mod._MSE_GQA_LAUNCH[(True, 4, 4)] = (hn, bkv, w)
        Pi = TB.hadamard(128).to(dev); cent = (torch.linspace(-2.5, 2.5, 16) / math.sqrt(128)).to(dev)
        c, bt, sl, kps = TB.make(B, L, 4, 128)
        q = torch.randn(B, 28, 128, device=dev, dtype=torch.bfloat16)
        TB.plat.force_cuda = True
        t = min(TB.timeit(lambda: TB.call(mod, q, c, bt, sl, Pi, cent, kps, sp)) for sp in (32, 64, 128))
        TB.plat.force_cuda = False
        return t
    except Exception as ex:
        print("TQ comparison unavailable:", repr(ex)[:200], flush=True)
        return None


def main():
    torch.manual_seed(0)
    print("torch", torch.__version__, "triton", triton.__version__, torch.cuda.get_device_name(0), flush=True)
    Hq, Hk, D = 28, 4, 128
    rows = []
    for (B, L) in ((1, 32768), (8, 16384), (16, 8192), (32, 8192), (64, 4096)):
        K = torch.randn(B, Hk, L, D, device=dev)
        K[..., :8] += 40 * torch.randn(1, Hk, 1, 8, device=dev)        # Qwen-like large per-channel key offsets
        V = torch.randn(B, Hk, L, D, device=dev)
        q = torch.randn(B, Hq, D, device=dev, dtype=torch.bfloat16)
        kc, kmn, ksc, vc, vmn, vsc, kdq, vdq = quantize(K, V)
        ref4 = ref_attn(q, kdq, vdq)
        Kb, Vb = K.bfloat16().contiguous(), V.bfloat16().contiguous()
        refb = ref_attn(q, Kb.float(), Vb.float())
        best4 = bestb = None
        for nsplit in (32, 64, 128):
            for bn in (32, 64, 128):
                for w in (4, 8):
                    try:
                        o4 = run(q, L, nsplit, bn, w, int4=(kc, kmn, ksc, vc, vmn, vsc)).float()
                        e4 = ((o4 - ref4).norm() / ref4.norm()).item()
                        t4 = timeit(lambda: run(q, L, nsplit, bn, w, int4=(kc, kmn, ksc, vc, vmn, vsc)), n=20)
                        if e4 < 1e-2 and (best4 is None or t4 < best4[0]):
                            best4 = (t4, nsplit, bn, w, e4)
                        ob = run(q, L, nsplit, bn, w, bf=(Kb, Vb)).float()
                        eb = ((ob - refb).norm() / refb.norm()).item()
                        tb = timeit(lambda: run(q, L, nsplit, bn, w, bf=(Kb, Vb)), n=20)
                        if eb < 1e-2 and (bestb is None or tb < bestb[0]):
                            bestb = (tb, nsplit, bn, w, eb)
                    except Exception as ex:
                        if best4 is None and bestb is None and nsplit == 32 and bn == 32 and w == 4:
                            print("first config failed:", repr(ex)[:300], flush=True)
        tq = tq_best(B, L)
        bytes4 = B * Hk * L * (D + 4) + B * Hk * (L // 128) * D * 4
        bytesb = B * Hk * L * D * 2 * 2
        r = {"B": B, "L": L,
             "pc_int4_us": round(best4[0], 1) if best4 else None, "pc_int4_cfg": best4[1:4] if best4 else None,
             "pc_int4_relerr": best4[4] if best4 else None,
             "bf16_us": round(bestb[0], 1) if bestb else None, "bf16_cfg": bestb[1:4] if bestb else None,
             "tq4_grouped_us": round(tq, 1) if tq else None}
        if best4: r["pc_int4_GBps"] = round(bytes4 / best4[0] / 1e3)
        if bestb: r["bf16_GBps"] = round(bytesb / bestb[0] / 1e3)
        rows.append(r); print("RESULT", json.dumps(r), flush=True)
    json.dump(rows, open("b6_result.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
