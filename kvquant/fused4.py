"""Fused decode attention over the live int4 KV cache (kvquant.live4), without dequantizing the cache.

Why: live4 holds K/V in 4 bits, but the plain path dequantizes the WHOLE layer cache to bf16 on every decode
step and then runs ordinary attention. That reads and writes the cache several times per token (measured on an
MI300X: 5.5-6x slower decoding than a bf16 cache). Decoding is memory bound, so the right path reads the 4-bit
codes once and dequantizes them in registers inside the attention kernel.

What this module does, per decode step (one new token) and per layer:

* region A, tokens [0, n_q): keys and values both 4-bit -> Triton kernel (flash-decoding: the region is split
  into chunks, each program does an online softmax over its chunk for all query heads that share one KV head,
  then the chunks are merged). Keys: per-channel scales per 128-token group; values: per-token scales.
* region B, tokens [n_q, n_old) plus the new token: keys still in full precision (the residual that has not
  filled a group), values 4-bit -> small, done in PyTorch.
* the two partial softmaxes are merged exactly (log-sum-exp), then the new token is appended to the store.

Prefill (many query tokens) is unchanged: it runs ordinary SDPA on the full-precision prompt (see live4's
prefill rule). Only batch size 1 without padding is supported on the fused path (the decode mask is ignored).

Usage::

    from kvquant.fused4 import enable
    enable(model)                       # registers attn_implementation "kvq_int4" and switches the model to it
    cache = make_live_int4(DynamicCache(config=model.config), fused=True)
    model.generate(ids, past_key_values=cache, ...)

``backend``: "triton" on a GPU, "torch" anywhere (reference implementation of the same math, used in tests).
"""
from __future__ import annotations
import math
import torch

BACKEND = "auto"        # "auto" | "triton" | "torch"
_ACTIVE = {"cache": None}
_STATS = {"fused_steps": 0}

try:  # Triton is optional (not available on macOS); the torch reference path covers CPU tests
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    _HAS_TRITON = False


# ----------------------------------------------------------------------------------------------- math helpers
def _partial(q, k, v, scaling):
    """softmax partial of q [B,H,R,D] over k,v [B,H,N,D] (fp32): returns m [B,H,R], l [B,H,R], o [B,H,R,D]"""
    s = torch.einsum("bhrd,bhnd->bhrn", q, k) * scaling
    m = s.amax(-1)
    p = torch.exp(s - m[..., None])
    return m, p.sum(-1), torch.einsum("bhrn,bhnd->bhrd", p, v)


def _merge(parts):
    m = torch.stack([p[0] for p in parts]).amax(0)
    l = sum(p[1] * torch.exp(p[0] - m) for p in parts)
    o = sum(p[2] * torch.exp(p[0] - m)[..., None] for p in parts)
    return o / l[..., None]


def region_torch(q, ks, vs, scaling, n):
    """reference for region A: tokens [0, n) with both K and V quantized (n a multiple of the key group)"""
    k = ks.dequant(0, n, torch.float32)
    v = vs.dequant(0, n, torch.float32)
    return _partial(q.float(), k, v, scaling)


# ----------------------------------------------------------------------------------------------- Triton kernel
if _HAS_TRITON:
    @triton.jit
    def _int4_decode_kernel(
        Q, KC, KS, KM, VC, VS, VM, OM, OL, OO,
        s_qb, s_qh, s_qr,
        s_kcb, s_kch, s_kcn,
        s_ksb, s_ksh, s_ksg,
        s_vcb, s_vch, s_vcn,
        s_vsb, s_vsh, s_vsn,
        N, R, H, S, CHUNK, scaling,
        G: tl.constexpr, D2: tl.constexpr, BN: tl.constexpr, RP: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_r = tl.arange(0, RP)
        offs_j = tl.arange(0, D2)
        offs_n = tl.arange(0, BN)
        rmask = offs_r < R
        qb = Q + b * s_qb + h * s_qh + offs_r[:, None] * s_qr
        q_e = (tl.load(qb + 2 * offs_j[None, :], mask=rmask[:, None], other=0.0) * scaling).to(tl.float16)
        q_o = (tl.load(qb + 2 * offs_j[None, :] + 1, mask=rmask[:, None], other=0.0) * scaling).to(tl.float16)

        m_i = tl.full([RP], float("-inf"), tl.float32)
        l_i = tl.zeros([RP], tl.float32)
        acc_e = tl.zeros([RP, D2], tl.float32)
        acc_o = tl.zeros([RP, D2], tl.float32)
        start = pid_s * CHUNK
        end = tl.minimum(start + CHUNK, N)
        for n0 in range(start, end, BN):
            n = n0 + offs_n
            nmask = n < end
            g = n0 // G                                   # BN divides G: one key group per block
            kc = tl.load(KC + b * s_kcb + h * s_kch + n[:, None] * s_kcn + offs_j[None, :],
                         mask=nmask[:, None], other=0)
            ksb = KS + b * s_ksb + h * s_ksh + g * s_ksg
            kmb = KM + b * s_ksb + h * s_ksh + g * s_ksg
            ks_e = tl.load(ksb + 2 * offs_j).to(tl.float32)
            ks_o = tl.load(ksb + 2 * offs_j + 1).to(tl.float32)
            km_e = tl.load(kmb + 2 * offs_j).to(tl.float32)
            km_o = tl.load(kmb + 2 * offs_j + 1).to(tl.float32)
            k_e = ((kc & 15).to(tl.float32) * ks_e[None, :] + km_e[None, :]).to(tl.float16)
            k_o = ((kc >> 4).to(tl.float32) * ks_o[None, :] + km_o[None, :]).to(tl.float16)
            s = tl.dot(q_e, tl.trans(k_e)) + tl.dot(q_o, tl.trans(k_o))          # [RP, BN] fp32
            s = tl.where(nmask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            vc = tl.load(VC + b * s_vcb + h * s_vch + n[:, None] * s_vcn + offs_j[None, :],
                         mask=nmask[:, None], other=0)
            vsc = tl.load(VS + b * s_vsb + h * s_vsh + n * s_vsn, mask=nmask, other=0.0).to(tl.float32)
            vmn = tl.load(VM + b * s_vsb + h * s_vsh + n * s_vsn, mask=nmask, other=0.0).to(tl.float32)
            v_e = ((vc & 15).to(tl.float32) * vsc[:, None] + vmn[:, None]).to(tl.float16)
            v_o = ((vc >> 4).to(tl.float32) * vsc[:, None] + vmn[:, None]).to(tl.float16)
            pp = p.to(tl.float16)
            acc_e = acc_e * alpha[:, None] + tl.dot(pp, v_e)
            acc_o = acc_o * alpha[:, None] + tl.dot(pp, v_o)
            m_i = m_new
        row = (pid_bh * S + pid_s) * RP + offs_r
        tl.store(OM + row, m_i)
        tl.store(OL + row, l_i)
        ob = OO + row[:, None] * (2 * D2)
        tl.store(ob + 2 * offs_j[None, :], acc_e)
        tl.store(ob + 2 * offs_j[None, :] + 1, acc_o)


if _HAS_TRITON:
    @triton.jit
    def _tail_merge_kernel(
        Q, TK, TV, OM, OL, OO, OUT,
        s_qb, s_qh, s_qr, s_tb, s_th, s_tn,
        T, R, H, S, scaling,
        D: tl.constexpr, BT: tl.constexpr, RP: tl.constexpr,
    ):
        """softmax partial over the full-precision tail (key residual + new token), merged with the S partials of
        region A, written as the final normalized output. One program per (batch, kv head)."""
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs_r = tl.arange(0, RP)
        offs_d = tl.arange(0, D)
        offs_t = tl.arange(0, BT)
        rmask = offs_r < R
        tmask = offs_t < T
        q = tl.load(Q + b * s_qb + h * s_qh + offs_r[:, None] * s_qr + offs_d[None, :], mask=rmask[:, None], other=0.0)
        q = q * scaling
        tb = b * s_tb + h * s_th + offs_t[:, None] * s_tn + offs_d[None, :]
        k = tl.load(TK + tb, mask=tmask[:, None], other=0.0)
        v = tl.load(TV + tb, mask=tmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_run = tl.max(s, 1)
        p = tl.exp(s - m_run[:, None])
        l_run = tl.sum(p, 1)
        o_run = tl.dot(p, v, input_precision="ieee")
        for i in range(0, S):
            row = (pid * S + i) * RP + offs_r
            m_s = tl.load(OM + row)
            l_s = tl.load(OL + row)
            o_s = tl.load(OO + row[:, None] * D + offs_d[None, :])
            m_new = tl.maximum(m_run, m_s)
            a = tl.exp(m_run - m_new)
            c = tl.exp(m_s - m_new)
            l_run = l_run * a + l_s * c
            o_run = o_run * a[:, None] + o_s * c[:, None]
            m_run = m_new
        out = o_run / l_run[:, None]
        tl.store(OUT + b * s_qb + h * s_qh + offs_r[:, None] * s_qr + offs_d[None, :], out, mask=rmask[:, None])


def region_triton(q, ks, vs, scaling, n, chunk=512, BN=64, raw=False):
    """region A on the GPU; same contract as region_torch (raw=True: per-chunk partials (OM, OL, OO, S, RP))"""
    B, H, R, D = q.shape
    G = ks.group
    assert G % BN == 0 and chunk % G == 0 and D % 2 == 0
    assert ks.codes.stride(-1) == 1 and ks.sc.stride(-1) == 1 and vs.codes.stride(-1) == 1
    RP = max(16, 1 << (R - 1).bit_length())
    S = -(-n // chunk)
    qf = q.float().contiguous()
    dev = q.device
    OM = torch.empty(B * H, S, RP, device=dev, dtype=torch.float32)
    OL = torch.empty_like(OM)
    OO = torch.empty(B * H, S, RP, D, device=dev, dtype=torch.float32)
    _int4_decode_kernel[(B * H, S)](
        qf, ks.codes, ks.sc, ks.mn, vs.codes, vs.sc, vs.mn, OM, OL, OO,
        qf.stride(0), qf.stride(1), qf.stride(2),
        ks.codes.stride(0), ks.codes.stride(1), ks.codes.stride(2),
        ks.sc.stride(0), ks.sc.stride(1), ks.sc.stride(2),
        vs.codes.stride(0), vs.codes.stride(1), vs.codes.stride(2),
        vs.sc.stride(0), vs.sc.stride(1), vs.sc.stride(2),
        n, R, H, S, chunk, scaling,
        G=G, D2=D // 2, BN=BN, RP=RP,
    )
    if raw:
        return OM, OL, OO, S, RP
    OM = OM.view(B, H, S, RP)[..., :R]
    OL = OL.view(B, H, S, RP)[..., :R]
    OO = OO.view(B, H, S, RP, D)[..., :R, :]
    m = OM.amax(2)
    w = torch.exp(OM - m[:, :, None])
    return m, (OL * w).sum(2), (OO * w[..., None]).sum(2)


def _use_triton(device):
    if BACKEND == "torch":
        return False
    if BACKEND == "triton":
        return True
    return _HAS_TRITON and device.type == "cuda"


# ----------------------------------------------------------------------------------------------- decode step
def decode_step(layer, query, kq, vq, scaling):
    """attention of query [B,Hq,1,D] over the layer's int4 store + the new token (kq, vq [B,Hkv,1,D]);
    appends the new token to the store; returns [B,Hq,1,D] in query.dtype"""
    B, Hq, _, D = query.shape
    Hkv = kq.shape[1]
    R = Hq // Hkv
    q = query[:, :, 0, :].reshape(B, Hkv, R, D).float()
    ks, vs = layer._k4, layer._v4
    nq, n_old = ks.n_q, vs.n_q
    parts = []
    if nq:
        parts.append(region_triton(q, ks, vs, scaling, nq) if _use_triton(query.device)
                     else region_torch(q, ks, vs, scaling, nq))
    tk = ([ks.resid] if ks.resid is not None else []) + [kq]
    tv = ([vs.dequant(nq, n_old, torch.float32)] if n_old > nq else []) + [vq]
    parts.append(_partial(q, torch.cat([t.float() for t in tk], -2), torch.cat([t.float() for t in tv], -2),
                          scaling))
    out = _merge(parts)
    ks.append(kq)
    vs.append(vq)
    _STATS["fused_steps"] += 1
    return out.reshape(B, Hq, 1, D).to(query.dtype)


TAIL_KERNEL = True     # triton path: tail + merge in one kernel (cuts ~25 small torch ops per layer per step)


def decode_step_v2(layer, query, kq, vq, scaling):
    """same contract as decode_step, triton only: region A kernel (raw partials) + tail/merge kernel"""
    B, Hq, _, D = query.shape
    Hkv = kq.shape[1]
    R = Hq // Hkv
    ks, vs = layer._k4, layer._v4
    nq, n_old = ks.n_q, vs.n_q
    q = query[:, :, 0, :].reshape(B, Hkv, R, D).float().contiguous()
    if nq:
        OM, OL, OO, S, RP = region_triton(q, ks, vs, scaling, nq, raw=True)
    else:
        RP = max(16, 1 << (R - 1).bit_length()); S = 0
        OM = OL = OO = q
    tk = torch.cat(([ks.resid] if ks.resid is not None else []) + [kq], -2).float().contiguous()
    tv = torch.cat(([vs.dequant(nq, n_old, torch.float32)] if n_old > nq else []) + [vq.float()], -2).contiguous()
    T = tk.shape[-2]
    assert T <= 128, T
    out = torch.empty_like(q)
    _tail_merge_kernel[(B * Hkv,)](
        q, tk, tv, OM, OL, OO, out,
        q.stride(0), q.stride(1), q.stride(2), tk.stride(0), tk.stride(1), tk.stride(2),
        T, R, Hkv, S, scaling, D=D, BT=128, RP=RP,
    )
    ks.append(kq)
    vs.append(vq)
    _STATS["fused_steps"] += 1
    return out.reshape(B, Hq, 1, D).to(query.dtype)


# ----------------------------------------------------------------------------------------------- transformers glue
def fused_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """attention function registered as "kvq_int4": fused int4 decode when the layer has a pending token,
    otherwise plain SDPA (prefill, non-int4 caches)"""
    cache = _ACTIVE["cache"]
    layer = None
    if cache is not None and getattr(module, "layer_idx", None) is not None:
        layer = cache.layers[module.layer_idx]
    pend = getattr(layer, "_pending", None) if layer is not None else None
    if pend is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    layer._pending = None
    kq, vq = pend
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise RuntimeError("fused int4 decode supports batch 1, one new token per step")
    scaling = scaling if scaling is not None else query.shape[-1] ** -0.5
    if TAIL_KERNEL and _use_triton(query.device):
        out = decode_step_v2(layer, query, kq, vq, scaling)
    else:
        out = decode_step(layer, query, kq, vq, scaling)
    return out.transpose(1, 2).contiguous(), None


def activate(cache):
    _ACTIVE["cache"] = cache


def register():
    from transformers import AttentionInterface
    AttentionInterface.register("kvq_int4", fused_attention)
    try:
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionMaskInterface.register("kvq_int4", sdpa_mask)
    except Exception:  # pragma: no cover
        pass


def enable(model):
    """register the attention function and switch `model` to it"""
    register()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("kvq_int4")
    else:  # pragma: no cover
        model.config._attn_implementation = "kvq_int4"
    return model
