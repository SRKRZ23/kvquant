"""Bit-identity + timing of the prefix-prefill early-exit patch, kernel level (context_attention_fwd).

Loads the installed prefix_prefill.py (unpatched) and a patched copy as two modules, runs both on the same mixed
batches (a short chunk with a long cached prefix next to a long chunk, plus decodes), and checks the outputs are
bitwise equal. Prints: RESULT PPID case equal max_abs_diff ms_unpatched ms_patched
"""
import importlib.util, os, sys, tempfile, torch
import vllm.v1.attention.ops.prefix_prefill as orig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(orig.__file__).read()
tmp = os.path.join(tempfile.mkdtemp(), "prefix_prefill_patched.py")
open(tmp, "w").write(src)
import subprocess
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_pp_early_exit.py"), tmp], check=True)
spec = importlib.util.spec_from_file_location("prefix_prefill_patched", tmp)
pat = importlib.util.module_from_spec(spec); spec.loader.exec_module(pat)

dev, dt = "cuda", torch.bfloat16
HS, BS, X = 128, 16, 8
torch.manual_seed(0)


def case(name, reqs, nh=28, kvh=4, alibi=False):
    """reqs: list of (query_len, ctx_len)."""
    nblk_tot = sum((q + c + BS - 1) // BS for q, c in reqs) + 1
    kc = torch.randn(nblk_tot, kvh, HS // X, BS, X, dtype=dt, device=dev)
    vc = torch.randn(nblk_tot, kvh, HS, BS, dtype=dt, device=dev)
    maxb = max((q + c + BS - 1) // BS for q, c in reqs)
    bt = torch.zeros(len(reqs), maxb, dtype=torch.int32, device=dev)
    nxt = 0
    for i, (q, c) in enumerate(reqs):
        n = (q + c + BS - 1) // BS
        bt[i, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32); nxt += n
    qlens = [q for q, _ in reqs]
    tot = sum(qlens)
    qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=dev)
    sl = torch.tensor([q + c for q, c in reqs], dtype=torch.int32, device=dev)
    q = torch.randn(tot, nh, HS, dtype=dt, device=dev)
    k = torch.randn(tot, kvh, HS, dtype=dt, device=dev)
    v = torch.randn(tot, kvh, HS, dtype=dt, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    slopes = torch.rand(nh, dtype=torch.float32, device=dev) if alibi else None
    outs, times = [], []
    for mod in (orig, pat):
        o = torch.zeros_like(q)
        def call():
            mod.context_attention_fwd(q, k, v, o, "auto", kc, vc, bt, qsl, sl, int(sl.max()), max(qlens), one, one,
                                      alibi_slopes=slopes, skip_decode=True)
        call(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            call()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 10); outs.append(o.clone())
    eq = torch.equal(outs[0], outs[1])
    d = (outs[0].float() - outs[1].float()).abs().max().item()
    print(f"RESULT PPID {name} equal={eq} max_abs_diff={d:.3g} ms_unpatched={times[0]:.3f} ms_patched={times[1]:.3f}",
          flush=True)
    return eq


ok = True
ok &= case("single_chunk_2k_ctx_0", [(2048, 0)])
ok &= case("short_chunk_long_ctx_next_to_long_chunk", [(16000, 0), (384, 29000)])
ok &= case("tail_plus_head_chunks", [(13616, 16384), (2768, 0)])
ok &= case("mixed_decodes_and_prefill", [(1, 30000)] * 8 + [(8000, 22000)])
ok &= case("many_short_chunks_long_ctx", [(64, 20000)] * 16 + [(4096, 0)])
ok &= case("alibi_short_chunk_long_ctx", [(4096, 0), (256, 8000)], alibi=True)
print("PPID_ALL_EQUAL" if ok else "PPID_MISMATCH", flush=True)
