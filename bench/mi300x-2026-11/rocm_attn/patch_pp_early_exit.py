"""Skip query tiles past a request's chunk in the prefix-prefill kernels (ROCM_ATTN chunked prefill).

Without it, every program (request, head, query tile) with block_start_loc >= query_len still loads the whole cached
context for that request before block_mask zeroes its output: when a batch mixes a short chunk with a long cached
prefix and a long chunk elsewhere (max_input_len), the short one's empty tiles re-read the full prefix for nothing.
usage: python3 patch_pp_early_exit.py <path to vllm/v1/attention/ops/prefix_prefill.py>
"""
import sys
p = sys.argv[1]; s = open(p).read()
old = "    block_start_loc = BLOCK_M * start_m\n"
new = ("    block_start_loc = BLOCK_M * start_m\n"
       "    # Query tiles past this request's chunk have no rows to compute; return before\n"
       "    # the pass over the cached context instead of masking the result afterwards.\n"
       "    if block_start_loc >= cur_batch_query_len:\n"
       "        return\n")
n = s.count(old)
if n != 2:
    print(f"PATCH_PP pattern count {n} != 2, skipped"); sys.exit(0)
s = s.replace(old, new)
open(p, "w").write(s); print("PATCH_PP patched both kernels", p)
