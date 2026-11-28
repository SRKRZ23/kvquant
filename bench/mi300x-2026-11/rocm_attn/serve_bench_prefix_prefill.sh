#!/bin/bash
# One-shot droplet run (MI300X, vllm/vllm-openai-rocm:nightly).
#  T: PR "[Test] split-K 64 segments long context": new test appended to the image commit's test file, pytest
#  S: PR "[ROCm][Perf] prefix_prefill skip empty tiles": vllm bench serve before/after (2 workloads x 3 runs)
cd /shared-docker/run10
export HF_HOME=/shared-docker/hf
M=Qwen/Qwen2.5-7B-Instruct
L(){ echo "PHASE $(date -u +%H:%M) $*"; }
python3 -c "import vllm,torch;print('ENV vllm',vllm.__version__,'torch',torch.__version__)"
SHA=$(python3 -c "import vllm,re;print(re.search(r'g([0-9a-f]{7,})',vllm.__version__).group(1))")
echo "IMAGE_SHA $SHA"

L "T split-K long-context test"
curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/$SHA/tests/kernels/attention/test_triton_unified_attention.py" -o tua_base.py \
  || curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/main/tests/kernels/attention/test_triton_unified_attention.py" -o tua_base.py
python3 - <<'PY'
s = open("tua_base.py").read()
if "import cdiv" not in s and " cdiv," not in s:
    s = s.replace("from vllm.utils.math_utils import next_power_of_2", "from vllm.utils.math_utils import cdiv, next_power_of_2")
s += "\n\n" + open("new_test_block.py").read()
open("test_tua_long.py", "w").write(s); print("TEST_FILE_READY")
PY
python3 -m pytest -q -p no:cacheprovider test_tua_long.py -k split_k_long_context --durations=5 2>&1 | tail -12
echo "--- 64-segment path sanity: existing short-context decode tests still pass at this commit"
python3 -m pytest -q -p no:cacheprovider test_tua_long.py -k "test_triton_unified_attn and not long_context and not bf16 and not fp16 and not use_td and not window and not clamped" -x 2>&1 | tail -3

serve_bench(){ local arm=$1
  vllm serve $M --max-model-len 32768 --port 8000 > serve_$arm.log 2>&1 &
  local SP=$!; local t=0
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; t=$((t+5)); [ $t -gt 900 ] && { echo "SERVE_TIMEOUT $arm"; tail -20 serve_$arm.log; return; }; done
  grep -m3 -E "attention backend|Backends selected|attention \|" serve_$arm.log | cut -c1-200
  vllm bench serve --model $M --port 8000 --dataset-name random --random-input-len 2000 --random-output-len 16 \
    --num-prompts 8 --max-concurrency 4 --seed 99 > warm_$arm.txt 2>&1
  for r in 1 2 3; do
    vllm bench serve --model $M --port 8000 --dataset-name prefix_repetition --prefix-repetition-prefix-len 16000 \
      --prefix-repetition-suffix-len 256 --prefix-repetition-num-prefixes 8 --prefix-repetition-output-len 64 \
      --num-prompts 128 --max-concurrency 32 --seed $r --save-result --result-filename W1_${arm}_$r.json > W1_${arm}_$r.txt 2>&1
    grep -E "Request throughput|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Benchmark duration|Error|error:" W1_${arm}_$r.txt | head -8 | sed "s/^/W1 $arm $r | /"
    vllm bench serve --model $M --port 8000 --dataset-name random --random-input-len 16000 --random-output-len 128 \
      --random-range-ratio 0.5 --num-prompts 64 --max-concurrency 16 --seed $r --save-result --result-filename W2_${arm}_$r.json > W2_${arm}_$r.txt 2>&1
    grep -E "Request throughput|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Benchmark duration|Error|error:" W2_${arm}_$r.txt | head -8 | sed "s/^/W2 $arm $r | /"
  done
  kill $SP; wait $SP 2>/dev/null; sleep 10; }

PP=$(python3 -c 'import vllm.v1.attention.ops.prefix_prefill as m;print(m.__file__)')
cp "$PP" "$PP.orig"
L "S before (main kernel)"; serve_bench before
L "S after (skip empty tiles)"; python3 patch_pp_early_exit.py "$PP"; serve_bench after
cp "$PP.orig" "$PP"
L "R10_DONE"
