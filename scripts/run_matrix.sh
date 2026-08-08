"""Run the full experiment matrix sequentially on the GPU.

Models (from guide §17):
  A  Standard Transformer     FP16
  B  Tensormatics Transformer FP16
  C  Standard Transformer     1-bit FFN
  D  Tensormatics Transformer 1-bit FFN
  E  Standard Transformer     1-bit core (QKV+FFN)
  F  Tensormatics Transformer 1-bit core (QKV+FFN)

Usage:
  bash scripts/run_matrix.sh [steps] [eval_interval]
"""
PY=${PYTHON:-/c/Users/antho/AppData/Local/Programs/Python/Python312/python.exe}
STEPS=${1:-5000}
EVAL=${2:-500}

run() {
  local ffn=$1 binary_ffn=$2 binary_attn=$3 tag=$4
  local flags="--ffn $ffn --steps $STEPS --eval-interval $EVAL --eval-iters 50 --out out --tag $tag"
  [ "$binary_ffn" = "1" ] && flags="$flags --binary-ffn"
  [ "$binary_attn" = "1" ] && flags="$flags --binary-attn"
  echo ""
  echo "########## $tag ##########"
  "$PY" scripts/train.py $flags 2>&1 | tee "out/run_${tag}.log" | tail -12
}

run standard 0 0 A-std-fp16
run tensormatics 0 0 B-tm-fp16
run standard 1 0 C-std-1bit-ffn
run tensormatics 1 0 D-tm-1bit-ffn
run standard 1 1 E-std-1bit-core
run tensormatics 1 1 F-tm-1bit-core
echo ""
echo "ALL DONE"
