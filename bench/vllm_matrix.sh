#!/usr/bin/env bash
# ============================================================================
# vLLM 启动参数矩阵 —— 量化配置对比用
#
# 用法：
#   ./bench/vllm_matrix.sh list                  # 打印所有配置与关键参数
#   ./bench/vllm_matrix.sh bf16                  # 启动某个配置
#   ./bench/vllm_matrix.sh fp8 --dry             # 只打印命令，不启动
#   ./bench/vllm_matrix.sh awq --bench           # 启动后自动跑一轮压测
#
# 覆盖模型路径：
#   MODELS_DIR=/root/models ./bench/vllm_matrix.sh list
#
# ⚠️ 一次只改一个变量。同一轮对比里，除「被测变量」外所有参数必须完全一致，
#    否则结论归因不了。下面的 COMMON_ARGS 就是这个"不变的部分"。
# ============================================================================
set -euo pipefail

# ---------- 可覆盖的变量 ----------
MODELS_DIR="${MODELS_DIR:-/root/models}"
BASE="${BASE:-$MODELS_DIR/Qwen3-32B}"              # BF16 原始（也是后训练合并后的落点）
AWQ_OFFICIAL="${AWQ_OFFICIAL:-$MODELS_DIR/Qwen3-32B-AWQ}"   # 官方 AWQ
FP8_OFFICIAL="${FP8_OFFICIAL:-$MODELS_DIR/Qwen3-32B-FP8}"   # 官方 FP8
MERGED="${MERGED:-$MODELS_DIR/merged-bf16}"        # 后训练合并后的权重

PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-./bench/logs}"

# ---------- 独立变量：TP size（直接决定卡数，必测）----------
TP="${TP:-1}"

# ---------- 独立变量：KV cache 精度 ----------
# ⚠️ 需要 Ada(L40S/4090) 或 Hopper(H100/H800/H20)，compute capability >= 8.9。
#    Ampere(A100/A800) 不支持 → 这条路走不通，只能降 max-num-seqs。
KV_DTYPE="${KV_DTYPE:-auto}"

# ---------- 不变的那部分（一次只改一个变量的前提）----------
# 所有参数都显式写出，不依赖版本默认值 —— 这样报告才可复现。
MAX_SEQS="${MAX_SEQS:-16}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"
SWAP="${SWAP:-4}"

COMMON_ARGS=(
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len 32768
  --gpu-memory-utilization 0.92
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$BATCH_TOKENS"
  --tensor-parallel-size "$TP"
  --kv-cache-dtype "$KV_DTYPE"
  --swap-space "$SWAP"
  --disable-log-requests
)

if [[ "${CHUNKED:-1}" == "1" ]]; then
  COMMON_ARGS+=(--enable-chunked-prefill)
else
  COMMON_ARGS+=(--no-enable-chunked-prefill)
fi

if [[ "${PREFIX:-1}" == "1" ]]; then
  COMMON_ARGS+=(--enable-prefix-caching)
else
  COMMON_ARGS+=(--no-enable-prefix-caching)
fi

# ---------- 配置矩阵 ----------
# 每个配置只声明「它变了什么」。这是整个实验的自变量清单。
config_model() {
  case "$1" in
    bf16)       echo "$BASE" ;;
    awq)        echo "$AWQ_OFFICIAL" ;;
    awq-g64)    echo "$MERGED-AWQ-G64" ;;
    awq-domain) echo "$MERGED-AWQ-domain" ;;
    fp8)        echo "$FP8_OFFICIAL" ;;
    fp8-merged) echo "$MERGED-FP8" ;;
    *)          echo "" ;;
  esac
}

config_extra() {
  case "$1" in
    # ---- 基线：不量化。所有对比的上界锚点 ----
    bf16)       echo "" ;;

    # ---- 官方预量化 checkpoint：不需要 --quantization ----
    # vLLM 从 config.json 的 quantization_config 自动识别格式
    awq)        echo "" ;;
    fp8)        echo "" ;;

    # ---- 自己量化的 checkpoint（后训练之后生成）----
    awq-g64)    echo "" ;;
    awq-domain) echo "" ;;
    fp8-merged) echo "" ;;

    *)          echo "" ;;
  esac
}

config_desc() {
  case "$1" in
    bf16)       echo "BF16 基线（上界锚点，不量化）" ;;
    awq)        echo "官方 AWQ W4A16 g128 —— decode 带宽红利" ;;
    awq-g64)    echo "自量化 AWQ g64 —— 对比 group_size 的影响" ;;
    awq-domain) echo "自量化 AWQ + 领域语料校准 —— ⭐ 只有你能做的那组" ;;
    fp8)        echo "官方 FP8 W8A8 —— prefill 算力红利" ;;
    fp8-merged) echo "自量化 FP8（对后训练权重）" ;;
    *)          echo "" ;;
  esac
}

# ---------- 打印参数表 ----------
do_list() {
  echo "被测变量 → 配置清单（COMMON_ARGS 之外的差异）"
  echo "================================================================================"
  printf "%-12s %-34s %s\n" "配置" "说明" "变化的参数"
  echo "--------------------------------------------------------------------------------"
  for k in bf16 awq awq-g64 awq-domain fp8 fp8-merged; do
    printf "%-12s %-34s %s\n" "$k" "$(config_desc "$k")" "$(config_extra "$k" | tr -s ' ' ' ')"
  done
  echo "================================================================================"
  echo "所有配置共用的参数（不变的部分）："
  printf '  %s\n' "${COMMON_ARGS[@]}"
  echo
  echo "可单独叠加的开关："
  echo "  KV_DTYPE=fp8   改 --kv-cache-dtype（需 Ada/Hopper，A800 不支持）"
  echo "  TP=2           改 --tensor-parallel-size"
  echo "  MAX_SEQS=4     改 --max-num-seqs（默认 16）"
  echo "  BATCH_TOKENS=2048  改 --max-num-batched-tokens（默认 8192）"
  echo "  PREFIX=0       换成 --no-enable-prefix-caching"
  echo "  CHUNKED=0      换成 --no-enable-chunked-prefill"
  echo "  SWAP=0         改 --swap-space（测速时设 0，避免 swap 掩盖真实瓶颈）"
  echo
  echo "启动后打流（另开一个终端）："
  echo "  python bench/gen_workload.py --out bench/workload/mix.jsonl"
  echo "  vllm bench serve --backend openai-chat --base-url http://localhost:$PORT \\"
  echo "      --model $SERVED_NAME --dataset-name custom \\"
  echo "      --dataset-path bench/workload/mix.jsonl --num-prompts 221 \\"
  echo "      --max-concurrency 8 --request-rate inf \\"
  echo "      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,99 \\"
  echo "      --save-result --result-dir ./bench/results"
}

# ---------- 打流 ----------
do_bench() {
  local conc="${CONC:-1,4,8,16}"
  IFS=',' read -ra CONCS <<< "$conc"
  mkdir -p ./bench/results
  for c in "${CONCS[@]}"; do
    echo ""
    echo ">>> 并发 $c"
    vllm bench serve \
      --backend openai-chat \
      --base-url "http://localhost:$PORT" \
      --model "$SERVED_NAME" \
      --dataset-name custom \
      --dataset-path "${WORKLOAD:-./bench/workload/mix.jsonl}" \
      --num-prompts "${NUM_PROMPTS:-221}" \
      --max-concurrency "$c" \
      --request-rate inf \
      --percentile-metrics ttft,tpot,e2el \
      --metric-percentiles 50,90,99 \
      --save-result \
      --result-dir ./bench/results \
      --result-filename "${1}-c${c}.json"
  done
  echo ""
  echo ">>> 结果已写入 ./bench/results/"
}

# ---------- 主入口 ----------
CFG="${1:-list}"
shift || true
DRY=0
BENCH=0
for a in "$@"; do
  [[ "$a" == "--dry" ]]   && DRY=1
  [[ "$a" == "--bench" ]] && BENCH=1
done

if [[ "$CFG" == "list" || "$CFG" == "-h" || "$CFG" == "--help" ]]; then
  do_list
  exit 0
fi

MODEL="$(config_model "$CFG")"
if [[ -z "$MODEL" ]]; then
  echo "未知配置：$CFG" >&2
  do_list >&2
  exit 2
fi

EXTRA="$(config_extra "$CFG")"
read -r -a EXTRA_ARR <<< "$EXTRA"

mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "配置：$CFG"
echo "说明：$(config_desc "$CFG")"
echo "权重：$MODEL"
echo "================================================================================"

CMD=(vllm serve "$MODEL" "${COMMON_ARGS[@]}")
[[ ${#EXTRA_ARR[@]} -gt 0 ]] && CMD+=("${EXTRA_ARR[@]}")

printf '>>> 执行：\n  '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "$DRY" == "1" ]]; then
  echo ""
  echo "[dry] 未启动服务。"
  exit 0
fi

# 预量化 checkpoint 不需要 --quantization（vLLM 从 config.json 自动识别）。
# 若要用在线量化（不预生成 checkpoint，仅 Ada/Hopper 支持）：
#   vllm serve $BASE --quantization fp8_per_tensor
# 注意在线量化是 dynamic，与预量化 static checkpoint 不等价，报告里要注明。

echo ""
echo ">>> 启动中（首次加载含 CUDA graph 捕获，需数分钟）..."
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/serve-$CFG.log" &

SERVER_PID=$!
trap 'echo ">>> 停止服务 $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true' EXIT

# 等待就绪
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo ">>> 服务就绪（${i}s）"
    break
  fi
  sleep 1
done

if [[ "$BENCH" == "1" ]]; then
  echo ""
  echo ">>> warmup（丢弃前 8 条，避免 CUDA graph 捕获污染 P99）"
  vllm bench serve \
    --backend openai-chat \
    --base-url "http://localhost:$PORT" \
    --model "$SERVED_NAME" \
    --dataset-name custom \
    --dataset-path "${WORKLOAD:-./bench/workload/mix.jsonl}" \
    --num-prompts 8 --max-concurrency 1 --request-rate inf \
    --save-result --result-dir /tmp/warmup-discard >/dev/null 2>&1 || true

  do_bench "$CFG"
fi

echo ""
echo ">>> 服务仍在运行（PID $SERVER_PID）。Ctrl-C 停止。"
echo ">>> 另开终端跑：./bench/vllm_matrix.sh $CFG --bench"
wait $SERVER_PID
