#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "VLLM_API_KEY is required; refusing to expose an unauthenticated server." >&2
  exit 2
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
MODEL_REVISION="${MODEL_REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

printf 'Starting authenticated vLLM\n'
printf '  model=%s\n' "$MODEL_ID"
printf '  revision=%s\n' "$MODEL_REVISION"
printf '  listen=0.0.0.0:%s\n' "$PORT"
printf '  max_model_len=%s\n' "$MAX_MODEL_LEN"
printf '  gpu_memory_utilization=%s\n' "$GPU_MEMORY_UTILIZATION"
printf '  api_key=configured (value hidden)\n'

exec vllm serve "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --language-model-only \
  --api-key "$VLLM_API_KEY"
