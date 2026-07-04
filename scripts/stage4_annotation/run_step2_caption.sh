#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DATA_DIR="${DATA_DIR:-/path/to/vidaforge_output}"
export DATA_DIR
SOURCE="${SOURCE:-example_source}"
SOURCE_BATCH="${SOURCE_BATCH:-default}"
INPUT_RUN_ID="${INPUT_RUN_ID:-example_run}"
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_caption}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
CLIENT_BATCH_SIZE="${CLIENT_BATCH_SIZE:-256}"
CLIENT_RAY_NUM_CPUS="${CLIENT_RAY_NUM_CPUS:-1.0}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
LIMIT="${LIMIT:-null}"
RESUME="${RESUME:-false}"
INPUT_PATH="${DATA_DIR}/meta/stage4_annotation/step1_camera/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage4_annotation/step2_caption/run_id_${RUN_ID}"

MODEL_PATH="${MODEL_PATH:-/path/to/vlm_model}"
MODEL_NAME="${MODEL_NAME:-vlm-model}"
REPLICAS="${REPLICAS:-auto}"
TP_SIZE="${TP_SIZE:-1}"
SERVE_RAY_NUM_CPUS="${SERVE_RAY_NUM_CPUS:-${CPU_PER_REPLICA:-8}}"
BASE_PORT="${BASE_PORT:-8100}"
HOST="${HOST:-0.0.0.0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
PLACEMENT_STRATEGY="${PLACEMENT_STRATEGY:-STRICT_PACK}"
LOG_DIR="${LOG_DIR:-null}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-${DATA_DIR}}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-[]}"
if [[ -z "${VLLM_ENV:-}" ]]; then
  VLLM_ENV="{VLLM_USE_DEEP_GEMM:0}"
fi

CAPTION_MODE="${CAPTION_MODE:-video_audio}"
MEDIA_INPUT="${MEDIA_INPUT:-local}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-1}"
TRUST_ENV="${TRUST_ENV:-false}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
STORE_PROMPT="${STORE_PROMPT:-false}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step2_caption"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "step.client.batch_size=${CLIENT_BATCH_SIZE}"
  "step.client.ray_num_cpus=${CLIENT_RAY_NUM_CPUS}"
  "ray_address=${RAY_ADDRESS}"
  "limit=${LIMIT}"
  "step.serve.model_path=${MODEL_PATH}"
  "step.serve.model_name=${MODEL_NAME}"
  "step.serve.replicas=${REPLICAS}"
  "step.serve.tp_size=${TP_SIZE}"
  "step.serve.ray_num_cpus=${SERVE_RAY_NUM_CPUS}"
  "step.serve.base_port=${BASE_PORT}"
  "step.serve.host=${HOST}"
  "step.serve.vllm_bin=${VLLM_BIN}"
  "step.serve.api_key=${API_KEY}"
  "step.serve.placement_strategy=${PLACEMENT_STRATEGY}"
  "step.serve.log_dir=${LOG_DIR}"
  "step.serve.allowed_local_media_path=${ALLOWED_LOCAL_MEDIA_PATH}"
  "step.serve.extra_args=${VLLM_EXTRA_ARGS}"
  "++step.serve.env=${VLLM_ENV}"
  "step.mode=${CAPTION_MODE}"
  "step.inference.media_input=${MEDIA_INPUT}"
  "step.inference.request_concurrency=${REQUEST_CONCURRENCY}"
  "step.inference.trust_env=${TRUST_ENV}"
  "step.inference.temperature=${TEMPERATURE}"
  "step.inference.top_p=${TOP_P}"
  "step.inference.presence_penalty=${PRESENCE_PENALTY}"
  "step.inference.max_tokens=${MAX_TOKENS}"
  "step.inference.store_prompt=${STORE_PROMPT}"
  "step.resume=${RESUME}"
  "++step.inference.extra_body.chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage4_annotation.py "${COMMON_ARGS[@]}" "$@"
