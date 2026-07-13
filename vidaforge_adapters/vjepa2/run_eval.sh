#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

VJEPA2_DIR="${VJEPA2_DIR:?Set VJEPA2_DIR to the official V-JEPA2 repository}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH to a V-JEPA2 training config}"
VALID_CSV="${VALID_CSV:?Set VALID_CSV to the Stage 5 validation manifest}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CKPT_PATH="${CKPT_PATH:-}"

if [[ -z "${CKPT_PATH}" && -z "${CHECKPOINT_DIR}" ]]; then
  echo "Set CKPT_PATH to one checkpoint or CHECKPOINT_DIR to a checkpoint directory" >&2
  exit 2
fi

if [[ -n "${PYTHON_SHARED_LIB:-}" ]]; then
  [[ -d "${PYTHON_SHARED_LIB}" ]] || {
    echo "PYTHON_SHARED_LIB does not exist: ${PYTHON_SHARED_LIB}" >&2
    exit 2
  }
  export LD_LIBRARY_PATH="${PYTHON_SHARED_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" ]]; then
  [[ -d "${TORCHCODEC_FFMPEG_LIB}" ]] || {
    echo "TORCHCODEC_FFMPEG_LIB does not exist: ${TORCHCODEC_FFMPEG_LIB}" >&2
    exit 2
  }
  export LD_LIBRARY_PATH="${TORCHCODEC_FFMPEG_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export VJEPA2_DIR
export PYTHONPATH="${REPO_DIR}:${VJEPA2_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_ROOT="${CHECKPOINT_DIR:-$(dirname "${CKPT_PATH}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/eval}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
MAX_BATCHES="${MAX_BATCHES:-}"
LAMBDA_MODE="${LAMBDA_MODE:-checkpoint}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "[setup] config=${CONFIG_PATH}"
echo "[setup] valid_csv=${VALID_CSV}"
echo "[setup] checkpoint_dir=${CHECKPOINT_DIR:-single checkpoint mode}"
echo "[setup] output_dir=${OUTPUT_DIR}"
echo "[setup] max_batches=${MAX_BATCHES:-full}"
echo "[setup] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

mkdir -p "${OUTPUT_DIR}"

run_one_ckpt() {
  local ckpt_path="$1"
  local output_path="$2"

  local eval_args=(
    --vjepa2-dir "${VJEPA2_DIR}"
    --fname "${CONFIG_PATH}"
    --ckpt "${ckpt_path}"
    --valid-csv "${VALID_CSV}"
    --output "${output_path}"
    --lambda-mode "${LAMBDA_MODE}"
  )

  if [[ -n "${MAX_BATCHES}" ]]; then
    eval_args+=(--max-batches "${MAX_BATCHES}")
  fi

  torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    -m vidaforge_adapters.vjepa2.eval_loss \
    "${eval_args[@]}"
}

if [[ -n "${CKPT_PATH}" ]]; then
  if [[ -z "${OUTPUT_PATH}" ]]; then
    ckpt_name="$(basename "${CKPT_PATH}")"
    OUTPUT_PATH="${OUTPUT_DIR}/${ckpt_name%.pth.tar}.json"
  fi
  echo "[eval] ${CKPT_PATH} -> ${OUTPUT_PATH}"
  run_one_ckpt "${CKPT_PATH}" "${OUTPUT_PATH}"
  exit 0
fi

mapfile -t CKPT_NAMES < <(
  find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type f -name 'e*.pth.tar' -printf '%f\n' \
    | awk 'match($0, /^e([0-9]+)\.pth\.tar$/, m) { print m[1] "\t" $0 }' \
    | sort -n -k1,1 \
    | cut -f2-
)

if [[ "${#CKPT_NAMES[@]}" -eq 0 ]]; then
  echo "no e*.pth.tar checkpoint files found under ${CHECKPOINT_DIR}" >&2
  exit 2
fi

for ckpt_name in "${CKPT_NAMES[@]}"; do
  ckpt_path="${CHECKPOINT_DIR}/${ckpt_name}"
  output_path="${OUTPUT_DIR}/${ckpt_name%.pth.tar}.json"
  if [[ -s "${output_path}" ]]; then
    echo "[eval] skip existing ${output_path}"
    continue
  fi
  echo "[eval] ${ckpt_path} -> ${output_path}"
  run_one_ckpt "${ckpt_path}" "${output_path}"
done
