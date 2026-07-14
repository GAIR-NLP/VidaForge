#!/usr/bin/env bash
set -euo pipefail

# Example:
# VJEPA2_DIR=/path/to/vjepa2 NPROC_PER_NODE=8 bash vidaforge_adapters/vjepa2/run.sh \
#   folder=/path/to/output \
#   data.datasets=[/path/to/train.csv] \
#   optimization.epochs=10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VJEPA2_DIR="${VJEPA2_DIR:?Set VJEPA2_DIR to the official V-JEPA2 repository}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/vidaforge_adapters/vjepa2/configs/vitg16/pretrain_vitg16.yaml}"

if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" ]]; then
  [[ -d "${TORCHCODEC_FFMPEG_LIB}" ]] || {
    echo "TORCHCODEC_FFMPEG_LIB does not exist: ${TORCHCODEC_FFMPEG_LIB}" >&2
    exit 2
  }
  export LD_LIBRARY_PATH="${TORCHCODEC_FFMPEG_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${PYTHON_SHARED_LIB:-}" ]]; then
  [[ -d "${PYTHON_SHARED_LIB}" ]] || {
    echo "PYTHON_SHARED_LIB does not exist: ${PYTHON_SHARED_LIB}" >&2
    exit 2
  }
  export LD_LIBRARY_PATH="${PYTHON_SHARED_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m vidaforge_adapters.vjepa2.pretrain \
  --vjepa2-dir "${VJEPA2_DIR}" \
  --fname "${CONFIG_PATH}" \
  "$@"
