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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_select}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
LIMIT="${LIMIT:-null}"
INPUT_PATH="${INPUT_PATH:-${DATA_DIR}/meta/stage3_selection/step3_dedup/run_id_${INPUT_RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${DATA_DIR}/meta/stage3_selection/step4_select/run_id_${RUN_ID}}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step4_select"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "limit=${LIMIT}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage3_selection.py "${COMMON_ARGS[@]}" "$@"
