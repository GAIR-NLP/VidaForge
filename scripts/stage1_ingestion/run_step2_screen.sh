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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
LIMIT="${LIMIT:-null}"
MIN_SHORT_SIDE="${MIN_SHORT_SIDE:-360}"
MIN_FPS="${MIN_FPS:-20.0}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-1.0}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-600.0}"
INPUT_PATH="${DATA_DIR}/meta/stage1_ingestion/step1_probe/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage1_ingestion/step2_screen/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step2_screen"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "limit=${LIMIT}"
  "parquet_size=${PARQUET_SIZE}"
  "step.rules.short_side.min=${MIN_SHORT_SIDE}"
  "step.rules.fps.min=${MIN_FPS}"
  "step.rules.duration.min=${MIN_DURATION_SEC}"
  "step.rules.duration.max=${MAX_DURATION_SEC}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage1_ingestion.py "${COMMON_ARGS[@]}" "$@"
