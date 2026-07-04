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
RAW_DIR="${RAW_DIR:-/path/to/raw_videos}"
export RAW_DIR
RUN_ID="${RUN_ID:-example_run}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-1}"
LIMIT="${LIMIT:-null}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
INPUT_PATH="${INPUT_PATH:-${RAW_DIR}}"
OUTPUT_PATH="${DATA_DIR}/meta/stage1_ingestion/step1_probe/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step1_probe"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "run_id=${RUN_ID}"
  "limit=${LIMIT}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "step.ffprobe_bin=${FFPROBE_BIN}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage1_ingestion.py "${COMMON_ARGS[@]}" "$@"
