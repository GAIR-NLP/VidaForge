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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_clip}"
MIN_LEN_SEC="${MIN_LEN_SEC:-1.0}"
MAX_LEN_SEC="${MAX_LEN_SEC:-10.0}"
OVERLONG_SPLIT_LEN_SEC="${OVERLONG_SPLIT_LEN_SEC:-10.0}"
BOUNDARY_TRIM_SEC="${BOUNDARY_TRIM_SEC:-0.3}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-1}"
LIMIT="${LIMIT:-null}"
RESUME="${RESUME:-false}"
INPUT_PATH="${DATA_DIR}/meta/stage2_segmentation/step1_detect/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage2_segmentation/step2_clip/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step2_clip"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "limit=${LIMIT}"
  "step.min_len_sec=${MIN_LEN_SEC}"
  "step.max_len_sec=${MAX_LEN_SEC}"
  "step.overlong_split_len_sec=${OVERLONG_SPLIT_LEN_SEC}"
  "step.boundary_trim_sec=${BOUNDARY_TRIM_SEC}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
  "step.ffmpeg_bin=${FFMPEG_BIN}"
  "step.resume=${RESUME}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage2_segmentation.py "${COMMON_ARGS[@]}" "$@"
