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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_filter}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
REPLICAS="${REPLICAS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-0}"
LIMIT="${LIMIT:-null}"
RESUME="${RESUME:-false}"
INPUT_PATH="${DATA_DIR}/meta/stage3_selection/step1_context/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage3_selection/step2_filter/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step2_filter"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "step.replicas=${REPLICAS}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
  "step.ray_num_gpus=${RAY_NUM_GPUS}"
  "limit=${LIMIT}"
  "step.resume=${RESUME}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage3_selection.py "${COMMON_ARGS[@]}" "$@"
