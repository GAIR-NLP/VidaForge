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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_dedup}"
DEDUPLICATORS="${DEDUPLICATORS:-[pdq]}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
APPLY_ENABLED="${APPLY_ENABLED:-true}"
APPLY_REPLICAS="${APPLY_REPLICAS:-auto}"
APPLY_RAY_NUM_CPUS="${APPLY_RAY_NUM_CPUS:-1}"
APPLY_RAY_NUM_GPUS="${APPLY_RAY_NUM_GPUS:-0}"
APPLY_BATCH_SIZE="${APPLY_BATCH_SIZE:-128}"
MATCH_REPLICAS="${MATCH_REPLICAS:-1}"
MATCH_RAY_NUM_CPUS="${MATCH_RAY_NUM_CPUS:-8}"
MATCH_RAY_NUM_GPUS="${MATCH_RAY_NUM_GPUS:-0}"
MATCH_BATCH_SIZE="${MATCH_BATCH_SIZE:-10000}"
LIMIT="${LIMIT:-null}"
INPUT_PATH="${DATA_DIR}/meta/stage3_selection/step2_filter/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage3_selection/step3_dedup/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step3_dedup"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "step.deduplicators=${DEDUPLICATORS}"
  "step.apply.enabled=${APPLY_ENABLED}"
  "step.apply.replicas=${APPLY_REPLICAS}"
  "step.apply.ray_num_cpus=${APPLY_RAY_NUM_CPUS}"
  "step.apply.ray_num_gpus=${APPLY_RAY_NUM_GPUS}"
  "step.apply.batch_size=${APPLY_BATCH_SIZE}"
  "step.match.replicas=${MATCH_REPLICAS}"
  "step.match.ray_num_cpus=${MATCH_RAY_NUM_CPUS}"
  "step.match.ray_num_gpus=${MATCH_RAY_NUM_GPUS}"
  "step.match.batch_size=${MATCH_BATCH_SIZE}"
  "limit=${LIMIT}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage3_selection.py "${COMMON_ARGS[@]}" "$@"
