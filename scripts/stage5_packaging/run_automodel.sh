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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_automodel}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
REPLICAS="${REPLICAS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"
LIMIT="${LIMIT:-null}"
RESUME="${RESUME:-false}"
INPUT_PATH="${INPUT_PATH:-${DATA_DIR}/meta/stage4_annotation/step3_tag/run_id_${INPUT_RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${DATA_DIR}/data/stage5_packaging/automodel/run_id_${RUN_ID}}"

CAPTION_FIELD="${CAPTION_FIELD:-caption_level_3}"
SELECT_PASS="${SELECT_PASS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DYNAMIC_FORWARD_BATCH_SIZE="${DYNAMIC_FORWARD_BATCH_SIZE:-4}"
METADATA_SHARD_SIZE="${METADATA_SHARD_SIZE:-10000}"

BUCKET_RESOLUTION="${BUCKET_RESOLUTION:-480p}"
BUCKET_UPSCALE="${BUCKET_UPSCALE:-false}"

ENCODER_CONFIG="${ENCODER_CONFIG:-wan}"
MODEL_NAME="${MODEL_NAME:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=automodel"
  "step/encoders=${ENCODER_CONFIG}"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "limit=${LIMIT}"
  "step.caption_field=${CAPTION_FIELD}"
  "step.select_pass=${SELECT_PASS}"
  "step.batch_size=${BATCH_SIZE}"
  "step.dynamic_forward_batch_size=${DYNAMIC_FORWARD_BATCH_SIZE}"
  "step.metadata_shard_size=${METADATA_SHARD_SIZE}"
  "step.replicas=${REPLICAS}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
  "step.ray_num_gpus=${RAY_NUM_GPUS}"
  "step.resume=${RESUME}"
  "step.bucket.resolution=${BUCKET_RESOLUTION}"
  "step.bucket.upscale=${BUCKET_UPSCALE}"
  "step.encoder.model_name=${MODEL_NAME}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage5_packaging.py "${COMMON_ARGS[@]}" "$@"
