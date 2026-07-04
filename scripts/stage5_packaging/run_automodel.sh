#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DEFAULT_TORCHCODEC_FFMPEG_LIB="${DEFAULT_TORCHCODEC_FFMPEG_LIB:-}"
DEFAULT_PYTHON_SHARED_LIB="${DEFAULT_PYTHON_SHARED_LIB:-}"

if [[ -z "${TORCHCODEC_FFMPEG_LIB:-}" && -d "${DEFAULT_TORCHCODEC_FFMPEG_LIB}" ]]; then
  TORCHCODEC_FFMPEG_LIB="${DEFAULT_TORCHCODEC_FFMPEG_LIB}"
fi
if [[ -z "${PYTHON_SHARED_LIB:-}" && -d "${DEFAULT_PYTHON_SHARED_LIB}" ]]; then
  PYTHON_SHARED_LIB="${DEFAULT_PYTHON_SHARED_LIB}"
fi

if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" || -n "${PYTHON_SHARED_LIB:-}" ]]; then
  if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" && ! -d "${TORCHCODEC_FFMPEG_LIB}" ]]; then
    echo "TORCHCODEC_FFMPEG_LIB does not exist: ${TORCHCODEC_FFMPEG_LIB}" >&2
    exit 2
  fi
  if [[ -n "${PYTHON_SHARED_LIB:-}" && ! -d "${PYTHON_SHARED_LIB}" ]]; then
    echo "PYTHON_SHARED_LIB does not exist: ${PYTHON_SHARED_LIB}" >&2
    exit 2
  fi

  CLEAN_LD_LIBRARY_PATH=""
  IFS=':' read -r -a LD_LIBRARY_PATH_ENTRIES <<< "${LD_LIBRARY_PATH:-}"
  for entry in "${LD_LIBRARY_PATH_ENTRIES[@]}"; do
    if [[ -z "${entry}" || "${entry}" == *"ffmpeg-8.0.1"* ]]; then
      continue
    fi
    if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" && "${entry}" == "${TORCHCODEC_FFMPEG_LIB}" ]]; then
      continue
    fi
    if [[ -n "${PYTHON_SHARED_LIB:-}" && "${entry}" == "${PYTHON_SHARED_LIB}" ]]; then
      continue
    fi
    if [[ -z "${CLEAN_LD_LIBRARY_PATH}" ]]; then
      CLEAN_LD_LIBRARY_PATH="${entry}"
    else
      CLEAN_LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}:${entry}"
    fi
  done

  LD_LIBRARY_PATH=""
  if [[ -n "${PYTHON_SHARED_LIB:-}" ]]; then
    LD_LIBRARY_PATH="${PYTHON_SHARED_LIB}"
  fi
  if [[ -n "${TORCHCODEC_FFMPEG_LIB:-}" ]]; then
    if [[ -n "${LD_LIBRARY_PATH}" ]]; then
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TORCHCODEC_FFMPEG_LIB}"
    else
      LD_LIBRARY_PATH="${TORCHCODEC_FFMPEG_LIB}"
    fi
  fi
  if [[ -n "${CLEAN_LD_LIBRARY_PATH}" ]]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CLEAN_LD_LIBRARY_PATH}"
  fi
  export LD_LIBRARY_PATH
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
MODEL_NAME="${MODEL_NAME:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"

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
