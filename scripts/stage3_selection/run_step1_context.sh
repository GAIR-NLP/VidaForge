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
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}_context}"
FRAME_SAMPLED_FPS="${FRAME_SAMPLED_FPS:-2.0}"
FRAME_SHORT_SIDE="${FRAME_SHORT_SIDE:-384}"
AUDIO_FORMAT="${AUDIO_FORMAT:-m4a}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-24000}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-1}"
LIMIT="${LIMIT:-null}"
RESUME="${RESUME:-false}"
INPUT_PATH="${DATA_DIR}/meta/stage2_segmentation/step2_clip/run_id_${INPUT_RUN_ID}"
OUTPUT_PATH="${DATA_DIR}/meta/stage3_selection/step1_context/run_id_${RUN_ID}"

mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  "step=step1_context"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
  "limit=${LIMIT}"
  "step.frame.sampled_fps=${FRAME_SAMPLED_FPS}"
  "step.frame.short_side=${FRAME_SHORT_SIDE}"
  "step.audio.format=${AUDIO_FORMAT}"
  "step.audio.sample_rate=${AUDIO_SAMPLE_RATE}"
  "step.audio.channels=${AUDIO_CHANNELS}"
  "step.ffmpeg_bin=${FFMPEG_BIN}"
  "step.resume=${RESUME}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage3_selection.py "${COMMON_ARGS[@]}" "$@"
