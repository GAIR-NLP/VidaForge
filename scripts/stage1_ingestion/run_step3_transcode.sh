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
INPUT_RUN_ID="${INPUT_RUN_ID:-example_run}"
RUN_ID="${RUN_ID:-${INPUT_RUN_ID}}"
PARQUET_SIZE="${PARQUET_SIZE:-10000}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
LIMIT="${LIMIT:-null}"

RESUME="${RESUME:-false}"

TARGET_SHORT_EDGE="${TARGET_SHORT_EDGE:-720}"
TARGET_FPS="${TARGET_FPS:-25}"
CRF="${CRF:-23}"
PIX_FMT="${PIX_FMT:-yuv420p}"
AUDIO_BITRATE="${AUDIO_BITRATE:-128k}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
FFMPEG_THREADS="${FFMPEG_THREADS:-4}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
INPUT_PATH="${INPUT_PATH:-${DATA_DIR}/meta/stage1_ingestion/step2_screen/run_id_${INPUT_RUN_ID}}"
OUTPUT_PATH="${DATA_DIR}/meta/stage1_ingestion/step3_transcode/run_id_${RUN_ID}"

COMMON_ARGS=(
  "step=step3_transcode"
  "input_path=${INPUT_PATH}"
  "output_path=${OUTPUT_PATH}"
  "source=${SOURCE}"
  "source_batch=${SOURCE_BATCH}"
  "input_run_id=${INPUT_RUN_ID}"
  "run_id=${RUN_ID}"
  "limit=${LIMIT}"
  "parquet_size=${PARQUET_SIZE}"
  "ray_address=${RAY_ADDRESS}"
  "step.resume=${RESUME}"
  "step.target_short_edge=${TARGET_SHORT_EDGE}"
  "step.target_fps=${TARGET_FPS}"
  "step.crf=${CRF}"
  "step.pix_fmt=${PIX_FMT}"
  "step.audio_bitrate=${AUDIO_BITRATE}"
  "step.ray_num_cpus=${RAY_NUM_CPUS}"
  "step.ffmpeg_threads=${FFMPEG_THREADS}"
  "step.ffmpeg_bin=${FFMPEG_BIN}"
  "step.ffprobe_bin=${FFPROBE_BIN}"
)

PYTHONPATH=. "${PYTHON_BIN}" recipe/stage1_ingestion.py "${COMMON_ARGS[@]}" "$@"


# Usage notes:
# 1. Default behavior reads all rows from step2_screen:
#      bash scripts/stage1_ingestion/run_step3_transcode.sh
# 1.1 To read one run and write another:
#      INPUT_RUN_ID=example_run RUN_ID=example_run_transcode bash scripts/stage1_ingestion/run_step3_transcode.sh
# 2. To transcode only Screen-pass samples:
#      INPUT_PATH=/path/to/step2_screen/run_id_x/pass bash scripts/stage1_ingestion/run_step3_transcode.sh
# 2.1 To tune CPU scheduling:
#      RAY_NUM_CPUS=4 FFMPEG_THREADS=4 bash scripts/stage1_ingestion/run_step3_transcode.sh
# 2.2 To resume within the same RUN_ID and skip already successful outputs:
#      INPUT_RUN_ID=example_run RUN_ID=example_run RESUME=true bash scripts/stage1_ingestion/run_step3_transcode.sh
# 3. To read step2_screen/reject and debug rejected samples:
#      INPUT_PATH=/path/to/step2_screen/run_id_x/reject bash scripts/stage1_ingestion/run_step3_transcode.sh
