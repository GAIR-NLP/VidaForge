#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export RUN_NAME="${RUN_NAME:-wan2_1_t2v_pretrain}"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-4}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${CACHE_DIR:-}" ]]; then
  echo "CACHE_DIR must be set, e.g. /path/to/stage5_packaging/automodel/datasets/after_selection" >&2
  exit 2
fi
export CACHE_DIR

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must be set, e.g. /path/to/Wan2.1-T2V-1.3B-Diffusers" >&2
  exit 2
fi
export MODEL_PATH

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "CHECKPOINT_DIR must be set, e.g. /path/to/Wan_Exp" >&2
  exit 2
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR%/}/${RUN_NAME}"
export CHECKPOINT_DIR
export RESTORE_FROM="${RESTORE_FROM:-null}"

export WANDB_PROJECT="${WANDB_PROJECT:-wan-t2v-flow-matching-pretrain}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-1000}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export MAX_STEPS="${MAX_STEPS:-null}"
export LOG_EVERY="${LOG_EVERY:-2}"

export MODEL_TYPE="${MODEL_TYPE:-wan}"
export DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-false}"
export SHUFFLE="${SHUFFLE:-true}"
export DROP_LAST="${DROP_LAST:-false}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PIN_MEMORY="${PIN_MEMORY:-true}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export MAP_LOCATION="${MAP_LOCATION:-cpu}"
export DATALOADER_LIMIT="${DATALOADER_LIMIT:-null}"

export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"

export TP_SIZE="${TP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export DP_REPLICATE_SIZE="${DP_REPLICATE_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-none}"

export DIST_BACKEND="${DIST_BACKEND:-nccl}"
export DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-30}"

export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export MODEL_SAVE_FORMAT="${MODEL_SAVE_FORMAT:-safetensors}"
export SAVE_CONSOLIDATED="${SAVE_CONSOLIDATED:-final}"
export DIFFUSERS_COMPATIBLE="${DIFFUSERS_COMPATIBLE:-true}"

mkdir -p "${CHECKPOINT_DIR}"

OPTIONAL_ARGS=()
if [[ "${MAX_STEPS}" != "null" ]]; then
  OPTIONAL_ARGS+=("--step_scheduler.max_steps" "${MAX_STEPS}")
fi

if [[ "${DATALOADER_LIMIT}" != "null" ]]; then
  OPTIONAL_ARGS+=("--data.dataloader.limit" "${DATALOADER_LIMIT}")
fi

if [[ "${RESTORE_FROM}" != "null" ]]; then
  OPTIONAL_ARGS+=("--checkpoint.restore_from" "${RESTORE_FROM}")
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  vidaforge_adapters/automodel/pretrain.py \
  --config vidaforge_adapters/automodel/configs/wan2_1_t2v_flow.yaml \
  --wandb.enable true \
  --wandb.mode offline \
  --wandb.project "${WANDB_PROJECT}" \
  --wandb.name "${WANDB_NAME}" \
  --dist_env.backend "${DIST_BACKEND}" \
  --dist_env.timeout_minutes "${DIST_TIMEOUT_MINUTES}" \
  --model.pretrained_model_name_or_path "${MODEL_PATH}" \
  --step_scheduler.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --step_scheduler.local_batch_size "${LOCAL_BATCH_SIZE}" \
  --step_scheduler.ckpt_every_steps "${CKPT_EVERY_STEPS}" \
  --step_scheduler.num_epochs "${NUM_EPOCHS}" \
  --step_scheduler.log_every "${LOG_EVERY}" \
  --data.dataloader._target_ vidaforge_adapters.automodel.VidaForgeVideoDataloaderConfig \
  --data.dataloader.cache_dir "${CACHE_DIR}" \
  --data.dataloader.model_type "${MODEL_TYPE}" \
  --data.dataloader.dynamic_batch_size "${DYNAMIC_BATCH_SIZE}" \
  --data.dataloader.shuffle "${SHUFFLE}" \
  --data.dataloader.drop_last "${DROP_LAST}" \
  --data.dataloader.num_workers "${NUM_WORKERS}" \
  --data.dataloader.pin_memory "${PIN_MEMORY}" \
  --data.dataloader.prefetch_factor "${PREFETCH_FACTOR}" \
  --data.dataloader.map_location "${MAP_LOCATION}" \
  --optim.learning_rate "${LEARNING_RATE}" \
  --lr_scheduler.min_lr "${MIN_LR}" \
  --lr_scheduler.lr_warmup_steps "${LR_WARMUP_STEPS}" \
  --fsdp.tp_size "${TP_SIZE}" \
  --fsdp.cp_size "${CP_SIZE}" \
  --fsdp.pp_size "${PP_SIZE}" \
  --fsdp.dp_replicate_size "${DP_REPLICATE_SIZE}" \
  --fsdp.dp_size "${DP_SIZE}" \
  --checkpoint.enabled "${CHECKPOINT_ENABLED}" \
  --checkpoint.checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint.model_save_format "${MODEL_SAVE_FORMAT}" \
  --checkpoint.save_consolidated "${SAVE_CONSOLIDATED}" \
  --checkpoint.diffusers_compatible "${DIFFUSERS_COMPATIBLE}" \
  "${OPTIONAL_ARGS[@]}"
