#!/usr/bin/env bash
#
# Evaluate validation loss for every checkpoint under CHECKPOINT_DIR.
#
# Smoke test a few batches:
#
#   MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
#   VALID_CACHE_DIR=/path/to/validation_cache \
#   CHECKPOINT_DIR=/path/to/checkpoints \
#   EVAL_MAX_BATCHES=5 \
#   vidaforge_adapters/automodel/run_eval.sh
#
# Override paths when needed:
#
#   CHECKPOINT_DIR=/path/to/Wan_Exp/after_selection_200k \
#   VALID_CACHE_DIR=/path/to/datasets/valid_selection_10k \
#   OUTPUT_DIR=/path/to/eval_json \
#   vidaforge_adapters/automodel/run_eval.sh
#
# The script sorts checkpoint directories by the numeric suffix in names like
# epoch_0_step_99 and epoch_0_step_199. Existing JSON outputs are skipped so the
# loop can be resumed.
#
# Set cluster-specific NCCL and Gloo environment variables before invoking this
# script when running across multiple nodes.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_NAME="${RUN_NAME:-wan2_1_t2v_eval}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Wan model directory}"
VALID_CACHE_DIR="${VALID_CACHE_DIR:?Set VALID_CACHE_DIR to the Stage 5 validation cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the training checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/eval}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
DP_REPLICATE_SIZE="${DP_REPLICATE_SIZE:-${NNODES}}"
DP_SIZE="${DP_SIZE:-${NPROC_PER_NODE}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((LOCAL_BATCH_SIZE * DP_REPLICATE_SIZE * DP_SIZE))}"

NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-20}"
EVAL_SEED="${EVAL_SEED:-12345}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
EVAL_FRESH_INIT="${EVAL_FRESH_INIT:-false}"

mkdir -p "${OUTPUT_DIR}"

echo "[eval] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[eval] run_name=${RUN_NAME}"
echo "[eval] valid_cache_dir=${VALID_CACHE_DIR}"
echo "[eval] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[eval] output_dir=${OUTPUT_DIR}"
echo "[eval] global_batch_size=${GLOBAL_BATCH_SIZE} local_batch_size=${LOCAL_BATCH_SIZE}"
echo "[eval] dp_replicate_size=${DP_REPLICATE_SIZE} dp_size=${DP_SIZE}"

COMMON_ARGS=(
  --config vidaforge_adapters/automodel/configs/wan2_1_t2v_flow.yaml
  --wandb.enable false
  --dist_env.backend nccl
  --dist_env.timeout_minutes 30
  --model.pretrained_model_name_or_path "${MODEL_PATH}"
  --data.dataloader._target_ vidaforge_adapters.automodel.VidaForgeVideoDataloaderConfig
  --data.dataloader.cache_dir "${VALID_CACHE_DIR}"
  --data.dataloader.model_type wan
  --data.dataloader.dynamic_batch_size false
  --data.dataloader.shuffle false
  --data.dataloader.drop_last true
  --data.dataloader.num_workers "${NUM_WORKERS}"
  --data.dataloader.pin_memory true
  --data.dataloader.prefetch_factor "${PREFETCH_FACTOR}"
  --data.dataloader.map_location cpu
  --step_scheduler.global_batch_size "${GLOBAL_BATCH_SIZE}"
  --step_scheduler.local_batch_size "${LOCAL_BATCH_SIZE}"
  --step_scheduler.ckpt_every_steps 100
  --step_scheduler.num_epochs 1
  --step_scheduler.max_steps 1
  --step_scheduler.log_every 1000000
  --optim.learning_rate 5e-5
  --lr_scheduler.min_lr 1e-6
  --lr_scheduler.lr_warmup_steps 0
  --checkpoint.checkpoint_dir "${CHECKPOINT_DIR}"
  --checkpoint.model_save_format safetensors
  --checkpoint.save_consolidated every
  --checkpoint.diffusers_compatible true
  --fsdp.dp_replicate_size "${DP_REPLICATE_SIZE}"
  --fsdp.dp_size "${DP_SIZE}"
  --eval.log_every "${EVAL_LOG_EVERY}"
  --eval.seed "${EVAL_SEED}"
)

if [[ -n "${EVAL_MAX_BATCHES}" ]]; then
  COMMON_ARGS+=(--eval.max_batches "${EVAL_MAX_BATCHES}")
fi

run_eval() {
  local output_path="$1"
  shift

  torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    vidaforge_adapters/automodel/eval_loss.py \
    "${COMMON_ARGS[@]}" \
    --eval.output_path "${output_path}" \
    "$@"
}

if [[ "${EVAL_FRESH_INIT}" == "true" ]]; then
  echo "[eval] fresh_init -> ${OUTPUT_DIR}/step_000000_fresh_init.json"
  run_eval "${OUTPUT_DIR}/step_000000_fresh_init.json" \
    --checkpoint.enabled false \
    --eval.require_checkpoint false
fi

mapfile -t CHECKPOINT_NAMES < <(
  find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*_step_*' -printf '%f\n' \
    | awk -F'_step_' 'NF == 2 && $2 ~ /^[0-9]+$/ { print $2 "\t" $0 }' \
    | sort -n -k1,1 \
    | cut -f2-
)

if [[ "${#CHECKPOINT_NAMES[@]}" -eq 0 ]]; then
  echo "no checkpoint directories found under ${CHECKPOINT_DIR}" >&2
  exit 2
fi

for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
  step="${checkpoint_name##*_step_}"
  printf -v step_label "%06d" "${step}"
  output_path="${OUTPUT_DIR}/step_${step_label}_${checkpoint_name}.json"

  if [[ -s "${output_path}" ]]; then
    echo "[eval] skip existing ${output_path}"
    continue
  fi

  echo "[eval] ${checkpoint_name} -> ${output_path}"
  run_eval "${output_path}" \
    --checkpoint.enabled true \
    --checkpoint.restore_from "${checkpoint_name}"
done
