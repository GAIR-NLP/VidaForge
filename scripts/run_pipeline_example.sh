#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PIPELINE_TARGETS="${PIPELINE_TARGETS:-${PIPELINE_STEPS:-probe}}"

export DATA_DIR="${DATA_DIR:-/path/to/vidaforge_output}"
export RAW_DIR="${RAW_DIR:-/path/to/raw_videos}"
export RUN_ID="${RUN_ID:-example_run}"
export INPUT_RUN_ID="${INPUT_RUN_ID:-${RUN_ID}}"
export SOURCE="${SOURCE:-example_source}"
export SOURCE_BATCH="${SOURCE_BATCH:-default}"
export VIDEO_LIMIT="${VIDEO_LIMIT:-20000000}"
export CLIP_LIMIT="${CLIP_LIMIT:-100000000}"
export PARQUET_SIZE="${PARQUET_SIZE:-10000}"
export RESUME=False

if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
  PIPELINE_TARGETS="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "Top-level pipeline runners do not accept Hydra overrides: $*" >&2
  echo "Edit scripts/run_pipeline_example.sh directly, or call the step runner under scripts/stage*/." >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline_example.sh <target>[,<target>...]

Targets:
  probe, screen, transcode
  detect, clip
  context, filter_quality, filter_aesthetic, filter_text
  dedup, dedup_pdq, dedup_cosmos, select
  camera, caption, tag
  pack, automodel, automodel_wan, vjepa2
  all                     run the main pipeline in order, ending with automodel pack

Notes:
  filter_quality writes step2_filter_quality/run_id_${RUN_ID}.
  filter_aesthetic reads filter_quality and writes step2_filter_aesthetic.
  filter_text reads filter_aesthetic and writes final step2_filter/run_id_${RUN_ID}.
  dedup_pdq reads final step2_filter and writes step3_dedup_pdq.
  dedup_cosmos reads step3_dedup_pdq and writes final step3_dedup.
  pack reads step3_tag and writes stage5_packaging/automodel.
  vjepa2 is an optional Stage 5 exporter; it reads step3_tag and writes stage5_packaging/vjepa2.
EOF
}

print_step() {
  echo "run_id=${RUN_ID}"
  echo "target=${PIPELINE_TARGETS}"
  echo "step=$1"
}

run_probe() {
  print_step "stage1_ingestion/step1_probe"
  bash scripts/stage1_ingestion/run_step1_probe.sh \
    limit="${VIDEO_LIMIT}" \
    step.batch_size=128 \
    step.ffprobe_bin="ffprobe"
}

run_screen() {
  print_step "stage1_ingestion/step2_screen"
  bash scripts/stage1_ingestion/run_step2_screen.sh \
    step.rules.short_side.min=256 \
    limit="${VIDEO_LIMIT}"
}

run_transcode() {
  print_step "stage1_ingestion/step3_transcode"
  bash scripts/stage1_ingestion/run_step3_transcode.sh \
    limit="${VIDEO_LIMIT}" \
    step.ffmpeg_bin="ffmpeg" \
    step.ffprobe_bin="ffprobe" \
    step.ray_num_cpus=1 \
    step.ffmpeg_threads=1
}

run_detect() {
  print_step "stage2_segmentation/step1_detect"
  bash scripts/stage2_segmentation/run_step1_detect.sh \
    limit="${VIDEO_LIMIT}" \
    step.ray_num_cpus=1 \
    step.min_len_sec=2.0 \
    step.detectors=['transnetv2'] \
    step.detector.transnetv2.weights_path="/path/to/transnetv2-pytorch-weights.pth"
}

run_clip() {
  print_step "stage2_segmentation/step2_clip"
  bash scripts/stage2_segmentation/run_step2_clip.sh \
    limit="${VIDEO_LIMIT}" \
    step.ffmpeg_bin="ffmpeg" \
    step.ray_num_cpus=2 \
    step.min_len_sec=2.0
}

run_context() {
  print_step "stage3_selection/step1_context"
  bash scripts/stage3_selection/run_step1_context.sh \
    limit="${CLIP_LIMIT}" \
    step.ray_num_cpus=1 \
    step.batch_size=4 \
    step.ffmpeg_bin="ffmpeg" \
    step.frame.sampled_fps=2 \
    step.frame.short_side=256
}

run_filter_quality() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step1_context/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step2_filter_quality/run_id_${RUN_ID}"

  print_step "stage3_selection/step2_filter_quality"
  bash scripts/stage3_selection/run_step2_filter.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.batch_size=32 \
    step.filters='[optical,motion]' \
    step.ray_num_gpus=0 \
    step.filter.motion.ffmpeg_bin="ffmpeg"
}

run_filter_aesthetic() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step2_filter_quality/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step2_filter_aesthetic/run_id_${RUN_ID}"

  print_step "stage3_selection/step2_filter_aesthetic"
  bash scripts/stage3_selection/run_step2_filter.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.batch_size=192 \
    step.filters='[aesthetic]' \
    step.ray_num_cpus=10 \
    step.ray_num_gpus=1 \
    step.filter.aesthetic.device=cuda \
    step.filter.aesthetic.forward_batch_size=512 \
    step.filter.aesthetic.prefetch_batches=2 \
    step.filter.aesthetic.predictor_path="/path/to/aesthetic_predictor_v2_5.pth" \
    step.filter.aesthetic.encoder_path="/path/to/siglip-so400m-patch14-384"
}

run_filter_text() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step2_filter_aesthetic/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step2_filter/run_id_${RUN_ID}"

  print_step "stage3_selection/step2_filter"
  bash scripts/stage3_selection/run_step2_filter.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.batch_size=192 \
    step.filters='[text]' \
    step.ray_num_cpus=10 \
    step.ray_num_gpus=1 \
    step.filter.text.device=cuda \
    step.filter.text.forward_batch_size=512 \
    step.filter.text.prefetch_batches=2 \
    step.filter.text.model_path="/path/to/PP-OCRv5_server_det_safetensors"
}

run_dedup_pdq() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step2_filter/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step3_dedup_pdq/run_id_${RUN_ID}"

  print_step "stage3_selection/step3_dedup_pdq"
  bash scripts/stage3_selection/run_step3_dedup.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.deduplicators='[pdq]' \
    step.apply.enabled=true \
    step.apply.replicas=auto \
    step.apply.ray_num_cpus=1 \
    step.apply.ray_num_gpus=0 \
    step.apply.batch_size=64 \
    step.match.replicas=auto \
    step.match.ray_num_cpus=16 \
    step.match.ray_num_gpus=0 \
    step.match.batch_size=512
}

run_dedup_cosmos() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step3_dedup_pdq/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step3_dedup/run_id_${RUN_ID}"

  print_step "stage3_selection/step3_dedup_cosmos"
  bash scripts/stage3_selection/run_step3_dedup.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.deduplicators='[cosmos]' \
    step.apply.enabled=true \
    step.apply.replicas=auto \
    step.apply.ray_num_cpus=5 \
    step.apply.ray_num_gpus=1 \
    step.apply.batch_size=768 \
    step.match.replicas=auto \
    step.match.ray_num_cpus=16 \
    step.match.ray_num_gpus=1 \
    step.match.batch_size=512 \
    step.deduplicator.cosmos.feature.model_name="/path/to/Cosmos-Embed1-336p" \
    step.deduplicator.cosmos.feature.forward_batch_size=256 \
    step.deduplicator.cosmos.feature.prefetch_batches=2 \
    step.deduplicator.cosmos.match.min_cosine_similarity=0.95 \
    step.deduplicator.cosmos.match.index_backend=gpu_cuvs
}

run_select() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step3_dedup/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage3_selection/step4_select/run_id_${RUN_ID}"

  print_step "stage3_selection/step4_select"
  bash scripts/stage3_selection/run_step4_select.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.filter.filter_ok.equals=1 \
    step.filter.optical.min=0.9 \
    step.filter.motion.min=0.1 \
    step.filter.aesthetic.min=0.1 \
    step.filter.text.min=0.5 \
    step.dedup.dedup_ok.equals=1 \
    step.dedup.pdq.keep_ratio=1.0 \
    step.dedup.pdq.min_keep=1 \
    step.dedup.pdq.max_keep=1 \
    step.dedup.cosmos.keep_ratio=0.2 \
    step.dedup.cosmos.min_keep=1 \
    step.dedup.cosmos.max_keep=20
}

run_camera() {
  local input_path="${DATA_DIR}/meta/stage3_selection/step4_select/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage4_annotation/step1_camera/run_id_${RUN_ID}"

  print_step "stage4_annotation/step1_camera"
  bash scripts/stage4_annotation/run_step1_camera.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.serve.model_path="/path/to/gemma-4-E4B-it" \
    step.serve.model_name="gemma-4-E4B-it" \
    step.resume=true \
    step.serve.replicas=auto \
    step.serve.tp_size=1 \
    step.serve.ray_num_cpus=5 \
    step.serve.allowed_local_media_path="${DATA_DIR}" \
    step.client.batch_size=256 \
    step.client.ray_num_cpus=5 \
    step.inference.media_input=local \
    step.inference.request_concurrency=64 \
    step.inference.max_tokens=512 \
    ++step.inference.extra_body.chat_template_kwargs.enable_thinking=false
}

run_caption() {
  local input_path="${DATA_DIR}/meta/stage4_annotation/step1_camera/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage4_annotation/step2_caption/run_id_${RUN_ID}"

  print_step "stage4_annotation/step2_caption"
  bash scripts/stage4_annotation/run_step2_caption.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.mode="video" \
    step.serve.model_path="/path/to/Qwen3.6-27B-FP8" \
    step.serve.model_name="Qwen3.6-27B-FP8" \
    step.resume=true \
    step.serve.replicas=auto \
    step.serve.tp_size=1 \
    step.serve.ray_num_cpus=5 \
    step.serve.allowed_local_media_path="${DATA_DIR}" \
    step.client.batch_size=256 \
    step.client.ray_num_cpus=5 \
    step.inference.media_input=local \
    step.inference.request_concurrency=64 \
    step.inference.max_tokens=4096 \
    ++step.inference.extra_body.chat_template_kwargs.enable_thinking=false
}

run_tag() {
  local input_path="${DATA_DIR}/meta/stage4_annotation/step2_caption/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/meta/stage4_annotation/step3_tag/run_id_${RUN_ID}"

  print_step "stage4_annotation/step3_tag"
  bash scripts/stage4_annotation/run_step3_tag.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.resume=true \
    step.serve.model_path="/path/to/Qwen3.6-27B-FP8" \
    step.serve.model_name="Qwen3.6-27B-FP8" \
    step.serve.replicas=auto \
    step.serve.tp_size=1 \
    step.serve.ray_num_cpus=5 \
    step.serve.allowed_local_media_path="${DATA_DIR}" \
    step.client.batch_size=256 \
    step.client.ray_num_cpus=5 \
    step.inference.media_input=local \
    step.inference.request_concurrency=64 \
    step.inference.max_tokens=2048 \
    ++step.inference.extra_body.chat_template_kwargs.enable_thinking=false
}

run_pack_automodel_wan() {
  local input_path="${DATA_DIR}/meta/stage4_annotation/step3_tag/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/data/stage5_packaging/automodel/run_id_${RUN_ID}"

  print_step "stage5_packaging/automodel"
  bash scripts/stage5_packaging/run_automodel.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.resume=true \
    step.batch_size=32 \
    step.replicas=auto \
    step.ray_num_cpus=8 \
    step.ray_num_gpus=1 \
    step.select_pass=null \
    step.caption_field=caption_level_3 \
    step.dynamic_forward_batch_size=4 \
    step.metadata_shard_size="${PARQUET_SIZE}" \
    step.encoder.model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
}

run_pack_vjepa2() {
  local input_path="${DATA_DIR}/meta/stage4_annotation/step3_tag/run_id_${RUN_ID}"
  local output_path="${DATA_DIR}/data/stage5_packaging/vjepa2/run_id_${RUN_ID}"

  print_step "stage5_packaging/vjepa2"
  bash scripts/stage5_packaging/run_vjepa2.sh \
    input_path="${input_path}" \
    output_path="${output_path}" \
    limit="${CLIP_LIMIT}" \
    step.select_pass=1 \
    step.duration_sec.min=2.0 \
    step.duration_sec.max=10.0 \
    step.resolution.min=256p \
    step.resolution.max=1080p \
    step.manifest_name=train.csv
}

run_all() {
  run_probe
  run_screen
  run_transcode
  run_detect
  run_clip
  run_context
  run_filter_quality
  run_filter_aesthetic
  run_filter_text
  run_dedup_pdq
  run_dedup_cosmos
  run_select
  run_camera
  run_caption
  run_tag
  run_pack_automodel_wan
}

run_step() {
  local step="$1"
  shift

  case "${step}" in
    probe|stage1_probe|step1_probe|stage1_step1_probe)
      run_probe
      ;;
    screen|stage1_screen|step2_screen|stage1_step2_screen)
      run_screen
      ;;
    transcode|stage1_transcode|step3_transcode|stage1_step3_transcode)
      run_transcode
      ;;
    detect|stage2_detect|step1_detect|stage2_step1_detect)
      run_detect
      ;;
    clip|stage2_clip|step2_clip|stage2_step2_clip)
      run_clip
      ;;
    context|stage3_context|step1_context|stage3_step1_context)
      run_context
      ;;
    filter_quality|stage3_filter_quality|step2_filter_quality|stage3_step2_filter_quality)
      run_filter_quality
      ;;
    filter_aesthetic|stage3_filter_aesthetic)
      run_filter_aesthetic
      ;;
    filter_text|stage3_filter_text)
      run_filter_text
      ;;
    dedup|stage3_dedup|step3_dedup|stage3_step3_dedup)
      run_dedup_pdq
      run_dedup_cosmos
      ;;
    dedup_pdq|stage3_dedup_pdq)
      run_dedup_pdq
      ;;
    dedup_cosmos|stage3_dedup_cosmos)
      run_dedup_cosmos
      ;;
    select|stage3_select|step4_select|stage3_step4_select)
      run_select
      ;;
    camera|stage4_camera|step1_camera|stage4_step1_camera)
      run_camera
      ;;
    caption|stage4_caption|step2_caption|stage4_step2_caption)
      run_caption
      ;;
    tag|stage4_tag|step3_tag|stage4_step3_tag)
      run_tag
      ;;
    pack|automodel|automodel_wan|stage5_pack|stage5_automodel|stage5_automodel_wan)
      run_pack_automodel_wan
      ;;
    vjepa2|pack_vjepa2|stage5_vjepa2|stage5_pack_vjepa2)
      run_pack_vjepa2
      ;;
    all)
      run_all
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown pipeline target: ${step}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a SELECTED_TARGETS <<< "${PIPELINE_TARGETS}"

for target in "${SELECTED_TARGETS[@]}"; do
  run_step "${target}"
done
