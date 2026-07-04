"""Stage 5 packaging Hydra entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from vidaforge.common.hydra import as_bool, object_dict, optional_float, optional_int, optional_string
from vidaforge.packaging.automodel.config import AutoModelPackConfig
from vidaforge.packaging.automodel.orchestrator import AutoModelPackOrchestrator
from vidaforge.packaging.automodel.wan import WanAutoModelEncoder
from vidaforge.packaging.vjepa2.config import VJEPA2PackConfig
from vidaforge.packaging.vjepa2.orchestrator import VJEPA2PackOrchestrator


def _build_automodel_config(cfg: DictConfig) -> AutoModelPackConfig:
    step = cfg.step
    bucket = object_dict(step.bucket)
    return AutoModelPackConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        source=str(cfg.source),
        source_batch=str(cfg.source_batch),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        name=str(step.name),
        caption_field=str(step.caption_field),
        select_pass=optional_int(step.select_pass),
        batch_size=int(step.batch_size),
        dynamic_forward_batch_size=int(step.dynamic_forward_batch_size),
        metadata_shard_size=int(step.metadata_shard_size),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        replicas=step.replicas,
        ray_num_cpus=float(step.ray_num_cpus),
        ray_num_gpus=float(step.ray_num_gpus),
        bucket_resolution=str(step.bucket.resolution),
        bucket_upscale=as_bool(step.bucket.upscale),
        bucket_durations_sec=[
            float(value)
            for value in bucket.get("durations_sec", [2, 3, 4, 5, 6, 8, 10])
        ],
        limit=optional_int(cfg.limit),
        resume=as_bool(step.resume),
    )


def _build_wan_encoder_kwargs(cfg: DictConfig) -> dict[str, object]:
    encoder = cfg.step.encoder
    return {
        "model_name": str(encoder.model_name),
        "device": str(encoder.device),
        "max_sequence_length": int(encoder.max_sequence_length),
        "resize_mode": str(encoder.resize_mode),
        "seek_mode": str(encoder.seek_mode),
        "center_crop": as_bool(encoder.center_crop),
        "deterministic_latents": as_bool(encoder.deterministic_latents),
    }


def _run_automodel(cfg: DictConfig) -> None:
    encoder_name = str(cfg.step.encoder.name)
    if encoder_name != "wan":
        raise ValueError(f"Unsupported AutoModel encoder: {encoder_name!r}")

    config = _build_automodel_config(cfg)
    result = AutoModelPackOrchestrator(
        stage_name="stage5_packaging",
        step_name="automodel",
    ).pack(
        config,
        encoder_cls=WanAutoModelEncoder,
        encoder_kwargs=_build_wan_encoder_kwargs(cfg),
    )

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"metadata_shard_count={result.metadata_shard_count}")
    print(f"output_path={result.output_path}")
    print(f"metadata_path={result.metadata_path}")
    print(f"summary_path={result.summary_path}")
    print(f"caption_field={config.caption_field}")
    print(f"select_pass={config.select_pass}")
    print(f"batch_size={config.batch_size}")
    print(f"dynamic_forward_batch_size={config.dynamic_forward_batch_size}")
    print(f"ray_address={config.ray_address}")
    print(f"replicas={config.replicas}")
    print(f"ray_num_cpus={config.ray_num_cpus}")
    print(f"ray_num_gpus={config.ray_num_gpus}")
    print(f"bucket_train_resolution={config.bucket_resolution}")
    print(f"bucket_upscale={config.bucket_upscale}")
    print(f"bucket_durations_sec={config.bucket_durations_sec}")


def _build_vjepa2_config(cfg: DictConfig) -> VJEPA2PackConfig:
    step = cfg.step
    return VJEPA2PackConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        source=str(cfg.source),
        source_batch=str(cfg.source_batch),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        name=str(step.name),
        select_pass=optional_int(step.select_pass),
        label=int(step.get("label", 0)),
        manifest_name=str(step.manifest_name),
        duration_min_sec=optional_float(step.duration_sec.min),
        duration_max_sec=optional_float(step.duration_sec.max),
        resolution_min=optional_string(step.resolution.min),
        resolution_max=optional_string(step.resolution.max),
        parquet_size=int(cfg.parquet_size),
        limit=optional_int(cfg.limit),
    )


def _run_vjepa2(cfg: DictConfig) -> None:
    config = _build_vjepa2_config(cfg)
    result = VJEPA2PackOrchestrator(
        stage_name="stage5_packaging",
        step_name="vjepa2",
    ).pack(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"rejected_count={result.rejected_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"manifest_path={result.manifest_path}")
    print(f"summary_path={result.summary_path}")
    print(f"select_pass={config.select_pass}")
    print(f"manifest_name={config.manifest_name}")
    print(f"duration_min_sec={config.duration_min_sec}")
    print(f"duration_max_sec={config.duration_max_sec}")
    print(f"resolution_min={config.resolution_min}")
    print(f"resolution_max={config.resolution_max}")


@hydra.main(version_base=None, config_path="../configs/stage5_packaging", config_name="config")
def main(cfg: DictConfig) -> None:
    step_name = str(cfg.step.name)
    if step_name == "automodel":
        _run_automodel(cfg)
        return
    if step_name == "vjepa2":
        _run_vjepa2(cfg)
        return
    raise ValueError(f"Unsupported Stage 5 packaging step: {step_name!r}")


if __name__ == "__main__":
    main()
