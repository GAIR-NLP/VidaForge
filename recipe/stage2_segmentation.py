"""Stage 2 segmentation Hydra entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from vidaforge.common.hydra import as_bool, optional_int
from vidaforge.common import replace_path_part
from vidaforge.segmentation import (
    ClipConfig,
    ClipOrchestrator,
    DetectConfig,
    DetectOrchestrator,
)
from vidaforge.segmentation.detect.registry import DETECTORS


def _parse_detector_names(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        return [value.lower()]
    if isinstance(raw, (list, tuple, ListConfig)):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    raise TypeError(f"Unsupported detectors value: {raw!r}")


def _instantiate_detectors(cfg: DictConfig) -> list[object]:
    detector_names = _parse_detector_names(cfg.step.detectors)
    if not detector_names:
        raise ValueError("detectors must contain at least one detector name.")

    detectors_cfg = cfg.step.detector or {}
    detectors: list[object] = []
    for detector_name in detector_names:
        detector_spec = DETECTORS.get(detector_name)
        if detector_spec is None:
            raise ValueError(
                f"Unsupported detector_name={detector_name!r}. "
                f"Supported: {sorted(DETECTORS)}"
            )
        detector_config_cfg = detectors_cfg.get(detector_name, {})
        detector_config = OmegaConf.to_container(detector_config_cfg, resolve=True)
        if not isinstance(detector_config, dict):
            raise TypeError(
                f"step.detector.{detector_name} must be a mapping, "
                f"got {type(detector_config)!r}"
            )
        if detector_name != "uniform" and detector_config.get("min_len_sec") is None:
            detector_config["min_len_sec"] = float(cfg.step.min_len_sec)
        detectors.append(detector_spec.config_type.model_validate(detector_config))
    return detectors


def _run_detect(cfg: DictConfig) -> None:
    detectors = _instantiate_detectors(cfg)
    pipeline = DetectOrchestrator(
        stage_name="stage2_segmentation",
        step_name="step1_detect",
    )
    config = DetectConfig(
        input_path=Path(cfg.input_path),
        output_path=Path(cfg.output_path),
        source=cfg.source,
        source_batch=cfg.source_batch,
        input_run_id=cfg.input_run_id,
        run_id=cfg.run_id,
        detectors=detectors,
        min_len_sec=float(cfg.step.min_len_sec),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )
    result = pipeline.detect(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")


def _run_clip(cfg: DictConfig) -> None:
    pipeline = ClipOrchestrator(
        stage_name="stage2_segmentation",
        step_name="step2_clip",
    )
    output_meta_path = Path(str(cfg.output_path))
    config = ClipConfig(
        input_path=Path(str(cfg.input_path)),
        output_data_path=replace_path_part(output_meta_path, old="meta", new="data"),
        output_meta_path=output_meta_path,
        source=cfg.source,
        source_batch=cfg.source_batch,
        run_id=cfg.run_id,
        input_run_id=cfg.input_run_id,
        min_len_sec=float(cfg.step.min_len_sec),
        max_len_sec=float(cfg.step.max_len_sec),
        overlong_split_len_sec=float(cfg.step.overlong_split_len_sec),
        boundary_trim_sec=float(cfg.step.boundary_trim_sec),
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        ffmpeg_bin=str(cfg.step.ffmpeg_bin),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )
    result = pipeline.clip(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_data_path={result.output_data_path}")
    print(f"output_meta_path={result.output_meta_path}")
    print(f"summary_path={result.summary_path}")


@hydra.main(version_base=None, config_path="../configs/stage2_segmentation", config_name="config")
def main(cfg: DictConfig) -> None:
    step_name = str(cfg.step.name)
    if step_name == "step1_detect":
        _run_detect(cfg)
        return
    if step_name == "step2_clip":
        _run_clip(cfg)
        return
    raise ValueError(f"Unsupported Stage 2 segmentation step: {step_name!r}")


if __name__ == "__main__":
    main()
