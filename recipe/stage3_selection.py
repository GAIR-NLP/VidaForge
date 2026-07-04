"""Stage 3 selection Hydra entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from vidaforge.common.hydra import as_bool, object_dict, optional_int
from vidaforge.common import replace_path_part
from vidaforge.selection.context import (
    AudioContextConfig,
    ContextConfig,
    ContextOrchestrator,
    FrameContextConfig,
)
from vidaforge.selection.dedup import DedupConfig, DedupOrchestrator
from vidaforge.selection.dedup.config import (
    DedupApplyConfig,
    DedupConfigBase,
    DedupMatchConfig,
)
from vidaforge.selection.dedup.registry import DEDUPLICATORS
from vidaforge.selection.filter import (
    FilterConfig,
    FilterOrchestrator,
)
from vidaforge.selection.filter.config import FilterConfigBase
from vidaforge.selection.filter.registry import FILTERS
from vidaforge.selection.select import SelectConfig, SelectOrchestrator


def _run_context(cfg: DictConfig) -> None:
    pipeline = ContextOrchestrator(
        stage_name="stage3_selection",
        step_name="step1_context",
    )
    output_meta_path = Path(str(cfg.output_path))
    frame = cfg.step.frame
    audio = cfg.step.audio
    config = ContextConfig(
        input_path=Path(str(cfg.input_path)),
        output_data_path=replace_path_part(output_meta_path, old="meta", new="data"),
        output_meta_path=output_meta_path,
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        name=str(cfg.step.name),
        frame=FrameContextConfig(
            sampled_fps=float(frame.sampled_fps),
            short_side=int(frame.short_side),
        ),
        audio=AudioContextConfig(
            format=str(audio.format),
            sample_rate=int(audio.sample_rate),
            channels=int(audio.channels),
        ),
        ffmpeg_bin=str(cfg.step.ffmpeg_bin),
        parquet_size=int(cfg.parquet_size),
        batch_size=int(cfg.step.batch_size),
        ray_address=str(cfg.ray_address),
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )
    result = pipeline.build_context(config)

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


def _run_dedup(cfg: DictConfig) -> None:
    pipeline = DedupOrchestrator(
        stage_name="stage3_selection",
        step_name="step3_dedup",
    )
    deduplicators: list[DedupConfigBase] = []
    for item in cfg.step.deduplicators:
        dedup_name = str(item)
        dedup_spec = DEDUPLICATORS.get(dedup_name)
        if dedup_spec is None:
            raise ValueError(f"unsupported deduplicator: {dedup_name}")
        dedup_config = object_dict(cfg.step.deduplicator.get(dedup_name))
        if not isinstance(dedup_config, dict):
            raise TypeError(f"deduplicator.{dedup_name} must be a dict")
        deduplicators.append(dedup_spec.config_type.model_validate(dedup_config))

    config = DedupConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        name=str(cfg.step.name),
        deduplicators=deduplicators,
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        apply=DedupApplyConfig(
            enabled=as_bool(cfg.step.apply.enabled),
            replicas=cfg.step.apply.replicas,
            ray_num_cpus=float(cfg.step.apply.ray_num_cpus),
            ray_num_gpus=float(cfg.step.apply.ray_num_gpus),
            batch_size=int(cfg.step.apply.batch_size),
        ),
        match=DedupMatchConfig(
            replicas=cfg.step.match.replicas,
            ray_num_cpus=float(cfg.step.match.ray_num_cpus),
            ray_num_gpus=float(cfg.step.match.ray_num_gpus),
            batch_size=int(cfg.step.match.batch_size),
        ),
        limit=optional_int(cfg.limit),
    )
    result = pipeline.dedup(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"pair_count={result.pair_count}")
    print(f"deduplicator_match_summary={result.deduplicator_match_summary}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")


def _run_filter(cfg: DictConfig) -> None:
    pipeline = FilterOrchestrator(
        stage_name="stage3_selection",
        step_name="step2_filter",
    )
    filters: list[FilterConfigBase] = []
    for item in cfg.step.filters:
        filter_name = str(item)
        filter_spec = FILTERS.get(filter_name)
        if filter_spec is None:
            raise ValueError(f"unsupported filter: {filter_name}")
        filter_config = object_dict(cfg.step.filter.get(filter_name))
        if not isinstance(filter_config, dict):
            raise TypeError(f"filter.{filter_name} must be a dict")
        filters.append(filter_spec.config_type.model_validate(filter_config))

    config = FilterConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        name=str(cfg.step.name),
        filters=filters,
        batch_size=int(cfg.step.batch_size),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        replicas=cfg.step.replicas,
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        ray_num_gpus=float(cfg.step.ray_num_gpus),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )
    result = pipeline.filter(config)

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


def _run_select(cfg: DictConfig) -> None:
    pipeline = SelectOrchestrator(
        stage_name="stage3_selection",
        step_name="step4_select",
    )
    config = SelectConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        run_id=str(cfg.run_id),
        input_run_id=str(cfg.input_run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        name=str(cfg.step.name),
        filter=object_dict(cfg.step.filter),
        dedup=object_dict(cfg.step.dedup),
        parquet_size=int(cfg.parquet_size),
        limit=optional_int(cfg.limit),
    )
    result = pipeline.select(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"pass_count={result.pass_count}")
    print(f"reject_count={result.reject_count}")
    print(f"shard_count={result.shard_count}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")


@hydra.main(version_base=None, config_path="../configs/stage3_selection", config_name="config")
def main(cfg: DictConfig) -> None:
    step_name = str(cfg.step.name)
    if step_name == "step1_context":
        _run_context(cfg)
        return
    if step_name == "step2_filter":
        _run_filter(cfg)
        return
    if step_name == "step3_dedup":
        _run_dedup(cfg)
        return
    if step_name == "step4_select":
        _run_select(cfg)
        return
    raise ValueError(f"Unsupported Stage 3 selection step: {step_name!r}")


if __name__ == "__main__":
    main()
