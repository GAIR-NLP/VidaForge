"""Stage 1 ingestion Hydra entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from vidaforge.common.hydra import as_bool, optional_int, optional_string, plain_dict
from vidaforge.common import replace_path_part
from vidaforge.ingestion.probe import ProbeConfig, ProbeOrchestrator
from vidaforge.ingestion.screen import ScreenConfig, ScreenOrchestrator
from vidaforge.ingestion.transcode import TranscodeConfig, TranscodeOrchestrator


def _run_probe(cfg: DictConfig) -> None:
    pipeline = ProbeOrchestrator(
        stage_name="stage1_ingestion",
        step_name="step1_probe",
    )
    temp_dir = optional_string(cfg.step.temp_dir)
    config = ProbeConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        source="" if cfg.source is None else str(cfg.source),
        source_batch="" if cfg.source_batch is None else str(cfg.source_batch),
        run_id="" if cfg.run_id is None else str(cfg.run_id),
        ffprobe_bin=str(cfg.step.ffprobe_bin),
        temp_dir=None if temp_dir is None else Path(temp_dir),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        batch_size=int(cfg.step.batch_size),
        limit=optional_int(cfg.limit),
    )
    result = pipeline.probe(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"elapsed_sec={result.elapsed_sec}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")


def _run_screen(cfg: DictConfig) -> None:
    pipeline = ScreenOrchestrator(
        stage_name="stage1_ingestion",
        step_name="step2_screen",
    )
    config = ScreenConfig(
        input_path=Path(str(cfg.input_path)),
        output_path=Path(str(cfg.output_path)),
        input_run_id=str(cfg.input_run_id),
        run_id=str(cfg.run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        parquet_size=int(cfg.parquet_size),
        rules=plain_dict(cfg.step.rules),
        limit=optional_int(cfg.limit),
    )
    result = pipeline.screen(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"pass_count={result.pass_count}")
    print(f"reject_count={result.reject_count}")
    print(f"shard_count={result.shard_count}")
    print(f"elapsed_sec={result.elapsed_sec}")
    print(f"output_path={result.output_path}")
    print(f"summary_path={result.summary_path}")


def _run_transcode(cfg: DictConfig) -> None:
    pipeline = TranscodeOrchestrator(
        stage_name="stage1_ingestion",
        step_name="step3_transcode",
    )
    output_meta_path = Path(str(cfg.output_path))
    config = TranscodeConfig(
        input_path=Path(str(cfg.input_path)),
        output_data_path=replace_path_part(output_meta_path, old="meta", new="data"),
        output_meta_path=output_meta_path,
        target_short_edge=int(cfg.step.target_short_edge),
        target_fps=int(cfg.step.target_fps),
        crf=int(cfg.step.crf),
        pix_fmt=str(cfg.step.pix_fmt),
        audio_bitrate=str(cfg.step.audio_bitrate),
        input_run_id=str(cfg.input_run_id),
        run_id=str(cfg.run_id),
        source=cfg.source,
        source_batch=cfg.source_batch,
        ffmpeg_bin=str(cfg.step.ffmpeg_bin),
        ffprobe_bin=str(cfg.step.ffprobe_bin),
        parquet_size=int(cfg.parquet_size),
        ray_address=str(cfg.ray_address),
        ray_num_cpus=float(cfg.step.ray_num_cpus),
        ffmpeg_threads=optional_int(cfg.step.ffmpeg_threads),
        limit=optional_int(cfg.limit),
        resume=as_bool(cfg.step.resume),
    )
    result = pipeline.transcode(config)

    print(f"input_path={result.input_path}")
    print(f"source_count={result.source_count}")
    print(f"input_count={result.input_count}")
    print(f"resumed_count={result.resumed_count}")
    print(f"output_count={result.output_count}")
    print(f"ok_count={result.ok_count}")
    print(f"failed_count={result.failed_count}")
    print(f"shard_count={result.shard_count}")
    print(f"elapsed_sec={result.elapsed_sec}")
    print(f"output_data_path={result.output_data_path}")
    print(f"output_meta_path={result.output_meta_path}")
    print(f"summary_path={result.summary_path}")


@hydra.main(version_base=None, config_path="../configs/stage1_ingestion", config_name="config")
def main(cfg: DictConfig) -> None:
    step_name = str(cfg.step.name)
    if step_name == "step1_probe":
        _run_probe(cfg)
        return
    if step_name == "step2_screen":
        _run_screen(cfg)
        return
    if step_name == "step3_transcode":
        _run_transcode(cfg)
        return
    raise ValueError(f"Unsupported Stage 1 step: {step_name!r}")


if __name__ == "__main__":
    main()
