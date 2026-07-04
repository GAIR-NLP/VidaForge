from __future__ import annotations

from functools import partial
import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import (
    count_parquet,
    run_ray_task_processing,
)

from .config import (
    DetectConfig,
    DetectResult,
    DetectorConfigBase,
)
from .registry import DETECTORS
from .worker import process_detect_row


def _validate_config(config: DetectConfig) -> None:
    detector_configs = config.detectors
    if not detector_configs:
        raise ValueError("detectors must contain at least one detector config")
    for detector_config in detector_configs:
        if not isinstance(detector_config, DetectorConfigBase):
            raise TypeError(
                "detectors must contain DetectorConfigBase instances; "
                f"got {type(detector_config)!r}"
            )
    detector_names = [
        detector_config.detector_name
        for detector_config in detector_configs
    ]
    if len(set(detector_names)) != len(detector_names):
        raise ValueError(f"detectors must be unique: {detector_names}")
    unsupported_detector_names = sorted(set(detector_names) - set(DETECTORS))
    if unsupported_detector_names:
        raise ValueError(f"unsupported detectors: {unsupported_detector_names}")
    metadata_only_detector_names = [
        detector_config.detector_name
        for detector_config in detector_configs
        if DETECTORS[detector_config.detector_name].metadata_only
    ]
    if metadata_only_detector_names and len(detector_configs) > 1:
        raise ValueError(
            f"{'+'.join(metadata_only_detector_names)} detector cannot be combined with other detectors"
        )
    if config.min_len_sec <= 0:
        raise ValueError("min_len_sec must be > 0")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")


class DetectOrchestrator:
    """Run boundary detectors over finalized Stage 1 video rows and emit video ticks."""

    def __init__(
        self,
        stage_name: str = "stage2_segmentation",
        step_name: str = "step1_detect",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def detect(self, config: DetectConfig) -> DetectResult:
        _validate_config(config)
        detector_specs = config.detectors
        detector_names = [
            detector_config.detector_name
            for detector_config in detector_specs
        ]
        detector_label = "+".join(detector_names)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="video")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)
        detect_worker = partial(
            process_detect_row,
            detectors=detector_specs,
            run_id=config.run_id,
            input_run_id=config.input_run_id,
        )

        stats, writer_summary = run_ray_task_processing(
            input_path=input_path,
            output_path=output_path,
            parquet_size=config.parquet_size,
            input_unit="video",
            output_unit="video",
            step="detect",
            ray_address=config.ray_address,
            ray_num_cpus=config.ray_num_cpus,
            worker=detect_worker,
            limit=config.limit,
            resume=config.resume,
            desc=f"{detector_label} detect",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "detectors": detector_names,
            "detector_configs": {
                detector_config.detector_name: detector_config.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for detector_config in detector_specs
            },
            "min_len_sec": config.min_len_sec,
            "input_run_id": config.input_run_id,
            "ray_address": config.ray_address,
            "ray_num_cpus": config.ray_num_cpus,
            "resume": config.resume,
            "parquet_size": config.parquet_size,
            "source_count": source_count,
            "input_count": stats.input_count,
            "resumed_count": stats.resumed_count,
            "output_count": stats.output_count,
            "ok_count": stats.ok_count,
            "failed_count": stats.failed_count,
            "shard_count": int(writer_summary["shard_count"]),
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "failed_examples": stats.failed_examples,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return DetectResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=stats.input_count,
            resumed_count=stats.resumed_count,
            output_count=stats.output_count,
            ok_count=stats.ok_count,
            failed_count=stats.failed_count,
            shard_count=int(writer_summary["shard_count"]),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
