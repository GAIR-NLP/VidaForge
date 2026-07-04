from __future__ import annotations

from functools import partial
from itertools import islice
import time

from vidaforge.common import scan_raw_videos, strip_raw_dir, utc_now_iso, write_summary_json
from vidaforge.index import run_ray_task_processing

from .config import ProbeConfig, ProbeResult
from .worker import process_probe_row


def _validate_config(config: ProbeConfig) -> None:
    if not config.source.strip():
        raise ValueError("source must be set")
    if not config.source_batch.strip():
        raise ValueError("source_batch must be set")
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")


class ProbeOrchestrator:
    """Run Stage 1 probe and emit one metadata row per raw video."""

    def __init__(
        self,
        stage_name: str = "stage1_ingestion",
        step_name: str = "step1_probe",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def probe(self, config: ProbeConfig) -> ProbeResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = sum(1 for _ in scan_raw_videos(input_path))
        input_count = (
            source_count
            if config.limit is None
            else min(source_count, max(0, int(config.limit)))
        )

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)

        probe_worker = partial(
            process_probe_row,
            source=config.source,
            source_batch=config.source_batch,
            run_id=config.run_id,
            ffprobe_bin=config.ffprobe_bin,
            temp_dir=config.temp_dir,
        )
        raw_items = scan_raw_videos(input_path)
        if config.limit is not None:
            raw_items = islice(raw_items, max(0, int(config.limit)))

        stats, writer_summary = run_ray_task_processing(
            rows=(
                {
                    "raw_type": str(item["raw_type"]),
                    "raw_path": strip_raw_dir(item["raw_path"]),
                    "raw_member_path": str(item["raw_member_path"]),
                }
                for item in raw_items
            ),
            row_count=input_count,
            output_path=output_path,
            parquet_size=config.parquet_size,
            output_unit="video",
            step="probe",
            ray_address=config.ray_address,
            ray_num_cpus=config.ray_num_cpus,
            task_batch_size=config.batch_size,
            worker=probe_worker,
            desc="stage1 probe",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "ffprobe_bin": config.ffprobe_bin,
            "temp_dir": None if config.temp_dir is None else str(config.temp_dir),
            "ray_address": config.ray_address,
            "ray_num_cpus": config.ray_num_cpus,
            "batch_size": config.batch_size,
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
            "source": config.source,
            "source_batch": config.source_batch,
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return ProbeResult(
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
