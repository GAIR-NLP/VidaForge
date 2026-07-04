from __future__ import annotations

from collections import Counter
import time
from pathlib import Path

from vidaforge.common import join_data_dir, utc_now_iso, write_summary_json
from vidaforge.index import StreamingParquetShardWriter, count_parquet, iter_parquet
from vidaforge.packaging.automodel.resolution import resolution_pixel_budget

from .config import VJEPA2PackConfig, VJEPA2PackResult


def validate_vjepa2_pack_config(config: VJEPA2PackConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if config.select_pass not in {None, 0, 1}:
        raise ValueError("select_pass must be one of: null, 0, 1")
    if (
        config.duration_min_sec is not None
        and config.duration_max_sec is not None
        and config.duration_min_sec > config.duration_max_sec
    ):
        raise ValueError("duration_sec.min must be <= duration_sec.max")
    resolution_min_pixels = (
        resolution_pixel_budget(config.resolution_min)
        if config.resolution_min is not None
        else None
    )
    resolution_max_pixels = (
        resolution_pixel_budget(config.resolution_max)
        if config.resolution_max is not None
        else None
    )
    if (
        resolution_min_pixels is not None
        and resolution_max_pixels is not None
        and resolution_min_pixels > resolution_max_pixels
    ):
        raise ValueError("resolution.min must be <= resolution.max")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be null or > 0")
    if not config.manifest_name.strip():
        raise ValueError("manifest_name must not be empty")
    if Path(config.manifest_name).name != config.manifest_name:
        raise ValueError("manifest_name must be a filename, not a path")


def vjepa2_input_row_filter(
    row: dict[str, object],
    *,
    select_pass: int | None,
    duration_min_sec: float | None = None,
    duration_max_sec: float | None = None,
    resolution_min_pixels: int | None = None,
    resolution_max_pixels: int | None = None,
) -> bool:
    return (
        vjepa2_reject_reason(
            row,
            select_pass=select_pass,
            duration_min_sec=duration_min_sec,
            duration_max_sec=duration_max_sec,
            resolution_min_pixels=resolution_min_pixels,
            resolution_max_pixels=resolution_max_pixels,
        )
        is None
    )


def vjepa2_reject_reason(
    row: dict[str, object],
    *,
    select_pass: int | None,
    duration_min_sec: float | None,
    duration_max_sec: float | None,
    resolution_min_pixels: int | None,
    resolution_max_pixels: int | None,
) -> str | None:
    if select_pass is None:
        pass
    elif int(row["select_pass"]) != int(select_pass):
        return "select_pass_mismatch"

    if duration_min_sec is not None or duration_max_sec is not None:
        try:
            duration_sec = float(row.get("duration_sec"))
        except (TypeError, ValueError):
            return "duration_missing"
        if duration_sec <= 0:
            return "duration_missing"
        if duration_min_sec is not None and duration_sec < duration_min_sec:
            return "duration_below_min"
        if duration_max_sec is not None and duration_sec > duration_max_sec:
            return "duration_above_max"

    if resolution_min_pixels is not None or resolution_max_pixels is not None:
        width = int(row.get("width", 0) or 0)
        height = int(row.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            return "resolution_missing"
        source_pixels = width * height
        if resolution_min_pixels is not None and source_pixels < resolution_min_pixels:
            return "resolution_below_min"
        if resolution_max_pixels is not None and source_pixels > resolution_max_pixels:
            return "resolution_above_max"

    return None


class VJEPA2PackOrchestrator:
    """Export Stage 5 V-JEPA2 video manifest without copying or decoding videos."""

    def __init__(
        self,
        stage_name: str = "stage5_packaging",
        step_name: str = "vjepa2",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def pack(self, config: VJEPA2PackConfig) -> VJEPA2PackResult:
        validate_vjepa2_pack_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / config.manifest_name
        if manifest_path.suffix.lower() != ".csv":
            raise ValueError(
                "manifest_name must end with .csv for V-JEPA2 VideoDataset"
            )

        source_count = count_parquet(input_path, unit="clip")
        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        resolution_min_pixels = (
            resolution_pixel_budget(config.resolution_min)
            if config.resolution_min is not None
            else None
        )
        resolution_max_pixels = (
            resolution_pixel_budget(config.resolution_max)
            if config.resolution_max is not None
            else None
        )

        writer = StreamingParquetShardWriter(
            output_path,
            unit="clip",
            parquet_size=config.parquet_size,
            reset=True,
        )
        source_resolution_counter: Counter[str] = Counter()
        select_pass_counter: Counter[str] = Counter()
        reject_reason_counter: Counter[str] = Counter()
        input_count = 0
        output_count = 0

        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in iter_parquet(
                input_path,
                unit="clip",
            ):
                input_count += 1
                reject_reason = vjepa2_reject_reason(
                    row,
                    select_pass=config.select_pass,
                    duration_min_sec=config.duration_min_sec,
                    duration_max_sec=config.duration_max_sec,
                    resolution_min_pixels=resolution_min_pixels,
                    resolution_max_pixels=resolution_max_pixels,
                )
                if reject_reason is not None:
                    reject_reason_counter[reject_reason] += 1
                    continue

                clip_path = Path(str(row["clip_path"]))
                if clip_path.is_absolute():
                    video_path = clip_path.expanduser().resolve()
                else:
                    video_path = join_data_dir(clip_path)
                video_path_text = str(video_path)
                if any(char.isspace() for char in video_path_text):
                    raise ValueError(
                        "V-JEPA2 space-delimited manifest cannot represent paths "
                        f"with whitespace: {video_path_text}"
                    )

                width = int(row.get("width", 0) or 0)
                height = int(row.get("height", 0) or 0)
                if width > 0 and height > 0:
                    source_resolution_counter[f"{width}x{height}"] += 1
                else:
                    source_resolution_counter["unknown"] += 1
                select_pass_counter[str(int(row.get("select_pass", -1) or 0))] += 1

                manifest.write(f"{video_path_text} {config.label}\n")
                output_row = dict(row)
                output_row.update(
                    {
                        "vjepa2_ok": 1,
                        "vjepa2_video_path": video_path_text,
                        "vjepa2_manifest_path": str(manifest_path),
                    }
                )
                writer.write(output_row)
                output_count += 1
                if config.limit is not None and output_count >= config.limit:
                    break

        writer.close()
        if output_count <= 0:
            manifest_path.unlink(missing_ok=True)
            raise ValueError(
                "V-JEPA2 export matched 0 rows. Check select_pass filters "
                f"for input_path={input_path}"
            )

        writer_summary = writer.summary()
        rejected_count = sum(reject_reason_counter.values())
        elapsed_sec = round(time.perf_counter() - started_perf, 3)
        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "parquet_size": config.parquet_size,
            "select_pass": config.select_pass,
            "manifest_name": config.manifest_name,
            "manifest_path": str(manifest_path),
            "duration_sec": {
                "min": config.duration_min_sec,
                "max": config.duration_max_sec,
            },
            "resolution": {
                "min": config.resolution_min,
                "max": config.resolution_max,
                "min_pixels": resolution_min_pixels,
                "max_pixels": resolution_max_pixels,
            },
            "source_resolution_distribution": {
                key: {"count": count, "ratio": round(count / output_count, 6)}
                for key, count in sorted(source_resolution_counter.items())
            },
            "select_pass_distribution": {
                key: {"count": count, "ratio": round(count / output_count, 6)}
                for key, count in sorted(select_pass_counter.items())
            },
            "reject_reason_counts": dict(sorted(reject_reason_counter.items())),
            "source_count": source_count,
            "input_count": input_count,
            "output_count": output_count,
            "ok_count": output_count,
            "rejected_count": rejected_count,
            "failed_count": 0,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_run_id": config.input_run_id,
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return VJEPA2PackResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=input_count,
            output_count=output_count,
            ok_count=output_count,
            rejected_count=rejected_count,
            failed_count=0,
            shard_count=int(writer_summary["shard_count"]),
            manifest_path=manifest_path,
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
