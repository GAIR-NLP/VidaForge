"""Stage 5 AutoModel dataset viewer."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .viewer_common import (
        DEFAULT_PROJECT_DIR,
        format_bytes,
        format_elapsed_seconds,
        load_summary,
        render_dataframe_preview,
        render_json_preview,
        render_bar_chart,
    )
except ImportError:
    from viewer_common import (
        DEFAULT_PROJECT_DIR,
        format_bytes,
        format_elapsed_seconds,
        load_summary,
        render_dataframe_preview,
        render_json_preview,
        render_bar_chart,
    )


STAGE_DIR = "stage5_packaging"
STEP_DIR = "automodel"
VIEWER_TITLE = "Stage 5 AutoModel Dataset"
SUMMARY_COUNT_FIELDS = (
    "source_count",
    "input_count",
    "resumed_count",
    "output_count",
    "ok_count",
    "failed_count",
    "packed_count",
    "metadata_row_count",
    "metadata_shard_count",
    "shard_count",
)


def _list_dataset_run_dirs(project_dir: str) -> list[str]:
    base = Path(project_dir).expanduser().resolve() / "data" / STAGE_DIR / STEP_DIR
    if not base.exists() or not base.is_dir():
        return []
    return sorted(
        [path.name for path in base.iterdir() if path.is_dir()],
        reverse=True,
    )


def _resolve_dataset_path(
    *,
    project_dir: str,
    run_dir: str,
    override: str,
) -> Path:
    override_path = Path(override).expanduser()
    if override.strip():
        return override_path.resolve()
    return (
        Path(project_dir).expanduser().resolve()
        / "data"
        / STAGE_DIR
        / STEP_DIR
        / run_dir
    )


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(int(value))
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _resolution_label(value: object) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "unknown"
    width = _safe_int(value[0])
    height = _safe_int(value[1])
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"
    return f"{width}x{height}"


def _source_resolution_label(value: object) -> str:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return "unknown"
    width = _safe_int(value[0])
    height = _safe_int(value[1])
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"
    short_side = min(width, height)
    if short_side < 360:
        return "short_side_<360"
    if short_side < 480:
        return "short_side_360_479"
    if short_side < 720:
        return "short_side_480_719"
    if short_side < 1080:
        return "short_side_720_1079"
    return "short_side_>=1080"


def _frame_bucket_label(value: object) -> str:
    frame_count = _safe_int(value)
    if frame_count is None or frame_count <= 0:
        return "unknown"
    return f"{frame_count}f"


def _latent_shape_label(value: object) -> str:
    if not isinstance(value, list | tuple) or not value:
        return "unknown"
    values = [_safe_int(item) for item in value]
    if any(item is None or item <= 0 for item in values):
        return "unknown"
    return "x".join(str(item) for item in values)


def _cache_path_from_item(item: dict[str, object]) -> Path | None:
    cache_file = str(item.get("cache_file", "") or "").strip()
    if not cache_file:
        return None
    return Path(cache_file).expanduser().resolve()


def _counter_frame(
    counter: Counter[str],
    *,
    total: int,
    name: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    rows = [
        {
            name: key,
            "count": int(count),
            "ratio": round(count / total, 6) if total > 0 else 0.0,
        }
        for key, count in counter.most_common(top_k)
    ]
    return pd.DataFrame.from_records(rows, columns=[name, "count", "ratio"])


def _caption_histogram(lengths: list[int]) -> pd.DataFrame:
    bins = [
        (0, 64),
        (65, 128),
        (129, 256),
        (257, 384),
        (385, 512),
        (513, 768),
        (769, 1024),
    ]
    counter: Counter[str] = Counter()
    for length in lengths:
        label = ">1024"
        for start, end in bins:
            if start <= length <= end:
                label = f"{start}-{end}"
                break
        counter[label] += 1

    labels = [f"{start}-{end}" for start, end in bins] + [">1024"]
    return pd.DataFrame.from_records(
        [
            {
                "range": label,
                "count": int(counter.get(label, 0)),
                "ratio": (
                    round(counter.get(label, 0) / len(lengths), 6)
                    if lengths
                    else 0.0
                ),
            }
            for label in labels
        ]
    )


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    series = pd.Series(values, dtype="float64")
    return round(float(series.quantile(percentile / 100.0)), 3)


def _metadata_preview_row(
    item: dict[str, object],
    *,
    shard_name: str,
    item_index: int,
) -> dict[str, object]:
    cache_path = _cache_path_from_item(item)
    return {
        "shard": shard_name,
        "item_index": item_index,
        "clip_id": str(item.get("clip_id", "") or ""),
        "video_id": str(item.get("video_id", "") or ""),
        "bucket_frame_count": _safe_int(item.get("bucket_frame_count")),
        "bucket_resolution": _resolution_label(item.get("bucket_resolution")),
        "latent_shape": _latent_shape_label(item.get("latent_shape")),
        "caption_token_length": _safe_int(item.get("caption_token_length")),
        "caption_token_truncated": _safe_bool(item.get("caption_token_truncated")),
        "cache_file": str(cache_path) if cache_path is not None else "",
        "original_video_path": str(item.get("original_video_path", "") or ""),
    }


@st.cache_data(show_spinner="Scanning AutoModel metadata shards...")
def inspect_automodel_dataset(
    dataset_path_str: str,
    *,
    verify_cache_files: bool,
    sample_limit: int,
) -> dict[str, object]:
    dataset_path = Path(dataset_path_str).expanduser().resolve()
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.is_file():
        return {
            "ok": False,
            "errors": [f"missing metadata.json: {metadata_path}"],
            "dataset_path": str(dataset_path),
        }

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "errors": [f"invalid metadata.json: {exc}"],
            "dataset_path": str(dataset_path),
        }

    shard_names = metadata.get("shards")
    if not isinstance(shard_names, list):
        return {
            "ok": False,
            "errors": [f"metadata.json must contain a shards list: {metadata_path}"],
            "dataset_path": str(dataset_path),
        }

    errors: list[str] = []
    frame_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    temporal_spatial_counter: Counter[str] = Counter()
    latent_shape_counter: Counter[str] = Counter()
    source_resolution_counter: Counter[str] = Counter()
    caption_lengths: list[int] = []
    caption_truncated_count = 0
    caption_max_length_counter: Counter[str] = Counter()
    shard_rows: list[dict[str, object]] = []
    preview_rows: list[dict[str, object]] = []

    metadata_item_count = 0
    invalid_item_count = 0
    cache_file_count = 0
    cache_file_missing_count = 0
    cache_file_existing_count = 0
    cache_file_total_bytes = 0

    for shard_index, shard_name_value in enumerate(shard_names):
        shard_name = str(shard_name_value or "").strip()
        shard_path = dataset_path / shard_name
        shard_row = {
            "shard_index": shard_index,
            "shard": shard_name,
            "path": str(shard_path),
            "rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "exists": shard_path.is_file(),
        }
        if not shard_name:
            errors.append(f"empty shard name at index {shard_index}")
            shard_rows.append(shard_row)
            continue
        if not shard_path.is_file():
            errors.append(f"missing metadata shard: {shard_path}")
            shard_rows.append(shard_row)
            continue

        try:
            shard_items = json.loads(shard_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid shard JSON {shard_path}: {exc}")
            shard_rows.append(shard_row)
            continue
        if not isinstance(shard_items, list):
            errors.append(f"metadata shard must contain a list: {shard_path}")
            shard_rows.append(shard_row)
            continue

        shard_row["rows"] = len(shard_items)
        for item_index, item in enumerate(shard_items):
            metadata_item_count += 1
            if not isinstance(item, dict):
                invalid_item_count += 1
                shard_row["invalid_rows"] = int(shard_row["invalid_rows"]) + 1
                continue

            shard_row["valid_rows"] = int(shard_row["valid_rows"]) + 1
            frame_label = _frame_bucket_label(item.get("bucket_frame_count"))
            resolution_label = _resolution_label(item.get("bucket_resolution"))
            latent_shape_label = _latent_shape_label(item.get("latent_shape"))

            frame_counter[frame_label] += 1
            resolution_counter[resolution_label] += 1
            temporal_spatial_counter[f"{frame_label}/{resolution_label}"] += 1
            latent_shape_counter[latent_shape_label] += 1
            source_resolution_counter[
                _source_resolution_label(item.get("source_resolution"))
            ] += 1

            caption_length = _safe_int(item.get("caption_token_length"))
            if caption_length is not None and caption_length >= 0:
                caption_lengths.append(caption_length)
            if _safe_bool(item.get("caption_token_truncated")):
                caption_truncated_count += 1
            max_length = _safe_int(item.get("caption_token_max_length"))
            caption_max_length_counter[str(max_length or "unknown")] += 1

            cache_path = _cache_path_from_item(item)
            if cache_path is not None:
                cache_file_count += 1
                if verify_cache_files:
                    try:
                        if cache_path.is_file():
                            cache_file_existing_count += 1
                            cache_file_total_bytes += cache_path.stat().st_size
                        else:
                            cache_file_missing_count += 1
                    except OSError:
                        cache_file_missing_count += 1

            if len(preview_rows) < sample_limit:
                preview_rows.append(
                    _metadata_preview_row(
                        item,
                        shard_name=shard_name,
                        item_index=item_index,
                    )
                )

        shard_rows.append(shard_row)

    caption_count = len(caption_lengths)
    caption_mean = round(sum(caption_lengths) / caption_count, 3) if caption_count else 0.0
    return {
        "ok": True,
        "errors": errors,
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "dataset_format": "AutoModel Stage 5 .meta cache",
        "declared_shard_count": len(shard_names),
        "metadata_item_count": metadata_item_count,
        "invalid_item_count": invalid_item_count,
        "cache_file_count": cache_file_count,
        "verify_cache_files": verify_cache_files,
        "cache_file_existing_count": cache_file_existing_count,
        "cache_file_missing_count": cache_file_missing_count,
        "cache_file_total_bytes": cache_file_total_bytes,
        "frame_distribution": _counter_frame(
            frame_counter,
            total=metadata_item_count,
            name="frame_bucket",
        ),
        "resolution_distribution": _counter_frame(
            resolution_counter,
            total=metadata_item_count,
            name="resolution_bucket",
        ),
        "temporal_spatial_distribution": _counter_frame(
            temporal_spatial_counter,
            total=metadata_item_count,
            name="bucket",
        ),
        "latent_shape_distribution": _counter_frame(
            latent_shape_counter,
            total=metadata_item_count,
            name="latent_shape",
            top_k=50,
        ),
        "source_resolution_distribution": _counter_frame(
            source_resolution_counter,
            total=metadata_item_count,
            name="source_resolution",
        ),
        "caption_histogram": _caption_histogram(caption_lengths),
        "caption_stats": {
            "count": caption_count,
            "truncated_count": caption_truncated_count,
            "truncated_ratio": (
                round(caption_truncated_count / caption_count, 6)
                if caption_count
                else 0.0
            ),
            "mean": caption_mean,
            "p95": _percentile(caption_lengths, 95),
            "p99": _percentile(caption_lengths, 99),
            "max": max(caption_lengths, default=0),
        },
        "caption_max_length_distribution": _counter_frame(
            caption_max_length_counter,
            total=metadata_item_count,
            name="caption_token_max_length",
        ),
        "shards": pd.DataFrame.from_records(shard_rows),
        "preview": pd.DataFrame.from_records(preview_rows),
    }


def _count_value(value: object) -> str:
    number = _safe_int(value)
    return "-" if number is None else f"{number:,}"


def _ratio_value(value: object) -> str:
    number = _safe_float(value)
    return "-" if number is None else f"{number * 100:.2f}%"


def _summary_counts_frame(summary: dict[str, object], scan: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field in SUMMARY_COUNT_FIELDS:
        if field in summary:
            rows.append({"field": field, "value": summary.get(field)})
    rows.extend(
        [
            {
                "field": "scanned_metadata_items",
                "value": scan.get("metadata_item_count", 0),
            },
            {
                "field": "declared_metadata_shards",
                "value": scan.get("declared_shard_count", 0),
            },
            {
                "field": "invalid_metadata_items",
                "value": scan.get("invalid_item_count", 0),
            },
        ]
    )
    return pd.DataFrame.from_records(rows)


def _render_distribution(
    frame: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    chart_height: int = 300,
    label_angle: int = 0,
) -> None:
    if frame.empty:
        st.info("No distribution data.")
        return
    render_bar_chart(
        frame,
        category_col=label_col,
        value_col="count",
        title=title,
        height=chart_height,
        sort="-y",
        label_angle=label_angle,
    )
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_overview(summary: dict[str, object], scan: dict[str, object]) -> None:
    st.caption(f"Dataset format: `{scan['dataset_format']}`")
    st.caption(f"Dataset path: `{scan['dataset_path']}`")
    st.caption(f"Metadata path: `{scan['metadata_path']}`")

    metadata_count = int(scan.get("metadata_item_count", 0) or 0)
    ok_count = summary.get("ok_count", summary.get("packed_count", metadata_count))
    failed_count = summary.get("failed_count")

    cols = st.columns(6)
    cols[0].metric("source", _count_value(summary.get("source_count")))
    cols[1].metric("input", _count_value(summary.get("input_count")))
    cols[2].metric("ok / packed", _count_value(ok_count))
    cols[3].metric("failed", _count_value(failed_count))
    cols[4].metric("metadata items", f"{metadata_count:,}")
    cols[5].metric("metadata shards", _count_value(scan.get("declared_shard_count")))

    cache_cols = st.columns(4)
    cache_cols[0].metric("cache_file refs", _count_value(scan.get("cache_file_count")))
    if scan.get("verify_cache_files"):
        cache_cols[1].metric(
            ".meta existing",
            _count_value(scan.get("cache_file_existing_count")),
        )
        cache_cols[2].metric(
            ".meta missing",
            _count_value(scan.get("cache_file_missing_count")),
        )
        cache_cols[3].metric(
            ".meta bytes",
            format_bytes(scan.get("cache_file_total_bytes")),
        )
    else:
        cache_cols[1].metric(".meta existing", "not checked")
        cache_cols[2].metric(".meta missing", "not checked")
        cache_cols[3].metric(".meta bytes", "not checked")

    if "elapsed_sec" in summary:
        st.caption(f"Packaging elapsed: {format_elapsed_seconds(summary.get('elapsed_sec'))}")

    metadata_row_count = _safe_int(summary.get("metadata_row_count"))
    packed_count = _safe_int(summary.get("packed_count"))
    if metadata_row_count is not None and metadata_row_count != metadata_count:
        st.warning(
            f"summary metadata_row_count={metadata_row_count:,} differs from "
            f"scanned metadata items={metadata_count:,}."
        )
    if packed_count is not None and packed_count != metadata_count:
        st.warning(
            f"summary packed_count={packed_count:,} differs from scanned metadata "
            f"items={metadata_count:,}."
        )

    errors = scan.get("errors")
    if isinstance(errors, list) and errors:
        st.error("Metadata scan found issues.")
        render_dataframe_preview(
            pd.DataFrame({"issue": [str(item) for item in errors]}),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Run Counts")
    st.dataframe(_summary_counts_frame(summary, scan), width="stretch", hide_index=True)

    with st.expander("summary.json", expanded=False):
        render_json_preview(summary)


def _render_bucket_tab(scan: dict[str, object]) -> None:
    frame_distribution = scan["frame_distribution"]
    resolution_distribution = scan["resolution_distribution"]
    temporal_spatial_distribution = scan["temporal_spatial_distribution"]
    latent_shape_distribution = scan["latent_shape_distribution"]
    source_resolution_distribution = scan["source_resolution_distribution"]

    top_n = st.slider(
        "Frame x resolution Top N",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )
    top_temporal_spatial = temporal_spatial_distribution.head(top_n)

    cols = st.columns(2)
    with cols[0]:
        _render_distribution(
            frame_distribution,
            label_col="frame_bucket",
            title="Frame bucket distribution",
        )
    with cols[1]:
        _render_distribution(
            resolution_distribution,
            label_col="resolution_bucket",
            title="Resolution bucket distribution",
        )

    st.subheader("Frame x Resolution Buckets")
    _render_distribution(
        top_temporal_spatial,
        label_col="bucket",
        title="Temporal-spatial bucket distribution",
        chart_height=360,
        label_angle=-35,
    )

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Latent Shapes")
        _render_distribution(
            latent_shape_distribution,
            label_col="latent_shape",
            title="Latent shape distribution",
            label_angle=-35,
        )
    with cols[1]:
        st.subheader("Source Resolution")
        _render_distribution(
            source_resolution_distribution,
            label_col="source_resolution",
            title="Source resolution distribution",
            label_angle=-25,
        )


def _render_caption_tab(scan: dict[str, object]) -> None:
    stats = scan["caption_stats"]
    cols = st.columns(6)
    cols[0].metric("token rows", _count_value(stats.get("count")))
    cols[1].metric("truncated", _count_value(stats.get("truncated_count")))
    cols[2].metric("truncated rate", _ratio_value(stats.get("truncated_ratio")))
    cols[3].metric("mean", str(stats.get("mean", "-")))
    cols[4].metric("p95 / p99", f"{stats.get('p95', '-')}/{stats.get('p99', '-')}")
    cols[5].metric("max", str(stats.get("max", "-")))

    cols = st.columns(2)
    with cols[0]:
        _render_distribution(
            scan["caption_histogram"],
            label_col="range",
            title="Caption token length histogram",
            label_angle=-25,
        )
    with cols[1]:
        _render_distribution(
            scan["caption_max_length_distribution"],
            label_col="caption_token_max_length",
            title="Caption max length distribution",
        )


def _render_shards_and_samples(scan: dict[str, object]) -> None:
    st.subheader("Metadata Shards")
    shards = scan["shards"]
    if isinstance(shards, pd.DataFrame) and not shards.empty:
        render_dataframe_preview(shards, width="stretch", hide_index=True)
    else:
        st.info("No shard rows.")

    st.subheader("Metadata Preview")
    preview = scan["preview"]
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        render_dataframe_preview(preview, width="stretch", hide_index=True, height=420)
    else:
        st.info("No preview rows.")


def render_page() -> None:
    st.subheader(VIEWER_TITLE)

    with st.sidebar:
        st.subheader(VIEWER_TITLE)
        project_dir = st.text_input(
            "Project data root",
            value=str(DEFAULT_PROJECT_DIR),
            key="stage5_automodel_project_dir",
        )
        os.environ["DATA_DIR"] = str(Path(project_dir).expanduser().resolve())
        run_dirs = _list_dataset_run_dirs(project_dir)
        run_dir = st.selectbox(
            "AutoModel output run directory",
            run_dirs or [""],
            index=0,
            disabled=not run_dirs,
            key="stage5_automodel_run_dir",
        )
        dataset_override = st.text_input(
            "Dataset path override (optional)",
            value="",
            key="stage5_automodel_dataset_override",
            help="Path to the Stage 5 AutoModel output directory containing metadata.json.",
        )
        verify_cache_files = st.checkbox(
            "Check .meta files exist (do not read latents)",
            value=False,
            key="stage5_automodel_verify_cache_files",
        )
        sample_limit = st.number_input(
            "Metadata preview rows",
            min_value=0,
            max_value=5000,
            value=200,
            step=50,
            key="stage5_automodel_sample_limit",
        )
        refresh = st.button("Refresh cache", key="stage5_automodel_refresh")

    if not run_dirs and not dataset_override.strip():
        st.error("No AutoModel output directory was found. Check data/stage5_packaging/automodel, or use the dataset path override.")
        return

    dataset_path = _resolve_dataset_path(
        project_dir=project_dir,
        run_dir=run_dir,
        override=dataset_override,
    )
    st.caption(f"Current AutoModel dataset path: `{dataset_path}`")
    if refresh:
        inspect_automodel_dataset.clear()
        load_summary.clear()

    if not dataset_path.exists():
        st.error("Dataset path does not exist. Check the run directory or override the path manually.")
        return
    if not dataset_path.is_dir():
        st.error("Dataset path must be a directory containing metadata.json.")
        return

    summary = load_summary(str(dataset_path / "summary.json"))
    scan = inspect_automodel_dataset(
        str(dataset_path),
        verify_cache_files=verify_cache_files,
        sample_limit=int(sample_limit),
    )
    if not scan.get("ok"):
        st.error("AutoModel metadata scan failed.")
        render_json_preview(scan)
        return

    tabs = st.tabs(["Overview", "Buckets", "Captions", "Shards / Samples"])
    with tabs[0]:
        _render_overview(summary, scan)
    with tabs[1]:
        _render_bucket_tab(scan)
    with tabs[2]:
        _render_caption_tab(scan)
    with tabs[3]:
        _render_shards_and_samples(scan)


def main() -> None:
    st.set_page_config(page_title=VIEWER_TITLE, layout="wide")
    st.title(VIEWER_TITLE)
    render_page()


if __name__ == "__main__":
    main()
