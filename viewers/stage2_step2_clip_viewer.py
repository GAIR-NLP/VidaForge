"""Stage 2 Step 2 clip viewer."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .viewer_common import (
        add_fixed_range_chart,
        add_value_counts_chart,
        apply_text_probe_filters,
        format_bytes,
        format_elapsed_seconds,
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
    )
except ImportError:
    from viewer_common import (
        add_fixed_range_chart,
        add_value_counts_chart,
        apply_text_probe_filters,
        format_bytes,
        format_elapsed_seconds,
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
    )

try:
    from .coerce import coerce_float, coerce_int
except ImportError:
    from coerce import coerce_float, coerce_int
from vidaforge.common import join_data_dir
from vidaforge.index import resolve_parquet_paths

STAGE_DIR = "stage2_segmentation"
STEP_DIR = "step2_clip"


def _resolve_clip_shards(metadata_path: Path, summary: dict[str, object]) -> tuple[tuple[str, int], ...]:
    metadata_root = metadata_path.expanduser().resolve()
    shards = summary.get("shards")
    resolved: list[tuple[str, int]] = []
    if isinstance(shards, list):
        for item in shards:
            if not isinstance(item, dict):
                continue
            shard_name = str(item.get("path") or "").strip()
            if not shard_name:
                continue
            shard_path = metadata_root / shard_name
            if not shard_path.exists() or not shard_path.is_file():
                continue
            rows = coerce_int(item.get("rows"), default=-1)
            if rows < 0:
                rows = pq.ParquetFile(shard_path).metadata.num_rows
            resolved.append((str(shard_path), rows))
    if resolved:
        return tuple(resolved)

    for path in resolve_parquet_paths(metadata_root, unit="clip"):
        resolved.append((str(path), pq.ParquetFile(path).metadata.num_rows))
    return tuple(resolved)


def _total_rows_from_shards(shards: tuple[tuple[str, int], ...]) -> int:
    return sum(rows for _, rows in shards)


def _build_video_index(
    shards: tuple[tuple[str, int], ...],
    summary: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clip_start = 0

    current_video_id = ""
    current_row: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current_row
        if current_row is not None:
            rows.append(current_row)
        current_row = None

    for shard_path, _ in shards:
        parquet_file = pq.ParquetFile(shard_path)
        available_columns = set(parquet_file.schema.names)
        columns = [
            column
            for column in (
                "video_id",
                "video_path",
                "clip_ok",
                "clip_error",
                "input_run_id",
                "run_id",
            )
            if column in available_columns
        ]
        frame = pq.read_table(shard_path, columns=columns).to_pandas()
        for _, item in frame.iterrows():
            video_id = str(item.get("video_id") or "")
            if current_row is None or video_id != current_video_id:
                flush_current()
                current_video_id = video_id
                current_row = {
                    "_video_index": len(rows),
                    "_clip_start_index": clip_start,
                    "clip_count": 0,
                    "video_id": video_id,
                    "video_path": str(item.get("video_path") or ""),
                    "clip_ok": 0,
                    "clip_failed": 0,
                    "video_clip_error": "",
                    "input_run_id": str(item.get("input_run_id") or summary.get("input_run_id") or ""),
                    "run_id": str(item.get("run_id") or summary.get("run_id") or ""),
                }

            clip_ok = coerce_int(item.get("clip_ok"))
            current_row["clip_count"] = coerce_int(current_row.get("clip_count")) + 1
            current_row["clip_ok"] = coerce_int(current_row.get("clip_ok")) + int(clip_ok == 1)
            current_row["clip_failed"] = (
                coerce_int(current_row.get("clip_failed")) + int(clip_ok != 1)
            )
            clip_error = str(item.get("clip_error") or "")
            if clip_error and not str(current_row.get("video_clip_error") or ""):
                current_row["video_clip_error"] = clip_error
            clip_start += 1

    flush_current()
    for video_index, row in enumerate(rows):
        row["_video_index"] = video_index
    return pd.DataFrame.from_records(rows)


@st.cache_data(show_spinner=False)
def _load_clip_page(
    shards: tuple[tuple[str, int], ...],
    *,
    start_index: int,
    page_size: int,
) -> pd.DataFrame:
    if page_size <= 0:
        return pd.DataFrame()

    page_end = start_index + page_size
    shard_start = 0
    frames: list[pd.DataFrame] = []
    for shard_path, shard_rows in shards:
        shard_end = shard_start + shard_rows
        if shard_end <= start_index:
            shard_start = shard_end
            continue
        if shard_start >= page_end:
            break

        local_start = max(0, start_index - shard_start)
        local_end = min(shard_rows, page_end - shard_start)
        table = pq.read_table(shard_path)
        frame = table.to_pandas()
        frame = frame.iloc[local_start:local_end].copy()
        frame.insert(0, "_clip_index", range(shard_start + local_start, shard_start + local_end))
        frames.append(frame)
        shard_start = shard_end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prepare_clip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for column in (
        "_clip_index",
        "start_sec",
        "end_sec",
        "duration_sec",
        "detect_start_sec",
        "detect_end_sec",
        "detect_duration_sec",
        "clip_index",
        "split_index",
        "ffmpeg_elapsed_sec",
        "filesize_bytes",
        "width",
        "height",
        "fps",
        "clip_ok",
    ):
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    if "detectors" in enriched.columns:
        enriched["_detectors_label"] = [
            _format_detectors(row.get("detectors"))
            for _, row in enriched.iterrows()
        ]
    return enriched.reset_index(drop=True)


def _format_seconds(value: object) -> str:
    number = coerce_float(value)
    return "-" if number is None else f"{number:.3f}s"


def _format_bytes(value: object) -> str:
    number = coerce_float(value)
    return "-" if number is None else format_bytes(number)


def _coerce_name_list(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _format_detectors(value: object) -> str:
    detector_names = _coerce_name_list(value)
    return "+".join(detector_names) if detector_names else "unknown"


def _build_clip_detail(row: pd.Series) -> dict[str, object]:
    fields = (
        "clip_id",
        "video_id",
        "clip_path",
        "video_path",
        "raw_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "detect_start_sec",
        "detect_end_sec",
        "detect_duration_sec",
        "clip_index",
        "split_index",
        "detectors",
        "ffmpeg_elapsed_sec",
        "filesize_bytes",
        "width",
        "height",
        "fps",
        "input_run_id",
        "run_id",
        "clip_ok",
        "clip_error",
    )
    detail: dict[str, object] = {}
    for field in fields:
        if field not in row:
            continue
        value = row.get(field)
        if field == "detectors":
            detail[field] = _coerce_name_list(value)
        elif pd.notna(value):
            detail[field] = value
    return detail


def _render_clip_video(row: pd.Series) -> None:
    clip_path = str(row.get("clip_path") or "").strip()
    if not clip_path:
        st.warning("Current clip is missing `clip_path`.")
        return

    path = join_data_dir(clip_path)
    if not path.exists() or not path.is_file():
        st.error(f"clip file does not exist: {clip_path}")
        return

    try:
        st.video(str(path), autoplay=False)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"clip file cannot be opened and was skipped: {clip_path}")
        st.caption(str(exc))
        return
    st.caption(clip_path)


def _render_clip_grid(frame: pd.DataFrame, *, columns: int) -> None:
    columns = max(1, columns)
    for start in range(0, len(frame), columns):
        block = frame.iloc[start : start + columns]
        cols = st.columns(len(block))
        for col, (_, row) in zip(cols, block.iterrows()):
            with col:
                st.caption(
                    f"clip {coerce_int(row.get('_clip_index')) + 1} | "
                    f"{_format_seconds(row.get('start_sec'))} -> {_format_seconds(row.get('end_sec'))} | "
                    f"{_format_seconds(row.get('duration_sec'))}"
                )
                _render_clip_video(row)
                if coerce_int(row.get("clip_ok")) != 1:
                    st.error(str(row.get("clip_error") or "clip failed"))
                with st.expander("meta", expanded=False):
                    st.json(_build_clip_detail(row))


def render_page() -> None:
    context = render_metadata_selection(
        stage_dir=STAGE_DIR,
        step_dir=STEP_DIR,
        title="Stage 2 Step 2 Clip",
        include_filter_scope=False,
        include_probe_filter=False,
        include_browse_layout_controls=False,
        load_frame=False,
    )
    if context is None:
        return

    metadata_path = Path(context["metadata_path"])
    summary = context["summary"]
    shards = _resolve_clip_shards(metadata_path, summary)
    total_rows = _total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No clip index records were loaded.")
        if summary:
            with st.expander("summary.json", expanded=False):
                render_json_preview(summary)
        return

    video_index = _build_video_index(shards, summary)
    if video_index.empty:
        st.warning("Current clip metadata cannot be paged by video.")
        with st.expander("summary.json", expanded=False):
            render_json_preview(summary)
        return

    video_index = apply_text_probe_filters(
        video_index,
        keyword=context["keyword"],
        probe_filter=context["probe_filter"],
        extra_text_cols=(
            "video_id",
            "video_clip_error",
            "host",
            "input_run_id",
            "run_id",
        ),
    )
    if video_index.empty:
        st.warning("No videos match the current filters.")
        return
    video_index = video_index.reset_index(drop=True)

    widget_prefix = f"{STAGE_DIR}_{STEP_DIR}"
    video_number_key = f"{widget_prefix}_video_number"
    video_delta_key = f"{widget_prefix}_video_delta"
    clip_columns_key = f"{widget_prefix}_clip_columns"
    if video_number_key not in st.session_state:
        st.session_state[video_number_key] = 1
    if video_delta_key not in st.session_state:
        st.session_state[video_delta_key] = 0
    if clip_columns_key not in st.session_state:
        st.session_state[clip_columns_key] = 2

    total_videos = len(video_index)
    pending_delta = int(st.session_state.get(video_delta_key) or 0)
    if pending_delta:
        st.session_state[video_number_key] = max(
            1,
            min(int(st.session_state[video_number_key]) + pending_delta, total_videos),
        )
        st.session_state[video_delta_key] = 0
    st.session_state[video_number_key] = max(
        1,
        min(int(st.session_state[video_number_key]), total_videos),
    )

    with context["extra_sidebar_container"]:
        clip_columns = int(
            st.selectbox(
                "Clips per row",
                (1, 2, 3, 4),
                key=clip_columns_key,
            )
        )
        video_number = int(
            st.number_input(
                "Video Page",
                min_value=1,
                max_value=total_videos,
                step=1,
                key=video_number_key,
            )
        )

    current_video_position = video_number - 1
    selected_video = video_index.iloc[current_video_position]
    clip_start_index = coerce_int(selected_video.get("_clip_start_index"))
    clip_count = coerce_int(selected_video.get("clip_count"))
    video_clips = _prepare_clip_frame(
        _load_clip_page(
            shards,
            start_index=clip_start_index,
            page_size=clip_count,
        )
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total Clips", summary.get("output_count", total_rows))
    metric_cols[1].metric("Clip OK", summary.get("ok_count", "-"))
    metric_cols[2].metric("Clip Failed", summary.get("failed_count", "-"))
    metric_cols[3].metric("Input Videos", summary.get("input_count", "-"))
    metric_cols[4].metric("Resumed Videos", summary.get("resumed_count", "-"))
    metric_cols[5].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    with st.expander("summary.json", expanded=False):
        render_json_preview(summary)

    tabs = st.tabs(["Video Clips", "Current Video Distribution", "Clip Details"])
    with tabs[0]:
        st.caption(f"Video {current_video_position + 1}/{total_videos}, current video clips: {clip_count}")
        nav_cols = st.columns([1, 1, 2])
        with nav_cols[0]:
            if st.button(
                "Previous video",
                width="stretch",
                disabled=current_video_position <= 0,
                key=f"{widget_prefix}_prev_video",
            ):
                st.session_state[video_delta_key] = -1
                st.rerun()
        with nav_cols[1]:
            if st.button(
                "Next video",
                width="stretch",
                disabled=current_video_position >= total_videos - 1,
                key=f"{widget_prefix}_next_video",
            ):
                st.session_state[video_delta_key] = 1
                st.rerun()

        summary_cols = st.columns(5)
        summary_cols[0].metric("Current Video Clips", clip_count)
        summary_cols[1].metric("Clip OK", coerce_int(selected_video.get("clip_ok")))
        summary_cols[2].metric("Clip Failed", coerce_int(selected_video.get("clip_failed")))
        summary_cols[3].metric("Status", "ok" if coerce_int(selected_video.get("clip_failed")) == 0 else "failed")
        summary_cols[4].metric("Video Index", current_video_position + 1)
        st.caption(str(selected_video.get("video_id") or ""))
        st.caption(str(selected_video.get("video_path") or ""))
        if str(selected_video.get("video_clip_error") or "").strip():
            st.error(str(selected_video.get("video_clip_error")))

        if video_clips.empty:
            st.warning("Current video has no clips to display.")
        else:
            _render_clip_grid(video_clips, columns=clip_columns)

    with tabs[1]:
        if video_clips.empty:
            st.warning("Current video has no clips.")
        else:
            status_counts = (
                pd.to_numeric(video_clips["clip_ok"], errors="coerce")
                .fillna(0)
                .astype(int)
                .map({1: "ok", 0: "failed"})
                .value_counts()
                .rename_axis("status")
                .reset_index(name="count")
            )
            render_bar_chart(
                status_counts,
                category_col="status",
                value_col="count",
                title="Clip Status Distribution (Current Video)",
            )
            add_value_counts_chart(video_clips, "_detectors_label", "Detector Distribution (Current Video)")
            add_fixed_range_chart(
                video_clips,
                "duration_sec",
                "Clip Duration Distribution (Current Video)",
                bins=[0, 2, 5, 10, 15, float("inf")],
                labels=["0~2s", "2~5s", "5~10s", "10~15s", ">15s"],
            )
            add_fixed_range_chart(
                video_clips,
                "filesize_bytes",
                "Clip File Size Distribution (Current Video)",
                bins=[
                    0,
                    1 * 1024 * 1024,
                    5 * 1024 * 1024,
                    20 * 1024 * 1024,
                    100 * 1024 * 1024,
                    float("inf"),
                ],
                labels=["0~1 MB", "1~5 MB", "5~20 MB", "20~100 MB", ">100 MB"],
            )

    with tabs[2]:
        if video_clips.empty:
            st.warning("Current video has no clips.")
            return
        display_cols = [
            col
            for col in (
                "_clip_index",
                "clip_id",
                "clip_ok",
                "duration_sec",
                "start_sec",
                "end_sec",
                "detect_start_sec",
                "detect_end_sec",
                "split_index",
                "filesize_bytes",
                "video_id",
                "clip_path",
                "video_path",
                "detectors",
                "clip_error",
            )
            if col in video_clips.columns
        ]
        table = video_clips[display_cols].copy()
        if "_clip_index" in table.columns:
            table["_clip_index"] = pd.to_numeric(table["_clip_index"], errors="coerce").astype("Int64") + 1
        if "filesize_bytes" in table.columns:
            table["filesize"] = table["filesize_bytes"].map(_format_bytes)
            table = table.drop(columns=["filesize_bytes"])
        render_dataframe_preview(table, width="stretch", height=460)
