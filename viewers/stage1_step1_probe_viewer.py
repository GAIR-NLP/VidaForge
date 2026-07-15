from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from .viewer_common import (
        add_fixed_range_chart,
        add_fps_chart,
        add_resolution_chart,
        add_value_counts_chart,
        apply_text_probe_filters,
        ensure_size_columns,
        format_elapsed_seconds,
        render_dataframe_preview,
        render_json_preview,
        summarize_total_size,
        render_metadata_selection,
        show_samples,
    )
except ImportError:
    from viewer_common import (
        add_fixed_range_chart,
        add_fps_chart,
        add_resolution_chart,
        add_value_counts_chart,
        apply_text_probe_filters,
        ensure_size_columns,
        format_elapsed_seconds,
        render_dataframe_preview,
        render_json_preview,
        summarize_total_size,
        render_metadata_selection,
        show_samples,
    )

STAGE_DIR = "stage1_ingestion"
STEP_DIR = "step1_probe"


def render_page() -> None:
    context = render_metadata_selection(
        stage_dir=STAGE_DIR,
        step_dir=STEP_DIR,
        title="Stage 1 Step 1: Probe",
        include_filter_scope=False,
        include_probe_filter=True,
        include_raw_dir=True,
    )
    if context is None:
        return

    frame = apply_text_probe_filters(
        context["frame"],
        keyword=context["keyword"],
        probe_filter=context["probe_filter"],
    )
    frame = ensure_size_columns(frame, size_bytes_col="filesize_bytes", size_mb_col="filesize_mb")
    summary = context["summary"]

    tabs = st.tabs(["Overview", "Sample Browser"])
    with tabs[0]:
        metric_cols = st.columns(6)
        metric_cols[0].metric("Rows", len(frame))
        metric_cols[1].metric("Probe Rows", summary.get("target_rows", "-"))
        metric_cols[2].metric("Probe OK", summary.get("probe_ok", "-"))
        metric_cols[3].metric("Probe Failed", summary.get("probe_failed", "-"))
        metric_cols[4].metric("Current Video Size", summarize_total_size(frame, size_bytes_col="filesize_bytes"))
        metric_cols[5].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

        preview_columns = [
            col
            for col in (
                "video_path",
                "filesize_mb",
                "duration_sec",
                "fps",
                "width",
                "height",
                "codec",
                "has_audio",
                "avg_frame_rate",
                "probe_ok",
                "probe_error",
            )
            if col in frame.columns
        ]
        render_dataframe_preview(frame[preview_columns], width="stretch", height=260)

        if summary:
            with st.expander("summary.json"):
                render_json_preview(summary)

        if "duration_sec" in frame.columns:
            duration_minutes = frame.copy()
            duration_minutes["duration_min"] = pd.to_numeric(
                duration_minutes["duration_sec"], errors="coerce"
            ) / 60.0
            add_fixed_range_chart(
                duration_minutes,
                "duration_min",
                "Duration Distribution (min)",
                bins=[0, 1, 10, 100, float("inf")],
                labels=["0~1 min", "1~10 min", "10~100 min", ">100 min"],
            )
        else:
            st.info("Duration Distribution (min): missing field `duration_sec`")

        add_resolution_chart(frame)
        add_value_counts_chart(frame, "codec", "Codec Distribution")
        add_fps_chart(frame)
        add_fixed_range_chart(
            frame,
            "filesize_mb",
            "File Size Distribution (MB)",
            bins=[0, 10, 100, 1024, 10 * 1024, float("inf")],
            labels=["0~10 MB", "10~100 MB", "100 MB~1 GB", "1~10 GB", ">10 GB"],
        )

    with tabs[1]:
        show_samples(
            frame,
            per_page=context["per_page"],
            columns=context["columns"],
            browse_order=context["browse_order"],
            widget_key_prefix=f"{STAGE_DIR}_{STEP_DIR}_samples",
            media_path_fields=("video_path", "raw_path"),
            raw_media_path_fields=("video_path", "raw_path"),
            show_local_paths=context["show_local_paths"],
            detail_fields=(
                "filesize_mb",
                "duration_sec",
                "fps",
                "width",
                "height",
                "codec",
                "has_audio",
                "avg_frame_rate",
                "probe_ok",
                "probe_error",
                "probe_elapsed_ms",
            ),
        )
