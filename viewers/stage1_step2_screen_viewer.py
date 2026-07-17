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
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
        show_samples,
        summarize_total_size,
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
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
        show_samples,
        summarize_total_size,
    )

STAGE_DIR = "stage1_ingestion"
STEP_DIR = "step2_screen"


def render_page() -> None:
    context = render_metadata_selection(
        stage_dir=STAGE_DIR,
        step_dir=STEP_DIR,
        title="Stage 1 Step 2: Screen",
        include_filter_scope=True,
        include_probe_filter=False,
        include_raw_dir=True,
    )
    if context is None:
        return

    frame = apply_text_probe_filters(
        context["frame"],
        keyword=context["keyword"],
        probe_filter=context["probe_filter"],
        extra_text_cols=("screen_reject_reason", "screen_json"),
    )
    frame = ensure_size_columns(frame, size_bytes_col="filesize_bytes", size_mb_col="filesize_mb")

    filter_scope = context["filter_scope"]
    screen_reject_reason = "all"
    if "screen_reject_reason" in frame.columns:
        available_screen_reject_reasons = sorted(
            {
                value
                for value in frame["screen_reject_reason"].dropna().astype(str).tolist()
                if value
            }
        )
        with context["extra_sidebar_container"]:
            selected_screen_reject_reason = st.selectbox(
                "Reject Reason",
                ("all", *available_screen_reject_reasons),
                index=0,
                key=f"{STAGE_DIR}_{STEP_DIR}_screen_reject_reason",
                disabled=filter_scope != "reject",
                help="Set Filter Result to reject first, then filter by a specific reject reason.",
            )
        if filter_scope == "reject":
            screen_reject_reason = selected_screen_reject_reason

    if filter_scope != "all" and "screen_pass" in frame.columns:
        screen_pass = pd.to_numeric(frame["screen_pass"], errors="coerce")
        if filter_scope == "keep":
            frame = frame[screen_pass == 1]
        else:
            frame = frame[screen_pass != 1]
        frame = frame.reset_index(drop=True)

    if screen_reject_reason != "all" and "screen_reject_reason" in frame.columns:
        frame = frame[
            frame["screen_reject_reason"].fillna("").astype(str)
            == screen_reject_reason
        ]
        frame = frame.reset_index(drop=True)

    summary = context["summary"]

    tabs = st.tabs(["Overview", "Sample Browser"])
    with tabs[0]:
        current_total = len(frame)
        current_kept = "-"
        current_rejected = "-"
        current_retention_rate = "-"
        if "screen_pass" in frame.columns:
            keep_series = pd.to_numeric(frame["screen_pass"], errors="coerce")
            kept_count = int((keep_series == 1).sum())
            rejected_count = int(current_total - kept_count)
            current_kept = kept_count
            current_rejected = rejected_count
            if current_total > 0:
                current_retention_rate = f"{(kept_count / current_total):.2%}"

        metric_cols = st.columns(6)
        metric_cols[0].metric("Current Rows", current_total)
        metric_cols[1].metric("Current Kept", current_kept)
        metric_cols[2].metric("Current Rejected", current_rejected)
        metric_cols[3].metric("Current Retention", current_retention_rate)
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
                "screen_pass",
                "screen_reject_reason",
                "screen_ok",
                "screen_error",
                "probe_ok",
                "probe_error",
            )
            if col in frame.columns
        ]
        render_dataframe_preview(frame[preview_columns], width="stretch", height=260)

        if "screen_reject_reason" in frame.columns:
            reject_chart = (
                frame["screen_reject_reason"]
                .fillna("")
                .astype(str)
                .replace("", "unknown")
                .value_counts()
                .rename_axis("reason")
                .reset_index(name="count")
            )
            reject_chart = reject_chart[reject_chart["reason"] != "unknown"]
        else:
            reject_chart = pd.DataFrame(columns=["reason", "count"])

        if not reject_chart.empty:
            render_bar_chart(
                reject_chart,
                category_col="reason",
                value_col="count",
                title="Reject Reason Distribution (Current View)",
            )

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
        add_value_counts_chart(frame, "screen_reject_reason", "Filter Reason Distribution", top_k=20)

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
                "avg_frame_rate",
                "screen_ok",
                "screen_error",
                "screen_pass",
                "screen_reject_reason",
                "screen_json",
                "probe_ok",
                "probe_error",
                "probe_elapsed_ms",
            ),
        )
