"""Stage 3 Step 1 context viewer."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .context_viewer_common import (
        LARGE_SUMMARY_SKIP_KEYS,
        coerce_int,
        context_row_detail,
        format_seconds,
        frame_timestamps,
        load_rows_by_indices,
        load_summary_light,
        page_indices,
        parse_json_object,
        render_audio_player,
        render_clip_video,
        render_frame_strip,
        resolve_context_shards,
        resolve_frame_paths,
        total_rows_from_shards,
    )
    from .viewer_common import (
        format_elapsed_seconds,
        render_metadata_selection,
        resolve_summary_path,
    )
except ImportError:
    from context_viewer_common import (
        LARGE_SUMMARY_SKIP_KEYS,
        coerce_int,
        context_row_detail,
        format_seconds,
        frame_timestamps,
        load_rows_by_indices,
        load_summary_light,
        page_indices,
        parse_json_object,
        render_audio_player,
        render_clip_video,
        render_frame_strip,
        resolve_context_shards,
        resolve_frame_paths,
        total_rows_from_shards,
    )
    from viewer_common import (
        format_elapsed_seconds,
        render_metadata_selection,
        resolve_summary_path,
    )

from vidaforge.common import replace_path_part

STAGE_DIR = "stage3_selection"
STEP_DIR = "step1_context"
VIEWER_TITLE = "Stage 3 Step 1 Context"


@st.cache_data(show_spinner=False)
def _load_context_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_context_row_index",
    )


def _render_context_clip(
    row: dict[str, object],
    *,
    clip_position: int,
    show_video: bool,
) -> None:
    frame_json = parse_json_object(row.get("frame_json"))
    frame_paths = resolve_frame_paths(frame_json=frame_json)
    timestamps = frame_timestamps(frame_json)

    st.markdown(f"#### Clip {clip_position}")
    cols = st.columns(4)
    cols[0].metric("duration", format_seconds(row.get("duration_sec")))
    cols[1].metric("frames", len(frame_paths))
    cols[2].metric("frame_ok", coerce_int(row.get("frame_ok")))
    cols[3].metric("audio_ok", coerce_int(row.get("audio_ok")))

    audio_json = parse_json_object(row.get("audio_json"))
    audio_cols = st.columns(3)
    audio_cols[0].metric("audio_format", audio_json.get("audio_format", "-"))
    audio_cols[1].metric("sampled_fps", frame_json.get("sampled_fps", "-"))
    audio_paths = str(audio_json.get("audio_paths") or "").strip()
    if audio_paths:
        audio_cols[2].caption(audio_paths)
    render_audio_player(row)

    if show_video:
        render_clip_video(row, missing_label="Current context row is missing `clip_path`.")
    else:
        clip_path = str(row.get("clip_path") or "").strip()
        if clip_path:
            st.caption(f"video unloaded: {clip_path}")

    st.caption(
        f"{len(frame_paths)} frames | "
        f"method={frame_json.get('sampling_method', '-')} | "
        f"clip_id={row.get('clip_id', '-')}"
    )
    render_frame_strip(frame_paths, timestamps=timestamps)

    with st.expander("context meta", expanded=False):
        st.json(context_row_detail(row, frame_json=frame_json))


def render_page() -> None:
    selection = render_metadata_selection(
        stage_dir=STAGE_DIR,
        step_dir=STEP_DIR,
        title=VIEWER_TITLE,
        include_filter_scope=False,
        include_probe_filter=False,
        include_browse_layout_controls=False,
        load_frame=False,
        load_summary_data=False,
    )
    if selection is None:
        return

    metadata_path = Path(selection["metadata_path"])
    output_data_path = replace_path_part(metadata_path, old="meta", new="data")
    if selection.get("refresh"):
        load_summary_light.clear()
        load_rows_by_indices.clear()
        _load_context_rows.clear()
    summary = load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = resolve_context_shards(metadata_path, summary)
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No context index records were loaded.")
        if summary:
            with st.expander("summary.json", expanded=False):
                st.json(summary)
        return

    widget_prefix = f"{STAGE_DIR}_{STEP_DIR}"
    page_number_key = f"{widget_prefix}_clip_page_number"
    page_delta_key = f"{widget_prefix}_clip_page_delta"
    per_page_key = f"{widget_prefix}_clips_per_page"
    show_video_key = f"{widget_prefix}_show_video"
    if page_number_key not in st.session_state:
        st.session_state[page_number_key] = 1
    if page_delta_key not in st.session_state:
        st.session_state[page_delta_key] = 0

    with selection["extra_sidebar_container"]:
        per_page = int(
            st.selectbox(
                "Clips per page",
                (5, 10, 20, 50),
                index=1,
                key=per_page_key,
            )
        )

    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    pending_delta = int(st.session_state.get(page_delta_key) or 0)
    if pending_delta:
        st.session_state[page_number_key] = max(
            1,
            min(int(st.session_state[page_number_key]) + pending_delta, total_pages),
        )
        st.session_state[page_delta_key] = 0
    st.session_state[page_number_key] = max(
        1,
        min(int(st.session_state[page_number_key]), total_pages),
    )

    with selection["extra_sidebar_container"]:
        page_number = int(
            st.number_input(
                "clip Page",
                min_value=1,
                max_value=total_pages,
                step=1,
                key=page_number_key,
            )
        )
        show_video = st.toggle(
            "Load video player",
            value=False,
            key=show_video_key,
            help="Disabled by default to avoid loading many videos at once; enable it when you need to inspect source video.",
        )

    if selection["keyword"]:
        st.info("The current clip paging mode does not run full keyword filtering. For filtering, create a smaller run or add an index table later.")

    metric_cols = st.columns(7)
    metric_cols[0].metric("output rows", summary.get("output_count", total_rows))
    metric_cols[1].metric("Context OK", summary.get("ok_count", "-"))
    metric_cols[2].metric("Context Failed", summary.get("failed_count", "-"))
    metric_cols[3].metric("input rows", summary.get("input_count", "-"))
    metric_cols[4].metric("resumed", summary.get("resumed_count", "-"))
    metric_cols[5].metric("Clip Pages", total_pages)
    metric_cols[6].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    summary_preview = {
        key: value
        for key, value in summary.items()
        if key not in LARGE_SUMMARY_SKIP_KEYS
    }
    with st.expander("summary preview", expanded=False):
        st.json(summary_preview)
    if st.checkbox(
        "Show full summary.json (may be slow)",
        value=False,
        key=f"{widget_prefix}_show_full_summary",
    ):
        st.json(summary)

    page_start = (page_number - 1) * per_page
    page_end = min(total_rows, page_start + per_page)
    st.caption(f"clips {page_start + 1}-{page_end} / {total_rows}")
    nav_cols = st.columns([1, 1, 3])
    with nav_cols[0]:
        if st.button(
            "Previous page",
            width="stretch",
            disabled=page_number <= 1,
            key=f"{widget_prefix}_prev_clip_page",
        ):
            st.session_state[page_delta_key] = -1
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "Next page",
            width="stretch",
            disabled=page_number >= total_pages,
            key=f"{widget_prefix}_next_clip_page",
        ):
            st.session_state[page_delta_key] = 1
            st.rerun()

    row_indices = page_indices(total_rows=total_rows, page_number=page_number, per_page=per_page)
    page_rows = _load_context_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Context", "Current Page Details"])
    with tabs[0]:
        if not page_rows:
            st.warning("No context rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            clip_position = page_start + offset + 1
            _render_context_clip(
                row,
                clip_position=clip_position,
                show_video=show_video,
            )
            st.divider()

    with tabs[1]:
        st.json(
            {
                "page": page_number,
                "per_page": per_page,
                "row_indices": list(row_indices),
                "loaded_context_rows": len(page_rows),
                "clips": [
                    context_row_detail(row, frame_json=parse_json_object(row.get("frame_json")))
                    for row in page_rows
                ],
            }
        )
