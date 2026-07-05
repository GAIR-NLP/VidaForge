"""Stage 4 Step 2 caption viewer."""

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


STAGE_DIR = "stage4_annotation"
STEP_DIR = "step2_caption"
VIEWER_TITLE = "Stage 4 Step 2 Caption"


@st.cache_data(show_spinner=False)
def _load_caption_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_caption_row_index",
    )


def _render_caption_text(row: dict[str, object]) -> None:
    caption_json = parse_json_object(row.get("caption_json"))
    levels = (
        ("level_0", row.get("caption_level_0") or caption_json.get("level_0")),
        ("level_1", row.get("caption_level_1") or caption_json.get("level_1")),
        ("level_2", row.get("caption_level_2") or caption_json.get("level_2")),
        ("level_3", row.get("caption_level_3") or caption_json.get("level_3")),
    )
    for level, text in levels:
        st.markdown(f"**{level}**")
        st.write(str(text or "").strip() or "-")


def _render_caption_clip(
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
    cols[0].metric("caption_ok", coerce_int(row.get("caption_ok")))
    cols[1].metric("mode", row.get("caption_mode", "-"))
    cols[2].metric("frames", len(frame_paths))
    cols[3].metric("audio_ok", coerce_int(row.get("audio_ok")))

    if show_video:
        render_clip_video(row, missing_label="Current caption row is missing `clip_path`.")

    render_frame_strip(frame_paths, timestamps=timestamps)
    render_audio_player(row)

    _render_caption_text(row)

    with st.expander("caption json", expanded=False):
        st.json(parse_json_object(row.get("caption_json")))
    prompt_json = parse_json_object(row.get("caption_prompt_json"))
    if prompt_json:
        with st.expander("caption prompt", expanded=False):
            st.markdown("**system_prompt**")
            st.code(str(prompt_json.get("system_prompt") or ""), language="text")
            st.markdown("**user_prompt**")
            st.code(str(prompt_json.get("user_prompt") or ""), language="text")
    with st.expander("row meta", expanded=False):
        st.json(
            {
                key: row.get(key)
                for key in (
                    "clip_id",
                    "video_id",
                    "clip_path",
                    "duration_sec",
                    "width",
                    "height",
                    "fps",
                    "caption_error",
                    "input_run_id",
                    "run_id",
                )
                if key in row
            }
        )


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
    if selection.get("refresh"):
        load_summary_light.clear()
        load_rows_by_indices.clear()
        _load_caption_rows.clear()
    summary = load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = resolve_context_shards(metadata_path, summary)
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No caption index records were loaded.")
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
        )

    metric_cols = st.columns(6)
    metric_cols[0].metric("caption rows", summary.get("caption_rows", total_rows))
    metric_cols[1].metric("Caption OK", summary.get("caption_ok", "-"))
    metric_cols[2].metric("Caption Failed", summary.get("caption_failed", "-"))
    metric_cols[3].metric("mode", summary.get("mode", "-"))
    metric_cols[4].metric("Clip Pages", total_pages)
    metric_cols[5].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    summary_preview = {
        key: value
        for key, value in summary.items()
        if key not in LARGE_SUMMARY_SKIP_KEYS
    }
    with st.expander("summary preview", expanded=False):
        st.json(summary_preview)

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
    page_rows = _load_caption_rows(shards, row_indices=row_indices)

    if not page_rows:
        st.warning("No caption rows were loaded on the current page.")
        return
    for offset, row in enumerate(page_rows, start=0):
        _render_caption_clip(
            row,
            clip_position=page_start + offset + 1,
            show_video=show_video,
        )
        st.divider()
