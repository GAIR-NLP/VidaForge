"""Stage 4 Step 3 tag viewer."""

from __future__ import annotations

import html
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
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

from vidaforge.annotation.tag import schema as tag_schema


STAGE_DIR = "stage4_annotation"
STEP_DIR = "step3_tag"
VIEWER_TITLE = "Stage 4 Step 3 Tag"
TAG_OVERVIEW_COLUMNS = (
    "clip_id",
    "tag_ok",
    "tag_json",
    "tag_domain",
    "tag_scene",
    "tag_subjects",
    "tag_actions",
    "tag_style",
    "tag_text",
    "tag_watermark",
)
SINGLE_VALUE_TAG_LABELS = {
    "domain": tuple(tag_schema.TAG_DOMAIN_LABELS),
    "scene": tuple(tag_schema.TAG_SCENE_LABELS),
    "style": tuple(tag_schema.TAG_STYLE_LABELS),
    "text": tuple(tag_schema.TAG_TEXT_LABELS),
    "watermark": tuple(tag_schema.TAG_WATERMARK_LABELS),
}
MULTI_VALUE_TAG_LABELS = {
    "subjects": tuple(tag_schema.TAG_SUBJECT_LABELS),
    "actions": tuple(tag_schema.TAG_ACTION_LABELS),
}
TAG_FIELD_ORDER = (
    "domain",
    "scene",
    "subjects",
    "actions",
    "style",
    "text",
    "watermark",
)


@st.cache_data(show_spinner=False)
def _load_tag_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_tag_row_index",
    )


def _tag_json_from_row(row: dict[str, object]) -> dict[str, Any]:
    tag_json = parse_json_object(row.get("tag_json"))
    if tag_json:
        return tag_json

    parsed: dict[str, Any] = {}
    for field in TAG_FIELD_ORDER:
        value = row.get(f"tag_{field}")
        if _is_missing(value):
            continue
        parsed[field] = _normalize_tag_value(value)
    return parsed


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _normalize_tag_value(value: object) -> object:
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted if str(item or "").strip()]
        return converted
    return value


def _selected_labels(value: object) -> set[str]:
    normalized = _normalize_tag_value(value)
    if isinstance(normalized, list):
        return {str(item).strip() for item in normalized if str(item or "").strip()}
    text = str(normalized or "").strip()
    return {text} if text else set()


def _label_badges(labels: tuple[str, ...], selected: set[str]) -> str:
    badges = []
    for label in labels:
        is_selected = label in selected
        color = "#ffffff" if is_selected else "#667085"
        background = "#175cd3" if is_selected else "#f2f4f7"
        border = "#175cd3" if is_selected else "#d0d5dd"
        weight = "700" if is_selected else "500"
        badges.append(
            "<span style="
            f"'display:inline-block;margin:0 4px 5px 0;padding:3px 8px;"
            f"border-radius:999px;border:1px solid {border};"
            f"background:{background};color:{color};font-weight:{weight};"
            "font-size:12px;line-height:1.4;white-space:nowrap;'>"
            f"{html.escape(label)}</span>"
        )
    return "".join(badges)


def _render_schema_label_row(title: str, labels: tuple[str, ...], selected: object) -> None:
    st.markdown(
        "<div style='margin:0 0 8px 0;'>"
        f"<div style='font-size:12px;font-weight:700;color:#344054;margin-bottom:3px;'>"
        f"{html.escape(title)}</div>"
        f"<div>{_label_badges(labels, _selected_labels(selected))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _empty_tag_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for field, labels in SINGLE_VALUE_TAG_LABELS.items():
        counts[field] = {label: 0 for label in labels}
    for field, labels in MULTI_VALUE_TAG_LABELS.items():
        counts[field] = {label: 0 for label in labels}
    return counts


def _add_label_count(
    counts: dict[str, dict[str, int]],
    *,
    field: str,
    label: object,
) -> None:
    label_text = str(label or "").strip()
    if not label_text:
        return
    field_counts = counts.setdefault(field, {})
    field_counts[label_text] = field_counts.get(label_text, 0) + 1


def _accumulate_tag_counts(
    counts: dict[str, dict[str, int]],
    *,
    tag_json: dict[str, Any],
) -> None:
    for field in SINGLE_VALUE_TAG_LABELS:
        _add_label_count(counts, field=field, label=tag_json.get(field))

    for field in MULTI_VALUE_TAG_LABELS:
        value = _normalize_tag_value(tag_json.get(field))
        if isinstance(value, list):
            for label in value:
                _add_label_count(counts, field=field, label=label)
        else:
            _add_label_count(counts, field=field, label=value)


@st.cache_data(show_spinner=False)
def _load_tag_overview(
    shards: tuple[tuple[str, int], ...],
) -> dict[str, object]:
    counts = _empty_tag_counts()
    latest_rows: dict[str, tuple[int, dict[str, object]]] = {}
    row_number = 0
    tag_failed = 0

    for shard_path, _ in shards:
        parquet_file = pq.ParquetFile(shard_path)
        columns = [
            column for column in TAG_OVERVIEW_COLUMNS if column in parquet_file.schema_arrow.names
        ]
        if not columns:
            row_number += parquet_file.metadata.num_rows
            continue

        table = pq.read_table(shard_path, columns=columns)
        for row in table.to_pylist():
            if int(row.get("tag_ok") or 0) != 1:
                tag_failed += 1
                row_number += 1
                continue
            clip_id = str(row.get("clip_id") or f"row-{row_number:08d}")
            latest_rows[clip_id] = (row_number, dict(row))
            row_number += 1

    for _, row in sorted(latest_rows.values(), key=lambda item: item[0]):
        _accumulate_tag_counts(
            counts,
            tag_json=_tag_json_from_row(row),
        )

    return {
        "rows_scanned": row_number,
        "unique_success_clips": len(latest_rows),
        "tag_failed_rows": tag_failed,
        "counts": counts,
    }


def _counts_dataframe(counts: dict[str, int], *, denominator: int) -> pd.DataFrame:
    rows = [
        {
            "label": str(label),
            "count": int(count),
            "percent": 0.0 if denominator <= 0 else round(100.0 * int(count) / denominator, 2),
        }
        for label, count in counts.items()
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["label"])))
    return pd.DataFrame(rows)


def _render_count_table(title: str, counts: dict[str, int], *, denominator: int) -> None:
    df = _counts_dataframe(counts, denominator=denominator)
    st.markdown(f"**{title}**")
    st.caption(f"denominator={denominator}")
    if df.empty:
        st.caption("No labels.")
        return
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "label": st.column_config.TextColumn("label"),
            "count": st.column_config.NumberColumn("count", format="%d"),
            "percent": st.column_config.NumberColumn("percent", format="%.2f%%"),
        },
    )


def _render_overview_tab(
    shards: tuple[tuple[str, int], ...],
    *,
    summary: dict[str, object],
) -> None:
    overview = _load_tag_overview(shards)

    metric_cols = st.columns(4)
    metric_cols[0].metric("scanned rows", overview["rows_scanned"])
    metric_cols[1].metric("unique success clips", overview["unique_success_clips"])
    metric_cols[2].metric("failed rows", overview["tag_failed_rows"])
    metric_cols[3].metric("model", summary.get("model", "-"))

    counts_by_field = overview["counts"]
    denominator = int(overview["unique_success_clips"] or 0)
    if not isinstance(counts_by_field, dict) or not counts_by_field:
        st.warning("No tag_json values to summarize.")
        return

    for left_field, right_field in zip(TAG_FIELD_ORDER[::2], TAG_FIELD_ORDER[1::2]):
        left_col, right_col = st.columns(2)
        with left_col:
            _render_count_table(
                left_field,
                dict(counts_by_field.get(left_field, {})),
                denominator=denominator,
            )
        with right_col:
            _render_count_table(
                right_field,
                dict(counts_by_field.get(right_field, {})),
                denominator=denominator,
            )
    if len(TAG_FIELD_ORDER) % 2:
        _render_count_table(
            TAG_FIELD_ORDER[-1],
            dict(counts_by_field.get(TAG_FIELD_ORDER[-1], {})),
            denominator=denominator,
        )


def _render_tag_label_panel(tag_json: dict[str, Any]) -> None:
    for field in ("domain", "scene"):
        _render_schema_label_row(
            field,
            SINGLE_VALUE_TAG_LABELS[field],
            tag_json.get(field),
        )
    for field in ("subjects", "actions"):
        _render_schema_label_row(
            field,
            MULTI_VALUE_TAG_LABELS[field],
            tag_json.get(field),
        )
    for field in ("style", "text", "watermark"):
        _render_schema_label_row(
            field,
            SINGLE_VALUE_TAG_LABELS[field],
            tag_json.get(field),
        )


def _render_tag_prompt(row: dict[str, object]) -> None:
    prompt_json = parse_json_object(row.get("tag_prompt_json"))
    if not prompt_json:
        st.caption("Current row has no `tag_prompt_json`. Enable store_prompt when running Tag to save prompts.")
        return

    meta = {
        field: prompt_json.get(field)
        for field in ("schema_version", "prompt_version", "image_count")
        if field in prompt_json
    }
    if meta:
        st.json(meta)

    system_prompt = str(prompt_json.get("system_prompt") or "").strip()
    if system_prompt:
        st.markdown("**system prompt**")
        st.code(system_prompt, language="markdown")

    user_prompt = str(prompt_json.get("user_prompt") or "").strip()
    if user_prompt:
        st.markdown("**user prompt**")
        st.code(user_prompt, language="markdown")

    image_paths = prompt_json.get("image_paths")
    timestamps_sec = prompt_json.get("timestamps_sec")
    if isinstance(image_paths, list):
        rows = []
        for index, image_path in enumerate(image_paths):
            timestamp = None
            if isinstance(timestamps_sec, list) and index < len(timestamps_sec):
                timestamp = timestamps_sec[index]
            rows.append(
                {
                    "index": index,
                    "timestamp_sec": timestamp,
                    "image_path": str(image_path),
                }
            )
        st.markdown("**prompt images**")
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )


def _render_caption_preview(row: dict[str, object]) -> None:
    fields = (
        ("caption_level_0", "level_0"),
        ("caption_level_1", "level_1"),
        ("caption_level_2", "level_2"),
        ("caption_level_3", "level_3"),
    )
    caption_json = parse_json_object(row.get("caption_json"))
    has_caption = False
    for row_field, json_field in fields:
        text = str(row.get(row_field) or caption_json.get(json_field) or "").strip()
        if not text:
            continue
        has_caption = True
        st.markdown(f"**{json_field}**")
        st.write(text)
    if not has_caption:
        st.caption("Current row has no caption fields.")


def _row_detail(row: dict[str, object], *, tag_json: dict[str, Any]) -> dict[str, object]:
    fields = (
        "clip_id",
        "video_id",
        "clip_path",
        "duration_sec",
        "width",
        "height",
        "fps",
        "frame_ok",
        "frame_error",
        "camera_ok",
        "caption_ok",
        "tag_ok",
        "tag_error",
        "tag_schema_version",
        "tag_prompt_version",
        "tag_prompt_image_count",
        "tag_prompt_timestamps_sec",
        "input_run_id",
        "run_id",
    )
    detail = {
        field: row.get(field)
        for field in fields
        if field in row and not _is_missing(row.get(field))
    }
    frame_json = parse_json_object(row.get("frame_json"))
    if frame_json:
        detail["frame"] = {
            key: frame_json.get(key)
            for key in (
                "sampling_method",
                "sampled_fps",
                "sampled_frame_count",
                "frame_width",
                "frame_height",
            )
        }
    detail["tag"] = tag_json
    return detail


def _format_seconds(value: object) -> str:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "-"


def _render_tag_row(
    row: dict[str, object],
    *,
    clip_position: int,
    show_video: bool,
) -> None:
    tag_json = _tag_json_from_row(row)
    frame_json = parse_json_object(row.get("frame_json"))
    frame_paths = resolve_frame_paths(frame_json=frame_json)
    timestamps = frame_timestamps(frame_json)

    st.markdown(f"#### Clip {clip_position}")
    cols = st.columns(6)
    cols[0].metric("duration", _format_seconds(row.get("duration_sec")))
    cols[1].metric("tag_ok", coerce_int(row.get("tag_ok")))
    cols[2].metric("domain", str(tag_json.get("domain") or "-"))
    cols[3].metric("scene", str(tag_json.get("scene") or "-"))
    cols[4].metric("style", str(tag_json.get("style") or "-"))
    cols[5].metric("text", str(tag_json.get("text") or "-"))

    st.caption(
        f"clip_id={row.get('clip_id', '-')} | "
        f"frames={len(frame_paths)} | "
        f"subjects={tag_json.get('subjects')} | "
        f"actions={tag_json.get('actions')}"
    )
    render_frame_strip(frame_paths, timestamps=timestamps)

    summary_cols = st.columns([3, 2])
    with summary_cols[0]:
        with st.expander("Tag labels", expanded=False):
            _render_tag_label_panel(tag_json)
    with summary_cols[1]:
        if show_video:
            render_clip_video(row, missing_label="Current tag row is missing `clip_path`.")
        else:
            clip_path = str(row.get("clip_path") or "").strip()
            if clip_path:
                st.caption(f"video unloaded: {clip_path}")

    with st.expander("upstream caption", expanded=False):
        _render_caption_preview(row)
    with st.expander("tag_json", expanded=False):
        st.json(tag_json)
    with st.expander("tag prompt", expanded=False):
        _render_tag_prompt(row)
    with st.expander("tag row meta", expanded=False):
        st.json(_row_detail(row, tag_json=tag_json))


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
        _load_tag_rows.clear()
        _load_tag_overview.clear()
    summary = load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = resolve_context_shards(metadata_path, summary)
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No tag index records were loaded.")
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

    metric_cols = st.columns(6)
    metric_cols[0].metric("tag rows", summary.get("output_count", total_rows))
    metric_cols[1].metric("Tag OK", summary.get("ok_count", "-"))
    metric_cols[2].metric("Tag Failed", summary.get("failed_count", "-"))
    metric_cols[3].metric("Clip Pages", total_pages)
    metric_cols[4].metric("Concurrency", summary.get("request_concurrency", "-"))
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
    page_rows = _load_tag_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Overview", "Tag", "Current Page Details"])
    with tabs[0]:
        _render_overview_tab(
            shards,
            summary=summary,
        )

    with tabs[1]:
        if not page_rows:
            st.warning("No tag rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            _render_tag_row(
                row,
                clip_position=page_start + offset + 1,
                show_video=show_video,
            )
            st.divider()

    with tabs[2]:
        st.json(
            {
                "page": page_number,
                "per_page": per_page,
                "row_indices": list(row_indices),
                "loaded_tag_rows": len(page_rows),
                "clips": [
                    _row_detail(
                        row,
                        tag_json=_tag_json_from_row(row),
                    )
                    for row in page_rows
                ],
            }
        )
