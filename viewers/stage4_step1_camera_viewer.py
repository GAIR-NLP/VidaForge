"""Stage 4 Step 1 camera viewer."""

from __future__ import annotations

import json
import html
from pathlib import Path
import sys
from typing import Any, get_args

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .context_viewer_common import (
        frame_timestamps as _frame_timestamps,
        load_summary_light as _load_summary_light,
        page_indices as _page_indices,
        render_frame_strip as _render_frame_strip,
        resolve_context_shards as _resolve_frame_shards,
        resolve_frame_paths as _resolve_frame_paths,
        total_rows_from_shards as _total_rows_from_shards,
    )
    from .viewer_common import (
        format_elapsed_seconds,
        render_metadata_selection,
        resolve_summary_path,
    )
except ImportError:
    from context_viewer_common import (
        frame_timestamps as _frame_timestamps,
        load_summary_light as _load_summary_light,
        page_indices as _page_indices,
        render_frame_strip as _render_frame_strip,
        resolve_context_shards as _resolve_frame_shards,
        resolve_frame_paths as _resolve_frame_paths,
        total_rows_from_shards as _total_rows_from_shards,
    )
    from viewer_common import (
        format_elapsed_seconds,
        render_metadata_selection,
        resolve_summary_path,
    )

from vidaforge.common import join_data_dir, replace_path_part
try:
    from .coerce import coerce_float, coerce_int
except ImportError:
    from coerce import coerce_float, coerce_int
from vidaforge.annotation.camera import schema as camera_schema

STAGE_DIR = "stage4_annotation"
STEP_DIR = "step1_camera"
VIEWER_TITLE = "Stage 4 Step 1 Camera"
OVERVIEW_COLUMNS = ("clip_id", "camera_ok", "camera_json")
SINGLE_VALUE_CAMERA_LABELS = {
    "motion_type": tuple(get_args(camera_schema.MotionType)),
    "steadiness": tuple(get_args(camera_schema.Steadiness)),
    "speed": tuple(get_args(camera_schema.CameraSpeed)),
    "scene_dynamics": tuple(get_args(camera_schema.SceneDynamics)),
}
GROUP_CAMERA_LABELS = {
    "rotation": {
        "pan": tuple(get_args(camera_schema.Pan)),
        "tilt": tuple(get_args(camera_schema.Tilt)),
        "roll": tuple(get_args(camera_schema.Roll)),
    },
    "translation": {
        "dolly": tuple(get_args(camera_schema.Dolly)),
        "pedestal": tuple(get_args(camera_schema.Pedestal)),
        "truck": tuple(get_args(camera_schema.Truck)),
    },
    "intrinsic": {
        "zoom": tuple(get_args(camera_schema.Zoom)),
    },
    "object_centric": {
        "arc": tuple(get_args(camera_schema.Arc)),
        "arc_tracking": tuple(get_args(camera_schema.ArcTracking)),
        "lead_tracking": tuple(get_args(camera_schema.LeadTracking)),
        "tail_tracking": tuple(get_args(camera_schema.TailTracking)),
        "side_tracking": tuple(get_args(camera_schema.SideTracking)),
        "aerial_tracking": tuple(get_args(camera_schema.AerialTracking)),
        "pan_tracking": tuple(get_args(camera_schema.PanTracking)),
        "tilt_tracking": tuple(get_args(camera_schema.TiltTracking)),
        "subject_size_change": tuple(get_args(camera_schema.SubjectSizeChange)),
    },
}
EFFECT_CAMERA_LABELS = tuple(get_args(camera_schema.CinematicMotionEffect))
CAMERA_FIELD_ORDER = (
    "motion_type",
    "steadiness",
    "rotation",
    "translation",
    "intrinsic",
    "object_centric",
    "speed",
    "effects",
    "scene_dynamics",
)


def _format_seconds(value: object) -> str:
    number = coerce_float(value)
    return "-" if number is None else f"{number:.3f}s"


@st.cache_data(show_spinner=False)
def _load_camera_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not row_indices:
        return rows

    requested = set(index for index in row_indices if index >= 0)
    shard_start = 0
    for shard_path, shard_rows in shards:
        shard_end = shard_start + shard_rows
        global_to_local = {
            index: index - shard_start
            for index in requested
            if shard_start <= index < shard_end
        }
        if global_to_local:
            frame = pq.read_table(shard_path).to_pandas()
            for global_index in sorted(global_to_local):
                local_index = global_to_local[global_index]
                if local_index < len(frame):
                    row = frame.iloc[local_index].to_dict()
                    row["_camera_row_index"] = global_index
                    rows.append(row)
        shard_start = shard_end
    return rows


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _camera_summary(camera_json: dict[str, Any]) -> dict[str, object]:
    return {
        "ok": camera_json.get("ok"),
        "motion_type": camera_json.get("motion_type"),
        "steadiness": camera_json.get("steadiness"),
        "speed": camera_json.get("speed"),
        "scene_dynamics": camera_json.get("scene_dynamics"),
        "effects": camera_json.get("effects"),
        "rotation": camera_json.get("rotation"),
        "translation": camera_json.get("translation"),
        "intrinsic": camera_json.get("intrinsic"),
    }


def _count_key(*, subfield: str | None, option: object) -> str:
    return json.dumps(
        {
            "subfield": subfield or "",
            "option": str(option or "").strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _parse_count_key(key: str) -> tuple[str, str]:
    try:
        parsed = json.loads(key)
    except json.JSONDecodeError:
        return "", key
    if not isinstance(parsed, dict):
        return "", key
    return str(parsed.get("subfield") or ""), str(parsed.get("option") or "")


def _empty_camera_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for field, labels in SINGLE_VALUE_CAMERA_LABELS.items():
        counts[field] = {
            _count_key(subfield=None, option=label): 0
            for label in labels
        }

    for group, subfield_labels in GROUP_CAMERA_LABELS.items():
        group_counts: dict[str, int] = {}
        for subfield, labels in subfield_labels.items():
            for label in labels:
                group_counts[_count_key(subfield=subfield, option=label)] = 0
        counts[group] = group_counts

    counts["effects"] = {
        _count_key(subfield=None, option=label): 0
        for label in EFFECT_CAMERA_LABELS
    }
    return counts


def _add_count(
    counts: dict[str, dict[str, int]],
    *,
    field: str,
    label: object,
    subfield: str | None = None,
) -> None:
    label_text = str(label or "").strip()
    if not label_text:
        return
    field_counts = counts.setdefault(field, {})
    key = _count_key(subfield=subfield, option=label_text)
    field_counts[key] = field_counts.get(key, 0) + 1


def _accumulate_camera_counts(
    counts: dict[str, dict[str, int]],
    *,
    camera_json: dict[str, Any],
) -> None:
    for field in SINGLE_VALUE_CAMERA_LABELS:
        _add_count(
            counts,
            field=field,
            label=camera_json.get(field),
        )

    for group, subfield_labels in GROUP_CAMERA_LABELS.items():
        group_value = camera_json.get(group)
        if not isinstance(group_value, dict):
            continue
        for subfield in subfield_labels:
            value = group_value.get(subfield)
            if str(value or "").strip():
                _add_count(
                    counts,
                    field=group,
                    label=value,
                    subfield=subfield,
                )

    effects = camera_json.get("effects")
    if isinstance(effects, list):
        for effect in effects:
            _add_count(
                counts,
                field="effects",
                label=effect,
            )
    else:
        _add_count(
            counts,
            field="effects",
            label=effects,
        )


@st.cache_data(show_spinner=False)
def _load_camera_overview(
    shards: tuple[tuple[str, int], ...],
) -> dict[str, object]:
    counts = _empty_camera_counts()
    latest_rows: dict[str, tuple[int, dict[str, object]]] = {}
    row_number = 0
    camera_failed = 0

    for shard_path, _ in shards:
        parquet_file = pq.ParquetFile(shard_path)
        columns = [column for column in OVERVIEW_COLUMNS if column in parquet_file.schema_arrow.names]
        if not columns:
            continue
        table = pq.read_table(shard_path, columns=columns)
        for row in table.to_pylist():
            if int(row.get("camera_ok") or 0) != 1:
                camera_failed += 1
                row_number += 1
                continue
            clip_id = str(row.get("clip_id") or f"row-{row_number:08d}")
            latest_rows[clip_id] = (row_number, dict(row))
            row_number += 1

    for _, row in sorted(latest_rows.values(), key=lambda item: item[0]):
        _accumulate_camera_counts(
            counts,
            camera_json=_parse_json_object(row.get("camera_json")),
        )

    return {
        "rows_scanned": row_number,
        "unique_success_clips": len(latest_rows),
        "camera_failed_rows": camera_failed,
        "counts": counts,
    }


def _counts_dataframe(counts: dict[str, int], *, denominator: int) -> pd.DataFrame:
    rows = []
    has_subfield = False
    for key, count in counts.items():
        subfield, option = _parse_count_key(key)
        has_subfield = has_subfield or bool(subfield)
        rows.append(
            {
                "subfield": subfield,
                "option": option,
                "count": count,
                "percent": 0.0 if denominator <= 0 else round(100.0 * count / denominator, 2),
            }
        )
    rows.sort(
        key=lambda item: (
            str(item["subfield"]),
            -int(item["count"]),
            str(item["option"]),
        )
    )
    if not has_subfield:
        for row in rows:
            row.pop("subfield", None)
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
            "subfield": st.column_config.TextColumn("subfield"),
            "option": st.column_config.TextColumn("option"),
            "count": st.column_config.NumberColumn("count", format="%d"),
            "percent": st.column_config.NumberColumn("percent", format="%.2f%%"),
        },
    )


def _render_overview_tab(
    shards: tuple[tuple[str, int], ...],
    *,
    summary: dict[str, object],
) -> None:
    overview = _load_camera_overview(shards)

    metric_cols = st.columns(4)
    metric_cols[0].metric("scanned rows", overview["rows_scanned"])
    metric_cols[1].metric("unique success clips", overview["unique_success_clips"])
    metric_cols[2].metric("failed rows", overview["camera_failed_rows"])
    metric_cols[3].metric("model", summary.get("model", "-"))

    counts_by_field = overview["counts"]
    denominator = int(overview["unique_success_clips"] or 0)
    if not isinstance(counts_by_field, dict) or not counts_by_field:
        st.warning("No camera_json values to summarize.")
        return

    for left_field, right_field in zip(CAMERA_FIELD_ORDER[::2], CAMERA_FIELD_ORDER[1::2]):
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
    if len(CAMERA_FIELD_ORDER) % 2:
        _render_count_table(
            CAMERA_FIELD_ORDER[-1],
            dict(counts_by_field.get(CAMERA_FIELD_ORDER[-1], {})),
            denominator=denominator,
        )


def _fallback_frame_json_from_prompt(row: dict[str, object]) -> dict[str, Any]:
    prompt_json = _parse_json_object(row.get("camera_prompt_json"))
    image_paths = prompt_json.get("image_paths")
    timestamps_sec = prompt_json.get("timestamps_sec")
    if not isinstance(image_paths, list):
        return {}

    frame_json: dict[str, Any] = {
        "frame_paths": [str(path) for path in image_paths],
    }
    if isinstance(timestamps_sec, list):
        frame_json["timestamps_sec"] = timestamps_sec
    return frame_json


def _resolve_camera_frame_paths(
    row: dict[str, object],
    *,
    frame_json: dict[str, Any],
) -> list[Path]:
    prompt_json = _parse_json_object(row.get("camera_prompt_json"))
    image_paths = prompt_json.get("image_paths")
    if isinstance(image_paths, list):
        paths = [Path(str(path)).expanduser() for path in image_paths if str(path or "").strip()]
        if paths:
            return paths

    return _resolve_frame_paths(frame_json=frame_json)


def _camera_detail(row: dict[str, object], *, camera_json: dict[str, Any]) -> dict[str, object]:
    fields = (
        "clip_id",
        "video_id",
        "clip_path",
        "duration_sec",
        "width",
        "height",
        "fps",
        "camera_ok",
        "camera_error",
        "camera_prompt_image_count",
        "camera_prompt_timestamps_sec",
        "label_version",
        "prompt_version",
        "prompt_mode",
        "input_run_id",
        "run_id",
    )
    detail = {field: row.get(field) for field in fields if field in row and pd.notna(row.get(field))}
    frame_json = _parse_json_object(row.get("frame_json"))
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
    detail["camera"] = camera_json
    return detail


def _render_camera_prompt(row: dict[str, object]) -> None:
    prompt_json = _parse_json_object(row.get("camera_prompt_json"))
    if not prompt_json:
        st.caption("Current row has no `camera_prompt_json`. Run Camera with `STORE_PROMPT=true` to save prompts.")
        return

    meta = {
        field: prompt_json.get(field)
        for field in ("prompt_version", "prompt_mode", "image_count")
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

    rendered_fields = {
        "prompt_version",
        "prompt_mode",
        "system_prompt",
        "user_prompt",
        "image_paths",
        "timestamps_sec",
        "image_count",
    }
    extra_fields = {
        field: value
        for field, value in prompt_json.items()
        if field not in rendered_fields
    }
    if extra_fields:
        with st.expander("camera prompt extra fields", expanded=False):
            st.json(extra_fields)


def _render_clip_video(row: dict[str, object]) -> None:
    clip_path = str(row.get("clip_path") or "").strip()
    if not clip_path:
        st.warning("Current camera row is missing `clip_path`.")
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


def _selected_labels(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item or "").strip()}
    text = str(value or "").strip()
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


def _render_camera_label_panel(camera_json: dict[str, Any]) -> None:
    for field in ("motion_type", "steadiness", "speed", "scene_dynamics"):
        _render_schema_label_row(
            field,
            SINGLE_VALUE_CAMERA_LABELS[field],
            camera_json.get(field),
        )

    _render_schema_label_row("effects", EFFECT_CAMERA_LABELS, camera_json.get("effects"))

    for group, subfield_labels in GROUP_CAMERA_LABELS.items():
        group_value = camera_json.get(group)
        if not isinstance(group_value, dict):
            group_value = {}
        st.markdown(f"**{group}**")
        for subfield, labels in subfield_labels.items():
            _render_schema_label_row(
                subfield,
                labels,
                group_value.get(subfield),
            )


def _render_camera_row(
    row: dict[str, object],
    *,
    clip_position: int,
    show_video: bool,
) -> None:
    camera_json = _parse_json_object(row.get("camera_json"))
    frame_json = _parse_json_object(row.get("frame_json"))
    if not frame_json:
        frame_json = _fallback_frame_json_from_prompt(row)
    frame_paths = _resolve_camera_frame_paths(
        row,
        frame_json=frame_json,
    )
    timestamps = _frame_timestamps(frame_json)
    camera_summary = _camera_summary(camera_json)

    st.markdown(f"#### Clip {clip_position}")
    cols = st.columns(5)
    cols[0].metric("duration", _format_seconds(row.get("duration_sec")))
    cols[1].metric("camera_ok", coerce_int(row.get("camera_ok")))
    cols[2].metric("motion", str(camera_summary.get("motion_type") or "-"))
    cols[3].metric("steady", str(camera_summary.get("steadiness") or "-"))
    cols[4].metric("speed", str(camera_summary.get("speed") or "-"))

    st.caption(
        f"clip_id={row.get('clip_id', '-')} | "
        f"frames={len(frame_paths)} | "
        f"effects={camera_summary.get('effects')}"
    )
    _render_frame_strip(frame_paths, timestamps=timestamps)

    summary_cols = st.columns([3, 2])
    with summary_cols[0]:
        with st.expander("Camera labels", expanded=False):
            _render_camera_label_panel(camera_json)
    with summary_cols[1]:
        if show_video:
            _render_clip_video(row)
        else:
            clip_path = str(row.get("clip_path") or "").strip()
            if clip_path:
                st.caption(f"video unloaded: {clip_path}")

    with st.expander("camera_json", expanded=False):
        st.json(camera_json)
    with st.expander("camera prompt", expanded=False):
        _render_camera_prompt(row)
    with st.expander("camera row meta", expanded=False):
        st.json(_camera_detail(row, camera_json=camera_json))


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
        _load_summary_light.clear()
        _load_camera_rows.clear()
        _load_camera_overview.clear()
    summary = _load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = _resolve_frame_shards(metadata_path, summary)
    total_rows = _total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No camera index records were loaded.")
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
    metric_cols[0].metric("camera rows", summary.get("camera_rows", total_rows))
    metric_cols[1].metric("Camera OK", summary.get("camera_ok", "-"))
    metric_cols[2].metric("Camera Failed", summary.get("camera_failed", "-"))
    metric_cols[3].metric("Clip Pages", total_pages)
    metric_cols[4].metric("Concurrency", summary.get("request_concurrency", "-"))
    metric_cols[5].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    with st.expander("summary preview", expanded=False):
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

    row_indices = _page_indices(total_rows=total_rows, page_number=page_number, per_page=per_page)
    page_rows = _load_camera_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Overview", "Camera", "Current Page Details"])
    with tabs[0]:
        _render_overview_tab(
            shards,
            summary=summary,
        )

    with tabs[1]:
        if not page_rows:
            st.warning("No camera rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            clip_position = page_start + offset + 1
            _render_camera_row(
                row,
                clip_position=clip_position,
                show_video=show_video,
            )
            st.divider()

    with tabs[2]:
        st.json(
            {
                "page": page_number,
                "per_page": per_page,
                "row_indices": list(row_indices),
                "loaded_camera_rows": len(page_rows),
                "clips": [
                    _camera_detail(
                        row,
                        camera_json=_parse_json_object(row.get("camera_json")),
                    )
                    for row in page_rows
                ],
            }
        )
