"""Stage 3 Step 3 dedup viewer."""

from __future__ import annotations

import json
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
        coerce_float,
        coerce_int,
        format_seconds,
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
        coerce_float,
        coerce_int,
        format_seconds,
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


STAGE_DIR = "stage3_selection"
STEP_DIR = "step3_dedup"
VIEWER_TITLE = "Stage 3 Step 3 Dedup"
KNOWN_DEDUPLICATOR_NAMES = ("pdq", "cosmos")


@st.cache_data(show_spinner=False)
def _load_dedup_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_dedup_row_index",
    )


@st.cache_data(show_spinner=False)
def _load_dedup_group_summaries(
    shards: tuple[tuple[str, int], ...],
    *,
    deduplicator: str,
) -> list[dict[str, object]]:
    group_id_column = f"{deduplicator}_group_id"
    candidate_columns = (
        "clip_id",
        "video_id",
        "duration_sec",
        group_id_column,
        f"{deduplicator}_group_size",
        f"{deduplicator}_is_best_clip_in_group",
        f"{deduplicator}_best_clip_id_in_group",
        f"{deduplicator}_json",
    )
    grouped: dict[str, dict[str, object]] = {}
    shard_start = 0
    for shard_path, shard_rows in shards:
        parquet_file = pq.ParquetFile(shard_path)
        available_columns = set(parquet_file.schema_arrow.names)
        if group_id_column not in available_columns:
            shard_start += shard_rows
            continue
        columns = [column for column in candidate_columns if column in available_columns]
        frame = pq.read_table(shard_path, columns=columns).to_pandas()
        for local_index, row in frame.iterrows():
            raw_group_id = row.get(group_id_column)
            if _is_missing(raw_group_id):
                continue
            group_id = str(raw_group_id).strip()
            if not group_id:
                continue
            payload = parse_json_object(row.get(f"{deduplicator}_json"))
            clip_id = str(row.get("clip_id") or "").strip()
            member = {
                "row_index": int(shard_start + local_index),
                "clip_id": clip_id,
                "video_id": str(row.get("video_id") or "").strip(),
                "duration_sec": coerce_float(row.get("duration_sec")),
                "is_best_clip_in_group": coerce_int(
                    row.get(f"{deduplicator}_is_best_clip_in_group"),
                    default=coerce_int(payload.get("is_best_clip_in_group"), default=0),
                ),
                "best_matched_clip": _best_matched_clip_text(payload),
            }
            group = grouped.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "declared_group_size": coerce_int(
                        row.get(f"{deduplicator}_group_size"),
                        default=coerce_int(payload.get("group_size"), default=0),
                    ),
                    "best_clip_id_in_group": str(
                        row.get(f"{deduplicator}_best_clip_id_in_group")
                        or payload.get("best_clip_id_in_group")
                        or ""
                    ),
                    "members": [],
                },
            )
            group["members"].append(member)  # type: ignore[index, union-attr]
        shard_start += shard_rows

    summaries: list[dict[str, object]] = []
    for group in grouped.values():
        members = sorted(
            group["members"],  # type: ignore[index]
            key=lambda member: (
                -int(member.get("is_best_clip_in_group") or 0),
                str(member.get("clip_id") or ""),
            ),
        )
        row_indices = tuple(int(member["row_index"]) for member in members)
        declared_size = coerce_int(group.get("declared_group_size"), default=0)
        summaries.append(
            {
                "group_id": group["group_id"],
                "group_size": declared_size or len(members),
                "loaded_member_count": len(members),
                "best_clip_id_in_group": group["best_clip_id_in_group"],
                "video_id_count": len(
                    {
                        str(member.get("video_id") or "").strip()
                        for member in members
                        if str(member.get("video_id") or "").strip()
                    }
                ),
                "row_indices": row_indices,
                "members": members,
            }
        )
    return sorted(
        summaries,
        key=lambda group: (
            -coerce_int(group.get("group_size"), default=0),
            str(group.get("group_id") or ""),
        ),
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _normalize_list(value: object) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return _normalize_list(parsed)
        return [text]
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if hasattr(value, "tolist"):
        return _normalize_list(value.tolist())
    return [str(value).strip()]


def _deduplicator_names_from_row(row: dict[str, object]) -> list[str]:
    names = _normalize_list(row.get("deduplicators"))
    for name in KNOWN_DEDUPLICATOR_NAMES:
        if any(
            f"{name}_{suffix}" in row
            for suffix in (
                "ok",
                "error",
                "group_id",
                "group_size",
                "is_best_clip_in_group",
                "best_clip_id_in_group",
                "json",
            )
        ):
            names.append(name)
    return list(dict.fromkeys(name for name in names if name))


def _deduplicator_json(row: dict[str, object], name: str) -> dict[str, Any]:
    payload = parse_json_object(row.get(f"{name}_json"))
    if payload:
        return payload
    dedup_json = parse_json_object(row.get("dedup_json"))
    nested = dedup_json.get(name)
    return nested if isinstance(nested, dict) else {}


def _best_matched_clip_text(payload: dict[str, Any]) -> str:
    match = payload.get("best_matched_clip")
    if not isinstance(match, dict) or not match:
        return "-"
    clip_id = str(match.get("clip_id") or "").strip()
    if "cosine_similarity" in match:
        score = coerce_float(match.get("cosine_similarity"))
        return f"{clip_id} ({score:.6f})" if score is not None else clip_id
    if "similar_frame_ratio" in match and "mean_hamming_distance" in match:
        ratio = coerce_float(match.get("similar_frame_ratio"))
        distance = coerce_float(match.get("mean_hamming_distance"))
        if ratio is not None and distance is not None:
            return f"{clip_id} (ratio={ratio:.3f}, hamming={distance:.2f})"
    return clip_id or "-"


def _deduplicator_result_rows(row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in _deduplicator_names_from_row(row):
        payload = _deduplicator_json(row, name)
        group_id = str(row.get(f"{name}_group_id") or payload.get("group_id") or "")
        rows.append(
            {
                "deduplicator": name,
                "ok": coerce_int(row.get(f"{name}_ok")),
                "group_id": group_id,
                "group_size": coerce_int(
                    row.get(f"{name}_group_size"),
                    default=coerce_int(payload.get("group_size"), default=1),
                ),
                "is_best_clip_in_group": coerce_int(
                    row.get(f"{name}_is_best_clip_in_group"),
                    default=coerce_int(payload.get("is_best_clip_in_group"), default=1),
                ),
                "best_clip_id_in_group": str(
                    row.get(f"{name}_best_clip_id_in_group")
                    or payload.get("best_clip_id_in_group")
                    or ""
                ),
                "best_matched_clip": _best_matched_clip_text(payload),
                "error": str(row.get(f"{name}_error") or "").strip(),
            }
        )
    return rows


def _format_group_value(row: dict[str, object], name: str) -> str:
    group_id = str(row.get(f"{name}_group_id") or "").strip()
    group_size = coerce_int(row.get(f"{name}_group_size"), default=1)
    if not group_id:
        return "unique"
    return f"{group_id} ({group_size})"


def _row_detail(row: dict[str, object]) -> dict[str, object]:
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
        "filter_ok",
        "filter_error",
        "dedup_ok",
        "dedup_error",
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
    dedup_results = _deduplicator_result_rows(row)
    if dedup_results:
        detail["deduplicator_results"] = dedup_results
    dedup_json = parse_json_object(row.get("dedup_json"))
    if dedup_json:
        detail["dedup_json"] = dedup_json
    return detail


def _page_table(page_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in page_rows:
        item: dict[str, object] = {
            "clip_id": row.get("clip_id"),
            "duration_sec": coerce_float(row.get("duration_sec")),
            "dedup_ok": coerce_int(row.get("dedup_ok")),
            "dedup_error": str(row.get("dedup_error") or "").strip(),
        }
        for name in _deduplicator_names_from_row(row):
            item[f"{name}_ok"] = coerce_int(row.get(f"{name}_ok"))
            item[f"{name}_group_id"] = str(row.get(f"{name}_group_id") or "")
            item[f"{name}_group_size"] = coerce_int(
                row.get(f"{name}_group_size"),
                default=1,
            )
            item[f"{name}_is_best_clip_in_group"] = coerce_int(
                row.get(f"{name}_is_best_clip_in_group"),
                default=1,
            )
            item[f"{name}_best_clip_id_in_group"] = str(
                row.get(f"{name}_best_clip_id_in_group") or ""
            )
        rows.append(item)
    return pd.DataFrame(rows)


def _render_deduplicator_table(row: dict[str, object]) -> None:
    result_rows = _deduplicator_result_rows(row)
    if not result_rows:
        st.caption("Current row has no deduplicator result fields.")
        return

    st.dataframe(
        pd.DataFrame(result_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "deduplicator": st.column_config.TextColumn("deduplicator"),
            "ok": st.column_config.NumberColumn("ok", format="%d"),
            "group_id": st.column_config.TextColumn("group_id"),
            "group_size": st.column_config.NumberColumn("group_size", format="%d"),
            "is_best_clip_in_group": st.column_config.NumberColumn(
                "is_best_clip_in_group",
                format="%d",
            ),
            "best_clip_id_in_group": st.column_config.TextColumn(
                "best_clip_id_in_group"
            ),
            "best_matched_clip": st.column_config.TextColumn("best_matched_clip"),
            "error": st.column_config.TextColumn("error"),
        },
    )


def _render_dedup_json(row: dict[str, object]) -> None:
    dedup_json = parse_json_object(row.get("dedup_json"))
    if dedup_json:
        with st.expander("dedup_json", expanded=False):
            st.json(dedup_json)

    for name in _deduplicator_names_from_row(row):
        payload = parse_json_object(row.get(f"{name}_json"))
        if not payload:
            continue
        with st.expander(f"{name}_json", expanded=False):
            st.json(payload)


def _render_dedup_clip(
    row: dict[str, object],
    *,
    clip_position: int,
    show_video: bool,
) -> None:
    frame_json = parse_json_object(row.get("frame_json"))
    frame_paths = resolve_frame_paths(frame_json=frame_json)
    timestamps = frame_timestamps(frame_json)

    st.markdown(f"#### Clip {clip_position}")
    metric_cols = st.columns(6)
    metric_cols[0].metric("duration", format_seconds(row.get("duration_sec")))
    metric_cols[1].metric("dedup_ok", coerce_int(row.get("dedup_ok")))
    metric_cols[2].metric("pdq group", _format_group_value(row, "pdq"))
    metric_cols[3].metric(
        "pdq best",
        coerce_int(row.get("pdq_is_best_clip_in_group"), default=1),
    )
    metric_cols[4].metric("cosmos group", _format_group_value(row, "cosmos"))
    metric_cols[5].metric(
        "cosmos best",
        coerce_int(row.get("cosmos_is_best_clip_in_group"), default=1),
    )

    dedup_error = str(row.get("dedup_error") or "").strip()
    caption = (
        f"clip_id={row.get('clip_id', '-')} | "
        f"frames={len(frame_paths)} | "
        f"deduplicators={_deduplicator_names_from_row(row)}"
    )
    if dedup_error:
        caption = f"{caption} | error={dedup_error}"
    st.caption(caption)

    render_frame_strip(frame_paths, timestamps=timestamps)

    cols = st.columns([3, 2])
    with cols[0]:
        _render_deduplicator_table(row)
    with cols[1]:
        if show_video:
            render_clip_video(row, missing_label="Current dedup row is missing `clip_path`.")
        else:
            clip_path = str(row.get("clip_path") or "").strip()
            if clip_path:
                st.caption(f"video unloaded: {clip_path}")

    _render_dedup_json(row)
    with st.expander("dedup row meta", expanded=False):
        st.json(_row_detail(row))


def _group_table(groups: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_id": group.get("group_id"),
                "group_size": group.get("group_size"),
                "loaded_member_count": group.get("loaded_member_count"),
                "video_id_count": group.get("video_id_count"),
                "best_clip_id_in_group": group.get("best_clip_id_in_group"),
            }
            for group in groups
        ]
    )


def _group_size_distribution_table(groups: list[dict[str, object]]) -> pd.DataFrame:
    counts: dict[int, int] = {}
    for group in groups:
        size = coerce_int(group.get("group_size"), default=0)
        if size <= 0:
            continue
        counts[size] = counts.get(size, 0) + 1
    return pd.DataFrame(
        [
            {
                "group_size": size,
                "group_count": counts[size],
                "clip_count": size * counts[size],
            }
            for size in sorted(counts)
        ]
    )


def _filter_groups(
    groups: list[dict[str, object]],
    *,
    min_group_size: int,
    max_group_size: int | None,
    cross_video_only: bool,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for group in groups:
        group_size = coerce_int(group.get("group_size"), default=0)
        if group_size < min_group_size:
            continue
        if max_group_size is not None and group_size > max_group_size:
            continue
        if cross_video_only and coerce_int(group.get("video_id_count"), default=0) <= 1:
            continue
        filtered.append(group)
    return filtered


def _member_table(group: dict[str, object]) -> pd.DataFrame:
    members = group.get("members")
    if not isinstance(members, list):
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "clip_id": member.get("clip_id"),
                "video_id": member.get("video_id"),
                "duration_sec": member.get("duration_sec"),
                "is_best_clip_in_group": member.get("is_best_clip_in_group"),
                "best_matched_clip": member.get("best_matched_clip"),
                "row_index": member.get("row_index"),
            }
            for member in members
            if isinstance(member, dict)
        ]
    )


def _render_group_view(
    shards: tuple[tuple[str, int], ...],
    *,
    widget_prefix: str,
    show_video: bool,
) -> None:
    deduplicator = st.radio(
        "deduplicator",
        KNOWN_DEDUPLICATOR_NAMES,
        horizontal=True,
        key=f"{widget_prefix}_group_deduplicator",
    )
    groups = _load_dedup_group_summaries(shards, deduplicator=deduplicator)
    if not groups:
        st.info(f"Current metadata has no `{deduplicator}` duplicate group。")
        return

    max_observed_group_size = max(
        coerce_int(group.get("group_size"), default=0)
        for group in groups
    )
    filter_cols = st.columns(3)
    min_group_size = int(
        filter_cols[0].number_input(
            "group_size >=",
            min_value=2,
            max_value=max(2, max_observed_group_size),
            value=2,
            step=1,
            key=f"{widget_prefix}_{deduplicator}_min_group_size",
        )
    )
    max_group_size_enabled = filter_cols[1].checkbox(
        "Enable group_size <=",
        value=False,
        key=f"{widget_prefix}_{deduplicator}_max_group_size_enabled",
    )
    max_group_size = None
    if max_group_size_enabled:
        max_group_size = int(
            filter_cols[1].number_input(
                "group_size <=",
                min_value=min_group_size,
                max_value=max(min_group_size, max_observed_group_size),
                value=max(min_group_size, max_observed_group_size),
                step=1,
                key=f"{widget_prefix}_{deduplicator}_max_group_size",
            )
        )
    cross_video_only = filter_cols[2].checkbox(
        "Only cross-video_id groups",
        value=False,
        key=f"{widget_prefix}_{deduplicator}_cross_video_only",
    )
    filtered_groups = _filter_groups(
        groups,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        cross_video_only=cross_video_only,
    )

    distribution = _group_size_distribution_table(filtered_groups)
    with st.expander("group_size distribution (after current filters)", expanded=True):
        st.dataframe(
            distribution,
            width="stretch",
            hide_index=True,
            column_config={
                "group_size": st.column_config.NumberColumn("group_size", format="%d"),
                "group_count": st.column_config.NumberColumn("group_count", format="%d"),
                "clip_count": st.column_config.NumberColumn("clip_count", format="%d"),
            },
        )
    if not filtered_groups:
        st.info("No duplicate group matches the current filters.")
        return

    group_page_size = int(
        st.selectbox(
            "Groups per page (controls how many groups are listed below)",
            (10, 25, 50, 100),
            index=1,
            key=f"{widget_prefix}_groups_per_page",
        )
    )
    total_group_pages = max(1, (len(filtered_groups) + group_page_size - 1) // group_page_size)
    group_page_key = f"{widget_prefix}_{deduplicator}_group_page_number"
    if group_page_key not in st.session_state:
        st.session_state[group_page_key] = 1
    st.session_state[group_page_key] = max(
        1,
        min(int(st.session_state[group_page_key]), total_group_pages),
    )
    group_page = int(
        st.number_input(
            "group Page",
            min_value=1,
            max_value=total_group_pages,
            step=1,
            key=group_page_key,
        )
    )
    group_start = (group_page - 1) * group_page_size
    group_end = min(len(filtered_groups), group_start + group_page_size)
    page_groups = filtered_groups[group_start:group_end]

    st.caption(f"groups {group_start + 1}-{group_end} / {len(filtered_groups)} filtered, {len(groups)} total")
    group_options = [str(group["group_id"]) for group in page_groups]
    selected_group_id = st.selectbox(
        "duplicate group",
        group_options,
        key=f"{widget_prefix}_selected_{deduplicator}_group",
        format_func=lambda group_id: next(
            (
                f"{group_id} ({group['loaded_member_count']} clips)"
                for group in page_groups
                if str(group.get("group_id")) == group_id
            ),
            group_id,
        ),
    )
    group_by_id = {str(group["group_id"]): group for group in page_groups}
    selected_group = group_by_id[selected_group_id]

    metric_cols = st.columns(4)
    metric_cols[0].metric("groups", len(filtered_groups), delta=f"{len(groups)} total")
    metric_cols[1].metric("group_size", selected_group.get("group_size"))
    metric_cols[2].metric("video_ids", selected_group.get("video_id_count"))
    metric_cols[3].metric("best clip", selected_group.get("best_clip_id_in_group") or "-")

    with st.expander("Groups on Current Page", expanded=False):
        st.dataframe(
            _group_table(page_groups),
            width="stretch",
            hide_index=True,
        )

    member_table = _member_table(selected_group)
    if not member_table.empty:
        st.dataframe(
            member_table,
            width="stretch",
            hide_index=True,
            column_config={
                "duration_sec": st.column_config.NumberColumn("duration_sec", format="%.3f"),
                "row_index": st.column_config.NumberColumn("row_index", format="%d"),
            },
        )

    row_indices = tuple(int(index) for index in selected_group.get("row_indices", ()))
    group_rows = _load_dedup_rows(shards, row_indices=row_indices)
    rows_by_index = {
        int(row["_dedup_row_index"]): row
        for row in group_rows
        if "_dedup_row_index" in row
    }
    ordered_rows = [rows_by_index[index] for index in row_indices if index in rows_by_index]
    for offset, row in enumerate(ordered_rows, start=1):
        _render_dedup_clip(
            row,
            clip_position=offset,
            show_video=show_video,
        )
        st.divider()


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
        _load_dedup_rows.clear()
        _load_dedup_group_summaries.clear()
    summary = load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = resolve_context_shards(metadata_path, summary)
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No dedup index records were loaded.")
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

    match_summary = summary.get("deduplicator_match_summary")
    metric_cols = st.columns(7)
    metric_cols[0].metric("dedup rows", summary.get("output_count", total_rows))
    metric_cols[1].metric("Dedup OK", summary.get("ok_count", "-"))
    metric_cols[2].metric("Dedup Failed", summary.get("failed_count", "-"))
    metric_cols[3].metric("duplicate pairs", summary.get("pair_count", "-"))
    metric_cols[4].metric("input rows", summary.get("input_count", "-"))
    metric_cols[5].metric("Clip Pages", total_pages)
    metric_cols[6].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    summary_preview = {
        key: value
        for key, value in summary.items()
        if key not in LARGE_SUMMARY_SKIP_KEYS
    }
    with st.expander("summary preview", expanded=False):
        st.json(summary_preview)
    if isinstance(match_summary, dict) and match_summary:
        with st.expander("deduplicator match summary", expanded=False):
            st.json(match_summary)
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
    page_rows = _load_dedup_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Group View", "Clip View", "Current Page Table"])
    with tabs[0]:
        _render_group_view(
            shards,
            widget_prefix=widget_prefix,
            show_video=show_video,
        )

    with tabs[1]:
        if not page_rows:
            st.warning("No dedup rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            clip_position = page_start + offset + 1
            _render_dedup_clip(
                row,
                clip_position=clip_position,
                show_video=show_video,
            )
            st.divider()

    with tabs[2]:
        table = _page_table(page_rows)
        if table.empty:
            st.warning("No dedup table is available on the current page.")
        else:
            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(column, format="%.6f")
                    for column in table.columns
                    if column == "duration_sec"
                },
            )
