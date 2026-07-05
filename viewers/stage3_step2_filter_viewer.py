"""Stage 3 Step 2 filter viewer."""

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
        render_bar_chart,
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
        render_bar_chart,
        render_metadata_selection,
        resolve_summary_path,
    )


STAGE_DIR = "stage3_selection"
STEP_DIR = "step2_filter"
VIEWER_TITLE = "Stage 3 Step 2 Filter"
KNOWN_FILTER_NAMES = ("optical", "motion", "aesthetic", "text")
SCORE_BIN_EDGES = (
    float("-inf"),
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.000000001,
    float("inf"),
)
SCORE_BIN_LABELS = (
    "<0",
    "0.0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
    ">1",
)


@st.cache_data(show_spinner=False)
def _load_filter_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_filter_row_index",
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _format_score(value: object) -> str:
    score = coerce_float(value)
    return "-" if score is None else f"{score:.6f}"


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


def _filter_names_from_row(row: dict[str, object]) -> list[str]:
    names = _normalize_list(row.get("filters"))
    for name in KNOWN_FILTER_NAMES:
        if any(
            f"{name}_{suffix}" in row
            for suffix in ("ok", "error", "score", "json")
        ):
            names.append(name)
    return list(dict.fromkeys(name for name in names if name))


def _filter_result_rows(row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in _filter_names_from_row(row):
        rows.append(
            {
                "filter": name,
                "ok": coerce_int(row.get(f"{name}_ok")),
                "score": coerce_float(row.get(f"{name}_score")),
                "error": str(row.get(f"{name}_error") or "").strip(),
            }
        )
    return rows


def _filter_json_by_name(row: dict[str, object]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name in _filter_names_from_row(row):
        payloads[name] = parse_json_object(row.get(f"{name}_json"))
    return payloads


def _score_column_sort_key(column: str) -> tuple[int, str]:
    filter_name = column.removesuffix("_score")
    if filter_name in KNOWN_FILTER_NAMES:
        return KNOWN_FILTER_NAMES.index(filter_name), filter_name
    return len(KNOWN_FILTER_NAMES), filter_name


@st.cache_data(show_spinner=False)
def _load_filter_score_columns(shards: tuple[tuple[str, int], ...]) -> tuple[str, ...]:
    score_columns: set[str] = set()
    for shard_path, _ in shards:
        parquet_file = pq.ParquetFile(shard_path)
        for column in parquet_file.schema_arrow.names:
            if column.endswith("_score"):
                score_columns.add(column)
    return tuple(sorted(score_columns, key=_score_column_sort_key))


@st.cache_data(show_spinner=False)
def _load_filter_score_overview(
    shards: tuple[tuple[str, int], ...],
) -> dict[str, object]:
    rows_scanned = 0
    for shard_path, shard_rows in shards:
        rows_scanned += int(shard_rows)

    ordered_score_columns = _load_filter_score_columns(shards)
    stats: dict[str, dict[str, object]] = {
        column: {
            "count": 0,
            "missing": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
            "bins": {label: 0 for label in SCORE_BIN_LABELS},
        }
        for column in ordered_score_columns
    }

    if not ordered_score_columns:
        return {
            "rows_scanned": rows_scanned,
            "score_columns": ordered_score_columns,
            "stats": stats,
        }

    for shard_path, _ in shards:
        parquet_file = pq.ParquetFile(shard_path)
        available_columns = [
            column
            for column in ordered_score_columns
            if column in parquet_file.schema_arrow.names
        ]
        if not available_columns:
            continue

        frame = pq.read_table(shard_path, columns=available_columns).to_pandas()
        for column in available_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            valid = series.dropna()
            column_stats = stats[column]
            column_stats["missing"] = int(column_stats["missing"]) + int(
                series.isna().sum()
            )
            if valid.empty:
                continue

            count = int(len(valid))
            column_stats["count"] = int(column_stats["count"]) + count
            column_stats["sum"] = float(column_stats["sum"]) + float(valid.sum())
            valid_min = float(valid.min())
            valid_max = float(valid.max())
            current_min = column_stats["min"]
            current_max = column_stats["max"]
            column_stats["min"] = (
                valid_min
                if current_min is None
                else min(float(current_min), valid_min)
            )
            column_stats["max"] = (
                valid_max
                if current_max is None
                else max(float(current_max), valid_max)
            )

            bin_counts = (
                pd.cut(
                    valid,
                    bins=SCORE_BIN_EDGES,
                    labels=SCORE_BIN_LABELS,
                    right=False,
                    include_lowest=True,
                )
                .value_counts(sort=False)
                .to_dict()
            )
            bins = column_stats["bins"]
            if isinstance(bins, dict):
                for label, value in bin_counts.items():
                    label_text = str(label)
                    bins[label_text] = int(bins.get(label_text, 0)) + int(value)

    for column_stats in stats.values():
        count = int(column_stats["count"])
        column_stats["mean"] = (
            None if count <= 0 else float(column_stats["sum"]) / count
        )
        column_stats.pop("sum", None)

    return {
        "rows_scanned": rows_scanned,
        "score_columns": ordered_score_columns,
        "stats": stats,
    }


def _score_stats_dataframe(overview: dict[str, object]) -> pd.DataFrame:
    stats = overview.get("stats")
    if not isinstance(stats, dict):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for column in overview.get("score_columns", ()):
        if not isinstance(column, str):
            continue
        column_stats = stats.get(column)
        if not isinstance(column_stats, dict):
            continue
        score_count = int(column_stats.get("count") or 0)
        rows.append(
            {
                "filter": column.removesuffix("_score"),
                "score_column": column,
                "valid_count": score_count,
                "missing_count": int(column_stats.get("missing") or 0),
                "mean": column_stats.get("mean"),
                "min": column_stats.get("min"),
                "max": column_stats.get("max"),
            }
        )
    return pd.DataFrame(rows)


def _score_distribution_dataframe(
    overview: dict[str, object],
    *,
    score_column: str,
) -> pd.DataFrame:
    stats = overview.get("stats")
    if not isinstance(stats, dict):
        return pd.DataFrame()
    column_stats = stats.get(score_column)
    if not isinstance(column_stats, dict):
        return pd.DataFrame()
    bins = column_stats.get("bins")
    if not isinstance(bins, dict):
        return pd.DataFrame()
    valid_count = int(column_stats.get("count") or 0)
    rows = []
    for label in SCORE_BIN_LABELS:
        count = int(bins.get(label, 0))
        rows.append(
            {
                "score_range": label,
                "count": count,
                "percent": (
                    0.0
                    if valid_count <= 0
                    else round(100.0 * count / valid_count, 2)
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _load_score_filtered_indices(
    shards: tuple[tuple[str, int], ...],
    *,
    score_conditions: tuple[tuple[str, str, float], ...],
) -> tuple[int, ...]:
    if not score_conditions:
        return ()

    condition_by_column = {
        column: (operator, threshold)
        for column, operator, threshold in score_conditions
    }
    score_columns = tuple(condition_by_column)
    matched_indices: list[int] = []
    shard_start = 0
    for shard_path, shard_rows in shards:
        parquet_file = pq.ParquetFile(shard_path)
        available_columns = [
            column
            for column in score_columns
            if column in parquet_file.schema_arrow.names
        ]
        if available_columns:
            frame = pq.read_table(shard_path, columns=available_columns).to_pandas()
        else:
            frame = pd.DataFrame(index=range(shard_rows))

        masks: list[pd.Series] = []
        for column in score_columns:
            if column in frame.columns:
                operator, threshold = condition_by_column[column]
                score_series = pd.to_numeric(frame[column], errors="coerce")
                if operator == "lt":
                    mask = score_series < threshold
                else:
                    mask = score_series > threshold
            else:
                mask = pd.Series(False, index=frame.index)
            masks.append(mask.fillna(False))

        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = combined & mask

        matched_indices.extend(
            shard_start + int(local_index)
            for local_index in combined[combined].index.tolist()
        )
        shard_start += shard_rows

    return tuple(matched_indices)


def _slice_indices(
    indices: tuple[int, ...],
    *,
    page_number: int,
    per_page: int,
) -> tuple[int, ...]:
    start = max(0, (page_number - 1) * per_page)
    end = min(len(indices), start + per_page)
    return indices[start:end]


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
    filter_results = _filter_result_rows(row)
    if filter_results:
        detail["filter_results"] = filter_results
    filter_payloads = _filter_json_by_name(row)
    if filter_payloads:
        detail["filter_json"] = filter_payloads
    return detail


def _page_table(page_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in page_rows:
        item: dict[str, object] = {
            "clip_id": row.get("clip_id"),
            "duration_sec": coerce_float(row.get("duration_sec")),
            "filter_ok": coerce_int(row.get("filter_ok")),
            "filter_error": str(row.get("filter_error") or "").strip(),
        }
        for name in _filter_names_from_row(row):
            item[f"{name}_ok"] = coerce_int(row.get(f"{name}_ok"))
            item[f"{name}_score"] = coerce_float(row.get(f"{name}_score"))
            error = str(row.get(f"{name}_error") or "").strip()
            if error:
                item[f"{name}_error"] = error
        rows.append(item)
    return pd.DataFrame(rows)


def _render_filter_scores(row: dict[str, object]) -> None:
    result_rows = _filter_result_rows(row)
    if not result_rows:
        st.caption("Current row has no filter result fields.")
        return

    st.dataframe(
        pd.DataFrame(result_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "filter": st.column_config.TextColumn("filter"),
            "ok": st.column_config.NumberColumn("ok", format="%d"),
            "score": st.column_config.NumberColumn("score", format="%.6f"),
            "error": st.column_config.TextColumn("error"),
        },
    )


def _render_filter_json(row: dict[str, object]) -> None:
    payloads = _filter_json_by_name(row)
    if not payloads:
        st.caption("Current row has no filter JSON field.")
        return
    for name, payload in payloads.items():
        with st.expander(f"{name}_json", expanded=False):
            st.json(payload)


def _render_score_overview(
    shards: tuple[tuple[str, int], ...],
    *,
    summary: dict[str, object],
) -> None:
    overview = _load_filter_score_overview(shards)
    score_columns = tuple(
        column
        for column in overview.get("score_columns", ())
        if isinstance(column, str)
    )

    st.markdown("### Score Overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("scanned rows", overview.get("rows_scanned", "-"))
    metric_cols[1].metric("score columns", len(score_columns))
    metric_cols[2].metric("Filter OK", summary.get("ok_count", "-"))
    metric_cols[3].metric("Filter Failed", summary.get("failed_count", "-"))

    stats_table = _score_stats_dataframe(overview)
    if stats_table.empty:
        st.warning("Current metadata has no `*_score` fields to summarize.")
        return

    st.dataframe(
        stats_table,
        width="stretch",
        hide_index=True,
        column_config={
            "valid_count": st.column_config.NumberColumn("valid_count", format="%d"),
            "missing_count": st.column_config.NumberColumn(
                "missing_count",
                format="%d",
            ),
            "mean": st.column_config.NumberColumn("mean", format="%.6f"),
            "min": st.column_config.NumberColumn("min", format="%.6f"),
            "max": st.column_config.NumberColumn("max", format="%.6f"),
        },
    )

    for left_column, right_column in zip(score_columns[::2], score_columns[1::2]):
        left, right = st.columns(2)
        with left:
            distribution = _score_distribution_dataframe(
                overview,
                score_column=left_column,
            )
            render_bar_chart(
                distribution,
                category_col="score_range",
                value_col="count",
                title=f"{left_column} Distribution",
                sort=list(SCORE_BIN_LABELS),
                label_angle=45,
            )
        with right:
            distribution = _score_distribution_dataframe(
                overview,
                score_column=right_column,
            )
            render_bar_chart(
                distribution,
                category_col="score_range",
                value_col="count",
                title=f"{right_column} Distribution",
                sort=list(SCORE_BIN_LABELS),
                label_angle=45,
            )
    if len(score_columns) % 2:
        score_column = score_columns[-1]
        distribution = _score_distribution_dataframe(
            overview,
            score_column=score_column,
        )
        render_bar_chart(
            distribution,
            category_col="score_range",
            value_col="count",
            title=f"{score_column} Distribution",
            sort=list(SCORE_BIN_LABELS),
            label_angle=45,
        )


def _render_filter_clip(
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
    metric_cols[1].metric("filter_ok", coerce_int(row.get("filter_ok")))
    metric_cols[2].metric("optical", _format_score(row.get("optical_score")))
    metric_cols[3].metric("motion", _format_score(row.get("motion_score")))
    metric_cols[4].metric("aesthetic", _format_score(row.get("aesthetic_score")))
    metric_cols[5].metric("text", _format_score(row.get("text_score")))

    filter_error = str(row.get("filter_error") or "").strip()
    caption = (
        f"clip_id={row.get('clip_id', '-')} | "
        f"frames={len(frame_paths)} | "
        f"filters={_filter_names_from_row(row)}"
    )
    if filter_error:
        caption = f"{caption} | error={filter_error}"
    st.caption(caption)

    render_frame_strip(frame_paths, timestamps=timestamps)

    cols = st.columns([3, 2])
    with cols[0]:
        _render_filter_scores(row)
    with cols[1]:
        if show_video:
            render_clip_video(row, missing_label="Current filter row is missing `clip_path`.")
        else:
            clip_path = str(row.get("clip_path") or "").strip()
            if clip_path:
                st.caption(f"video unloaded: {clip_path}")

    _render_filter_json(row)
    with st.expander("filter row meta", expanded=False):
        st.json(_row_detail(row))


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
        _load_filter_score_columns.clear()
        _load_filter_rows.clear()
        _load_filter_score_overview.clear()
        _load_score_filtered_indices.clear()
    summary = load_summary_light(str(resolve_summary_path(metadata_path)))
    shards = resolve_context_shards(metadata_path, summary)
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No filter index records were loaded.")
        if summary:
            with st.expander("summary.json", expanded=False):
                st.json(summary)
        return

    widget_prefix = f"{STAGE_DIR}_{STEP_DIR}"
    page_number_key = f"{widget_prefix}_clip_page_number"
    page_delta_key = f"{widget_prefix}_clip_page_delta"
    per_page_key = f"{widget_prefix}_clips_per_page"
    show_video_key = f"{widget_prefix}_show_video"
    score_threshold_prefix = f"{widget_prefix}_score_filter_threshold"
    score_operator_prefix = f"{widget_prefix}_score_filter_operator"
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
        score_columns = _load_filter_score_columns(shards)
        st.caption("Score threshold filters: 0 means this filter is ignored.")
        score_condition_rows: list[tuple[str, str, float]] = []
        for column in score_columns:
            op_col, value_col = st.columns([1, 2])
            with op_col:
                operator = st.selectbox(
                    column,
                    ("gt", "lt"),
                    index=0,
                    format_func=lambda value: ">" if value == "gt" else "<",
                    key=f"{score_operator_prefix}_{column}",
                )
            with value_col:
                threshold = float(
                    st.number_input(
                        "Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.3f",
                        key=f"{score_threshold_prefix}_{column}",
                        label_visibility="collapsed",
                    )
                )
            if threshold > 0:
                score_condition_rows.append((column, str(operator), threshold))
        score_conditions = tuple(score_condition_rows)

    score_filter_active = bool(score_conditions)
    if score_filter_active:
        filtered_row_indices = _load_score_filtered_indices(
            shards,
            score_conditions=score_conditions,
        )
        display_total_rows = len(filtered_row_indices)
    else:
        filtered_row_indices = ()
        display_total_rows = total_rows

    total_pages = max(1, (display_total_rows + per_page - 1) // per_page)
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
    metric_cols[0].metric("Current Rows", display_total_rows)
    metric_cols[1].metric("Compute OK", summary.get("ok_count", "-"))
    metric_cols[2].metric("Compute Failed", summary.get("failed_count", "-"))
    metric_cols[3].metric("input rows", summary.get("input_count", "-"))
    metric_cols[4].metric("resumed", summary.get("resumed_count", "-"))
    metric_cols[5].metric("Clip Pages", total_pages)
    metric_cols[6].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))
    if score_filter_active:
        threshold_text = ", ".join(
            f"{column} {'<' if operator == 'lt' else '>'} {threshold:.3f}"
            for column, operator, threshold in score_conditions
        )
        st.caption(
            "Score filters: "
            f"all satisfy {threshold_text} "
            f"| all rows={summary.get('output_count', total_rows)}"
        )

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
    page_end = min(display_total_rows, page_start + per_page)
    if display_total_rows > 0:
        st.caption(f"clips {page_start + 1}-{page_end} / {display_total_rows}")
    else:
        st.caption("clips 0 / 0")
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
            disabled=page_number >= total_pages or display_total_rows <= 0,
            key=f"{widget_prefix}_next_clip_page",
        ):
            st.session_state[page_delta_key] = 1
            st.rerun()

    if score_filter_active:
        row_indices = _slice_indices(
            filtered_row_indices,
            page_number=page_number,
            per_page=per_page,
        )
    else:
        row_indices = page_indices(
            total_rows=total_rows,
            page_number=page_number,
            per_page=per_page,
        )
    page_rows = _load_filter_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Overview", "Filter", "Current Page Table"])
    with tabs[0]:
        _render_score_overview(shards, summary=summary)

    with tabs[1]:
        if not page_rows:
            st.warning("No filter rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            clip_position = page_start + offset + 1
            _render_filter_clip(
                row,
                clip_position=clip_position,
                show_video=show_video,
            )
            st.divider()

    with tabs[2]:
        table = _page_table(page_rows)
        if table.empty:
            st.warning("No filter table is available on the current page.")
        else:
            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(column, format="%.6f")
                    for column in table.columns
                    if column.endswith("_score") or column == "duration_sec"
                },
            )
