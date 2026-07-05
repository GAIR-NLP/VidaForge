"""Stage 3 Step 4 select viewer."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
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
STEP_DIR = "step4_select"
VIEWER_TITLE = "Stage 3 Step 4 Select"


@st.cache_data(show_spinner=False)
def _load_select_rows(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    return load_rows_by_indices(
        shards,
        row_indices=row_indices,
        row_index_key="_select_row_index",
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


def _select_partition_path(metadata_path: Path, partition: str) -> Path:
    if partition == "all":
        return metadata_path
    return metadata_path / partition


def _rules_dataframe(summary: dict[str, object]) -> pd.DataFrame:
    rules = summary.get("rules")
    if not isinstance(rules, dict):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for rule_name, rule in rules.items():
        if not isinstance(rule, dict):
            continue
        condition = ""
        if "equals" in rule:
            condition = f"equals {rule['equals']}"
        elif "min" in rule:
            condition = f"min {rule['min']}"
        elif "max" in rule:
            condition = f"max {rule['max']}"
        rows.append(
            {
                "rule": str(rule_name),
                "field": str(rule.get("field") or ""),
                "condition": condition,
                "reject_reason": str(rule.get("reject_reason") or ""),
                "missing_reject_reason": str(rule.get("missing_reject_reason") or ""),
            }
        )
    return pd.DataFrame(rows)


def _reject_reason_dataframe(summary: dict[str, object]) -> pd.DataFrame:
    counts = summary.get("reject_reason_counts")
    if not isinstance(counts, dict):
        return pd.DataFrame()
    rows = [
        {"reject_reason": str(reason), "count": int(count)}
        for reason, count in counts.items()
    ]
    rows.sort(key=lambda item: (-int(item["count"]), str(item["reject_reason"])))
    return pd.DataFrame(rows)


def _rule_result_rows(row: dict[str, object]) -> list[dict[str, object]]:
    select_json = parse_json_object(row.get("select_json"))
    rows: list[dict[str, object]] = []
    for rule_name, result in select_json.items():
        if not isinstance(result, dict):
            continue
        condition = ""
        if "equals" in result:
            condition = f"equals {result['equals']}"
        elif "min" in result:
            condition = f"min {result['min']}"
        elif "max" in result:
            condition = f"max {result['max']}"
        rows.append(
            {
                "rule": str(rule_name),
                "field": str(result.get("field") or ""),
                "value": result.get("value"),
                "pass": coerce_int(result.get("pass")),
                "condition": condition,
                "reject_reason": str(result.get("reject_reason") or ""),
            }
        )
    return rows


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
        "select_ok",
        "select_error",
        "select_pass",
        "select_reject_reason",
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
    rule_results = _rule_result_rows(row)
    if rule_results:
        detail["rule_results"] = rule_results
    select_json = parse_json_object(row.get("select_json"))
    if select_json:
        detail["select_json"] = select_json
    return detail


def _page_table(page_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in page_rows:
        rows.append(
            {
                "clip_id": row.get("clip_id"),
                "duration_sec": coerce_float(row.get("duration_sec")),
                "select_ok": coerce_int(row.get("select_ok")),
                "select_pass": coerce_int(row.get("select_pass")),
                "select_reject_reason": str(row.get("select_reject_reason") or ""),
                "optical_score": coerce_float(row.get("optical_score")),
                "motion_score": coerce_float(row.get("motion_score")),
                "aesthetic_score": coerce_float(row.get("aesthetic_score")),
                "dedup_ok": coerce_int(row.get("dedup_ok")),
                "pdq_is_best_clip_in_group": coerce_int(
                    row.get("pdq_is_best_clip_in_group"),
                    default=1,
                ),
                "cosmos_is_best_clip_in_group": coerce_int(
                    row.get("cosmos_is_best_clip_in_group"),
                    default=1,
                ),
                "pdq_group_id": str(row.get("pdq_group_id") or ""),
                "cosmos_group_id": str(row.get("cosmos_group_id") or ""),
            }
        )
    return pd.DataFrame(rows)


def _render_rule_results(row: dict[str, object]) -> None:
    rule_rows = _rule_result_rows(row)
    if not rule_rows:
        st.caption("Current row has no select_json rule results.")
        return
    st.dataframe(
        pd.DataFrame(rule_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "rule": st.column_config.TextColumn("rule"),
            "field": st.column_config.TextColumn("field"),
            "value": st.column_config.TextColumn("value"),
            "pass": st.column_config.NumberColumn("pass", format="%d"),
            "condition": st.column_config.TextColumn("condition"),
            "reject_reason": st.column_config.TextColumn("reject_reason"),
        },
    )


def _render_select_json(row: dict[str, object]) -> None:
    select_json = parse_json_object(row.get("select_json"))
    if not select_json:
        st.caption("Current row has no select_json field.")
        return
    with st.expander("select_json", expanded=False):
        st.json(select_json)


def _render_select_clip(
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
    metric_cols[1].metric("select_pass", coerce_int(row.get("select_pass")))
    metric_cols[2].metric("optical", _format_score(row.get("optical_score")))
    metric_cols[3].metric("motion", _format_score(row.get("motion_score")))
    metric_cols[4].metric("dedup_ok", coerce_int(row.get("dedup_ok")))
    metric_cols[5].metric(
        "pdq best",
        coerce_int(row.get("pdq_is_best_clip_in_group"), default=1),
    )

    reject_reason = str(row.get("select_reject_reason") or "").strip()
    caption = (
        f"clip_id={row.get('clip_id', '-')} | "
        f"frames={len(frame_paths)} | "
        f"select_ok={coerce_int(row.get('select_ok'))}"
    )
    if reject_reason:
        caption = f"{caption} | reject_reason={reject_reason}"
    st.caption(caption)

    render_frame_strip(frame_paths, timestamps=timestamps)

    cols = st.columns([3, 2])
    with cols[0]:
        _render_rule_results(row)
    with cols[1]:
        if show_video:
            render_clip_video(row, missing_label="Current select row is missing `clip_path`.")
        else:
            clip_path = str(row.get("clip_path") or "").strip()
            if clip_path:
                st.caption(f"video unloaded: {clip_path}")
        group_rows = [
            {
                "deduplicator": "pdq",
                "group_id": str(row.get("pdq_group_id") or ""),
                "group_size": coerce_int(row.get("pdq_group_size"), default=1),
                "is_best_clip_in_group": coerce_int(
                    row.get("pdq_is_best_clip_in_group"),
                    default=1,
                ),
                "best_clip_id_in_group": str(
                    row.get("pdq_best_clip_id_in_group") or ""
                ),
            },
            {
                "deduplicator": "cosmos",
                "group_id": str(row.get("cosmos_group_id") or ""),
                "group_size": coerce_int(row.get("cosmos_group_size"), default=1),
                "is_best_clip_in_group": coerce_int(
                    row.get("cosmos_is_best_clip_in_group"),
                    default=1,
                ),
                "best_clip_id_in_group": str(
                    row.get("cosmos_best_clip_id_in_group") or ""
                ),
            },
        ]
        group_table = pd.DataFrame(
            [
                item
                for item in group_rows
                if item["group_id"]
                or f"{item['deduplicator']}_group_size" in row
                or f"{item['deduplicator']}_is_best_clip_in_group" in row
            ]
        )
        if not group_table.empty:
            st.dataframe(group_table, width="stretch", hide_index=True)

    _render_select_json(row)
    with st.expander("select row meta", expanded=False):
        st.json(_row_detail(row))


def _render_overview(summary: dict[str, object], *, partition: str) -> None:
    st.markdown("### Overview")
    metric_cols = st.columns(5)
    metric_cols[0].metric("partition", partition)
    metric_cols[1].metric("pass", summary.get("pass_count", "-"))
    metric_cols[2].metric("reject", summary.get("reject_count", "-"))
    metric_cols[3].metric("ok", summary.get("ok_count", "-"))
    metric_cols[4].metric("failed", summary.get("failed_count", "-"))

    rules = _rules_dataframe(summary)
    reject_reasons = _reject_reason_dataframe(summary)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**rules**")
        if rules.empty:
            st.caption("summary.json has no rules.")
        else:
            st.dataframe(rules, width="stretch", hide_index=True)
    with cols[1]:
        st.markdown("**reject reasons**")
        if reject_reasons.empty:
            st.caption("summary.json has no reject_reason_counts.")
        else:
            st.dataframe(reject_reasons, width="stretch", hide_index=True)


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

    base_metadata_path = Path(selection["metadata_path"])
    with selection["extra_sidebar_container"]:
        partition = st.selectbox(
            "Output partition",
            ("all", "pass", "reject"),
            index=0,
            key=f"{STAGE_DIR}_{STEP_DIR}_partition",
            help="all reads the main output directory; pass/reject read the corresponding subdirectories.",
        )
    metadata_path = _select_partition_path(base_metadata_path, partition)
    if not metadata_path.exists():
        st.error(f"Output partition does not exist: `{metadata_path}`")
        return

    if selection.get("refresh"):
        load_summary_light.clear()
        load_rows_by_indices.clear()
        _load_select_rows.clear()
    summary = load_summary_light(str(resolve_summary_path(base_metadata_path)))
    shards = resolve_context_shards(metadata_path, summary if partition == "all" else {})
    total_rows = total_rows_from_shards(shards)
    if total_rows <= 0:
        st.warning("No select index records were loaded.")
        if summary:
            with st.expander("summary.json", expanded=False):
                st.json(summary)
        return

    widget_prefix = f"{STAGE_DIR}_{STEP_DIR}_{partition}"
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
    metric_cols[0].metric("partition rows", total_rows)
    metric_cols[1].metric("all rows", summary.get("output_count", "-"))
    metric_cols[2].metric("pass", summary.get("pass_count", "-"))
    metric_cols[3].metric("reject", summary.get("reject_count", "-"))
    metric_cols[4].metric("select failed", summary.get("failed_count", "-"))
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
    page_rows = _load_select_rows(shards, row_indices=row_indices)

    tabs = st.tabs(["Overview", "Select", "Current Page Table"])
    with tabs[0]:
        _render_overview(summary, partition=partition)

    with tabs[1]:
        if not page_rows:
            st.warning("No select rows were loaded on the current page.")
        for offset, row in enumerate(page_rows, start=0):
            clip_position = page_start + offset + 1
            _render_select_clip(
                row,
                clip_position=clip_position,
                show_video=show_video,
            )
            st.divider()

    with tabs[2]:
        table = _page_table(page_rows)
        if table.empty:
            st.warning("No select table is available on the current page.")
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
