from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from vidaforge.common import join_data_dir, join_raw_dir
from vidaforge.index import load_parquet

DEFAULT_PROJECT_DIR = Path(
    os.environ.get(
        "DATA_DIR",
        str(Path.cwd() / "examples" / "vidaforge_output"),
    )
)
DEFAULT_DATAFRAME_PREVIEW_ROWS = 1000
DEFAULT_JSON_PREVIEW_ITEMS = 50
DEFAULT_JSON_STRING_MAX_CHARS = 2000
LARGE_JSON_KEYS = {
    "task_outputs",
    "failed_task_outputs",
    "failed_examples",
}


@st.cache_data(show_spinner=False)
def load_rows(
    metadata_path: str,
    *,
    unit: str | None = None,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    return load_parquet(
        metadata_path,
        unit=unit,
        columns=list(columns) if columns is not None else None,
    )


@st.cache_data(show_spinner=False)
def load_summary(summary_path: str) -> dict[str, object]:
    path = Path(summary_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_elapsed_seconds(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    seconds = float(value)
    if seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remain = int(round(seconds % 60))
        return f"{minutes}m {remain:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


def format_bytes(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    size = float(value)
    if size < 0:
        return "-"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


@st.cache_data(show_spinner=False)
def load_file_sizes(file_paths: tuple[str, ...]) -> list[int | None]:
    sizes: list[int | None] = []
    for value in file_paths:
        path_str = str(value or "").strip()
        if not path_str:
            sizes.append(None)
            continue
        try:
            path = join_data_dir(path_str)
            if not path.exists() or not path.is_file():
                sizes.append(None)
                continue
            sizes.append(path.stat().st_size)
        except OSError:
            sizes.append(None)
    return sizes


def list_run_ids(project_dir: str, stage_dir: str, step_dir: str) -> list[str]:
    meta_dir = Path(project_dir).expanduser().resolve() / "meta" / stage_dir / step_dir
    if not meta_dir.exists() or not meta_dir.is_dir():
        return []

    run_ids: list[str] = []
    for path in meta_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("run_id="):
            run_ids.append(path.name.removeprefix("run_id="))
            continue
        run_ids.append(path.name)
    return sorted(run_ids, reverse=True)


def resolve_metadata_path(
    project_dir: str,
    stage_dir: str,
    step_dir: str,
    run_id: str,
    override: str,
) -> Path:
    override_path = Path(override).expanduser()
    if override.strip():
        return override_path.resolve()
    base_dir = Path(project_dir).expanduser().resolve() / "meta" / stage_dir / step_dir
    legacy_path = base_dir / f"run_id={run_id}"
    if legacy_path.exists():
        return legacy_path
    return base_dir / run_id


def resolve_summary_path(metadata_path: Path) -> Path:
    path = metadata_path.expanduser().resolve()
    if path.is_dir():
        return path / "summary.json"
    return path.parent / "summary.json"


def rows_to_dataframe(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    for column in (
        "filesize_bytes",
        "filesize_mb",
        "duration_sec",
        "fps",
        "width",
        "height",
        "bit_rate",
        "probe_ok",
        "probe_elapsed_ms",
        "filter_keep",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def dataframe_preview(
    frame: pd.DataFrame,
    *,
    max_rows: int = DEFAULT_DATAFRAME_PREVIEW_ROWS,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.head(max_rows)


def render_dataframe_preview(
    frame: pd.DataFrame,
    *,
    max_rows: int = DEFAULT_DATAFRAME_PREVIEW_ROWS,
    **kwargs: object,
) -> None:
    preview = dataframe_preview(frame, max_rows=max_rows)
    if len(preview) < len(frame):
        st.caption(f"Showing the first {len(preview):,} / {len(frame):,} rows to avoid loading too much data in the browser.")
    st.dataframe(preview, **kwargs)


def _compact_json_for_display(value: object) -> object:
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in LARGE_JSON_KEYS:
                if isinstance(item, list | tuple):
                    compacted[key_text] = {
                        "omitted": True,
                        "count": len(item),
                        "preview": [
                            _compact_json_for_display(element)
                            for element in list(item)[:DEFAULT_JSON_PREVIEW_ITEMS]
                        ],
                    }
                else:
                    compacted[key_text] = "<omitted large field>"
                continue
            compacted[key_text] = _compact_json_for_display(item)
        return compacted
    if isinstance(value, list | tuple):
        items = list(value)
        if len(items) <= DEFAULT_JSON_PREVIEW_ITEMS:
            return [_compact_json_for_display(item) for item in items]
        return {
            "omitted": True,
            "count": len(items),
            "preview": [
                _compact_json_for_display(item)
                for item in items[:DEFAULT_JSON_PREVIEW_ITEMS]
            ],
        }
    if isinstance(value, str) and len(value) > DEFAULT_JSON_STRING_MAX_CHARS:
        return value[:DEFAULT_JSON_STRING_MAX_CHARS] + "... <truncated>"
    return value


def render_json_preview(value: object) -> None:
    st.json(_compact_json_for_display(value))


def ensure_size_columns(
    frame: pd.DataFrame,
    *,
    size_bytes_col: str,
    size_mb_col: str,
    path_col: str | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()
    if size_bytes_col in enriched.columns:
        enriched[size_bytes_col] = pd.to_numeric(enriched[size_bytes_col], errors="coerce")
    elif size_mb_col in enriched.columns:
        size_mb = pd.to_numeric(enriched[size_mb_col], errors="coerce")
        enriched[size_mb_col] = size_mb
        enriched[size_bytes_col] = size_mb * 1024 * 1024
    elif path_col and path_col in enriched.columns:
        sizes = load_file_sizes(tuple(enriched[path_col].fillna("").astype(str).tolist()))
        enriched[size_bytes_col] = pd.to_numeric(pd.Series(sizes, index=enriched.index), errors="coerce")
    else:
        enriched[size_bytes_col] = pd.Series([pd.NA] * len(enriched), index=enriched.index, dtype="Float64")

    if size_mb_col not in enriched.columns:
        enriched[size_mb_col] = pd.to_numeric(enriched[size_bytes_col], errors="coerce") / (1024 * 1024)
    else:
        enriched[size_mb_col] = pd.to_numeric(enriched[size_mb_col], errors="coerce")
    return enriched


def summarize_total_size(frame: pd.DataFrame, *, size_bytes_col: str) -> str:
    if size_bytes_col not in frame.columns:
        return "-"
    series = pd.to_numeric(frame[size_bytes_col], errors="coerce").dropna()
    series = series[series >= 0]
    if series.empty:
        return "-"
    return format_bytes(float(series.sum()))


def render_bar_chart(
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    title: str,
    height: int = 320,
    sort: str | list[str] | None = None,
    label_angle: int = 0,
) -> None:
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{category_col}:N",
                sort=sort,
                axis=alt.Axis(
                    labelAngle=label_angle,
                    labelLimit=220,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(f"{value_col}:Q", title="count"),
            tooltip=[category_col, value_col],
        )
        .properties(height=height)
    )
    st.caption(title)
    st.altair_chart(chart, width="stretch")


def add_fixed_range_chart(
    frame: pd.DataFrame,
    column: str,
    title: str,
    *,
    bins: list[float],
    labels: list[str],
) -> None:
    if column not in frame.columns:
        st.info(f"{title}: missing field `{column}`")
        return

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    series = series[series >= 0]
    if series.empty:
        st.info(f"{title}: no data to summarize")
        return

    counts = (
        pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
        .value_counts(sort=False)
        .rename_axis("range")
        .reset_index(name="count")
    )
    counts = counts[counts["count"] > 0]
    if counts.empty:
        st.info(f"{title}: no data to summarize")
        return

    render_bar_chart(counts, category_col="range", value_col="count", title=title)


def add_value_counts_chart(
    frame: pd.DataFrame,
    column: str,
    title: str,
    *,
    top_k: int | None = None,
    empty_label: str = "unknown",
    height: int = 320,
) -> None:
    if column not in frame.columns:
        st.info(f"{title}: missing field `{column}`")
        return

    counts = (
        frame[column]
        .fillna("")
        .astype(str)
        .replace("", empty_label)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    if top_k is not None:
        counts = counts.head(top_k)
    if counts.empty:
        st.info(f"{title}: no data to summarize")
        return

    render_bar_chart(
        counts,
        category_col=column,
        value_col="count",
        title=title,
        height=height,
    )


def add_resolution_chart(frame: pd.DataFrame, top_k: int = 20) -> None:
    if "width" not in frame.columns or "height" not in frame.columns:
        st.info("Resolution Distribution: missing `width` or `height` field")
        return

    width = pd.to_numeric(frame["width"], errors="coerce")
    height = pd.to_numeric(frame["height"], errors="coerce")
    valid = frame[width.notna() & height.notna()].copy()
    if valid.empty:
        st.info("Resolution Distribution: no data to summarize")
        return

    valid["resolution"] = (
        width[width.notna()].astype(int).astype(str)
        + "x"
        + height[height.notna()].astype(int).astype(str)
    )
    counts = (
        valid["resolution"]
        .value_counts()
        .head(top_k)
        .rename_axis("resolution")
        .reset_index(name="count")
    )
    render_bar_chart(
        counts,
        category_col="resolution",
        value_col="count",
        title="Resolution Distribution（Top K）",
        height=360,
        sort="-y",
        label_angle=45,
    )


def add_fps_chart(frame: pd.DataFrame, top_k: int = 12) -> None:
    if "fps" not in frame.columns:
        st.info("FPS Distribution (Top K): missing field `fps`")
        return

    series = pd.to_numeric(frame["fps"], errors="coerce").dropna()
    series = series[series > 0]
    if series.empty:
        st.info("FPS Distribution (Top K): no data to summarize")
        return

    counts = (
        series.round(2)
        .map(lambda value: f"{value:g}")
        .value_counts()
        .head(top_k)
        .rename_axis("fps")
        .reset_index(name="count")
    )
    render_bar_chart(counts, category_col="fps", value_col="count", title="FPS Distribution (Top K)")


def apply_text_probe_filters(
    frame: pd.DataFrame,
    *,
    keyword: str,
    probe_filter: str,
    extra_text_cols: Sequence[str] = (),
) -> pd.DataFrame:
    filtered = frame
    if keyword:
        keyword_lower = keyword.lower()
        text_cols = [
            col
            for col in (
                "video_path",
                "raw_path",
                "codec",
                "probe_error",
                *extra_text_cols,
            )
            if col in filtered.columns
        ]
        if text_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in text_cols:
                mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(
                    keyword_lower, regex=False
                )
            filtered = filtered[mask]

    if probe_filter != "all" and "probe_ok" in filtered.columns:
        if probe_filter == "ok":
            filtered = filtered[filtered["probe_ok"] == 1]
        elif probe_filter == "failed":
            filtered = filtered[filtered["probe_ok"] != 1]

    return filtered.reset_index(drop=True)


def paginate(frame: pd.DataFrame, page: int, per_page: int) -> pd.DataFrame:
    start = (page - 1) * per_page
    end = start + per_page
    return frame.iloc[start:end]


def show_samples(
    frame: pd.DataFrame,
    *,
    per_page: int,
    columns: int,
    detail_fields: Sequence[str],
    media_path_fields: Sequence[str] = ("video_path",),
    raw_media_path_fields: Sequence[str] = (),
    browse_order: str = "ordered",
    widget_key_prefix: str = "samples",
    show_local_paths: bool = False,
) -> None:
    total = len(frame)
    if total == 0:
        st.warning("No samples match the current filters.")
        return

    display_frame = frame
    if browse_order == "shuffled":
        display_frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    raw_media_fields = set(raw_media_path_fields)

    pages = math.ceil(total / per_page)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key=f"{widget_key_prefix}_page",
        )
    )
    st.caption(f"Page {page}/{pages}")

    current = paginate(display_frame, page, per_page)
    for i in range(0, len(current), columns):
        block = current.iloc[i : i + columns]
        cols = st.columns(len(block))
        for col, (_, row) in zip(cols, block.iterrows()):
            with col:
                media_path = ""
                media_field = ""
                fallback_media_path = ""
                fallback_media_field = ""
                for field in media_path_fields:
                    value = str(row.get(field, "") or "").strip()
                    if not value:
                        continue
                    if not fallback_media_path:
                        fallback_media_path = value
                        fallback_media_field = field
                    if value.startswith(("http://", "https://")):
                        media_path = value
                        media_field = field
                        break
                    candidate = (
                        join_raw_dir(value)
                        if field in raw_media_fields
                        else join_data_dir(value)
                    )
                    if candidate.exists():
                        media_path = str(candidate)
                        media_field = field
                        break

                if media_path:
                    try:
                        st.video(media_path, autoplay=False)
                        st.caption(f"{media_field}: {Path(media_path).name}")
                    except Exception as exc:  # noqa: BLE001
                        display_path = media_path if show_local_paths else Path(media_path).name
                        st.warning(f"Video file cannot be opened and was skipped: {display_path}")
                        st.caption(str(exc) if show_local_paths else type(exc).__name__)
                elif fallback_media_path:
                    display_path = (
                        fallback_media_path
                        if show_local_paths
                        else Path(fallback_media_path).name
                    )
                    st.caption(f"{fallback_media_field} missing: {display_path}")
                detail = {
                    key: value
                    for key, value in row.to_dict().items()
                    if key in detail_fields and pd.notna(value)
                }
                with st.expander("meta", expanded=False):
                    st.json(detail)


def render_metadata_selection(
    *,
    stage_dir: str,
    step_dir: str,
    title: str,
    include_filter_scope: bool,
    include_probe_filter: bool = True,
    include_browse_layout_controls: bool = True,
    include_raw_dir: bool = False,
    load_frame: bool = True,
    load_summary_data: bool = True,
) -> dict[str, object] | None:
    st.subheader(title)
    default_project_dir = str(DEFAULT_PROJECT_DIR)
    widget_prefix = f"{stage_dir}_{step_dir}"

    with st.sidebar:
        st.subheader(title)
        show_local_paths = st.toggle(
            "Show filesystem paths",
            value=False,
            key=f"{widget_prefix}_show_local_paths",
            help="Disabled by default to avoid exposing local paths in screenshots or recordings.",
        )
        path_input_type = "default" if show_local_paths else "password"
        project_dir = st.text_input(
            "Project data root",
            value=default_project_dir,
            key=f"{widget_prefix}_project_dir",
            type=path_input_type,
        )
        os.environ["DATA_DIR"] = str(Path(project_dir).expanduser().resolve())
        if include_raw_dir:
            default_raw_dir = os.environ.get(
                "RAW_DIR",
                str(Path.cwd() / "examples" / "raw_videos"),
            )
            raw_dir = st.text_input(
                "Raw data root",
                value=default_raw_dir,
                key=f"{widget_prefix}_raw_dir",
                help="Root directory used to resolve raw_path values.",
                type=path_input_type,
            )
            os.environ["RAW_DIR"] = str(Path(raw_dir).expanduser().resolve())
        available_run_ids = list_run_ids(project_dir, stage_dir, step_dir)
        run_id_options = available_run_ids or [""]
        run_id = st.selectbox(
            "run_id",
            run_id_options,
            index=0,
            disabled=not available_run_ids,
            key=f"{widget_prefix}_run_id",
        )
        metadata_override = st.text_input(
            "Metadata path override (optional)",
            value="",
            key=f"{widget_prefix}_metadata_override",
            type=path_input_type,
        )
        if include_browse_layout_controls:
            per_page = st.selectbox("Videos per page", [4, 8, 12, 16, 24], index=2, key=f"{widget_prefix}_per_page")
            columns = st.selectbox("Columns", [1, 2, 3, 4], index=1, key=f"{widget_prefix}_columns")
        else:
            per_page = 12
            columns = 2
        browse_order = st.selectbox(
            "Sample order",
            ("ordered", "shuffled"),
            index=0,
            format_func=lambda value: "ordered" if value == "ordered" else "shuffled",
            key=f"{widget_prefix}_browse_order",
        )
        keyword = st.text_input("Keyword filter", value="", key=f"{widget_prefix}_keyword").strip()
        probe_filter = "all"
        if include_probe_filter:
            probe_filter = st.selectbox(
                "Probe filter",
                ("all", "ok", "failed"),
                index=0,
                key=f"{widget_prefix}_probe_filter",
            )
        filter_scope = "all"
        if include_filter_scope:
            filter_scope = st.selectbox(
                "Filter result",
                ("all", "keep", "reject"),
                index=0,
                key=f"{widget_prefix}_filter_scope",
            )
        extra_sidebar_container = st.container()
        refresh = st.button(
            "Refresh cache",
            key=f"{widget_prefix}_refresh",
            help="Clear Streamlit cache and reload current metadata and summary files.",
        )

    if not available_run_ids and not metadata_override.strip():
        st.error("No run_id directory found for this step. Check the data directory, or use the metadata path override.")
        return None

    metadata_path = resolve_metadata_path(project_dir, stage_dir, step_dir, run_id, metadata_override)
    if show_local_paths:
        metadata_path_label = str(metadata_path)
    else:
        try:
            metadata_path_label = str(
                metadata_path.relative_to(Path(project_dir).expanduser().resolve())
            )
        except ValueError:
            metadata_path_label = metadata_path.name
    st.caption(f"Current metadata path: `{metadata_path_label}`")

    if refresh:
        load_rows.clear()
        load_summary.clear()

    if not metadata_path.exists():
        st.error("Metadata path does not exist. Check run_id or override the path manually.")
        return None

    frame = None
    if load_frame:
        rows = load_rows(str(metadata_path))
        if not rows:
            st.warning("No records were loaded.")
            return None
        frame = rows_to_dataframe(rows)

    return {
        "metadata_path": metadata_path,
        "summary": load_summary(str(resolve_summary_path(metadata_path))) if load_summary_data else {},
        "frame": frame,
        "per_page": per_page,
        "columns": columns,
        "browse_order": browse_order,
        "keyword": keyword,
        "probe_filter": probe_filter,
        "filter_scope": filter_scope,
        "extra_sidebar_container": extra_sidebar_container,
        "refresh": refresh,
        "show_local_paths": show_local_paths,
    }
