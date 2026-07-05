"""Stage 2 Step 1 detect viewer."""

from __future__ import annotations
import json
import mimetypes
import os
import pandas as pd
import secrets
import shutil
import streamlit as st
import streamlit.components.v1 as components
import streamlit.runtime as runtime
import sys
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))

try:
    from .viewer_common import (
        apply_text_probe_filters,
        format_elapsed_seconds,
        load_rows,
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
    )
except ImportError:
    from viewer_common import (
        apply_text_probe_filters,
        format_elapsed_seconds,
        load_rows,
        render_bar_chart,
        render_dataframe_preview,
        render_json_preview,
        render_metadata_selection,
    )

try:
    from .coerce import coerce_float
except ImportError:
    from coerce import coerce_float
from vidaforge.common import join_data_dir

STAGE_DIR = "stage2_segmentation"
STEP_DIR = "step1_detect"
STATIC_DIR = Path(__file__).parent / "static"
PLAYER_TEMPLATE_PATH = STATIC_DIR / "player.html"
CACHE_BUST_TOKEN = secrets.token_hex(8)


def _bucket_duration(duration_sec: float) -> str:
    if duration_sec < 2.0:
        return "<2s"
    if duration_sec <= 10.0:
        return "2~10s"
    return ">10s"


def _prepare_detect_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for column in ("start_sec", "end_sec", "duration_sec", "range_index", "num_ranges_in_video"):
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    if "duration_sec" not in enriched.columns and {"start_sec", "end_sec"} <= set(enriched.columns):
        enriched["duration_sec"] = enriched["end_sec"] - enriched["start_sec"]
    enriched["duration_bucket"] = (
        pd.to_numeric(enriched["duration_sec"], errors="coerce")
        .fillna(-1)
        .map(lambda value: _bucket_duration(float(value)) if value >= 0 else "unknown")
    )
    return enriched.reset_index(drop=True)


def _merge_ticks(ticks: list[float]) -> list[float]:
    merged: list[float] = []
    for tick in sorted(ticks):
        if merged and abs(tick - merged[-1]) <= 1e-6:
            continue
        merged.append(round(float(tick), 6))
    return merged


def _coerce_tick_list(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, list | tuple):
        ticks: list[float] = []
        for item in value:
            try:
                ticks.append(float(item))
            except (TypeError, ValueError):
                continue
        return ticks
    return []


def _coerce_name_list(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _format_detectors(value: object) -> str:
    detector_names = _coerce_name_list(value)
    return "+".join(detector_names) if detector_names else "unknown"


def _ticks_for_video_row(row: pd.Series) -> list[float]:
    detect_ok = row.get("detect_ok", 1)
    if pd.isna(detect_ok) or int(detect_ok) != 1:
        return []

    ticks = _coerce_tick_list(row.get("ticks_sec"))
    if not ticks:
        return []

    duration = coerce_float(row.get("duration_sec"))
    if duration is not None and duration > 0:
        ticks = [min(max(tick, 0.0), duration) for tick in ticks]
        ticks.extend([0.0, duration])
    else:
        ticks = [max(tick, 0.0) for tick in ticks]
    return _merge_ticks(ticks)


def _video_range_stats(row: pd.Series) -> dict[str, float | int | None]:
    ticks = _ticks_for_video_row(row)
    durations = [
        max(0.0, float(end_sec) - float(start_sec))
        for start_sec, end_sec in zip(ticks, ticks[1:])
        if float(end_sec) - float(start_sec) > 1e-6
    ]
    if not durations:
        return {
            "count": 0,
            "long_count": 0,
            "short_count": 0,
            "avg_duration": None,
        }
    return {
        "count": len(durations),
        "long_count": sum(1 for duration_sec in durations if duration_sec > 10.0),
        "short_count": sum(1 for duration_sec in durations if duration_sec < 2.0),
        "avg_duration": sum(durations) / len(durations),
    }


def _attach_video_option_labels(video_summary: pd.DataFrame) -> pd.DataFrame:
    if video_summary.empty:
        return video_summary
    enriched = video_summary.copy()
    stats_by_row = [_video_range_stats(row) for _, row in enriched.iterrows()]
    enriched["_range_count"] = [stats["count"] for stats in stats_by_row]
    enriched["_long_range_count"] = [stats["long_count"] for stats in stats_by_row]
    enriched["_short_range_count"] = [stats["short_count"] for stats in stats_by_row]
    enriched["_avg_range_duration"] = [stats["avg_duration"] for stats in stats_by_row]
    if "detectors" in enriched.columns:
        enriched["_detectors_label"] = [
            _format_detectors(row.get("detectors"))
            for _, row in enriched.iterrows()
        ]
    option_labels: list[str] = []
    for _, row in enriched.iterrows():
        video_id = str(row.get("video_id", "") or "")
        video_path = str(row.get("video_path", "") or "")
        label_parts = []
        if video_id:
            label_parts.append(video_id)
        label_parts.append(f"ranges={int(row.get('_range_count') or 0)}")
        label_parts.append(f"long={int(row.get('_long_range_count') or 0)}")
        label_parts.append(video_path)
        option_labels.append(" | ".join(label_parts))
    enriched["_option_label"] = option_labels
    return enriched


@st.cache_data(show_spinner=False)
def _load_video_summary(metadata_path: str) -> pd.DataFrame:
    metadata_root = Path(metadata_path).expanduser().resolve()
    video_records = load_rows(str(metadata_root), unit="video")
    if video_records:
        return _attach_video_option_labels(pd.DataFrame.from_records(video_records))
    return pd.DataFrame()


def _build_detect_ranges_from_video_row(row: pd.Series) -> pd.DataFrame:
    ticks = _ticks_for_video_row(row)
    if len(ticks) < 2:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    total_ranges = len(ticks) - 1
    for index in range(total_ranges):
        start_sec = float(ticks[index])
        end_sec = float(ticks[index + 1])
        rows.append(
            {
                "video_id": row.get("video_id"),
                "video_path": row.get("video_path"),
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "duration_sec": round(max(0.0, end_sec - start_sec), 6),
                "range_index": index,
                "num_ranges_in_video": total_ranges,
                "detectors": row.get("detectors"),
                "_detectors_label": _format_detectors(row.get("detectors")),
                "input_run_id": row.get("input_run_id"),
                "run_id": row.get("run_id"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return _prepare_detect_ranges(pd.DataFrame.from_records(rows))


def _order_video_options(
    video_summary: pd.DataFrame,
    *,
    browse_order: str,
) -> pd.DataFrame:
    if video_summary.empty:
        return video_summary
    if browse_order == "shuffled":
        return video_summary.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return video_summary.reset_index(drop=True)


def _ensure_static_video(source: Path) -> Path:
    STATIC_DIR.mkdir(exist_ok=True)
    target = STATIC_DIR / source.name

    if target.exists():
        same_file = False
        try:
            same_file = source.samefile(target)
        except FileNotFoundError:
            same_file = False
        if same_file:
            return target
        target.unlink()

    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "video/mp4"


def _apply_base_url_path(url: str) -> str:
    base_url_path = (st.get_option("server.baseUrlPath") or "").strip("/")
    prefix = f"/{base_url_path}" if base_url_path else ""
    if url.startswith(("http://", "https://", "data:")):
        return url
    return f"{prefix}{url if url.startswith('/') else f'/{url}'}"


def _append_query_params(url: str, params: dict[str, str]) -> str:
    split = urlsplit(url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def _build_video_url(path: Path, mimetype: str) -> str:
    if runtime.exists():
        media_url = runtime.get_instance().media_file_mgr.add(
            str(path),
            mimetype,
            f"custom-video-player::{path.resolve()}",
        )
        return _append_query_params(
            _apply_base_url_path(media_url),
            {"v": CACHE_BUST_TOKEN},
        )

    served_video = _ensure_static_video(path)
    return _append_query_params(
        _apply_base_url_path(f"/app/static/{quote(served_video.name)}"),
        {"v": CACHE_BUST_TOKEN},
    )


def _render_inline_player(*, height: int, player_config: dict[str, object], video_url: str) -> None:
    player_html = PLAYER_TEMPLATE_PATH.read_text(encoding="utf-8")
    bootstrap_script = (
        "<script>"
        f"window.__PLAYER_VIDEO_URL__ = {json.dumps(video_url)};"
        f"window.__PLAYER_CONFIG__ = {json.dumps(player_config)};"
        "</script>"
    )
    inline_html = player_html.replace("<head>", f"<head>\n  {bootstrap_script}", 1)
    components.html(inline_html, height=height, scrolling=False)


def _render_html_range_player(
    *,
    video_path: str,
    tick_points: list[float],
    timeline_end: float | None,
    autoplay: bool = False,
    play_token: int = 0,
) -> None:
    source_video = join_data_dir(video_path)
    if not source_video.exists() or not source_video.is_file():
        st.warning(f"Video file does not exist; player skipped: {video_path}")
        return
    video_mime = _guess_mime_type(source_video)
    try:
        video_url = _build_video_url(source_video, video_mime)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Video file cannot be opened; player skipped: {video_path}")
        st.caption(str(exc))
        return

    player_config = {
        "video": video_url,
        "ticks": tick_points,
        "selected_start": "",
        "selected_end": "",
        "autoplay": "1" if autoplay else "0",
        "play_token": str(play_token),
        "timeline_end": f"{timeline_end:.6f}" if timeline_end is not None else "",
    }
    _render_inline_player(height=720, player_config=player_config, video_url=video_url)


def render_page() -> None:
    context = render_metadata_selection(
        stage_dir=STAGE_DIR,
        step_dir=STEP_DIR,
        title="Stage 2 Step 1 Detect",
        include_filter_scope=False,
        include_probe_filter=False,
        include_browse_layout_controls=False,
        load_frame=False,
    )
    if context is None:
        return

    video_summary = _load_video_summary(str(context["metadata_path"]))
    if video_summary.empty:
        st.warning("No video-level index records were loaded.")
        return

    video_summary = apply_text_probe_filters(
        video_summary,
        keyword=context["keyword"],
        probe_filter=context["probe_filter"],
        extra_text_cols=(
            "video_id",
            "raw_path",
            "_detectors_label",
            "input_run_id",
            "run_id",
        ),
    )
    if video_summary.empty:
        st.warning("No videos match the current filters.")
        return

    summary = context["summary"]
    total_ranges = int(pd.to_numeric(video_summary["_range_count"], errors="coerce").fillna(0).sum())
    total_videos = len(video_summary)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Detect Ranges", total_ranges)
    metric_cols[1].metric("Videos", total_videos)
    metric_cols[2].metric("Detect OK", summary.get("ok_count", "-"))
    metric_cols[3].metric("Detect Failed", summary.get("failed_count", "-"))
    metric_cols[4].metric("Elapsed", format_elapsed_seconds(summary.get("elapsed_sec")))

    with st.expander("summary.json", expanded=False):
        render_json_preview(summary)

    video_summary = _order_video_options(
        video_summary,
        browse_order=str(context["browse_order"]),
    )
    option_labels = video_summary["_option_label"].tolist()
    video_label_key = f"{STAGE_DIR}_{STEP_DIR}_video_label"
    video_page_key = f"{STAGE_DIR}_{STEP_DIR}_video_page"
    video_page_size_key = f"{STAGE_DIR}_{STEP_DIR}_video_page_size"
    video_nav_delta_key = f"{STAGE_DIR}_{STEP_DIR}_video_nav_delta"
    if video_nav_delta_key not in st.session_state:
        st.session_state[video_nav_delta_key] = 0
    if video_page_size_key not in st.session_state:
        st.session_state[video_page_size_key] = 100
    pending_video_delta = int(st.session_state[video_nav_delta_key])
    current_video_label = str(st.session_state.get(video_label_key, option_labels[0]))
    if current_video_label not in option_labels:
        current_video_label = option_labels[0]
    current_video_index = option_labels.index(current_video_label)
    page_size = int(st.session_state.get(video_page_size_key, 100) or 100)
    total_pages = max(1, math.ceil(len(option_labels) / page_size))
    current_page = int(st.session_state.get(video_page_key, current_video_index // page_size + 1) or 1)
    current_page = max(1, min(current_page, total_pages))
    st.session_state[video_page_key] = current_page
    if pending_video_delta != 0:
        next_video_index = max(0, min(current_video_index + pending_video_delta, len(option_labels) - 1))
        st.session_state[video_label_key] = option_labels[next_video_index]
        st.session_state[video_page_key] = next_video_index // page_size + 1
        st.session_state[video_nav_delta_key] = 0
        current_video_label = option_labels[next_video_index]
        current_video_index = next_video_index

    with context["extra_sidebar_container"]:
        page_size = st.selectbox(
            "Videos per page",
            (100, 200, 500),
            index=(100, 200, 500).index(int(st.session_state.get(video_page_size_key, 100) or 100)),
            key=video_page_size_key,
        )
        total_pages = max(1, math.ceil(len(option_labels) / int(page_size)))
        current_page = int(st.session_state.get(video_page_key, current_video_index // int(page_size) + 1) or 1)
        current_page = max(1, min(current_page, total_pages))
        st.session_state[video_page_key] = current_page
        current_page = int(
            st.number_input(
                "Video Page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                key=video_page_key,
            )
        )
        start_index = (current_page - 1) * int(page_size)
        end_index = min(start_index + int(page_size), len(option_labels))
        current_page_labels = option_labels[start_index:end_index]
        if current_video_label not in current_page_labels:
            current_video_label = current_page_labels[0]
            st.session_state[video_label_key] = current_video_label
        selected_label = st.selectbox(
            "Video Selection",
            current_page_labels,
            key=video_label_key,
        )
        st.caption(f"Current page video range: {start_index + 1}-{end_index} / {len(option_labels)}")

    current_video_index = option_labels.index(selected_label)

    selected_video_row = video_summary[video_summary["_option_label"] == selected_label].iloc[0]
    selected_video_path = str(selected_video_row["video_path"])
    selected_ticks = _ticks_for_video_row(selected_video_row)
    video_duration = coerce_float(selected_video_row.get("duration_sec"))
    selected_ranges = _build_detect_ranges_from_video_row(selected_video_row)
    if not selected_ranges.empty:
        selected_ranges = selected_ranges.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    tabs = st.tabs(["Detect Range Review", "Global Distribution", "Detect Range Details"])
    with tabs[0]:
        preview_cols = st.columns([1.4, 1.0])
        with preview_cols[0]:
            if selected_ticks:
                _render_html_range_player(
                    video_path=selected_video_path,
                    tick_points=selected_ticks,
                    timeline_end=video_duration,
                    autoplay=False,
                    play_token=0,
                )
            st.caption(
                "The player shows the full timeline and true cut boundaries for the current video."
            )
            st.caption(selected_video_path)
        with preview_cols[1]:
            nav_cols = st.columns(2)
            with nav_cols[0]:
                if st.button(
                    "Previous video",
                    width="stretch",
                    disabled=current_video_index <= 0,
                    key=f"{STAGE_DIR}_{STEP_DIR}_prev_video",
                ):
                    st.session_state[video_nav_delta_key] = -1
                    st.rerun()
            with nav_cols[1]:
                if st.button(
                    "Next video",
                    width="stretch",
                    disabled=current_video_index >= len(option_labels) - 1,
                    key=f"{STAGE_DIR}_{STEP_DIR}_next_video",
                ):
                    st.session_state[video_nav_delta_key] = 1
                    st.rerun()

            st.subheader("Current Video Summary")
            st.metric("Detect Ranges", int(selected_video_row["_range_count"]))
            st.metric("Long Detect Ranges (>10s)", int(selected_video_row["_long_range_count"]))
            st.metric("Short Detect Ranges (<2s)", int(selected_video_row["_short_range_count"]))
            avg_duration = selected_video_row.get("_avg_range_duration")
            st.metric(
                "Mean Detect Range Duration",
                f"{float(avg_duration):.2f}s" if pd.notna(avg_duration) else "-",
            )
            st.metric(
                "Video Duration",
                f"{float(video_duration):.2f}s" if video_duration is not None else "-",
            )
            if "input_run_id" in selected_video_row or "run_id" in selected_video_row:
                st.caption(
                    "run: "
                    f"input={selected_video_row.get('input_run_id', '') or '-'} / "
                    f"current={selected_video_row.get('run_id', '') or '-'}"
                )
            if selected_ranges.empty:
                st.info("Current video has no playable detect range.")
    with tabs[1]:
        duration_counts = pd.DataFrame(
            {
                "duration_bucket": ["<2s", "2~10s", ">10s"],
                "count": [
                    int(pd.to_numeric(video_summary["_short_range_count"], errors="coerce").fillna(0).sum()),
                    int(
                        (
                            pd.to_numeric(video_summary["_range_count"], errors="coerce").fillna(0)
                            - pd.to_numeric(video_summary["_short_range_count"], errors="coerce").fillna(0)
                            - pd.to_numeric(video_summary["_long_range_count"], errors="coerce").fillna(0)
                        ).sum()
                    ),
                    int(pd.to_numeric(video_summary["_long_range_count"], errors="coerce").fillna(0).sum()),
                ],
            }
        )
        render_bar_chart(
            duration_counts,
            category_col="duration_bucket",
            value_col="count",
            title="Detect Range Duration Buckets",
        )

        if "_detectors_label" in video_summary.columns:
            detector_counts = (
                video_summary["_detectors_label"]
                .fillna("")
                .astype(str)
                .replace("", "unknown")
                .value_counts(dropna=False)
                .rename_axis("detectors")
                .reset_index(name="count")
            )
            render_bar_chart(
                detector_counts,
                category_col="detectors",
                value_col="count",
                title="Detector Distribution",
            )

    with tabs[2]:
        display_cols = [
            col
            for col in (
                "range_id",
                "range_index",
                "start_sec",
                "end_sec",
                "duration_sec",
                "duration_bucket",
                "detectors",
                "input_run_id",
                "run_id",
                "video_id",
                "video_path",
            )
            if col in selected_ranges.columns
        ]
        render_dataframe_preview(selected_ranges[display_cols], width="stretch", height=420)
