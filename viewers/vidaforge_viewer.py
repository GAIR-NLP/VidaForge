"""Unified Streamlit entrypoint for all stage viewers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class StepViewer:
    key: str
    label: str
    module: str
    function: str = "render_page"


@dataclass(frozen=True)
class StageViewer:
    key: str
    label: str
    steps: tuple[StepViewer, ...]


VIEWERS: tuple[StageViewer, ...] = (
    StageViewer(
        key="stage1",
        label="Stage 1 Ingestion",
        steps=(
            StepViewer("step1_probe", "Step 1 Probe", "viewers.stage1_step1_probe_viewer"),
            StepViewer("step2_screen", "Step 2 Screen", "viewers.stage1_step2_screen_viewer"),
            StepViewer("step3_transcode", "Step 3 Transcode", "viewers.stage1_step3_transcode_viewer"),
        ),
    ),
    StageViewer(
        key="stage2",
        label="Stage 2 Segmentation",
        steps=(
            StepViewer("step1_detect", "Step 1 Detect", "viewers.stage2_step1_detect_viewer"),
            StepViewer("step2_clip", "Step 2 Clip", "viewers.stage2_step2_clip_viewer"),
        ),
    ),
    StageViewer(
        key="stage3",
        label="Stage 3 Selection",
        steps=(
            StepViewer("step1_context", "Step 1 Context", "viewers.stage3_step1_context_viewer"),
            StepViewer("step2_filter", "Step 2 Filter", "viewers.stage3_step2_filter_viewer"),
            StepViewer("step3_dedup", "Step 3 Dedup", "viewers.stage3_step3_dedup_viewer"),
            StepViewer("step4_select", "Step 4 Select", "viewers.stage3_step4_select_viewer"),
        ),
    ),
    StageViewer(
        key="stage4",
        label="Stage 4 Annotation",
        steps=(
            StepViewer("step1_camera", "Step 1 Camera", "viewers.stage4_step1_camera_viewer"),
            StepViewer("step2_caption", "Step 2 Caption", "viewers.stage4_step2_caption_viewer"),
            StepViewer("step3_tag", "Step 3 Tag", "viewers.stage4_step3_tag_viewer"),
        ),
    ),
    StageViewer(
        key="stage5",
        label="Stage 5 Packaging",
        steps=(
            StepViewer(
                "automodel_dataset",
                "AutoModel Dataset",
                "viewers.stage5_automodel_dataset_viewer",
            ),
        ),
    ),
)


STAGES_BY_KEY = {stage.key: stage for stage in VIEWERS}


def _load_render_page(step: StepViewer) -> Callable[[], None]:
    module = import_module(step.module)
    render_page = getattr(module, step.function)
    if not callable(render_page):
        raise TypeError(f"{step.module}.{step.function} is not callable")
    return render_page


def main() -> None:
    st.set_page_config(page_title="VidaForge Viewer", layout="wide")
    st.title("VidaForge Viewer")

    with st.sidebar:
        st.header("Navigation")
        stage_key = st.radio(
            "Stage",
            tuple(STAGES_BY_KEY),
            index=0,
            format_func=lambda key: STAGES_BY_KEY[key].label,
            key="videoforge_viewer_stage",
        )
        stage = STAGES_BY_KEY[stage_key]
        steps_by_key = {step.key: step for step in stage.steps}
        step_key = st.radio(
            "Step",
            tuple(steps_by_key),
            index=0,
            format_func=lambda key: steps_by_key[key].label,
            key=f"videoforge_viewer_step_{stage.key}",
        )
        st.divider()

    step = steps_by_key[step_key]
    try:
        render_page = _load_render_page(step)
        render_page()
    except Exception as exc:  # pragma: no cover - visible Streamlit failure boundary.
        st.error(f"Failed to render {stage.label} / {step.label}.")
        st.exception(exc)


if __name__ == "__main__":
    main()
