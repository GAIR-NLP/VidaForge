from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage2_step1_detect_viewer import render_page as render_stage2_step1_detect
    from .stage2_step2_clip_viewer import render_page as render_stage2_step2_clip
except ImportError:
    from stage2_step1_detect_viewer import render_page as render_stage2_step1_detect
    from stage2_step2_clip_viewer import render_page as render_stage2_step2_clip


def main() -> None:
    st.set_page_config(page_title="Stage2 Metadata Viewer", layout="wide")
    st.title("Stage2 Metadata Viewer")

    with st.sidebar:
        st.header("Navigation")
        step = st.radio(
            "Step",
            ("step1_detect", "step2_clip"),
            index=0,
            format_func=lambda value: {
                "step1_detect": "Step 1 Detect",
                "step2_clip": "Step 2 Clip",
            }.get(value, value),
            key="stage2_viewer_step",
        )
        st.divider()

    if step == "step2_clip":
        render_stage2_step2_clip()
    else:
        render_stage2_step1_detect()


if __name__ == "__main__":
    main()
