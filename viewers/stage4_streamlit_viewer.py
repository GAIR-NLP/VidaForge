from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage4_step1_camera_viewer import render_page as render_step1_camera
    from .stage4_step2_caption_viewer import render_page as render_step2_caption
    from .stage4_step3_tag_viewer import render_page as render_step3_tag
except ImportError:
    from stage4_step1_camera_viewer import render_page as render_step1_camera
    from stage4_step2_caption_viewer import render_page as render_step2_caption
    from stage4_step3_tag_viewer import render_page as render_step3_tag


def main() -> None:
    st.set_page_config(page_title="Stage4 Annotation Viewer", layout="wide")
    st.title("Stage4 Annotation Viewer")

    with st.sidebar:
        st.header("Navigation")
        step = st.radio(
            "Step",
            ("step1_camera", "step2_caption", "step3_tag"),
            index=0,
            format_func=lambda value: {
                "step1_camera": "Step 1 Camera",
                "step2_caption": "Step 2 Caption",
                "step3_tag": "Step 3 Tag",
            }.get(value, value),
            key="stage4_viewer_step",
        )
        st.divider()

    if step == "step1_camera":
        render_step1_camera()
    elif step == "step2_caption":
        render_step2_caption()
    elif step == "step3_tag":
        render_step3_tag()


if __name__ == "__main__":
    main()
