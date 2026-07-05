from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage1_step1_probe_viewer import render_page as render_step1_probe
    from .stage1_step2_screen_viewer import render_page as render_step2_screen
    from .stage1_step3_transcode_viewer import render_page as render_step3_transcode
except ImportError:
    from stage1_step1_probe_viewer import render_page as render_step1_probe
    from stage1_step2_screen_viewer import render_page as render_step2_screen
    from stage1_step3_transcode_viewer import render_page as render_step3_transcode


def main() -> None:
    st.set_page_config(page_title="Stage1 Metadata Viewer", layout="wide")
    st.title("Stage1 Metadata Viewer")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select page",
            (
                "Stage 1 Step 1: Probe",
                "Stage 1 Step 2: Screen",
                "Stage 1 Step 3: Transcode",
            ),
            index=0,
        )
        st.divider()

    if page == "Stage 1 Step 1: Probe":
        render_step1_probe()
    elif page == "Stage 1 Step 2: Screen":
        render_step2_screen()
    else:
        render_step3_transcode()


if __name__ == "__main__":
    main()
