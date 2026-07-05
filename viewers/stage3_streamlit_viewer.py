from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .stage3_step1_context_viewer import render_page as render_step1_context
    from .stage3_step2_filter_viewer import render_page as render_step2_filter
    from .stage3_step3_dedup_viewer import render_page as render_step3_dedup
    from .stage3_step4_select_viewer import render_page as render_step4_select
except ImportError:
    from stage3_step1_context_viewer import render_page as render_step1_context
    from stage3_step2_filter_viewer import render_page as render_step2_filter
    from stage3_step3_dedup_viewer import render_page as render_step3_dedup
    from stage3_step4_select_viewer import render_page as render_step4_select


def main() -> None:
    st.set_page_config(page_title="Stage3 Selection Viewer", layout="wide")
    st.title("Stage3 Selection Viewer")

    with st.sidebar:
        st.header("Navigation")
        step = st.radio(
            "Step",
            ("step1_context", "step2_filter", "step3_dedup", "step4_select"),
            index=0,
            format_func=lambda value: {
                "step1_context": "Step 1 Context",
                "step2_filter": "Step 2 Filter",
                "step3_dedup": "Step 3 Dedup",
                "step4_select": "Step 4 Select",
            }.get(value, value),
            key="stage3_viewer_step",
        )
        st.divider()

    if step == "step1_context":
        render_step1_context()
    elif step == "step2_filter":
        render_step2_filter()
    elif step == "step3_dedup":
        render_step3_dedup()
    elif step == "step4_select":
        render_step4_select()


if __name__ == "__main__":
    main()
