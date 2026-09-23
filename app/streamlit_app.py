"""Demo UI: ask a question, see the answer next to the chart it came from.

Run: uv run --extra ml --extra app streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from fieldnote.config import get_settings
from fieldnote.pipeline import load_pipeline
from fieldnote.types import format_ts

st.set_page_config(page_title="Fieldnote", layout="wide")


@st.cache_resource
def get_pipeline(mode: str):
    return load_pipeline(get_settings(), mode)  # type: ignore[arg-type]


st.title("Fieldnote")
st.caption("Ask about the equipment. Answers cite the manual page, photo or support call.")

mode = st.sidebar.radio("System", ["native", "caption"], help="caption = the baseline")
question = st.text_input(
    "Question",
    placeholder="Unit 7 keeps tripping on high pressure. What should head pressure be at 95°F?",
)

if question:
    with st.spinner("Searching manuals, photos and calls..."):
        ans = get_pipeline(mode).ask(question)
    st.markdown(ans.text)
    st.caption(" · ".join(f"{k.removesuffix('_s')} {v:.2f}s" for k, v in ans.timings.items()))

    if ans.citations:
        st.subheader("Evidence")
        cols = st.columns(min(3, len(ans.citations)))
        for col, cit in zip(cols * 3, ans.citations, strict=False):
            chunk = cit.candidate.chunk
            with col:
                st.markdown(f"**[{cit.sid}]** {cit.label}")
                if cit.image_path:
                    st.image(cit.image_path, use_container_width=True)
                elif chunk.source_type == "call":
                    audio = get_settings().data_dir / "calls" / chunk.source
                    if audio.exists():
                        st.audio(str(audio), start_time=int(chunk.start_s or 0))
                    st.caption(f"{format_ts(chunk.start_s or 0)}: {chunk.text[:300]}")
