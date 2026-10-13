"""Lookup a unit and see the page or call the answer came from.

Run: uv run --extra ml --extra app streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from fieldnote.config import get_settings
from fieldnote.pipeline import load_pipeline
from fieldnote.types import format_ts

st.set_page_config(page_title="Fieldnote", layout="wide", initial_sidebar_state="collapsed")

EXAMPLES = [
    {
        "label": "38AU at 400 psig. What liquid temp?",
        "question": (
            "I'm charging a Carrier 38AUZA07 6-ton condensing unit (R-410A) and my gauge "
            "reads 400 psig at the liquid valve. What liquid-line temperature at the valve "
            "should I be seeing if the charge is correct?"
        ),
    },
    {
        "label": "Penn P70E. Which terminal is line, alarm, load?",
        "question": (
            "I'm wiring a Penn P70E SPDT low-pressure control to run a load and an alarm. "
            "Which switch terminal gets line (L1), which one gets the alarm, and which one "
            "gets the load?"
        ),
    },
    {
        "label": "CoreSense code 3. What split means a dirty coil?",
        "question": (
            "On a Copeland CoreSense code 3 lockout, what condensing-to-ambient split did "
            "Copeland support say points to a dirty coil or airflow problem instead of a "
            "bad pressure switch?"
        ),
    },
]

KIND = {
    "manual_figure": "Figure",
    "manual_page": "Page",
    "manual_text": "Text",
    "photo": "Photo",
    "call": "Call",
}

LANE = {
    "text": "In the manuals",
    "image": "In the figures",
    "audio": "In the calls",
    "page": "On a full page",
}

st.markdown(
    """
    <style>
      [data-testid="stToolbar"] { display: none; }
      div[data-testid="stImage"] img {
        border: 1px solid #ddd;
        background: white;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline(mode: str):
    return load_pipeline(get_settings(), mode)  # type: ignore[arg-type]


def pick(question: str) -> None:
    st.session_state.question = question


if "question" not in st.session_state:
    st.session_state.question = ""

st.title("Fieldnote")
st.write("Look up a unit. The page or the call is next to the answer.")

for example in EXAMPLES:
    st.button(example["label"], on_click=pick, args=(example["question"],))

question = st.text_area(
    "Question",
    key="question",
    height=88,
    placeholder="Unit 7 keeps tripping on high pressure. What should head pressure be at 95°F?",
)
captions = st.checkbox(
    "Search captions instead of the picture",
    help="Uses a written description of each chart instead of the image.",
)
mode = "caption" if captions else "native"

if not question.strip():
    st.stop()

with st.spinner("Looking through the manuals and calls…"):
    try:
        pipe = get_pipeline(mode)
        ans = pipe.ask(question.strip())
    except (RuntimeError, BlockingIOError, OSError) as exc:
        if "already accessed" in str(exc) or isinstance(exc, BlockingIOError):
            st.error("The index is open in another window. Close that window and ask again.")
        else:
            st.error(str(exc))
        st.stop()

answer, evidence = st.columns([1.05, 1], gap="large")
with answer:
    st.subheader("Answer")
    st.markdown(ans.text)
    total = ans.timings.get("total_s", 0)
    st.caption(f"{total:.0f} seconds")

with evidence:
    st.subheader("Source")
    if not ans.citations:
        st.write("Nothing was cited.")
    for cit in ans.citations:
        chunk = cit.candidate.chunk
        kind = KIND.get(chunk.source_type, chunk.source_type)
        st.markdown(f"**{cit.label}**")
        st.caption(kind)
        if cit.image_path:
            st.image(cit.image_path, width="stretch")
        elif chunk.source_type == "call":
            audio = get_settings().data_dir / "calls" / chunk.source
            if audio.exists():
                st.audio(str(audio), start_time=int(chunk.start_s or 0))
            window = format_ts(chunk.start_s or 0)
            if chunk.end_s is not None:
                window += f"–{format_ts(chunk.end_s)}"
            st.caption(f"{window}. {chunk.text[:360]}")

retrieval = pipe.last_retrieval
if retrieval is not None and any(retrieval.per_lane.values()):
    st.subheader("Also looked at")
    lane_cols = st.columns(len(retrieval.per_lane) or 1)
    for col, (name, hits) in zip(lane_cols, retrieval.per_lane.items(), strict=False):
        with col:
            st.markdown(f"**{LANE.get(name, name)}**")
            if not hits:
                st.caption("Nothing.")
                continue
            top = hits[0].chunk
            st.caption(top.citation_label())
