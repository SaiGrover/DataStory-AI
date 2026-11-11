"""Step 8 — dataset assistant grounded in the computed analysis."""
from __future__ import annotations

import html

import streamlit as st

from ai import assistant, llm
from components import ui
from core import state

if not ui.require_data("ask questions about it"):
    st.stop()

ui.page_header("Step 8 · Agent", "DataStory AI assistant",
               "Ask about the active dataset and its analysis. Answers come from computed summaries and cite the calculation they rely on.")
ui.synthetic_notice()

cfg = llm.get_config()
if cfg:
    st.html(f'<span class="ds-pill ok">LLM connected · {html.escape(cfg.model)}</span> '
            '<span style="font-size:.85rem;color:#697468">The model only sees verified facts; answers with untraceable numbers are replaced by the facts themselves.</span>')
else:
    st.html('<span class="ds-pill muted">Rule-based mode</span> '
            '<span style="font-size:.85rem;color:#697468">No LLM key configured. The assistant answers by retrieving the relevant computed facts. '
            'Add GROQ_API_KEY to the app secrets to enable conversational phrasing.</span>')

r = state.results()
chat = st.session_state["chat"]

SUGGESTIONS = ["What are the major quality issues?", "Which columns contain the most missing values?", "Is the dataset imbalanced?",
               "Which features are correlated?", "Which model performed best?", "What do the evaluation metrics mean?",
               "What cleaning steps should I consider?", "Which features influenced model predictions?"]


def ask(q: str) -> None:
    chat.append({"role": "user", "content": q})
    with st.spinner("Looking up the analysis…"):
        chat.append({"role": "assistant", **assistant.answer(q, r)})


if not chat:
    st.markdown("**Try a question**")
    cols = st.columns(4)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 4].button(s, key=f"sugg_{i}", width="stretch"):
            ask(s)
            st.rerun()

for msg in chat:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:"):
        st.markdown(msg.get("answer") or msg.get("content", ""))
        if msg.get("note"):
            st.caption(f":material/shield: {msg['note']}")
        if msg.get("sources"):
            with st.expander(f"Supported by {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

q = st.chat_input("Ask about this dataset…")
if q:
    ask(q)
    st.rerun()

if chat:
    c1, c2 = st.columns([1, 5])
    if c1.button("Clear chat", icon=":material/delete:"):
        st.session_state["chat"] = []
        st.rerun()
    c2.caption("Answers reflect the analysis as it is now. The assistant cannot see individual rows and will say so when a question is outside the computed analysis.")
