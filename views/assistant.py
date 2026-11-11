"""Step 8 — analytical copilot: questions in, verified computations (text, tables, charts) out."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ai import copilot, llm
from components import ui
from core import state

if not ui.require_data("ask questions about it"):
    st.stop()

ui.page_header("Step 8 · Agent", "DataStory copilot",
               "Ask questions or request charts in plain language. Each answer comes from a named analysis tool run on the active "
               "dataset — the copilot never executes generated code.")
ui.provenance_notice()

r = state.results()
chat: list = st.session_state["chat"]
cfg = llm.get_config()

top_l, top_r = st.columns([3, 1.4], vertical_alignment="center")
with top_l:
    if cfg:
        st.html(f'<span class="ds-pill ok">LLM connected · {html.escape(cfg.model)}</span> '
                '<span class="ds-meta">The model picks a tool and rephrases its verified output; answers with untraceable numbers fall back to the computed result.</span>')
    else:
        st.html('<span class="ds-pill muted">Rule-based mode</span> '
                '<span class="ds-meta">No LLM key configured — questions are routed to tools by rules. Add GROQ_API_KEY to the app secrets for conversational routing.</span>')
with top_r:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if chat:
            def _transcript() -> str:
                lines = [f"# DataStory copilot — {state.meta().name} ({state.data_version()})", ""]
                for m in chat:
                    who = "You" if m["role"] == "user" else f"Copilot ({m.get('tool') or 'retrieval'})"
                    lines += [f"**{who}:** {m.get('answer') or m.get('content', '')}", ""]
                    if m.get("sources"):
                        lines += ["Sources: " + "; ".join(m["sources"]), ""]
                return "\n".join(lines)
            st.download_button("Export chat", _transcript().encode("utf-8"), "datastory_chat.md", "text/markdown", icon=":material/download:")
            if st.button("New conversation", icon=":material/add_comment:"):
                st.session_state["chat"] = []
                st.session_state["chat_spec"] = None
                st.rerun()


def _ctx() -> copilot.Ctx:
    return copilot.Ctx(state.analysis_df(), r, r["profile"]["kinds"], st.session_state.get("chat_spec"))


def ask(q: str) -> None:
    chat.append({"role": "user", "content": q, "at": datetime.now(timezone.utc).strftime("%H:%M")})
    with st.spinner("Running the analysis…"):
        ans = copilot.ask(q, _ctx(), history=chat)
    ans["data_version"] = state.data_version()
    chat.append(ans)
    if ans.get("spec"):
        st.session_state["chat_spec"] = ans["spec"]


def suggestions() -> list[str]:
    ov = state.overview()
    out = ["What are the biggest issues in this dataset?"]
    if r.get("target"):
        out.append(f"Which features are most strongly associated with {r['target']}?")
    for rec in ov["chart_recommendations"][:2]:
        out.append(f"Show {rec['title'][0].lower() + rec['title'][1:]}")
    if r["profile"]["missing_cells"]:
        out.append("Show missing values by column")
    out.append("Which model performed best?" if r.get("modeling") else "What cleaning steps should I perform?")
    out.append("Which segments are present?" if r.get("clustering") else "Summarize the main findings")
    return out[:6]


prefill = st.session_state.pop("_copilot_prefill", None)
if prefill:
    st.session_state["chat_spec"] = prefill.get("spec")
    ask(prefill["question"] + " — describe what it shows")

if not chat:
    with st.container(border=True):
        st.html('<div style="text-align:center;padding:14px 0 4px"><div class="ds-title" style="font-size:1.45rem">What would you like to explore?</div>'
                f'<div class="ds-sub" style="margin:0 auto">Questions are answered from <b>{html.escape(state.meta().name)}</b> '
                f'({r["profile"]["rows"]:,} rows, version {state.data_version()}). Try one of these:</div></div>')
        cols = st.columns(3)
        for i, s in enumerate(suggestions()):
            if cols[i % 3].button(s, key=f"sugg_{i}", width="stretch"):
                ask(s)
                st.rerun()
        st.caption("Supported: dataset & column summaries, quality issues, missing values, grouped statistics, row counts with conditions, "
                   "correlations, target associations, charts from plain language, model results, confusion matrix, feature importance, "
                   "cleaning suggestions, segments and findings.")

for i, msg in enumerate(chat):
    if msg["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(msg.get("content", ""))
        continue
    with st.chat_message("assistant", avatar=":material/insights:"):
        tool = msg.get("tool")
        if tool and tool not in ("retrieval", "refused", "clarification"):
            via = " · routed by LLM" if msg.get("routed_by") == "llm" else ""
            st.html(f'<span class="ds-tool">⚙ computed with tool <b>{html.escape(tool)}</b>{via} · data {html.escape(msg.get("data_version", ""))}</span>')
        st.markdown(msg.get("answer") or "")
        if msg.get("spec"):
            ui.spec_chart(msg["spec"], key=f"chat_{i}", source="Copilot", show_table=msg.get("table") is None)
        elif msg.get("figure") is not None:
            ui.chart(msg["figure"], f"chatfig_{i}")
        if isinstance(msg.get("table"), pd.DataFrame) and not msg["table"].empty and not msg.get("spec"):
            with st.expander("Table"):
                st.dataframe(msg["table"], width="stretch", hide_index=True)
        if msg.get("note"):
            st.caption(f":material/shield: {msg['note']}")
        if msg.get("data_version") and msg["data_version"] != state.data_version():
            st.caption(f":material/history: Computed on {msg['data_version']}; the data is now {state.data_version()}. Ask again for current values.")
        if msg.get("sources"):
            with st.expander(f"Evidence · {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
        if msg.get("navigate"):
            st.page_link(msg["navigate"][0], label=msg["navigate"][1], icon=":material/arrow_forward:")
        if i == len(chat) - 1 and msg.get("followups"):
            with st.container(horizontal=True):
                for j, f in enumerate(msg["followups"][:3]):
                    if st.button(f, key=f"fu_{i}_{j}"):
                        ask(f)
                        st.rerun()

q = st.chat_input("Ask about this dataset, or describe a chart…")
if q:
    ask(q)
    st.rerun()

if chat:
    st.caption("Answers reflect the data version shown on each message. The copilot sees computed summaries and runs fixed analyses; "
               "it does not browse individual rows and says so when a request is outside its tools.")
