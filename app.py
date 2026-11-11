"""DataStory AI — Streamlit entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from components import navigation, styling
from core import state

st.set_page_config(page_title="DataStory AI", page_icon=":material/bar_chart:", layout="wide",
                   initial_sidebar_state="collapsed")

state.init()
styling.inject()

pages = navigation.build_pages()
current = st.navigation(pages, position="hidden")

navigation.header()
navigation.stepper(pages, current)

try:
    current.run()
except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
    st.error(f"Something went wrong while rendering this page: {type(exc).__name__}: {exc}")
    st.caption("Your dataset and previous results are still in memory. Try another step, or reload the dataset.")
