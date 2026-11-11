"""Light/dark theme toggle (Streamlit custom component v2, inline).

How theming works
* .streamlit/config.toml defines ``[theme.light]`` and ``[theme.dark]``, so
  every native Streamlit widget is styled by Streamlit itself in both modes.
  Streamlit remembers the chosen appearance in the browser and defaults to the
  operating-system preference.
* This component reads the *active* Streamlit theme from the ``--st-*`` CSS
  variables Streamlit provides to custom components and mirrors it onto
  ``<html data-ds-theme>``, which switches DataStory's own CSS tokens. It never
  touches Streamlit's internal controls.
* The sun/moon button stores the requested appearance under the browser key
  Streamlit itself reads for its theme preference and reloads the page. A
  reload starts a new Streamlit session, so the current analysis is first
  parked in memory under a one-time random token (core/handoff.py) and claimed
  by the new session — nothing is written to disk and the entry expires after
  two minutes. If browser storage is blocked, the button reports that instead.
"""
from __future__ import annotations

import streamlit as st

_HTML = """
<button class="ds-theme" type="button" aria-label="Toggle light or dark theme" title="Toggle light / dark theme">
  <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/>
  <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
  <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
</button>
"""

_CSS = """
.ds-theme{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:10px;cursor:pointer;
  border:1px solid var(--st-border-color, #DDE3D3);background:var(--st-secondary-background-color, #F0F2E8);
  color:var(--st-text-color, #1B221B);transition:transform .15s, border-color .15s}
.ds-theme:hover{transform:translateY(-1px);border-color:var(--st-primary-color, #5E7845)}
.ds-theme:focus-visible{outline:2px solid var(--st-primary-color, #5E7845);outline-offset:2px}
.ds-theme svg{width:18px;height:18px}
.ds-theme .sun{display:none}
.ds-theme.dark .sun{display:block}
.ds-theme.dark .moon{display:none}
@media (prefers-reduced-motion: reduce){.ds-theme{transition:none}}
"""

_JS = """
function luminance(color) {
  const c = document.createElement("canvas").getContext("2d");
  if (!c) return 1;
  c.fillStyle = color; const hex = c.fillStyle;
  if (!hex.startsWith("#") || hex.length < 7) return 1;
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const btn = parentElement.querySelector(".ds-theme");
  if (!btn) return;
  btn.disabled = false;
  const detect = () => {
    const bg = getComputedStyle(btn).getPropertyValue("--st-background-color").trim();
    const mode = bg ? (luminance(bg) < 0.4 ? "dark" : "light") : null;
    if (mode) {
      if (document.documentElement.dataset.dsTheme !== mode) document.documentElement.dataset.dsTheme = mode;
      btn.classList.toggle("dark", mode === "dark");
    }
    return mode;
  };
  detect();
  if (!window.__dsThemeTimer) window.__dsThemeTimer = setInterval(detect, 700);
  const pending = data && data.pending;
  if (pending && window.__dsThemeHandled !== pending.token) {
    window.__dsThemeHandled = pending.token;
    try {
      const paths = (data && data.paths) || [""];
      const path = window.location.pathname;
      let base = path.endsWith("/") ? path : path + "/";
      for (const p of [...paths].sort((a, b) => b.length - a.length)) {
        if (p && (path === "/" + p || path.endsWith("/" + p))) { base = path.slice(0, path.length - p.length); break; }
      }
      if (!base.endsWith("/")) base += "/";
      const keys = new Set([path, base, base.replace(/\/+$/, "") || "/"]);
      for (const p of paths) keys.add((base + p).replace(/\/+$/, "") || "/");
      for (const k of keys) window.localStorage.setItem(`stActiveTheme-${k}-v2`, JSON.stringify(pending.theme));
      document.documentElement.dataset.dsTheme = pending.theme.toLowerCase();
      const url = new URL(window.location.href);
      url.searchParams.set("restore", pending.token);
      window.location.replace(url.toString());
    } catch (e) {
      btn.title = "Your browser blocked theme storage, so the theme could not be switched.";
    }
    return;
  }
  btn.onclick = () => {
    btn.disabled = true;
    setTriggerValue("toggle", detect() === "dark" ? "Light" : "Dark");
  };
}
"""

_TOGGLE = st.components.v2.component("ds_theme_toggle", html=_HTML, css=_CSS, js=_JS)


HANDOFF_KEYS = ("_report", "_nb", "previous_training_results", "_rec_open", "_nl_spec", "_nl_assume")


def _on_toggle() -> None:
    """Park this session's analysis for the reload that applies the new theme."""
    from core import handoff, state

    raw = st.session_state.get("ds_theme")
    val = raw.get("toggle") if hasattr(raw, "get") else getattr(raw, "toggle", None)
    if val not in ("Light", "Dark"):
        return
    keys = list(state._DEFAULTS) + list(HANDOFF_KEYS)
    payload = {k: st.session_state[k] for k in keys if k in st.session_state}
    st.session_state["_theme_pending"] = {"token": handoff.park(payload), "theme": val}


def restore_after_reload() -> bool:
    """Claim the analysis parked by the theme toggle (called once at app start)."""
    from core import handoff

    token = st.query_params.get("restore")
    if not token:
        return False
    payload = handoff.claim(str(token))
    del st.query_params["restore"]
    if payload is None:
        return False
    for k, v in payload.items():
        st.session_state[k] = v
    return True


def theme_toggle(paths: list[str], key: str = "ds_theme") -> None:
    """Render the sun/moon toggle. ``paths`` are the url paths of every page (default page = "")."""
    pending = st.session_state.pop("_theme_pending", None)
    _TOGGLE(data={"paths": paths, "pending": pending}, key=key, on_toggle_change=_on_toggle)


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - not running in a browser session (tests)
        return False
