"""DataStory design system for Streamlit.

* Colours live in one place: CSS custom properties for light (default) and dark
  (``html[data-ds-theme="dark"]``, set by the theme component in
  components/theme.py, plus a ``prefers-color-scheme`` fallback before it runs).
  Streamlit's own widgets are themed by .streamlit/config.toml, which mirrors
  the same values.
* Typography: Plus Jakarta Sans (UI and body), Manrope (display, headings,
  metrics), DM Serif Display used sparingly for the hero accent only. Numbers
  use tabular figures.
* Custom selectors target stable hooks only: our own class names and the
  ``st-key-*`` classes Streamlit adds for keyed containers.
"""
from __future__ import annotations

import streamlit as st

TOKENS = {
    "light": {
        "bg": "#FAF8F1", "bg-grad": "#F4F5EC", "surface": "#FFFFFF", "surface-2": "#F4F5EE", "elevated": "#FFFFFF",
        "text": "#1B221B", "text-2": "#5D685C", "text-3": "#7F897D", "border": "#E1E5D8", "border-strong": "#CBD3BF",
        "accent": "#5E7845", "accent-hover": "#4F6A39", "accent-soft": "#EEF2E6", "accent-ink": "#3F5530",
        "active": "#5E7845", "success": "#3F7A4A", "success-soft": "#E8F2E6", "warning": "#8A6414", "warning-soft": "#FBF3DF",
        "warning-border": "#E6CF9A", "error": "#A13D35", "error-soft": "#FBEAE7", "info-soft": "#EFF3F5",
        "shadow": "0 1px 2px rgba(27,34,27,.04), 0 6px 20px rgba(27,34,27,.06)",
        "shadow-lg": "0 2px 6px rgba(27,34,27,.06), 0 16px 36px rgba(27,34,27,.10)",
        "line": "#D7DECB", "glow": "rgba(147,173,116,.55)", "glow-strong": "rgba(255,255,255,.95)",
    },
    "dark": {
        "bg": "#121612", "bg-grad": "#141A14", "surface": "#1A201A", "surface-2": "#20271F", "elevated": "#232B22",
        "text": "#ECEADF", "text-2": "#B4BBAA", "text-3": "#8D9685", "border": "#2E372D", "border-strong": "#3D483B",
        "accent": "#8FAE6E", "accent-hover": "#A5C285", "accent-soft": "#243020", "accent-ink": "#C9DDB1",
        "active": "#8FAE6E", "success": "#8CC48F", "success-soft": "#1E2C1F", "warning": "#E3C27A", "warning-soft": "#2C2617",
        "warning-border": "#5A4A22", "error": "#E59A90", "error-soft": "#331E1B", "info-soft": "#1C2427",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.30)",
        "shadow-lg": "0 2px 6px rgba(0,0,0,.4), 0 18px 40px rgba(0,0,0,.45)",
        "line": "#38433A", "glow": "rgba(165,194,133,.45)", "glow-strong": "rgba(220,240,200,.9)",
    },
}


def _vars(theme: str) -> str:
    return ";".join(f"--ds-{k}:{v}" for k, v in TOKENS[theme].items())


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Manrope:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
:root{%(light)s;--ds-font:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --ds-display:Manrope,'Plus Jakarta Sans',system-ui,sans-serif;--ds-serif:'DM Serif Display',Georgia,serif;
  --ds-radius:14px;--ds-radius-sm:10px;--ds-ease:cubic-bezier(.2,.8,.2,1)}
@media (prefers-color-scheme: dark){html:not([data-ds-theme]){%(dark)s}}
html[data-ds-theme="dark"]{%(dark)s}
html[data-ds-theme="light"]{%(light)s}

.stApp{background:linear-gradient(180deg,var(--ds-bg) 0%%,var(--ds-bg-grad) 100%%);color:var(--ds-text)}
.stApp, .stApp p, .stApp li, .stApp label{font-family:var(--ds-font)}
.stApp p, .stApp li{line-height:1.6}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:var(--ds-display);color:var(--ds-text);letter-spacing:-.01em}
.stApp h2{padding-top:.5rem}.stApp h3{padding-top:.25rem}
[data-testid="stMetricValue"], .ds-num, .ds-stat .v, .stApp td{font-variant-numeric:tabular-nums}
.stMainBlockContainer{max-width:1320px;padding-left:2rem;padding-right:2rem}
[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none}
:focus-visible{outline:2px solid var(--ds-accent)!important;outline-offset:2px}

/* ------------------------------------------------------------------ header */
.st-key-ds_header{align-items:center;gap:12px;padding:2px 0 10px;border-bottom:1px solid var(--ds-border)}
.ds-brand{display:flex;align-items:center;gap:11px;line-height:1;color:var(--ds-text);white-space:nowrap}
.ds-brand svg{width:30px;height:30px;color:var(--ds-accent);flex:none}
.ds-brand .w{font-family:var(--ds-display);font-weight:800;font-size:1.42rem;letter-spacing:-.02em}
.ds-brand .w b{color:var(--ds-accent);font-weight:800}
.ds-brand .d{display:block;font-family:var(--ds-font);font-size:.72rem;font-weight:600;color:var(--ds-text-3);letter-spacing:.04em;
  text-transform:uppercase;margin-top:3px}
.ds-context{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;justify-content:flex-end;font-size:.8rem;color:var(--ds-text-2);
  min-width:0}
.ds-context b{color:var(--ds-text);font-weight:650}
.ds-context .name{max-width:28ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block;vertical-align:bottom}
.ds-dot{width:7px;height:7px;border-radius:50%%;display:inline-block;background:var(--ds-accent);margin-right:6px}
.ds-dot.warn{background:var(--ds-warning)}
.st-key-ds_header [data-testid="stPopover"] button{min-height:36px;padding:4px 12px;font-size:.82rem}

/* -------------------------------------------------------- flowchart nav */
.st-key-ds_flow{flex-wrap:nowrap!important;overflow-x:auto;overflow-y:visible;gap:0!important;padding:14px 4px 6px;
  scrollbar-width:thin;align-items:center}
.st-key-ds_flow > div{flex:0 0 auto!important;width:auto!important;min-width:0}
[class*="st-key-flow_"]{position:relative;margin-right:30px;width:auto!important;flex:0 0 auto!important}
[class*="st-key-flow_"]:last-child{margin-right:0}
[class*="st-key-flow_"] a{display:flex;align-items:center;gap:6px;border:1px solid var(--ds-border);background:var(--ds-surface);
  border-radius:999px;padding:6px 13px 6px 10px!important;min-height:36px;white-space:nowrap;box-shadow:none;
  transition:transform .18s var(--ds-ease),box-shadow .18s var(--ds-ease),border-color .18s,background .18s}
[class*="st-key-flow_"] a p{font-size:.8rem!important;font-weight:600!important;color:var(--ds-text-2)!important;margin:0}
[class*="st-key-flow_"] a span[data-testid="stIconMaterial"]{font-size:1.05rem;color:var(--ds-text-3)}
[class*="st-key-flow_"] a:hover{transform:translateY(-1px);border-color:var(--ds-accent);box-shadow:var(--ds-shadow)}
[class*="st-key-flow_"] a:hover p{color:var(--ds-text)!important}
/* connector: base line + arrowhead */
[class*="st-key-flow_"]::after{content:"";position:absolute;top:50%%;right:-27px;width:22px;height:2px;margin-top:-1px;
  border-radius:2px;background:var(--ds-line);pointer-events:none}
[class*="st-key-flow_"]::before{content:"";position:absolute;top:50%%;right:-28px;margin-top:-4px;border:4px solid transparent;
  border-left:5px solid var(--ds-line);border-right:0;pointer-events:none;z-index:1}
[class*="st-key-flow_"]:last-child::after,[class*="st-key-flow_"]:last-child::before{display:none}
/* done */
[class*="st-key-flow_done_"] a{background:var(--ds-accent-soft);border-color:color-mix(in srgb,var(--ds-accent) 45%%,var(--ds-border))}
[class*="st-key-flow_done_"] a p{color:var(--ds-accent-ink)!important}
[class*="st-key-flow_done_"] a span[data-testid="stIconMaterial"]{color:var(--ds-accent)}
[class*="st-key-flow_done_"]::after,[class*="st-key-flow_active_"]::after{
  background:linear-gradient(90deg,var(--ds-accent) 0%%,var(--ds-accent) 35%%,var(--ds-glow-strong) 50%%,var(--ds-accent) 65%%,var(--ds-accent) 100%%);
  background-size:300%% 100%%;animation:ds-flow 2.8s linear infinite}
[class*="st-key-flow_done_"]::before,[class*="st-key-flow_active_"]::before{border-left-color:var(--ds-accent)}
[class*="st-key-flow_active_"]::after{animation-duration:1.9s;box-shadow:0 0 8px var(--ds-glow)}
/* active */
[class*="st-key-flow_active_"] a{background:var(--ds-accent);border-color:var(--ds-accent);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--ds-accent) 18%%,transparent),var(--ds-shadow)}
[class*="st-key-flow_active_"] a p{color:#fff!important;font-weight:700!important}
html[data-ds-theme="dark"] [class*="st-key-flow_active_"] a p{color:#10150F!important}
[class*="st-key-flow_active_"] a span[data-testid="stIconMaterial"]{color:inherit;opacity:.9}
html[data-ds-theme="dark"] [class*="st-key-flow_active_"] a span[data-testid="stIconMaterial"]{color:#10150F}
[class*="st-key-flow_active_"] a span[data-testid="stIconMaterial"]{color:#fff}
/* locked */
[class*="st-key-flow_locked_"] a{background:transparent;border-style:dashed;opacity:.62}
[class*="st-key-flow_locked_"]:hover a{opacity:.8}
/* completion tick */
[class*="st-key-flow_done_"] a::after{content:"✓";font-size:.72rem;font-weight:800;color:var(--ds-accent);margin-left:2px}
@keyframes ds-flow{0%%{background-position:100%% 0}100%%{background-position:0%% 0}}

.st-key-ds_branch{gap:8px!important;align-items:center;padding:0 4px 10px;flex-wrap:wrap}
.st-key-ds_branch > div{width:auto!important;flex:0 0 auto!important}
.ds-branch-label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--ds-text-3);
  display:flex;align-items:center;gap:6px;white-space:nowrap}
.ds-branch-label svg{width:14px;height:14px;color:var(--ds-text-3)}
[class*="st-key-branch_"] a{border:1px dashed var(--ds-border-strong);border-radius:999px;padding:3px 11px 3px 8px!important;
  min-height:30px;background:transparent;transition:border-color .18s,background .18s}
[class*="st-key-branch_"] a p{font-size:.76rem!important;font-weight:600!important;color:var(--ds-text-2)!important}
[class*="st-key-branch_"] a:hover{border-color:var(--ds-accent);background:var(--ds-accent-soft)}
[class*="st-key-branch_done_"] a{border-style:solid;border-color:var(--ds-accent);background:var(--ds-accent-soft)}
[class*="st-key-branch_active_"] a{border-style:solid;border-color:var(--ds-accent);background:var(--ds-accent)}
[class*="st-key-branch_active_"] a p,[class*="st-key-branch_active_"] a span{color:#fff!important}
[class*="st-key-branch_locked_"] a{opacity:.55}
.ds-progress{font-size:.75rem;color:var(--ds-text-3);white-space:nowrap;margin-left:auto}
.ds-progress b{color:var(--ds-text-2)}

.st-key-ds_flow_mobile{display:none}
@media (max-width: 860px){
  .st-key-ds_flow,.st-key-ds_branch{display:none!important}
  .st-key-ds_flow_mobile{display:flex;padding:10px 0 6px}
  .st-key-ds_flow_mobile [data-testid="stPopover"]{width:100%%}
  .ds-context{justify-content:flex-start}
  .stMainBlockContainer{padding-left:1rem;padding-right:1rem}
}
@media (max-width: 1100px) and (min-width: 861px){
  [class*="st-key-flow_"] a p{font-size:.74rem!important}
  [class*="st-key-flow_"]{margin-right:24px}
  [class*="st-key-flow_"]::after{width:16px;right:-21px}
  [class*="st-key-flow_"]::before{right:-22px}
}

/* ------------------------------------------------------------ page header */
.ds-eyebrow{display:inline-flex;align-items:center;gap:6px;border-radius:999px;background:var(--ds-accent-soft);color:var(--ds-accent-ink);
  font-weight:700;font-size:.7rem;padding:4px 11px;letter-spacing:.07em;text-transform:uppercase}
.ds-title{font-family:var(--ds-display);font-size:clamp(1.5rem,2.4vw,1.95rem);font-weight:800;margin:10px 0 4px;line-height:1.15;
  color:var(--ds-text);letter-spacing:-.02em}
.ds-sub{color:var(--ds-text-2);font-size:.98rem;margin-bottom:12px;max-width:72ch;line-height:1.6}
.hero-title{font-family:var(--ds-display);font-size:clamp(2.2rem,4.4vw,3.6rem);font-weight:800;line-height:1.06;margin:16px 0 14px;
  color:var(--ds-text);letter-spacing:-.035em}
.hero-script{font-family:var(--ds-serif);font-style:italic;font-weight:400;color:var(--ds-accent);letter-spacing:-.01em;
  display:inline-block;font-size:1.06em}
.hero-lead{font-size:1.06rem;line-height:1.75;color:var(--ds-text-2);max-width:56ch}
.ds-hero-points{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:14px;font-size:.84rem;color:var(--ds-text-2)}
.ds-hero-points span::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%%;background:var(--ds-accent);
  margin-right:7px;vertical-align:middle}

/* ------------------------------------------------------- cards & surfaces */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--ds-radius)!important}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--ds-surface);
  border-color:var(--ds-border)!important;box-shadow:var(--ds-shadow)}
.stApp [data-testid="stExpander"] details{background:var(--ds-surface);border-color:var(--ds-border)!important;border-radius:var(--ds-radius)}
[data-testid="stMetric"]{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:var(--ds-radius);padding:13px 16px;
  box-shadow:var(--ds-shadow);height:100%%}
[data-testid="stMetricLabel"] p{font-size:.74rem!important;color:var(--ds-text-2)!important;font-weight:650;text-transform:uppercase;
  letter-spacing:.05em}
[data-testid="stMetricValue"]{font-family:var(--ds-display);color:var(--ds-text);letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis}
.ds-callout{border-radius:var(--ds-radius-sm);padding:11px 14px;font-size:.9rem;line-height:1.6;margin:6px 0 12px;color:var(--ds-text)}
.ds-callout.syn{background:var(--ds-warning-soft);border:1px solid var(--ds-warning-border)}
.ds-callout.syn strong{color:var(--ds-warning)}
.ds-callout.read{background:var(--ds-accent-soft);border-left:3px solid var(--ds-accent)}
.ds-callout.na{background:var(--ds-surface-2);border:1px dashed var(--ds-border-strong);color:var(--ds-text-2)}
.ds-callout.prov{background:var(--ds-info-soft);border:1px solid var(--ds-border)}
.ds-callout.err{background:var(--ds-error-soft);border:1px solid color-mix(in srgb,var(--ds-error) 35%%,transparent)}
.ds-empty{border:1.5px dashed var(--ds-border-strong);border-radius:18px;padding:40px 24px;text-align:center;background:var(--ds-surface-2)}
.ds-empty h3{margin:0 0 6px;font-size:1.12rem}.ds-empty p{color:var(--ds-text-2);margin:0}
.ds-pill{display:inline-flex;align-items:center;gap:4px;border-radius:999px;padding:2px 10px;font-size:.72rem;font-weight:700;
  letter-spacing:.01em;white-space:nowrap}
.ds-pill.syn{background:var(--ds-warning-soft);border:1px solid var(--ds-warning-border);color:var(--ds-warning)}
.ds-pill.ok{background:var(--ds-success-soft);border:1px solid color-mix(in srgb,var(--ds-success) 30%%,transparent);color:var(--ds-success)}
.ds-pill.muted{background:var(--ds-surface-2);border:1px solid var(--ds-border);color:var(--ds-text-2)}
.ds-pill.err{background:var(--ds-error-soft);border:1px solid color-mix(in srgb,var(--ds-error) 30%%,transparent);color:var(--ds-error)}
.ds-pill.info{background:var(--ds-info-soft);border:1px solid var(--ds-border);color:var(--ds-text-2)}
.ds-meta{font-size:.78rem;color:var(--ds-text-3)}
.ds-section{font-family:var(--ds-display);font-weight:750;font-size:1.08rem;margin:18px 0 8px;color:var(--ds-text);display:flex;
  align-items:center;gap:8px}
.ds-section .n{font-family:var(--ds-font);font-size:.72rem;font-weight:700;color:var(--ds-text-3);border:1px solid var(--ds-border);
  border-radius:999px;padding:1px 8px}
.ds-list{margin:0;padding-left:18px}.ds-list li{margin:3px 0}

/* insight (editorial) */
.ds-insight{padding:4px 2px}
.ds-insight .cat{font-size:.7rem;font-weight:700;color:var(--ds-accent);text-transform:uppercase;letter-spacing:.08em;display:flex;gap:8px;
  align-items:center}
.ds-insight h4{margin:6px 0 6px;font-size:1.06rem;line-height:1.35}
.ds-insight .obs{font-size:.98rem;color:var(--ds-text);margin:0 0 10px;max-width:80ch}
.ds-insight .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px 18px;font-size:.86rem}
.ds-insight .grid div b{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:var(--ds-text-3);margin-bottom:2px}
.ds-insight .ev{background:var(--ds-surface-2);border-radius:10px;padding:8px 12px;margin:0 0 10px;font-size:.85rem;color:var(--ds-text-2)}
.ds-insight .ev ul{margin:0;padding-left:16px}
.ds-insight .src{font-size:.74rem;color:var(--ds-text-3);margin-top:8px}
.ds-insight.warning{border-left:3px solid var(--ds-warning);padding-left:12px}
.ds-insight.attention{border-left:3px solid var(--ds-warning-border);padding-left:12px}

/* samples & recommendation cards */
.ds-sample .name{font-family:var(--ds-display);font-weight:750;font-size:.98rem}
.ds-sample .desc{color:var(--ds-text-2);font-size:.82rem;margin:2px 0 8px;min-height:2.4em}
.ds-sample .dims{font-size:.76rem;color:var(--ds-text-3);margin-top:6px}
.ds-rec .t{font-weight:700;font-size:.92rem;margin-bottom:2px}.ds-rec .w{font-size:.82rem;color:var(--ds-text-2)}
.ds-rec .m{font-size:.74rem;color:var(--ds-text-3);margin-top:4px}

/* chat */
[data-testid="stChatMessage"]{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:var(--ds-radius);padding:10px 14px}
.ds-tool{display:inline-flex;gap:6px;align-items:center;font-size:.72rem;color:var(--ds-text-3);border:1px solid var(--ds-border);
  border-radius:999px;padding:1px 9px;margin-bottom:4px}

/* report preview */
.ds-doc{background:#FFFFFF;color:#1B221B;border-radius:12px;padding:30px 36px;box-shadow:var(--ds-shadow-lg);max-width:860px;margin:0 auto;
  font-family:var(--ds-font);line-height:1.65}
.ds-doc h1,.ds-doc h2,.ds-doc h3{font-family:var(--ds-display);color:#1B221B}
.ds-doc .user{background:#F4F1E6;border-left:3px solid #C98B3A;padding:8px 12px;border-radius:6px}

/* controls */
.stButton > button, .stDownloadButton > button{font-weight:650;transition:transform .15s var(--ds-ease),box-shadow .15s,filter .15s}
.stButton > button:hover, .stDownloadButton > button:hover{transform:translateY(-1px)}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"]{box-shadow:0 6px 16px color-mix(in srgb,var(--ds-accent) 28%%,transparent)}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed var(--ds-border-strong);border-radius:var(--ds-radius);padding:22px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ds-accent)}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--ds-border)}
.stTabs [data-baseweb="tab"]{font-weight:650;padding:8px 14px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{max-width:220px}
[data-testid="stPlotlyChart"]{border-radius:var(--ds-radius);background:var(--ds-surface);border:1px solid var(--ds-border);padding:6px 6px 2px}
@media (max-width: 860px){.ds-title{font-size:1.45rem}.hero-title{font-size:2.1rem}}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
}
</style>
""" % {"light": _vars("light"), "dark": _vars("dark")}

LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"/><path d="M8 17v-5"/><path d="M13 17V7"/><path d="M18 17v-8"/></svg>')
BRANCH = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
          'aria-hidden="true"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/>'
          '<path d="M6 8v8"/><path d="M8 6h3a5 5 0 0 1 5 5"/></svg>')


def inject() -> None:
    st.html(CSS)
