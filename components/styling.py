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
        "c-sage": "#5E7845", "c-teal": "#4F7C8A", "c-amber": "#B97F2E", "c-olive": "#7E8F3A", "c-clay": "#A65D57",
        "c-moss": "#3F7A4A",
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
        "c-sage": "#9DBB7C", "c-teal": "#73A6B4", "c-amber": "#E0A75A", "c-olive": "#B9C66E", "c-clay": "#D08A82",
        "c-moss": "#8CC48F",
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
.stMainBlockContainer{max-width:1480px;padding-left:2rem;padding-right:2rem}
[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none}
:focus-visible{outline:2px solid var(--ds-accent)!important;outline-offset:2px}

/* ------------------------------------------------------------------ header */
.st-key-ds_header{position:relative;align-items:center;gap:14px;padding:4px 0 14px}
.st-key-ds_header::after{content:"";position:absolute;left:0;right:0;bottom:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--ds-c-sage),var(--ds-c-teal),var(--ds-c-amber),var(--ds-c-olive),transparent);
  background-size:240%% 100%%;animation:ds-sweep 12s linear infinite;opacity:.85}
@keyframes ds-sweep{from{background-position:0%% 0}to{background-position:240%% 0}}
.ds-brand{display:flex;align-items:center;gap:13px;line-height:1;color:var(--ds-text);white-space:nowrap}
.ds-mark{position:relative;width:46px;height:46px;flex:none;display:grid;place-items:center;border-radius:15px;isolation:isolate;
  border:1px solid var(--ds-border);background:radial-gradient(circle at 32%% 26%%,color-mix(in srgb,var(--ds-accent) 26%%,transparent),
  color-mix(in srgb,var(--ds-c-teal) 10%%,transparent) 58%%,transparent 74%%),var(--ds-surface);
  box-shadow:0 0 30px -10px var(--ds-glow),inset 0 1px 0 rgba(255,255,255,.08)}
.ds-mark::before{content:"";position:absolute;inset:-5px;border-radius:20px;z-index:-1;
  background:conic-gradient(from 0deg,var(--ds-c-sage),var(--ds-c-teal),var(--ds-c-amber),var(--ds-c-olive),var(--ds-c-sage));
  filter:blur(12px);opacity:.28;animation:ds-spin 9s linear infinite}
.ds-mark svg{width:32px;height:32px;overflow:visible}
.ds-mark .ring{transform-origin:24px 24px;animation:ds-spin 17s linear infinite}
.ds-mark .bars{animation:ds-pulse 3.2s ease-in-out infinite}
@keyframes ds-spin{to{transform:rotate(360deg)}}
@keyframes ds-pulse{0%%,100%%{opacity:.75}50%%{opacity:1}}
.ds-brand .w{font-family:var(--ds-display);font-weight:800;font-size:1.5rem;letter-spacing:-.035em;line-height:1.1;
  background:linear-gradient(100deg,var(--ds-text) 0%%,var(--ds-c-sage) 26%%,var(--ds-c-teal) 44%%,var(--ds-c-amber) 60%%,var(--ds-text) 84%%);
  background-size:270%% 100%%;-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent;
  animation:ds-wm 13s ease-in-out infinite}
@keyframes ds-wm{0%%,100%%{background-position:0%% 50%%}50%%{background-position:100%% 50%%}}
.ds-brand .d{display:block;font-family:var(--ds-font);font-size:.66rem;font-weight:600;color:var(--ds-text-3);letter-spacing:.2em;
  text-transform:uppercase;margin-top:5px}
.ds-brand .d b{color:var(--ds-accent);font-weight:700}
.ds-context{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;justify-content:flex-end;font-size:.8rem;color:var(--ds-text-2);
  min-width:0}
.ds-context b{color:var(--ds-text);font-weight:650}
.ds-context .name{max-width:28ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block;vertical-align:bottom}
.ds-dot{width:7px;height:7px;border-radius:50%%;display:inline-block;background:var(--ds-accent);margin-right:6px;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--ds-accent) 22%%,transparent)}
.st-key-ds_header [data-testid="stPopover"] button{min-height:36px;padding:4px 12px;font-size:.82rem}

/* ------------------------------------------------- pipeline navigation rail */
div[data-testid="stLayoutWrapper"]:has(> div.st-key-ds_rail){position:sticky;top:0;z-index:99}
.st-key-ds_rail{margin:2px 0 18px;padding:12px 0 8px;gap:6px!important;
  background:color-mix(in srgb,var(--ds-bg) 86%%,transparent);backdrop-filter:blur(12px) saturate(1.15);
  -webkit-backdrop-filter:blur(12px) saturate(1.15);border-bottom:1px solid var(--ds-border);
  box-shadow:0 14px 22px -20px rgba(0,0,0,.45)}
.st-key-ds_flow{flex-wrap:nowrap!important;overflow-x:auto;overflow-y:visible;gap:0!important;padding:4px 2px 6px;
  scrollbar-width:thin;align-items:stretch}
.st-key-ds_flow > div{min-width:0;flex:1 1 0!important}
[class*="st-key-flow_"] [data-testid="stElementContainer"],[class*="st-key-flow_"] [data-testid="stPageLink"],
[class*="st-key-flow_"] [data-testid="stPageLink"] > div,[class*="st-key-flow_"] [data-testid="stPageLink"] > div > div{width:100%%!important;max-width:none!important}
[class*="st-key-flow_"]{position:relative;flex:1 1 0!important;width:auto!important;min-width:96px;overflow:visible}
.st-key-ds_flow > div{margin-right:26px;overflow:visible}
.st-key-ds_flow > div:last-child{margin-right:0}
[class*="st-key-flow_"] [data-testid="stPageLink"], [class*="st-key-flow_"] [data-testid="stPageLink"] > div{width:100%%;height:100%%}
[class*="st-key-flow_"] a{position:relative;overflow:hidden;display:block!important;width:100%%;min-height:70px;height:100%%;
  padding:9px 11px 9px!important;border-radius:13px;border:1px solid var(--ds-border);
  background:linear-gradient(160deg,color-mix(in srgb,var(--ds-surface) 96%%,var(--c)),var(--ds-surface));
  transition:transform .18s var(--ds-ease),box-shadow .18s var(--ds-ease),border-color .18s,background .18s}
[class*="st-key-flow_"] a::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--c),transparent)}
[class*="st-key-flow_"] a span[data-testid="stIconMaterial"]{display:none}
[class*="st-key-flow_"] a p{margin:0;font-size:.6rem!important;line-height:1.45;letter-spacing:.11em;text-transform:uppercase;
  font-weight:650!important;color:var(--ds-text-3)!important;white-space:normal;text-align:left}
[class*="st-key-flow_"] a strong{display:block;font-family:var(--ds-display);font-size:.92rem;font-weight:750;letter-spacing:-.01em;
  text-transform:none;color:var(--ds-text);margin:2px 0 1px}
[class*="st-key-flow_"] a em{display:block;font-style:normal;font-size:.69rem;letter-spacing:0;text-transform:none;font-weight:500;
  color:var(--ds-text-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
[class*="st-key-flow_"] a:hover{transform:translateY(-1px);border-color:color-mix(in srgb,var(--c) 65%%,transparent);
  box-shadow:0 0 0 1px color-mix(in srgb,var(--c) 20%%,transparent),0 8px 20px -10px var(--c)}
/* connector between nodes: base line + arrowhead; flowing highlight once the path is reached */
[class*="st-key-flow_"]::after{content:"";position:absolute;top:50%%;right:-23px;width:18px;height:2px;margin-top:-1px;
  border-radius:2px;background:var(--ds-line);pointer-events:none}
[class*="st-key-flow_"]::before{content:"";position:absolute;top:50%%;right:-24px;margin-top:-4px;border:4px solid transparent;
  border-left:5px solid var(--ds-line);border-right:0;pointer-events:none;z-index:1}
.st-key-ds_flow > div:last-child [class*="st-key-flow_"]::after,.st-key-ds_flow > div:last-child [class*="st-key-flow_"]::before{display:none}
[class*="st-key-flow_done_"]::after,[class*="st-key-flow_active_"]::after{
  background:linear-gradient(90deg,var(--c) 0%%,var(--c) 35%%,var(--ds-glow-strong) 50%%,var(--c) 65%%,var(--c) 100%%);
  background-size:300%% 100%%;animation:ds-flow 2.8s linear infinite}
[class*="st-key-flow_done_"]::before,[class*="st-key-flow_active_"]::before{border-left-color:var(--c)}
[class*="st-key-flow_active_"]::after{animation-duration:1.9s;box-shadow:0 0 8px var(--ds-glow)}
[class*="st-key-flow_done_"] a{border-color:color-mix(in srgb,var(--c) 40%%,var(--ds-border))}
[class*="st-key-flow_done_"] a::after{content:"✓";position:absolute;top:7px;right:9px;font-size:.7rem;font-weight:800;color:var(--c)}
[class*="st-key-flow_active_"] a{background:color-mix(in srgb,var(--c) 14%%,var(--ds-surface));border-color:var(--c);
  box-shadow:0 0 0 1px color-mix(in srgb,var(--c) 40%%,transparent),0 10px 26px -12px var(--c)}
[class*="st-key-flow_active_"] a::before{height:3px}
[class*="st-key-flow_active_"] a strong{color:var(--c)}
[class*="st-key-flow_locked_"] a{background:transparent;border-style:dashed;opacity:.6}
[class*="st-key-flow_locked_"] a::before{opacity:.35}
@keyframes ds-flow{0%%{background-position:100%% 0}100%%{background-position:0%% 0}}

.st-key-ds_branch{gap:8px!important;align-items:center;padding:0 2px;flex-wrap:wrap}
.st-key-ds_branch > div{width:auto!important;flex:0 0 auto!important}
.ds-branch-label{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:var(--ds-text-3);
  display:flex;align-items:center;gap:6px;white-space:nowrap}
.ds-branch-label svg{width:14px;height:14px;color:var(--ds-text-3)}
[class*="st-key-branch_"] a{border:1px dashed var(--ds-border-strong);border-radius:999px;padding:3px 11px 3px 8px!important;
  min-height:30px;background:transparent;transition:border-color .18s,background .18s}
[class*="st-key-branch_"] a p{font-size:.76rem!important;font-weight:600!important;color:var(--ds-text-2)!important}
[class*="st-key-branch_"] a p em{font-style:normal;color:var(--ds-text-3);font-weight:500}
[class*="st-key-branch_"] a span[data-testid="stIconMaterial"]{color:var(--c)}
[class*="st-key-branch_"] a:hover{border-color:var(--c);background:color-mix(in srgb,var(--c) 10%%,transparent)}
[class*="st-key-branch_done_"] a{border-style:solid;border-color:var(--c);background:color-mix(in srgb,var(--c) 10%%,transparent)}
[class*="st-key-branch_active_"] a{border-style:solid;border-color:var(--c);background:color-mix(in srgb,var(--c) 18%%,transparent);
  box-shadow:0 6px 18px -10px var(--c)}
[class*="st-key-branch_active_"] a p{color:var(--ds-text)!important}
[class*="st-key-branch_locked_"] a{opacity:.55}
.ds-progress{font-size:.72rem;color:var(--ds-text-3);white-space:nowrap;margin-left:auto;letter-spacing:.02em}
.ds-progress b{color:var(--ds-text-2)}
.ds-progress .bar{display:inline-block;vertical-align:middle;width:90px;height:5px;border-radius:9px;background:var(--ds-border);
  margin:0 8px 2px 0;overflow:hidden}
.ds-progress .bar i{display:block;height:100%%;border-radius:9px;background:linear-gradient(90deg,var(--ds-c-sage),var(--ds-c-teal))}

.st-key-ds_flow_mobile{display:none}
@media (max-width: 1180px){[class*="st-key-flow_"] a em{display:none}[class*="st-key-flow_"] a{min-height:54px}}
@media (max-width: 860px){
  div[data-testid="stLayoutWrapper"]:has(> div.st-key-ds_rail){position:static}
  .st-key-ds_rail{display:none!important}
  .st-key-ds_flow_mobile{display:flex;padding:10px 0 6px}
  .st-key-ds_flow_mobile [data-testid="stPopover"]{width:100%%}
  .ds-context{justify-content:flex-start}
  .stMainBlockContainer{padding-left:1rem;padding-right:1rem}
}

/* floating copilot dock */
div[class*="st-key-askdock"]{position:fixed;left:20px;bottom:20px;z-index:900;width:auto!important}
div[class*="st-key-askdock"] a{border-radius:999px!important;padding:9px 16px!important;border:0!important;
  background:linear-gradient(135deg,var(--ds-c-sage),var(--ds-c-teal))!important;
  box-shadow:0 10px 30px -10px var(--ds-c-sage),inset 0 0 0 1px rgba(255,255,255,.1)!important}
div[class*="st-key-askdock"] a p, div[class*="st-key-askdock"] a span{color:#fff!important;font-weight:700!important;font-size:.82rem!important}
html[data-ds-theme="dark"] div[class*="st-key-askdock"] a p, html[data-ds-theme="dark"] div[class*="st-key-askdock"] a span{color:#0E150F!important}
div[class*="st-key-askdock"] a:hover{filter:brightness(1.06);transform:translateY(-1px)}

/* tiles (KPI) */
.ds-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin:4px 0 16px}
.ds-tile{position:relative;overflow:hidden;border-radius:14px;padding:13px 15px 12px;border:1px solid var(--ds-border);
  background:linear-gradient(160deg,color-mix(in srgb,var(--c) 7%%,var(--ds-surface)),var(--ds-surface));box-shadow:var(--ds-shadow)}
.ds-tile::before{content:"";position:absolute;top:0;left:0;width:100%%;height:2px;background:linear-gradient(90deg,var(--c),transparent)}
.ds-tile .l{color:var(--ds-text-2);font-size:.64rem;letter-spacing:.13em;text-transform:uppercase;font-weight:650}
.ds-tile .v{font-family:var(--ds-display);font-size:1.7rem;font-weight:800;line-height:1.1;margin:6px 0 3px;letter-spacing:-.025em;
  color:var(--c);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ds-tile .n{color:var(--ds-text-3);font-size:.74rem;line-height:1.4}
.ds-tile:hover{border-color:color-mix(in srgb,var(--c) 45%%,var(--ds-border))}

/* chart / content cards */
div[class*="st-key-card_"]{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:16px;padding:12px 14px 6px;
  box-shadow:var(--ds-shadow);margin-bottom:4px}
div[class*="st-key-card_"] [data-testid="stPlotlyChart"]{border:0!important;padding:0!important;background:transparent!important}
div[class*="st-key-card_"] div[data-testid="stExpander"] details{border:0!important;border-top:1px solid var(--ds-border)!important;
  border-radius:0!important;background:transparent!important;box-shadow:none}
div[class*="st-key-card_"] .ds-callout.read{background:transparent;border-left:2px solid var(--ds-accent);padding:4px 0 4px 12px;
  margin:2px 0 8px;color:var(--ds-text-2);font-size:.86rem}

/* every st.subheader gets the eyebrow rule */
[data-testid="stHeading"] h3{padding:0 0 0 11px!important;margin:18px 0 6px!important;border-left:3px solid transparent;
  border-image:linear-gradient(180deg,var(--ds-c-sage),var(--ds-c-teal)) 1 100%%;font-size:1.12rem!important}

/* st.metric reads as a KPI tile: accent strip + coloured value, accent cycles across a row */
[data-testid="stMetric"]{--m:var(--ds-c-sage);position:relative;overflow:hidden;
  background:linear-gradient(160deg,color-mix(in srgb,var(--m) 7%%,var(--ds-surface)),var(--ds-surface))!important}
[data-testid="stMetric"]::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--m),transparent)}
[data-testid="stMetricValue"]{color:var(--m)!important}
div[data-testid="stColumn"]:nth-child(6n+2) [data-testid="stMetric"]{--m:var(--ds-c-teal)}
div[data-testid="stColumn"]:nth-child(6n+3) [data-testid="stMetric"]{--m:var(--ds-c-amber)}
div[data-testid="stColumn"]:nth-child(6n+4) [data-testid="stMetric"]{--m:var(--ds-c-olive)}
div[data-testid="stColumn"]:nth-child(6n+5) [data-testid="stMetric"]{--m:var(--ds-c-clay)}
div[data-testid="stColumn"]:nth-child(6n+6) [data-testid="stMetric"]{--m:var(--ds-c-moss)}

/* method flow (vertical numbered pipeline) + artefacts panel */
.ds-mflow{margin:.3rem 0 1rem}
.ds-mflow .step{position:relative;display:grid;grid-template-columns:42px 1fr;gap:14px;padding-bottom:14px}
.ds-mflow .step:last-child{padding-bottom:0}
.ds-mflow .step:not(:last-child)::after{content:"";position:absolute;left:20px;top:48px;bottom:-2px;width:2px;
  background:linear-gradient(180deg,var(--c),var(--ds-border))}
.ds-mflow .node{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-family:var(--ds-display);font-weight:800;
  font-size:.88rem;color:var(--c);background:var(--ds-surface);border:1.5px solid var(--c);box-shadow:0 0 18px -8px var(--c);
  font-variant-numeric:tabular-nums}
.ds-mflow .sbody{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:13px;padding:10px 13px 11px}
.ds-mflow h5{font-family:var(--ds-display);margin:0 0 3px;font-size:.95rem;font-weight:750;color:var(--ds-text)}
.ds-mflow p{margin:0;color:var(--ds-text-2);font-size:.84rem;line-height:1.55}
.ds-chip{display:inline-block;margin:7px 6px 0 0;padding:2px 10px;border-radius:999px;font-size:.7rem;font-weight:650;color:var(--c);
  background:color-mix(in srgb,var(--c) 9%%,transparent);border:1px solid color-mix(in srgb,var(--c) 55%%,transparent)}
.ds-io{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:16px;padding:15px 16px;box-shadow:var(--ds-shadow)}
.ds-io .h{font-size:.64rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--ds-text-2);margin-bottom:10px}
.ds-io .k{font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:var(--c);margin:2px 0 6px}
.ds-io .row{display:flex;justify-content:space-between;gap:10px;background:var(--ds-surface-2);border:1px solid var(--ds-border);
  border-radius:9px;padding:6px 10px;margin-bottom:5px;font-size:.78rem}
.ds-io .row span:last-child{color:var(--ds-text-3);white-space:nowrap;font-variant-numeric:tabular-nums}
.ds-io .arrow{text-align:center;color:var(--c);font-size:1.05rem;line-height:1;margin:8px 0}
.ds-io .engine{text-align:center;font-family:var(--ds-display);font-weight:750;font-size:.9rem;color:var(--c);border:1.5px solid var(--c);
  border-radius:12px;padding:9px 8px;background:color-mix(in srgb,var(--c) 6%%,var(--ds-surface));box-shadow:0 0 24px -12px var(--c)}
.ds-io .f{margin-top:10px;font-size:.68rem;color:var(--ds-text-3)}

/* ------------------------------------------------------------ page header */
.ds-eyebrow{display:inline-flex;align-items:center;gap:6px;border-radius:999px;background:var(--ds-accent-soft);color:var(--ds-accent-ink);
  font-weight:700;font-size:.66rem;padding:4px 11px;letter-spacing:.16em;text-transform:uppercase;
  border:1px solid color-mix(in srgb,var(--ds-accent) 30%%,transparent)}
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
.ds-section{display:flex;align-items:baseline;gap:10px;margin:26px 0 12px;padding-left:11px;border-left:3px solid transparent;
  border-image:linear-gradient(180deg,var(--ds-c-sage),var(--ds-c-teal)) 1 100%%}
.ds-section h3{font-family:var(--ds-display);font-weight:750;font-size:1.14rem;margin:0!important;padding:0!important;
  color:var(--ds-text);letter-spacing:-.02em}
.ds-section .s{color:var(--ds-text-3);font-size:.82rem}
.ds-section .n{font-family:var(--ds-font);font-size:.68rem;font-weight:700;color:var(--ds-text-3);border:1px solid var(--ds-border);
  border-radius:999px;padding:1px 8px}
div[data-testid="stWidgetLabel"] p{color:var(--ds-text-2)!important;font-size:.7rem!important;letter-spacing:.11em;text-transform:uppercase;
  font-weight:650!important}
div[data-testid="stCaptionContainer"] p{color:var(--ds-text-3)!important;font-size:.78rem!important}
.stTabs [aria-selected="true"]{background:color-mix(in srgb,var(--ds-c-teal) 14%%,transparent);border-radius:10px 10px 0 0}
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
  .st-key-ds_header::after,.ds-mark::before,.ds-mark .ring,.ds-mark .bars,.ds-brand .w{animation:none!important}
  *,*::before,*::after{animation:none!important;transition:none!important}
}
</style>
""" % {"light": _vars("light"), "dark": _vars("dark")}

MARK = ('<div class="ds-mark"><svg viewBox="0 0 48 48" aria-hidden="true"><defs><linearGradient id="dsg" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="var(--ds-c-sage)"/><stop offset="55%" stop-color="var(--ds-c-teal)"/>'
        '<stop offset="100%" stop-color="var(--ds-c-amber)"/></linearGradient></defs>'
        '<circle class="ring" cx="24" cy="24" r="19" fill="none" stroke="url(#dsg)" stroke-width="2.3" stroke-linecap="round" stroke-dasharray="30 14"/>'
        '<g class="bars" stroke="url(#dsg)" stroke-width="3" stroke-linecap="round"><path d="M17 30v-5"/><path d="M24 30V17"/><path d="M31 30v-9"/></g>'
        '<path d="M14 33h20" stroke="var(--ds-c-sage)" stroke-opacity=".55" stroke-width="1.6" stroke-linecap="round"/></svg></div>')

LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"/><path d="M8 17v-5"/><path d="M13 17V7"/><path d="M18 17v-8"/></svg>')
BRANCH = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
          'aria-hidden="true"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/>'
          '<path d="M6 8v8"/><path d="M8 6h3a5 5 0 0 1 5 5"/></svg>')


def inject() -> None:
    st.html(CSS)
