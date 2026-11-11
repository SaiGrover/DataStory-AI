"""DataStory visual identity for Streamlit: sage-green palette, soft cards, serif-italic accents."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ds-bg:#fbfbf5;--ds-surface:#ffffff;--ds-soft:#f4f7ef;--ds-border:#e1e8d8;--ds-text:#172018;--ds-muted:#697468;
--ds-strong:#39463a;--ds-accent:#667f4d;--ds-accent-bright:#93ad74;--ds-warn-bg:#fbf3df;--ds-warn-border:#e6cf9a;}
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea{
  font-family:Inter,"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;}
.stApp{background:linear-gradient(180deg,#fbfbf5 0%,#f5f8ef 100%);color:var(--ds-text);}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:"Segoe UI Variable Display",Inter,"Segoe UI",system-ui,sans-serif;
  color:var(--ds-text);letter-spacing:0;font-weight:750;}
.stApp h2{font-size:1.35rem;padding-top:.4rem}.stApp h3{font-size:1.08rem}
[data-testid="stAppViewContainer"] .block-container, .stMainBlockContainer{max-width:1380px;padding-top:1.1rem;padding-bottom:4rem;}
[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none}

/* header */
.ds-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:6px 2px 12px;border-bottom:1px solid var(--ds-border);}
.ds-brand{display:flex;align-items:center;gap:12px;font-size:1.7rem;font-weight:800;line-height:1;color:var(--ds-text)}
.ds-brand svg{width:34px;height:34px;color:var(--ds-accent)}
.ds-tagline{font-weight:600;color:var(--ds-muted);font-size:.95rem}
.ds-strip{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 2px;font-size:.86rem;color:var(--ds-muted);
  border-bottom:1px solid var(--ds-border);margin-bottom:6px}
.ds-strip strong{color:var(--ds-text)}
.ds-pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.74rem;font-weight:700;letter-spacing:.01em}
.ds-pill.syn{background:var(--ds-warn-bg);border:1px solid var(--ds-warn-border);color:#6b4e12}
.ds-pill.ok{background:#eef3e6;border:1px solid #d5e2c4;color:#3f5b2f}
.ds-pill.muted{background:#f0f1ec;border:1px solid var(--ds-border);color:var(--ds-muted)}

/* stepper navigation */
.st-key-ds_nav{padding:8px 0 2px}
.st-key-ds_nav [data-testid="stHorizontalBlock"]{gap:6px}
.st-key-ds_nav a{border:1px solid var(--ds-border);background:var(--ds-surface);border-radius:12px;padding:7px 8px!important;
  justify-content:center;width:100%;min-height:40px;transition:background .15s,border-color .15s}
.st-key-ds_nav a:hover{background:var(--ds-soft);border-color:var(--ds-accent-bright)}
.st-key-ds_nav a p{font-size:.8rem!important;font-weight:650!important;color:var(--ds-strong)!important;white-space:nowrap}
.st-key-ds_nav a span[data-testid="stIconMaterial"]{color:var(--ds-accent)}
.ds-step-active{display:flex;align-items:center;justify-content:center;gap:6px;border-radius:12px;min-height:40px;padding:7px 8px;
  background:var(--ds-accent);color:#fff;font-size:.8rem;font-weight:700;white-space:nowrap;border:1px solid var(--ds-accent);
  box-shadow:0 8px 20px rgba(102,127,77,.22)}
.ds-step-active .n{opacity:.75;font-weight:600}

/* page header */
.ds-eyebrow{display:inline-block;border-radius:999px;background:#eef1ea;color:var(--ds-accent);font-weight:700;font-size:.74rem;
  padding:5px 12px;letter-spacing:.04em;text-transform:uppercase}
.ds-title{font-size:1.9rem;font-weight:800;margin:10px 0 2px;line-height:1.15;color:var(--ds-text)}
.ds-sub{color:var(--ds-muted);font-size:1rem;margin-bottom:10px;max-width:70ch}
.hero-title{font-size:clamp(2.1rem,4.2vw,3.4rem);font-weight:800;line-height:1.1;margin:14px 0 10px;color:var(--ds-text)}
.hero-script{font-family:Georgia,"Times New Roman",serif;font-style:italic;font-weight:600;color:var(--ds-accent);
  letter-spacing:-.03em;display:inline-block;transform:rotate(-1deg);font-size:1.08em}
.hero-lead{font-size:1.08rem;line-height:1.7;color:var(--ds-strong);max-width:56ch}

/* cards and surfaces */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"],
.stApp [data-testid="stExpander"] details{background:var(--ds-surface);border-color:var(--ds-border)!important;}
[data-testid="stMetric"]{background:var(--ds-surface);border:1px solid var(--ds-border);border-radius:14px;padding:14px 16px}
[data-testid="stMetricLabel"] p{font-size:.8rem!important;color:var(--ds-muted)!important;font-weight:600}
[data-testid="stMetricValue"]{font-size:1.55rem!important;font-weight:750;color:var(--ds-text)}
.ds-callout{border-radius:12px;padding:11px 14px;font-size:.9rem;line-height:1.55;margin:6px 0 12px}
.ds-callout.syn{background:var(--ds-warn-bg);border:1px solid var(--ds-warn-border);color:#5b430f}
.ds-callout.read{background:var(--ds-soft);border-left:3px solid var(--ds-accent-bright);color:var(--ds-strong)}
.ds-callout.na{background:#f3f3ef;border:1px dashed #cfd4c6;color:var(--ds-muted)}
.ds-empty{border:1.5px dashed #d2d9c6;border-radius:18px;padding:44px 24px;text-align:center;background:rgba(255,255,255,.6)}
.ds-empty h3{margin:0 0 6px;font-size:1.2rem}.ds-empty p{color:var(--ds-muted);margin:0}
.ds-insight h4{margin:0 0 2px;font-size:1.02rem}
.ds-insight .cat{font-size:.72rem;font-weight:700;color:var(--ds-accent);text-transform:uppercase;letter-spacing:.05em}
.ds-insight dl{margin:8px 0 0;display:grid;grid-template-columns:120px 1fr;gap:6px 14px;font-size:.9rem}
.ds-insight dt{font-weight:700;color:var(--ds-muted)}.ds-insight dd{margin:0;color:var(--ds-text)}
.ds-insight ul{margin:0;padding-left:18px}
.ds-sample{min-height:118px}.ds-sample .name{font-weight:750;font-size:1rem}.ds-sample .desc{color:var(--ds-muted);font-size:.84rem;margin:2px 0 6px}
.ds-sample .dims{font-size:.78rem;color:var(--ds-strong)}

/* controls */
.stButton > button, .stDownloadButton > button{border-radius:10px;font-weight:650;border-color:var(--ds-border)}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"]{
  background:linear-gradient(180deg,#8ca66f,#667f4d);border:1px solid #5d7646;color:#fff;box-shadow:0 8px 20px rgba(102,127,77,.2)}
.stButton > button[kind="primary"]:hover{filter:brightness(1.05)}
[data-testid="stFileUploaderDropzone"]{background:var(--ds-surface);border:1.5px dashed #cfd8c2;border-radius:14px}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--ds-border)}
.stTabs [data-baseweb="tab"]{font-weight:650;padding:8px 14px}
[data-testid="stDataFrame"]{border:1px solid var(--ds-border);border-radius:12px;overflow:hidden}
@media (max-width: 900px){.ds-tagline{display:none}.ds-title{font-size:1.5rem}}
</style>
"""

LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true"><path d="M3 3v18h18"/><path d="M8 17v-5"/><path d="M13 17V7"/><path d="M18 17v-8"/></svg>')


def inject() -> None:
    st.html(CSS)
