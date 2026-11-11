"""Optional LLM access through Groq's OpenAI-compatible chat-completions API.

Set ``GROQ_API_KEY`` (environment variable or Streamlit secret) to enable it.
``GROQ_MODEL`` picks a different Groq model. Keys are never written to disk,
logged, or echoed to the UI.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
REASONING_HEADROOM = 1024  # extra completion tokens so gpt-oss's hidden reasoning cannot crowd out the answer
_ERRORS = {
    401: "Groq rejected the API key (HTTP 401). Check GROQ_API_KEY.",
    404: "Groq does not recognise the configured model (HTTP 404). Check GROQ_MODEL.",
    413: "The request was too large for the Groq model (HTTP 413).",
    429: "Groq's rate limit was reached (HTTP 429); free-tier limits reset within a minute.",
}


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    source: str  # which variable supplied the key (the key itself is never exposed)


def _setting(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        try:
            import streamlit as st

            value = str(st.secrets.get(name, ""))
        except Exception:  # no secrets file, or not running under Streamlit
            value = ""
    return value.strip()


def get_config() -> LLMConfig | None:
    key = _setting("GROQ_API_KEY")
    if not key:
        return None
    return LLMConfig(api_key=key, base_url=GROQ_BASE_URL, model=_setting("GROQ_MODEL") or DEFAULT_MODEL,
                     source="GROQ_API_KEY")


def chat(messages: list[dict[str, str]], max_tokens: int = 500, temperature: float = 0.2,
         config: LLMConfig | None = None) -> str:
    cfg = config or get_config()
    if cfg is None:
        raise LLMError("GROQ_API_KEY is not configured.")
    payload: dict = {"model": cfg.model, "messages": messages, "temperature": temperature,
                     "max_completion_tokens": max_tokens}
    if cfg.model.startswith("openai/gpt-oss"):
        # reasoning model: keep thinking short, return only the final answer, leave room for both
        payload.update(reasoning_effort="low", include_reasoning=False,
                       max_completion_tokens=max_tokens + REASONING_HEADROOM)
    try:
        r = requests.post(
            f"{cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
            json=payload, timeout=60,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Groq could not be reached ({type(exc).__name__}).") from exc
    if r.status_code != 200:
        raise LLMError(_ERRORS.get(r.status_code, f"Groq returned HTTP {r.status_code}."))
    try:
        text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise LLMError("Groq returned an unexpected response.") from exc
    if not text:
        raise LLMError("Groq returned an empty answer.")
    return text


_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", "").replace("−", "-").lstrip("+") for m in _NUM.finditer(text)]


def unverified_numbers(text: str, facts: str) -> list[str]:
    """Numbers in ``text`` that cannot be traced to ``facts``.

    A number is accepted when it equals a fact number, or equals one rounded to
    the same number of decimals. Whole numbers from 0 to 10 are accepted without
    a match because they occur naturally in prose ("two columns", list numbering).
    """
    fact_vals = []
    for f in _numbers(facts):
        try:
            fact_vals.append(float(f))
        except ValueError:
            pass
    bad = []
    for n in _numbers(text):
        try:
            v = float(n)
        except ValueError:
            continue
        decimals = len(n.split(".")[1]) if "." in n else 0
        if decimals == 0 and 0 <= abs(v) <= 10:
            continue
        if any(abs(round(f, decimals) - v) < 1e-9 or abs(abs(round(f, decimals)) - abs(v)) < 1e-9 for f in fact_vals):
            continue
        bad.append(n)
    return bad
