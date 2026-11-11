"""Optional LLM access through any OpenAI-compatible chat-completions endpoint.

Configuration comes from environment variables or Streamlit secrets; keys are
never written to disk, logged, or echoed to the UI.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

_PROVIDERS = [  # (key variable, default base URL, default model)
    ("DATASTORY_LLM_API_KEY", "https://openrouter.ai/api/v1", "mistralai/mistral-7b-instruct"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "mistralai/mistral-7b-instruct"),
    ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
]


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
    for key_var, base, model in _PROVIDERS:
        key = _setting(key_var)
        if key:
            return LLMConfig(
                api_key=key,
                base_url=(_setting("DATASTORY_LLM_BASE_URL") or base).rstrip("/"),
                model=_setting("DATASTORY_LLM_MODEL") or _setting("OPENROUTER_MODEL") or model,
                source=key_var,
            )
    return None


def chat(messages: list[dict[str, str]], max_tokens: int = 500, temperature: float = 0.2,
         config: LLMConfig | None = None) -> str:
    cfg = config or get_config()
    if cfg is None:
        raise LLMError("No LLM API key is configured.")
    try:
        r = requests.post(
            f"{cfg.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
            json={"model": cfg.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=45,
        )
    except requests.RequestException as exc:
        raise LLMError(f"The LLM provider could not be reached ({type(exc).__name__}).") from exc
    if r.status_code != 200:
        raise LLMError(f"The LLM provider returned HTTP {r.status_code}.")
    try:
        return r.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise LLMError("The LLM provider returned an unexpected response.") from exc


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
