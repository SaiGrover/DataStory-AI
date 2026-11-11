"""Central number formatting used by the UI, the insight engine, the assistant and reports.

Rules
* A missing value is shown as "Unavailable", never as 0.
* A non-zero share that would round to zero is shown as "<0.1%" (or the
  equivalent for the requested precision), never as "0.0%".
* Exactly zero is shown as "0%"; exactly 100 as "100%".
"""
from __future__ import annotations

import math
from typing import Any

UNAVAILABLE = "Unavailable"


def _missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return isinstance(x, float) and math.isnan(x)
    except TypeError:
        return False


def pct(value: float | None, digits: int = 1) -> str:
    """Format a percentage that is already on the 0–100 scale."""
    if _missing(value):
        return UNAVAILABLE
    v = float(value)
    if v == 0:
        return "0%"
    if v == 100:
        return "100%"
    step = 10 ** -digits
    if 0 < abs(v) < step / 2:
        return f"<{step:.{digits}f}%" if v > 0 else f">-{step:.{digits}f}%"
    if 100 - step / 2 < v < 100:
        return f">{100 - step:.{digits}f}%"
    return f"{v:.{digits}f}%"


def share(part: float, whole: float, digits: int = 1) -> str:
    """Percentage of ``part`` in ``whole`` with the denominator made explicit by the caller."""
    if _missing(part) or _missing(whole) or not whole:
        return UNAVAILABLE
    return pct(100.0 * float(part) / float(whole), digits)


def num(value: Any, digits: int = 4) -> str:
    """General number: integers with thousands separators, floats with significant digits."""
    if _missing(value):
        return UNAVAILABLE
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return UNAVAILABLE
    if v.is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    a = abs(v)
    if a >= 1e15 or a < 1e-4:
        return f"{v:.{digits}g}"
    if a >= 1000:
        return f"{v:,.0f}"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(a)))) if a >= 1 else digits
    text = f"{v:,.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def metric(value: Any, digits: int = 4) -> str:
    """Model metric with fixed precision."""
    if _missing(value):
        return UNAVAILABLE
    return f"{float(value):.{digits}f}"


def count(value: Any) -> str:
    if _missing(value):
        return UNAVAILABLE
    return f"{int(value):,}"


def mean_sd(mean: Any, sd: Any, digits: int = 4) -> str:
    if _missing(mean):
        return UNAVAILABLE
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}" if not _missing(sd) else f"{float(mean):.{digits}f}"
