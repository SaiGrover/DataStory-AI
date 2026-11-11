"""Exact impact of a transformation, computed by comparing the before and after frames.

Row identity is the DataFrame index: cleaning operations never reset it, so a
row keeps its label through every step. Cell changes are measured only on rows
and columns present in both frames, so deleting rows never shows up as
"changed cells" on the rows that shift up.

Comparison is null-safe and dtype-aware: 2019 and 2019.0 are equal; missing →
value and value → missing are both changes; NaN, None and pd.NA are all
treated as missing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


def _is_numeric(s: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s)


def _equal_values(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality of two aligned series, assuming neither side is missing."""
    if _is_numeric(a) and _is_numeric(b):
        x = a.to_numpy(dtype=float, na_value=np.nan)
        y = b.to_numpy(dtype=float, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            return x == y
    if ptypes.is_datetime64_any_dtype(a) and ptypes.is_datetime64_any_dtype(b):
        return (a.to_numpy() == b.to_numpy())
    xa = a.astype(object).to_numpy()
    xb = b.astype(object).to_numpy()
    try:
        fast = xa == xb
        if isinstance(fast, np.ndarray) and fast.dtype == bool and fast.shape == xa.shape:
            return fast
    except (TypeError, ValueError):
        pass
    out = np.empty(len(xa), dtype=bool)
    for i, (u, v) in enumerate(zip(xa, xb)):
        try:
            r = u == v
            out[i] = bool(r) if not isinstance(r, np.ndarray) else bool(r.all())
        except (TypeError, ValueError):
            out[i] = False
    return out


def compute_impact(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    """Precise structural and cell-level impact of turning ``before`` into ``after``."""
    if not before.index.is_unique or not after.index.is_unique:
        raise ValueError("Row identity requires a unique index on both frames.")
    rows_b, rows_a = before.index, after.index
    common_rows = rows_b.intersection(rows_a, sort=False)
    rows_removed = rows_b.difference(rows_a, sort=False)
    rows_added = rows_a.difference(rows_b, sort=False)
    cols_b, cols_a = list(before.columns), list(after.columns)
    common_cols = [c for c in cols_b if c in set(cols_a)]
    cols_removed = [c for c in cols_b if c not in set(cols_a)]
    cols_added = [c for c in cols_a if c not in set(cols_b)]

    changed_rows = np.zeros(len(common_rows), dtype=bool)
    per_column: dict[str, int] = {}
    dtype_changes: dict[str, tuple[str, str]] = {}
    cells_changed = introduced = resolved = 0
    if len(common_rows) and common_cols:
        b_sub = before.loc[common_rows, common_cols]
        a_sub = after.loc[common_rows, common_cols]
        for c in common_cols:
            sb, sa = b_sub[c], a_sub[c]
            if str(sb.dtype) != str(sa.dtype):
                dtype_changes[c] = (str(sb.dtype), str(sa.dtype))
            na_b = sb.isna().to_numpy()
            na_a = sa.isna().to_numpy()
            both = ~na_b & ~na_a
            diff = na_b != na_a
            if both.any():
                eq = _equal_values(sb[both], sa[both])
                diff[np.flatnonzero(both)[~eq]] = True
            n = int(diff.sum())
            if n:
                per_column[c] = n
                changed_rows |= diff
            cells_changed += n
            introduced += int((~na_b & na_a).sum())
            resolved += int((na_b & ~na_a).sum())

    return {
        "cells_examined": int(len(common_rows) * len(common_cols)),
        "cells_changed": int(cells_changed),
        "rows_affected": int(changed_rows.sum()),
        "columns_affected": len(per_column),
        "changed_by_column": per_column,
        "rows_removed": int(len(rows_removed)),
        "rows_added": int(len(rows_added)),
        "columns_removed": cols_removed,
        "columns_added": cols_added,
        "dtype_changes": {k: {"before": v[0], "after": v[1]} for k, v in dtype_changes.items()},
        "missing_introduced": int(introduced),
        "missing_resolved": int(resolved),
        "removed_row_labels_sample": [str(x) for x in list(rows_removed[:20])],
        "changed_row_labels_sample": [str(x) for x in list(common_rows[changed_rows][:20])],
    }


def changed_mask(before: pd.DataFrame, after: pd.DataFrame, rows: pd.Index, columns: list[str]) -> pd.DataFrame:
    """Boolean frame (rows × columns) marking cells whose value changed, for highlighting a preview."""
    out = pd.DataFrame(False, index=rows, columns=columns)
    for c in columns:
        if c not in before.columns or c not in after.columns:
            continue
        sb, sa = before.loc[rows, c], after.loc[rows, c]
        na_b, na_a = sb.isna().to_numpy(), sa.isna().to_numpy()
        diff = na_b != na_a
        both = ~na_b & ~na_a
        if both.any():
            eq = _equal_values(sb[both], sa[both])
            diff[np.flatnonzero(both)[~eq]] = True
        out[c] = diff
    return out


def summary_lines(imp: dict[str, Any], detected: int | None = None) -> list[str]:
    lines = []
    if detected is not None:
        lines.append(f"Values flagged by the operation: {detected:,}")
    lines += [
        f"Cells examined: {imp['cells_examined']:,}",
        f"Values modified: {imp['cells_changed']:,}",
        f"Rows affected: {imp['rows_affected']:,}",
        f"Columns affected: {imp['columns_affected']:,}",
        f"Rows removed / added: {imp['rows_removed']:,} / {imp['rows_added']:,}",
        f"Columns removed / added: {len(imp['columns_removed'])} / {len(imp['columns_added'])}",
        f"Missing values introduced / resolved: {imp['missing_introduced']:,} / {imp['missing_resolved']:,}",
    ]
    return lines
