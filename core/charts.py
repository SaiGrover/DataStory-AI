"""Chart specifications: a strict, data-only schema for every generated chart.

A ``ChartSpec`` is validated against the active dataset and then executed by
``render``. Specs never contain code: chart type, columns, aggregation, filters
and display options are plain values checked against allow-lists. The same
spec drives the EDA recommendations, natural-language charts, the copilot,
saved charts in reports, session files and notebook export.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core import eda
from core.formatting import num, pct
from core.profiler import infer_kinds, multi_value_separator

CHART_TYPES = {
    "histogram": "Histogram", "box": "Box plot", "violin": "Violin plot", "bar": "Bar chart",
    "line": "Line chart", "scatter": "Scatter plot", "heatmap": "Correlation heatmap",
    "stacked_bar": "Stacked share bar", "missing": "Missing values by column",
}
AGGS = ["count", "mean", "median", "sum", "min", "max", "rate"]
FREQS = {"D": "day", "W": "week", "MS": "month", "QS": "quarter", "YS": "year"}
FILTER_OPS = ["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains"]
MAX_TOP_N = 50
LABEL_MAX = 28


class ChartSpecError(ValueError):
    """The specification cannot be executed against the active dataset."""


@dataclass
class ChartSpec:
    chart: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str = "count"
    freq: str | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    sort: str = "desc"  # desc | asc | none
    top_n: int | None = 15
    other: bool = True  # group remaining categories into "Other"
    dropna: bool = True
    explode: bool = False  # split comma-separated values ("United States, India")
    columns: list[str] = field(default_factory=list)  # heatmap
    normalize: bool = True  # stacked_bar: shares within each bar (True) or counts (False)
    bins: int = 30
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChartSpec":
        if not isinstance(d, dict):
            raise ChartSpecError("A chart specification must be an object.")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ChartSpecError(f"Unknown chart field(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:10]

    def describe(self) -> str:
        bits = [CHART_TYPES.get(self.chart, self.chart)]
        if self.chart == "heatmap":
            bits.append(f"of {len(self.columns)} columns")
        else:
            if self.y and self.agg not in ("count",):
                bits.append(f"{self.agg} of {self.y}")
            if self.x:
                bits.append(f"by {self.x}")
            if self.color:
                bits.append(f"split by {self.color}")
        if self.filters:
            bits.append("where " + " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in self.filters))
        return " ".join(bits)


@dataclass
class ChartOutput:
    figure: go.Figure | None
    table: pd.DataFrame
    interpretation: str
    notes: list[str]
    rows_used: int


# --------------------------------------------------------------- validation
def _kinds(df: pd.DataFrame, kinds: dict[str, str] | None) -> dict[str, str]:
    return kinds if kinds is not None else infer_kinds(df)


def is_yearlike(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return bool(len(x)) and bool((x == x.round()).all()) and x.between(1800, 2100).all() and x.nunique() > 3


def validate(df: pd.DataFrame, spec: ChartSpec, kinds: dict[str, str] | None = None) -> list[str]:
    """Return a list of readable problems; empty means the spec can run."""
    k = _kinds(df, kinds)
    errs: list[str] = []
    if spec.chart not in CHART_TYPES:
        return [f"Unsupported chart type '{spec.chart}'. Supported: {', '.join(CHART_TYPES)}."]
    if spec.agg not in AGGS:
        errs.append(f"Unsupported aggregation '{spec.agg}'. Supported: {', '.join(AGGS)}.")
    if spec.freq is not None and spec.freq not in FREQS:
        errs.append(f"Unsupported time frequency '{spec.freq}'.")
    if spec.sort not in ("desc", "asc", "none"):
        errs.append("Sort must be desc, asc or none.")
    if spec.top_n is not None and not (1 <= int(spec.top_n) <= MAX_TOP_N):
        errs.append(f"Top-N must be between 1 and {MAX_TOP_N}.")
    if not (5 <= int(spec.bins) <= 200):
        errs.append("Bins must be between 5 and 200.")
    for c in [spec.x, spec.y, spec.color, *spec.columns]:
        if c is not None and c not in df.columns:
            errs.append(f"Column '{c}' is not in the dataset.")
    for f in spec.filters:
        if not isinstance(f, dict) or {"column", "op", "value"} - set(f):
            errs.append("Each filter needs column, op and value.")
            continue
        if f["column"] not in df.columns:
            errs.append(f"Filter column '{f['column']}' is not in the dataset.")
        if f["op"] not in FILTER_OPS:
            errs.append(f"Unsupported filter operator '{f['op']}'.")
        if f["op"] in (">", ">=", "<", "<=") and f["column"] in df.columns and k.get(f["column"]) not in ("numeric", "datetime"):
            errs.append(f"'{f['op']}' needs a numeric or date column; '{f['column']}' is {k.get(f['column'])}.")
    if errs:
        return errs

    def need(col, allowed, role):
        want = " or ".join(allowed)
        if col is None:
            errs.append(f"This chart needs {role}.")
        elif k[col] not in allowed:
            errs.append(f"'{col}' is {k[col]}, but this chart needs a {want} column here.")

    c = spec.chart
    if c in ("histogram",):
        need(spec.x, ("numeric", "datetime"), "a numeric column")
    elif c in ("box", "violin"):
        need(spec.y, ("numeric",), "a numeric value column")
        if spec.x is not None and k[spec.x] not in ("categorical", "numeric"):
            errs.append(f"Grouping column '{spec.x}' must be categorical.")
    elif c == "bar":
        if spec.x is None:
            errs.append("A bar chart needs a category column.")
        if spec.agg != "count":
            if spec.y is None:
                errs.append(f"'{spec.agg}' needs a value column.")
            elif spec.agg == "rate":
                if df[spec.y].nunique(dropna=True) != 2:
                    errs.append(f"A rate needs a two-valued column; '{spec.y}' has {df[spec.y].nunique(dropna=True)} values.")
            else:
                need(spec.y, ("numeric",), "a numeric value column")
    elif c == "line":
        if spec.x is None:
            errs.append("A line chart needs a date (or year) column.")
        elif k[spec.x] != "datetime" and not (k[spec.x] == "numeric" and is_yearlike(df[spec.x])):
            errs.append(f"'{spec.x}' is not a date or year column, so it cannot be plotted over time.")
        if spec.agg != "count":
            need(spec.y, ("numeric",), "a numeric value column")
    elif c == "scatter":
        need(spec.x, ("numeric",), "a numeric x column")
        need(spec.y, ("numeric",), "a numeric y column")
    elif c == "heatmap":
        cols = spec.columns or [x for x, kk in k.items() if kk == "numeric"]
        if len([x for x in cols if k.get(x) == "numeric"]) < 2:
            errs.append("A correlation heatmap needs at least two numeric columns.")
    elif c == "stacked_bar":
        if spec.x is None or spec.color is None:
            errs.append("A stacked share bar needs a category column and a column to split by.")
    return errs


# --------------------------------------------------------------- execution
def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    out = df
    for f in filters:
        s, op, v = out[f["column"]], f["op"], f["value"]
        if op in (">", ">=", "<", "<="):
            if pd.api.types.is_datetime64_any_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                s = pd.to_datetime(s, errors="coerce", format="mixed")
                v = pd.to_datetime(v)
            else:
                v = float(v)
            mask = {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
        elif op in ("in", "not_in"):
            vals = [str(x) for x in (v if isinstance(v, list) else [v])]
            mask = s.astype(str).isin(vals)
            mask = ~mask if op == "not_in" else mask
        elif op == "contains":
            mask = s.astype(str).str.contains(str(v), case=False, regex=False, na=False)
        else:
            eq = s.astype(str).str.lower() == str(v).lower()
            if pd.api.types.is_numeric_dtype(s):
                try:
                    eq = eq | (s == float(v))
                except (TypeError, ValueError):
                    pass
            mask = eq if op == "==" else ~eq
        out = out[mask.fillna(False)]
    return out


def _short(label: Any) -> str:
    t = str(label)
    return t if len(t) <= LABEL_MAX else t[: LABEL_MAX - 1] + "…"


def _categories(s: pd.Series, explode: bool, dropna: bool) -> pd.Series:
    x = s.astype("object")
    if explode:
        x = x.dropna().astype(str).str.split(",").explode().str.strip()
        x = x[x != ""]
        tmp = x.to_frame("v")
        tmp["_row"] = tmp.index
        x = tmp.drop_duplicates(["_row", "v"])["v"]  # a value counts at most once per row
    if dropna:
        return x.dropna().astype(str)
    return x.where(x.notna(), "(missing)").astype(str)


def _bar_figure(table: pd.DataFrame, cat: str, val: str, title: str, value_label: str) -> go.Figure:
    labels = table[cat].astype(str).tolist()
    long = max((len(x) for x in labels), default=0) > 12 or len(labels) > 8
    short = [_short(x) for x in labels]
    if long:  # horizontal, largest on top
        fig = go.Figure(go.Bar(x=table[val][::-1], y=short[::-1], orientation="h", customdata=labels[::-1],
                               marker_color=eda.SAGE, hovertemplate="%{customdata}: %{x:,.4g}<extra></extra>"))
        fig.update_layout(xaxis_title=value_label, yaxis_title=None)
        fig.update_yaxes(type="category", automargin=True)
        height = max(300, 26 * len(labels) + 110)
    else:
        fig = go.Figure(go.Bar(x=short, y=table[val], customdata=labels, marker_color=eda.SAGE,
                               hovertemplate="%{customdata}: %{y:,.4g}<extra></extra>"))
        fig.update_layout(yaxis_title=value_label, xaxis_title=cat)
        fig.update_xaxes(type="category", tickangle=0 if len(labels) <= 6 else -30, automargin=True)
        height = 380
    return eda.style(fig, title, height=height)


def _positive_class(s: pd.Series) -> Any:
    vals = list(pd.Series(s.dropna().unique()))
    for cand in vals:
        if str(cand).strip().lower() in ("yes", "true", "1", "1.0", "y", "churned", "left", "positive"):
            return cand
    return sorted(vals, key=str)[-1]


def render(df: pd.DataFrame, spec: ChartSpec, kinds: dict[str, str] | None = None) -> ChartOutput:
    """Execute a validated spec. Raises ChartSpecError with a readable message."""
    k = _kinds(df, kinds)
    errs = validate(df, spec, k)
    if errs:
        raise ChartSpecError(" ".join(errs))
    data = apply_filters(df, spec.filters) if spec.filters else df
    notes: list[str] = []
    if spec.filters:
        notes.append(f"Filters kept {len(data):,} of {len(df):,} rows.")
    if data.empty:
        raise ChartSpecError("No rows match the filters, so there is nothing to plot.")
    c, title = spec.chart, spec.title
    top = spec.top_n

    if c == "histogram":
        x = data[spec.x]
        if k[spec.x] == "datetime":
            x = pd.to_datetime(x, errors="coerce", format="mixed")
        used = x.dropna()
        fig = eda.histogram(data.assign(**{spec.x: x}), spec.x, spec.bins, spec.color if spec.color in data else None)
        eda.style(fig, title or f"Distribution of {spec.x}")
        if k[spec.x] == "datetime":
            table = pd.DataFrame({"min": [used.min()], "max": [used.max()], "rows": [len(used)]})
            interp = f"**{spec.x}** spans {used.min():%Y-%m-%d} to {used.max():%Y-%m-%d} across {len(used):,} dated rows."
        else:
            xs = pd.to_numeric(used, errors="coerce").dropna()
            table = xs.describe().to_frame("value").T
            f = (lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}") if is_yearlike(xs) else num
            interp = (f"Across {len(xs):,} non-missing values, **{spec.x}** ranges from {f(xs.min())} to {f(xs.max())}; "
                      f"median {f(xs.median())}, mean {f(xs.mean())}.")
        if len(used) < len(data):
            notes.append(f"{len(data) - len(used):,} missing values are excluded.")
        return ChartOutput(fig, table.reset_index(drop=True), interp, notes, len(used))

    if c in ("box", "violin"):
        g = spec.x
        d = data[[spec.y] + ([g] if g else [])].dropna(subset=[spec.y])
        if g:
            keep = d[g].astype(str).value_counts().head(top or 15).index
            if d[g].nunique() > len(keep):
                notes.append(f"Showing the {len(keep)} most frequent groups of {d[g].nunique():,}.")
            d = d[d[g].astype(str).isin(keep)]
        fig = eda.box_or_violin(d, spec.y, g, "violin" if c == "violin" else "box")
        eda.style(fig, title or f"{spec.y}" + (f" by {g}" if g else ""), height=420)
        if g:
            table = d.groupby(d[g].astype(str))[spec.y].agg(["median", "mean", "count"]).sort_values("median", ascending=False).reset_index()
            hi, lo = table.iloc[0], table.iloc[-1]
            interp = (f"Median **{spec.y}** is highest for '{hi[g]}' ({num(hi['median'])}, n = {int(hi['count']):,}) and lowest for "
                      f"'{lo[g]}' ({num(lo['median'])}, n = {int(lo['count']):,}). These are observed differences, not causal effects.")
        else:
            table = d[spec.y].describe().to_frame("value").T.reset_index(drop=True)
            interp = f"**{spec.y}**: median {num(d[spec.y].median())}, interquartile range {num(d[spec.y].quantile(.25))}–{num(d[spec.y].quantile(.75))}."
        return ChartOutput(fig, table, interp, notes, len(d))

    if c == "bar":
        cats = _categories(data[spec.x], spec.explode, spec.dropna) if k[spec.x] != "datetime" else \
            pd.to_datetime(data[spec.x], errors="coerce", format="mixed").dt.year.dropna().astype(int).astype(str)
        if spec.explode:
            notes.append(f"Comma-separated values in {spec.x} were split, so one row can count toward several categories.")
        if spec.agg == "count":
            vc = cats.value_counts()
            table = vc.rename_axis(spec.x).reset_index(name="count")
            val, label = "count", "Rows"
            denom = int(data[spec.x].notna().sum()) if spec.explode or spec.dropna else len(data)
        else:
            vals = data.loc[cats.index, spec.y] if not spec.explode else data.loc[cats.index, spec.y]
            frame = pd.DataFrame({spec.x: cats.values, "v": vals.values})
            if spec.agg == "rate":
                pos = _positive_class(data[spec.y])
                frame["v"] = (frame["v"].astype(str) == str(pos)).astype(float).where(frame["v"].notna())
                frame = frame.dropna()
                table = frame.groupby(spec.x)["v"].agg(["mean", "count"]).rename(columns={"mean": "rate"})
                table["rate"] = table["rate"] * 100
                val, label = "rate", f"% {spec.y} = '{pos}'"
                notes.append(f"Rate = share of rows in each group where {spec.y} is '{pos}'.")
            else:
                frame["v"] = pd.to_numeric(frame["v"], errors="coerce")
                frame = frame.dropna()
                table = frame.groupby(spec.x)["v"].agg([spec.agg, "count"]).rename(columns={spec.agg: f"{spec.agg}_{spec.y}"})
                val, label = f"{spec.agg}_{spec.y}", f"{spec.agg.title()} of {spec.y}"
            table = table.rename(columns={"count": "n"}).reset_index()
            denom = None
        if spec.sort != "none":
            table = table.sort_values(val, ascending=spec.sort == "asc", kind="stable")
        n_cat = len(table)
        if top and n_cat > top:
            rest = table.iloc[top:]
            table = table.iloc[:top]
            if spec.other and spec.agg in ("count", "sum"):
                table = pd.concat([table, pd.DataFrame({spec.x: [f"Other ({len(rest):,} categories)"], val: [rest[val].sum()]})],
                                  ignore_index=True)
            notes.append(f"Showing the top {top} of {n_cat:,} categories" + (" plus an 'Other' bucket." if spec.other and spec.agg in ("count", "sum") else "."))
        if spec.agg == "count" and denom:
            table["share_pct"] = 100 * table["count"] / denom
        default_title = (f"Rows by {spec.x}" if spec.agg == "count" else f"{label} by {spec.x}")
        fig = _bar_figure(table, spec.x, val, title or default_title, label)
        first = table.iloc[0]
        if spec.agg == "count":
            interp = (f"**{spec.x}** has {n_cat:,} distinct values. The most frequent is '{first[spec.x]}' in "
                      f"{int(first['count']):,} rows ({pct(first['share_pct'])} of the {denom:,} rows with a {spec.x} value).")
            if spec.explode:
                interp += " A row listing several values counts once for each, so shares can sum to more than 100%."
        elif spec.agg == "rate":
            interp = (f"'{first[spec.x]}' has the {'highest' if spec.sort != 'asc' else 'lowest'} rate shown: {pct(first['rate'])} "
                      f"(n = {int(first['n']):,}). Rates in small groups are unstable; differences are associations, not causes.")
        else:
            interp = (f"'{first[spec.x]}' has the {'highest' if spec.sort != 'asc' else 'lowest'} {spec.agg} of {spec.y} shown "
                      f"({num(first[val])}, n = {int(first['n']):,}).")
            small = int((table.get("n", pd.Series(dtype=float)) < 30).sum())
            if small:
                interp += f" {small} group(s) have fewer than 30 rows, so their values are less stable."
        return ChartOutput(fig, table, interp, notes, len(cats))

    if c == "line":
        is_dt = k[spec.x] == "datetime"
        t = pd.to_datetime(data[spec.x], errors="coerce", format="mixed") if is_dt else pd.to_numeric(data[spec.x], errors="coerce")
        v = pd.Series(1.0, index=data.index) if spec.agg == "count" else pd.to_numeric(data[spec.y], errors="coerce")
        d = pd.DataFrame({"t": t, "v": v}).dropna()
        if d.empty:
            raise ChartSpecError("No rows have both a usable time value and a value to aggregate.")
        agg = "sum" if spec.agg == "count" else ("mean" if spec.agg == "rate" else spec.agg)
        if is_dt:
            freq = spec.freq or "MS"
            table = d.set_index("t")["v"].resample(freq).agg(agg).reset_index()
            if spec.agg != "count":
                table = table.dropna()
            unit = FREQS[freq]
        else:
            table = d.groupby("t")["v"].agg(agg).reset_index().sort_values("t")
            unit = "year"
        ylabel = "Rows" if spec.agg == "count" else f"{spec.agg.title()} of {spec.y}"
        table.columns = [spec.x, ylabel]
        fig = px.line(table, x=spec.x, y=ylabel, markers=len(table) <= 60, color_discrete_sequence=[eda.SAGE])
        eda.style(fig, title or (f"Rows per {unit} by {spec.x}" if spec.agg == "count" else f"{ylabel} per {unit}"), height=400)
        hi = table.loc[table[ylabel].idxmax()]
        fmt_t = (lambda x: f"{x:%Y-%m-%d}") if is_dt else (lambda x: f"{int(x)}")
        interp = (f"Across {len(table):,} {unit}s from {fmt_t(table[spec.x].min())} to {fmt_t(table[spec.x].max())}, the peak is "
                  f"{num(hi[ylabel])} at {fmt_t(hi[spec.x])}. This describes recorded periods only; it is not a forecast.")
        if is_dt and spec.agg == "count" and len(table) > 2 and table[ylabel].iloc[-1] < 0.5 * table[ylabel].median():
            notes.append("The last period may be incomplete (the data may end part-way through it).")
        return ChartOutput(fig, table, interp, notes, len(d))

    if c == "scatter":
        d = data[[x for x in dict.fromkeys([spec.x, spec.y, spec.color]) if x]].dropna(subset=[spec.x, spec.y])
        n_all = len(d)
        if n_all > 5000:
            d = d.sample(5000, random_state=42)
            notes.append(f"Plotting a seeded random sample of 5,000 of {n_all:,} points; the correlation uses all {n_all:,}.")
        fig = eda.scatter(d, spec.x, spec.y, spec.color)
        eda.style(fig, title or f"{spec.y} vs {spec.x}", height=440)
        full = data[[spec.x, spec.y]].apply(pd.to_numeric, errors="coerce").dropna()
        r = full[spec.x].corr(full[spec.y]) if len(full) > 2 else np.nan
        table = pd.DataFrame({"pearson_r": [r], "rows": [len(full)]})
        interp = eda.interpret_scatter(full, spec.x, spec.y)
        return ChartOutput(fig, table, interp, notes, n_all)

    if c == "heatmap":
        cols = [x for x in (spec.columns or [x for x, kk in k.items() if kk == "numeric"]) if k.get(x) == "numeric"]
        cols = [x for x in cols if data[x].nunique() > 1][:25]
        fig, corr = eda.correlation_heatmap(data, cols)
        if title:
            eda.style(fig, title, height=fig.layout.height)
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if pd.notna(corr.loc[a, b])]
        pairs.sort(key=lambda p: -abs(p[2]))
        interp = ("Strongest pairs: " + "; ".join(f"{a} ↔ {b} (r = {r:.2f})" for a, b, r in pairs[:3]) + ". " + eda.CAUSAL_NOTE) \
            if pairs else "No correlations could be computed."
        return ChartOutput(fig, corr.round(3).reset_index().rename(columns={"index": "column"}), interp, notes, len(data))

    if c == "stacked_bar":
        d = data[[spec.x, spec.color]].dropna()
        if spec.explode:
            xs = _categories(d[spec.x], True, True)
            d = pd.DataFrame({spec.x: xs.values, spec.color: d.loc[xs.index, spec.color].astype(str).values})
            notes.append(f"Comma-separated values in {spec.x} were split, so one row can count toward several bars.")
        else:
            d = d.astype(str)
        keep = d[spec.x].value_counts().head(top or 12).index
        if d[spec.x].nunique() > len(keep):
            notes.append(f"Showing the {len(keep)} most frequent values of {spec.x} (of {d[spec.x].nunique():,}).")
        d = d[d[spec.x].isin(keep)]
        if d[spec.color].nunique() > 12:
            raise ChartSpecError(f"'{spec.color}' has {d[spec.color].nunique()} values; choose a split column with at most 12.")
        counts_tbl = pd.crosstab(d[spec.x], d[spec.color]).loc[list(keep)]
        table = counts_tbl.div(counts_tbl.sum(axis=1), axis=0).mul(100) if spec.normalize else counts_tbl
        vname = "share_pct" if spec.normalize else "rows"
        long = table.reset_index().melt(id_vars=spec.x, var_name=spec.color, value_name=vname)
        long["label"] = long[spec.x].map(_short)
        fig = px.bar(long, x="label", y=vname, color=spec.color, barmode="stack" if spec.normalize else "group",
                     color_discrete_sequence=eda.PALETTE,
                     hover_data={spec.x: True, "label": False, vname: ":.1f" if spec.normalize else ":,"})
        fig.update_xaxes(type="category", title=spec.x, automargin=True, tickangle=0 if len(keep) <= 6 else -30)
        fig.update_layout(yaxis_title=f"% of rows within each {spec.x}" if spec.normalize else "Rows")
        eda.style(fig, title or (f"{spec.color} mix within {spec.x}" if spec.normalize else f"Rows by {spec.x} and {spec.color}"), height=420)
        sizes = counts_tbl.sum(axis=1)
        if spec.normalize:
            interp = (f"Each bar shows the mix of **{spec.color}** within a value of **{spec.x}** "
                      f"(bars cover {int(sizes.min()):,} to {int(sizes.max()):,} rows). Differences are associations, not causal effects.")
        else:
            lead = counts_tbl.sum(axis=0).sort_values(ascending=False)
            unit = f"{spec.x} entries" if spec.explode else "rows"
            interp = (f"Across the {len(keep)} {spec.x} values shown, '{lead.index[0]}' accounts for {int(lead.iloc[0]):,} {unit} and "
                      f"'{lead.index[-1]}' for {int(lead.iloc[-1]):,}. The largest total is '{sizes.idxmax()}' ({int(sizes.max()):,} {unit}).")
        return ChartOutput(fig, table.round(1).reset_index(), interp, notes, len(d))

    if c == "missing":
        miss = data.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            raise ChartSpecError("No column has missing values.")
        table = pd.DataFrame({"column": miss.index, "missing": miss.values, "missing_pct": 100 * miss.values / len(data)})
        fig = _bar_figure(table, "column", "missing_pct", title or "Missing values by column", "% of rows missing")
        top_row = table.iloc[0]
        interp = (f"{len(table)} column(s) have missing values. '{top_row['column']}' is missing in {int(top_row['missing']):,} of "
                  f"{len(data):,} rows ({pct(top_row['missing_pct'])}).")
        return ChartOutput(fig, table, interp, notes, len(data))

    raise ChartSpecError(f"Unsupported chart type '{c}'.")


# ------------------------------------------------------- recommendations
_MEASURE = ("revenue", "sales", "amount", "price", "profit", "cost", "income", "salary", "spend", "charges", "total", "quantity", "cnt", "count")
_GROUP = ("region", "category", "segment", "department", "product", "country", "city", "state", "type", "channel", "gender",
          "contract", "rating", "class", "store", "genre", "listed_in")


def _has(name: str, words: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(w in n for w in words)


def usable_categoricals(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
    """Categorical columns worth charting: not identifiers, not near-unique, not constant."""
    details = {c["name"]: c for c in profile["column_details"]}
    out = []
    for c, kd in profile["kinds"].items():
        d = details[c]
        if c in profile["id_candidates"] or c in profile["constant_columns"]:
            continue
        if kd == "categorical" and d["unique"] <= 60:
            out.append(c)
        elif kd in ("categorical", "text") and multi_value_separator(df[c]) and d["unique"] > 1:
            out.append(c)
    return out


def chart_suitability(df: pd.DataFrame, col: str, profile: dict[str, Any]) -> str | None:
    """Reason a column should not get a frequency chart, or None when it is fine."""
    d = next((x for x in profile["column_details"] if x["name"] == col), None)
    if d is None:
        return "Column not found."
    if col in profile["id_candidates"]:
        return f"'{col}' looks like an identifier ({d['unique']:,} distinct values); a frequency chart would show one bar per row."
    if d.get("unique_pct") and d["unique_pct"] > 90 and d["unique"] > 50 and not multi_value_separator(df[col]):
        return f"'{col}' is almost unique per row ({d['unique']:,} distinct values, {pct(d['unique_pct'])} of non-missing rows)."
    return None


def recommend(df: pd.DataFrame, profile: dict[str, Any], target: str | None = None, domain: str | None = None,
              limit: int = 8) -> list[dict[str, Any]]:
    """Ranked, deduplicated chart suggestions with the reasoning behind each."""
    kinds = profile["kinds"]
    details = {c["name"]: c for c in profile["column_details"]}
    ids = set(profile["id_candidates"]) | set(profile["constant_columns"])
    nums = [c for c, k in kinds.items() if k == "numeric" and c not in ids and details[c]["unique"] > 1]
    years = [c for c in nums if "year" in c.lower() and is_yearlike(df[c])]
    cont = [c for c in nums if c not in years and details[c]["unique"] > 12]
    dates = [c for c, k in kinds.items() if k == "datetime"]
    cats = usable_categoricals(df, profile)
    multi = {c for c in cats if multi_value_separator(df[c])}
    recs: list[dict[str, Any]] = []

    def add(score, spec, why, prereq, limits):
        recs.append({"id": spec.key(), "title": spec.title or spec.describe(), "chart_type": CHART_TYPES[spec.chart],
                     "spec": spec, "columns": [x for x in [spec.x, spec.y, spec.color, *spec.columns] if x],
                     "why": why, "prerequisites": prereq, "limitations": limits, "score": round(score, 3)})

    tdisc = None
    if target and target in df.columns:
        nt = df[target].nunique(dropna=True)
        tdisc = kinds[target] != "numeric" or nt <= 12
        if tdisc and nt <= 30:
            add(1.0, ChartSpec("bar", x=target, top_n=30, other=False, title=f"Class balance of {target}"),
                f"{target} is the selected target; its class balance decides which metrics are meaningful.",
                [f"{target} with ≤30 classes"], ["Counts only; says nothing about why classes differ."])
            for c in cont[:2]:
                add(0.9, ChartSpec("box", x=target, y=c, title=f"{c} by {target}"),
                    f"Compares the distribution of {c} across {target} classes.", [f"numeric {c}"],
                    ["Shows association with the target, not causation."])
            for c in [x for x in cats if x != target and details[x]["unique"] <= 20 and x not in multi][:2]:
                if nt == 2:
                    add(0.88, ChartSpec("bar", x=c, y=target, agg="rate", sort="desc", top_n=15, title=f"{target} rate by {c}"),
                        f"Shows how the share of {target} differs between {c} groups.", [f"{c} with few categories"],
                        ["Small groups give unstable rates."])
                else:
                    add(0.85, ChartSpec("stacked_bar", x=c, color=target, title=f"{target} mix within {c}"),
                        f"Shows the {target} mix within each {c} value.", [f"{c} with few categories"], ["Associations only."])
        elif kinds[target] == "numeric":
            add(1.0, ChartSpec("histogram", x=target, title=f"Distribution of {target}"),
                f"{target} is the selected target; its distribution affects which regression errors matter.", [], [])
            pairs = sorted(((abs(pd.to_numeric(df[c], errors="coerce").corr(pd.to_numeric(df[target], errors="coerce"))), c)
                            for c in cont if c != target), reverse=True)
            for r, c in [p for p in pairs if p[0] == p[0]][:2]:
                add(0.9, ChartSpec("scatter", x=c, y=target, title=f"{target} vs {c}"),
                    f"{c} has the strongest linear association with {target} (|r| = {r:.2f}).", [], ["Correlation is not causation."])

    measures = [c for c in cont if _has(c, _MEASURE)]
    groups = sorted([c for c in cats if c != target], key=lambda c: (not _has(c, _GROUP), details[c]["unique"]))
    for d in dates[:2]:
        add(0.86, ChartSpec("line", x=d, agg="count", freq="MS", title=f"Records per month by {d}"),
            f"{d} holds dates; volume over time reveals growth, seasonality and gaps.", [f"parseable dates in {d}"],
            ["The first and last months may be partial."])
        for m in measures[:1]:
            add(0.87, ChartSpec("line", x=d, y=m, agg="sum", freq="MS", title=f"Total {m} per month"),
                f"{m} looks like a measure; its monthly total is a core trend view.", [], ["Sums depend on complete recording."])
    for yv in years[:1]:
        add(0.84, ChartSpec("line", x=yv, agg="count", title=f"Rows by {yv}"),
            f"{yv} holds years; counting rows per year shows how the records are spread over time.", [], [])
    for m in measures[:2]:
        for g in groups[:2]:
            add(0.83, ChartSpec("bar", x=g, y=m, agg="sum", top_n=15, explode=g in multi, title=f"Total {m} by {g}"),
                f"Compares the total of {m} across {g} groups.", [], ["Totals reflect group size as well as per-row values."])
    for g in groups[:4]:
        n = details[g]["unique"]
        s = 0.8 - 0.02 * groups.index(g) + (0.04 if _has(g, _GROUP) else 0)
        add(s, ChartSpec("bar", x=g, top_n=15, explode=g in multi, title=f"Rows by {g}"),
            f"{g} has {n:,} categories" + (" (comma-separated lists are split)" if g in multi else "") + "; frequencies show its composition.",
            [], ["Top 15 shown; the rest are grouped into 'Other'."] if n > 15 else [])
    for c in sorted(cont, key=lambda c: -(abs(details[c].get("skew") or 0)))[:2]:
        add(0.7, ChartSpec("histogram", x=c, title=f"Distribution of {c}"),
            f"{c} is numeric" + (f" with skewness {details[c]['skew']:.2f}" if details[c].get("skew") is not None else "") + ".", [], [])
    if len(cont) >= 3:
        add(0.72, ChartSpec("heatmap", columns=cont[:15], title="Correlation between numeric columns"),
            f"{len(cont)} continuous numeric columns; the heatmap surfaces redundant or related measures.", [],
            ["Pearson r only captures linear association.", eda.CAUSAL_NOTE])
    if len(cont) >= 2 and not target:
        add(0.65, ChartSpec("scatter", x=cont[0], y=cont[1], title=f"{cont[1]} vs {cont[0]}"),
            "Pairwise relationship between two numeric columns.", [], ["Correlation is not causation."])
    if profile["missing_cells"]:
        add(0.75, ChartSpec("missing", title="Missing values by column"),
            f"{profile['columns_with_missing']} column(s) contain missing values.", [], [])
    for c in [x["column"] for x in profile["outlier_columns"][:1] if x["column"] in cont]:
        add(0.6, ChartSpec("box", y=c, title=f"Spread and outliers of {c}"),
            f"{c} has values outside the Tukey fences.", [], ["Flagged values are not necessarily errors."])

    seen, out = set(), []
    for r in sorted(recs, key=lambda r: -r["score"]):
        if r["id"] in seen:
            continue
        if validate(df, r["spec"], kinds):
            continue
        seen.add(r["id"])
        out.append(r)
    return out[:limit]
