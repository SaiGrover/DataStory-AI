"""Interactive EDA: Plotly figure builders and evidence-grounded interpretations.

Interpretation helpers only restate numbers computed from the data they are
given. They describe association, never causation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.formatting import pct as _pct

SAGE, MOSS, INK, MUTED, LINE = "#667F4D", "#93AD74", "#172018", "#697468", "#E1E8D8"
PALETTE = ["#667F4D", "#C98B3A", "#4F7C8A", "#A65D57", "#93AD74", "#8A6FA8", "#D2B24C", "#3F5B45", "#7D8B99", "#B9745F"]
SEQUENTIAL = [[0, "#F4F7EF"], [0.5, "#93AD74"], [1, "#2F4A2A"]]
DIVERGING = [[0, "#A65D57"], [0.5, "#FBFBF5"], [1, "#4E6B3A"]]
AGGREGATIONS = ["mean", "median", "sum", "count", "min", "max", "std"]
CAUSAL_NOTE = "Correlation describes how values move together in this dataset; it does not show that one variable causes the other."


def style(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=INK, size=12),
        colorway=PALETTE, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna().astype(float)


# ------------------------------------------------------------------ figures
def histogram(df: pd.DataFrame, col: str, bins: int = 30, color: str | None = None, kde: bool = False) -> go.Figure:
    data = df[[col] + ([color] if color else [])].dropna(subset=[col])
    fig = px.histogram(data, x=col, color=color, nbins=bins, histnorm="probability density" if kde else None,
                       barmode="overlay" if color else "relative", opacity=0.75 if color else 0.9,
                       color_discrete_sequence=PALETTE)
    if kde and not color:
        x = _num(data[col])
        if len(x) > 2 and x.nunique() > 1:
            from scipy.stats import gaussian_kde

            grid = np.linspace(x.min(), x.max(), 200)
            fig.add_trace(go.Scatter(x=grid, y=gaussian_kde(x)(grid), mode="lines", name="KDE",
                                     line=dict(color="#C98B3A", width=2)))
    fig.update_layout(bargap=0.03, yaxis_title="Density" if kde else "Count")
    return style(fig, f"Distribution of {col}")


def box_or_violin(df: pd.DataFrame, col: str, group: str | None = None, kind: str = "box") -> go.Figure:
    data = df[[col] + ([group] if group else [])].dropna(subset=[col])
    if group:
        data = data.assign(**{group: data[group].astype(str)})
    f = px.violin if kind == "violin" else px.box
    kwargs: dict[str, Any] = dict(box=True) if kind == "violin" else {}
    fig = f(data, y=col, x=group, color=group, color_discrete_sequence=PALETTE, points="outliers", **kwargs)
    fig.update_layout(showlegend=False)
    return style(fig, f"{col}" + (f" by {group}" if group else ""))


def category_counts(df: pd.DataFrame, col: str, top_n: int = 15) -> tuple[go.Figure, pd.DataFrame]:
    vc = df[col].astype("object").where(df[col].notna(), "(missing)").astype(str).value_counts()
    table = vc.head(top_n).rename_axis(col).reset_index(name="count")
    table["share_pct"] = 100 * table["count"] / len(df)
    fig = px.bar(table.iloc[::-1], x="count", y=col, orientation="h", color_discrete_sequence=[SAGE],
                 hover_data={"share_pct": ":.1f"})
    fig.update_yaxes(type="category")
    return style(fig, f"Counts of {col}" + (f" (top {top_n} of {len(vc)})" if len(vc) > top_n else ""),
                 height=max(280, 28 * len(table) + 90)), table


def correlation_heatmap(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> tuple[go.Figure, pd.DataFrame]:
    corr = df[cols].corr(method=method, numeric_only=True)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=corr.round(2).values, texttemplate="%{text}" if len(cols) <= 14 else None,
                               hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, f"{method.title()} correlation", height=max(380, 30 * len(cols) + 120)), corr


def missing_bar(profile: dict[str, Any]) -> go.Figure | None:
    rows = [(c["name"], c["missing_pct"], c["missing"]) for c in profile["column_details"] if c["missing"]]
    if not rows:
        return None
    t = pd.DataFrame(rows, columns=["column", "missing_pct", "missing"]).sort_values("missing_pct")
    fig = px.bar(t, x="missing_pct", y="column", orientation="h", color_discrete_sequence=["#C98B3A"],
                 hover_data={"missing": ":,"})
    fig.update_layout(xaxis_title="% of rows missing", yaxis_title=None)
    return style(fig, "Missing values by column", height=max(260, 28 * len(t) + 90))


def missing_matrix(df: pd.DataFrame, max_rows: int = 400, seed: int = 42) -> go.Figure | None:
    cols = [c for c in df.columns if df[c].isna().any()]
    if not cols:
        return None
    sub = df[cols] if len(df) <= max_rows else df[cols].sample(max_rows, random_state=seed).sort_index()
    fig = go.Figure(go.Heatmap(z=sub.isna().astype(int).values, x=cols, y=sub.index.astype(str), showscale=False,
                               colorscale=[[0, "#EAF0E2"], [1, "#C98B3A"]],
                               hovertemplate="row %{y}, %{x}: %{z}<extra>1 = missing</extra>"))
    fig.update_yaxes(showticklabels=False, title="rows" + (f" (seeded sample of {max_rows})" if len(df) > max_rows else ""))
    return style(fig, "Where values are missing (orange = missing)", height=380)


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, max_points: int = 5000, seed: int = 42) -> go.Figure:
    data = df[[c for c in dict.fromkeys([x, y, color]) if c]].dropna(subset=[x, y])
    if len(data) > max_points:
        data = data.sample(max_points, random_state=seed)
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7, color_discrete_sequence=PALETTE,
                     color_continuous_scale=SEQUENTIAL)
    return style(fig, f"{y} vs {x}", height=440)


def scatter_matrix(df: pd.DataFrame, cols: list[str], color: str | None = None, max_points: int = 1500, seed: int = 42) -> go.Figure:
    data = df[[c for c in dict.fromkeys(cols + ([color] if color else []))]].dropna()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=seed)
    fig = px.scatter_matrix(data, dimensions=cols, color=color, opacity=0.6, color_discrete_sequence=PALETTE)
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    return style(fig, "Scatter matrix", height=620)


def grouped_aggregation(df: pd.DataFrame, group: str, value: str, agg: str = "mean", top_n: int = 20) -> tuple[go.Figure, pd.DataFrame]:
    g = df[[group, value]].dropna()
    table = g.groupby(g[group].astype(str))[value].agg([agg, "count"]).rename(columns={agg: f"{agg}_{value}", "count": "n"})
    table = table.sort_values(f"{agg}_{value}", ascending=False).head(top_n).reset_index().rename(columns={"index": group})
    fig = px.bar(table, x=group, y=f"{agg}_{value}", color_discrete_sequence=[SAGE], hover_data={"n": ":,"})
    fig.update_xaxes(type="category")
    return style(fig, f"{agg.title()} of {value} by {group}"), table


def time_series(df: pd.DataFrame, date_col: str, value: str, freq: str = "D", agg: str = "mean") -> tuple[go.Figure, pd.DataFrame]:
    d = pd.DataFrame({"t": pd.to_datetime(df[date_col], errors="coerce", format="mixed"), "v": pd.to_numeric(df[value], errors="coerce")}).dropna()
    table = d.set_index("t")["v"].resample(freq).agg(agg).dropna().reset_index()
    table.columns = [date_col, value]
    fig = px.line(table, x=date_col, y=value, color_discrete_sequence=[SAGE])
    return style(fig, f"{agg.title()} of {value} over time"), table


def class_balance(counts: dict[str, int], target: str) -> go.Figure:
    t = pd.DataFrame({"class": list(counts), "count": list(counts.values())})
    fig = px.bar(t, x="class", y="count", color_discrete_sequence=[SAGE], text="count")
    fig.update_xaxes(type="category")
    return style(fig, f"Class distribution of {target}", height=340)


def category_vs_class(df: pd.DataFrame, feature: str, target: str, top_n: int = 12) -> tuple[go.Figure, pd.DataFrame]:
    d = df[[feature, target]].dropna().astype(str)
    keep = d[feature].value_counts().head(top_n).index
    d = d[d[feature].isin(keep)]
    table = pd.crosstab(d[feature], d[target], normalize="index").mul(100)
    long = table.reset_index().melt(id_vars=feature, var_name=target, value_name="share_pct")
    fig = px.bar(long, x=feature, y="share_pct", color=target, barmode="stack", color_discrete_sequence=PALETTE)
    fig.update_xaxes(type="category")
    fig.update_layout(yaxis_title=f"% of rows within each {feature}")
    return style(fig, f"{target} mix within {feature}"), table


# --------------------------------------------------------- interpretations
def interpret_numeric(c: dict[str, Any]) -> str:
    if c.get("mean") is None:
        return f"**{c['name']}** has no numeric values to summarise."
    parts = [f"**{c['name']}** ranges from {c['min']:.4g} to {c['max']:.4g} with mean {c['mean']:.4g} and median {c['median']:.4g}."]
    sk = c.get("skew")
    if sk is not None:
        shape = "roughly symmetric" if abs(sk) < 0.5 else ("right-skewed (long tail of high values)" if sk > 0 else "left-skewed (long tail of low values)")
        parts.append(f"Skewness is {sk:.2f}, i.e. {shape}.")
    if c.get("outliers"):
        parts.append(f"{c['outliers']:,} values ({_pct(c['outlier_pct'], 1)}) lie outside the Tukey fences and may be influential.")
    elif c.get("outliers") == 0:
        parts.append("No values fall outside the Tukey fences.")
    if c["missing"]:
        parts.append(f"{c['missing']:,} values ({_pct(c['missing_pct'], 1)}) are missing and are excluded from the chart.")
    return " ".join(parts)


def interpret_counts(col: str, table: pd.DataFrame, n_unique: int) -> str:
    top = table.iloc[0]
    txt = f"**{col}** has {n_unique:,} distinct values. The most frequent is '{top[col]}' with {int(top['count']):,} rows ({_pct(top['share_pct'], 1)})."
    if len(table) > 1:
        second = table.iloc[1]
        txt += f" Next is '{second[col]}' with {int(second['count']):,} ({_pct(second['share_pct'], 1)})."
    return txt


def interpret_correlations(pairs: list[dict[str, Any]], threshold: float = 0.7) -> str:
    if not pairs:
        return "Fewer than two numeric columns with variation are available, so correlations were not computed."
    strong = [p for p in pairs if abs(p["r"]) >= threshold]
    if strong:
        listed = "; ".join(f"{p['a']} ↔ {p['b']} (r = {p['r']:.2f})" for p in strong[:5])
        txt = f"{len(strong)} pair(s) have |r| ≥ {threshold:g}: {listed}."
    else:
        p = pairs[0]
        txt = f"No pair reaches |r| ≥ {threshold:g}. The strongest is {p['a']} ↔ {p['b']} (r = {p['r']:.2f})."
    return txt + " " + CAUSAL_NOTE


def interpret_missing(profile: dict[str, Any]) -> str:
    if not profile["missing_cells"]:
        return "No missing values were found in any column."
    cols = sorted([c for c in profile["column_details"] if c["missing"]], key=lambda c: -c["missing"])
    listed = ", ".join(f"{c['name']} ({_pct(c['missing_pct'], 1)})" for c in cols[:4])
    return (f"{profile['missing_cells']:,} cells ({_pct(profile['missing_pct'], 2)} of all cells) are missing, spread over "
            f"{len(cols)} column(s). Missingness is concentrated in: {listed}.")


def interpret_balance(t: dict[str, Any], name: str) -> str:
    counts = t.get("class_counts")
    if not counts:
        return f"**{name}** is treated as continuous, so class balance does not apply."
    big, small = max(counts, key=counts.get), min(counts, key=counts.get)
    ratio = t["imbalance_ratio"]
    verdict = "appears balanced" if ratio < 1.5 else ("is moderately imbalanced" if ratio < 3 else "is strongly imbalanced")
    return (f"**{name}** has {len(counts)} classes. The largest ('{big}') has {counts[big]:,} rows and the smallest "
            f"('{small}') has {counts[small]:,} — a ratio of {ratio:.2f}:1, so the target {verdict}.")


def interpret_scatter(df: pd.DataFrame, x: str, y: str) -> str:
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return "Not enough paired numeric values to measure a linear relationship."
    r = d[x].corr(d[y])
    strength = "weak" if abs(r) < 0.3 else ("moderate" if abs(r) < 0.7 else "strong")
    direction = "positive" if r > 0 else "negative"
    return f"Across {len(d):,} paired rows, Pearson r = {r:.2f} — a {strength} {direction} linear association. {CAUSAL_NOTE}"


def interpret_groups(table: pd.DataFrame, group: str, value: str, agg: str) -> str:
    col = f"{agg}_{value}"
    if table.empty:
        return "No rows are available for this grouping."
    hi, lo = table.iloc[0], table.iloc[-1]
    txt = f"'{hi[group]}' has the highest {agg} of {value} ({hi[col]:.4g}, n = {int(hi['n']):,})"
    if len(table) > 1:
        txt += f"; '{lo[group]}' has the lowest shown ({lo[col]:.4g}, n = {int(lo['n']):,})"
    small = int((table["n"] < 30).sum())
    return txt + "." + (f" {small} group(s) have fewer than 30 rows, so their values are less stable." if small else "")
