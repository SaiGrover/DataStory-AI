"""Natural-language → ChartSpec.

Deterministic parsing: resolve column mentions (exact names, spacing/case
variants, synonyms, close spellings), detect intent words, pick a chart type
compatible with the column types, then validate the spec against the data.
Nothing is executed except the validated spec.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.charts import ChartSpec, is_yearlike, validate
from core.profiler import infer_kinds, multi_value_separator

SYNONYMS = {
    "revenue": ["sales", "amount", "turnover", "income", "total"],
    "sales": ["revenue", "amount"],
    "salary": ["income", "pay", "wage", "monthlyincome", "compensation"],
    "income": ["salary", "monthlyincome"],
    "age": ["years_old"],
    "department": ["dept", "division", "team"],
    "country": ["nation", "location"],
    "year": ["release_year", "yr"],
    "release year": ["release_year"],
    "date": ["order_date", "date_added", "timestamp", "dteday"],
    "attrition": ["left", "churn", "turnover"],
    "churn": ["churned", "attrition", "exited"],
    "price": ["cost", "value", "saleprice"],
    "category": ["type", "segment", "genre", "listed_in"],
    "genre": ["listed_in", "category"],
    "rating": ["rated", "score"],
    "region": ["area", "zone", "territory"],
    "gender": ["sex"],
    "tenure": ["yearsatcompany", "months"],
    "movies": ["type"],
    "shows": ["type"],
}
_STOP = {"show", "me", "the", "a", "an", "of", "by", "and", "vs", "versus", "between", "across", "for", "per", "in", "on", "plot",
         "chart", "graph", "create", "make", "draw", "distribution", "relationship", "compare", "top", "over", "time", "with",
         "what", "is", "are", "how", "many", "counts", "count", "number", "rate", "rates", "trend", "monthly", "yearly", "weekly",
         "daily", "average", "mean", "median", "total", "sum", "histogram", "scatter", "bar", "line", "box", "violin", "heatmap",
         "correlation", "correlations", "matrix", "missing", "values", "value", "where", "only", "each", "all", "from", "to", "do",
         "does", "which", "their", "its", "my", "this", "dataset", "data", "generate", "visualize", "visualise", "please", "can",
         "you", "against", "split", "group", "grouped", "highest", "lowest", "largest", "smallest", "most", "least"}
_CHART_WORDS = {"histogram": "histogram", "scatter": "scatter", "box plot": "box", "boxplot": "box", "violin": "violin",
                "heatmap": "heatmap", "line chart": "line", "line": "line", "bar chart": "bar", "bar": "bar", "stacked": "stacked_bar"}
_FREQ_WORDS = {"daily": "D", "per day": "D", "weekly": "W", "per week": "W", "monthly": "MS", "per month": "MS", "by month": "MS",
               "quarterly": "QS", "per quarter": "QS", "yearly": "YS", "annual": "YS", "per year": "YS", "by year": "YS"}


@dataclass
class ParseResult:
    spec: ChartSpec | None
    assumptions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    clarification: str | None = None
    columns: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def resolve_columns(text: str, columns: list[str]) -> list[str]:
    """Columns mentioned in ``text``, in order of first mention."""
    low = text.lower()
    found: list[tuple[int, str]] = []
    norm_map = {_norm(c): c for c in columns}

    # 1) literal names (longest first so 'release_year' wins over 'year')
    taken = [False] * len(low)
    for c in sorted(columns, key=len, reverse=True):
        variants = {c.lower(), c.lower().replace("_", " "), c.lower().replace("_", "")}
        for v in variants:
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", low):
                if not any(taken[m.start():m.end()]):
                    found.append((m.start(), c))
                    for i in range(m.start(), m.end()):
                        taken[i] = True
                    break
    # 2) synonyms and close spellings for remaining words
    words = [(m.start(), m.group(0)) for m in re.finditer(r"[a-z][a-z0-9_]*", low)]
    for pos, w in words:
        if any(taken[pos:pos + len(w)]) or w in _STOP or len(w) < 3:
            continue
        cands = [w, _singular(w)] + SYNONYMS.get(w, []) + SYNONYMS.get(_singular(w), [])
        hit = None
        for cand in cands:
            n = _norm(cand)
            if n in norm_map:
                hit = norm_map[n]
                break
            part = [c for c in columns if n and len(n) >= 4 and (n in _norm(c).split("_") or _norm(c).endswith(n) or _norm(c).startswith(n))]
            if len(part) == 1:
                hit = part[0]
                break
        if hit is None:
            close = difflib.get_close_matches(_norm(w), list(norm_map), n=1, cutoff=0.85)
            hit = norm_map[close[0]] if close else None
        if hit:
            found.append((pos, hit))
    out: list[str] = []
    for _, c in sorted(found):
        if c not in out:
            out.append(c)
    return out


def _value_filters(text: str, df: pd.DataFrame, kinds: dict[str, str], used: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """'for movies', 'in India', 'where region is West', 'where age > 30'."""
    filters, notes = [], []
    low = text.lower()
    for m in re.finditer(r"where\s+([a-z0-9_ ]+?)\s*(>=|<=|>|<|=|==|is not|is|!=)\s*['\"]?([\w .\-]+?)['\"]?(?:\s+and\b|$|[,.?])", low):
        col = resolve_columns(m.group(1), list(df.columns))
        if col:
            op = {"=": "==", "is": "==", "is not": "!=", "==": "==", "!=": "!="}.get(m.group(2), m.group(2))
            filters.append({"column": col[0], "op": op, "value": m.group(3).strip()})
    for m in re.finditer(r"\b(?:for|in|only|among)\s+(?:the\s+)?([a-z0-9][\w\- ]{1,30}?)(?=$|[,.?]|\s+(?:by|and|over|per|vs|across)\b)", low):
        phrase = m.group(1).strip()
        for c, k in kinds.items():
            if k not in ("categorical", "text") or c in [f["column"] for f in filters]:
                continue
            s = df[c].dropna().astype(str)
            if multi_value_separator(df[c]):
                continue
            vals = {v.lower(): v for v in s.unique()[:500]}
            for cand in (phrase, _singular(phrase), phrase.rstrip("s")):
                if cand in vals:
                    filters.append({"column": c, "op": "==", "value": vals[cand]})
                    notes.append(f"Interpreted “{phrase}” as {c} = '{vals[cand]}'.")
                    break
            else:
                continue
            break
    return filters, notes


def _mentioned_value_column(text: str, df: pd.DataFrame, kinds: dict[str, str], exclude: set[str]) -> str | None:
    """A low-cardinality column whose values (at least two of them) are named in the text, e.g. 'movie and TV show'."""
    low = text.lower()
    best = None
    for c, k in kinds.items():
        if k != "categorical" or c in exclude or df[c].nunique() > 12:
            continue
        vals = [str(v) for v in df[c].dropna().unique()]
        hits = [v for v in vals if len(v) > 1 and re.search(r"(?<![a-z0-9])" + re.escape(v.lower()) + r"s?(?![a-z0-9])", low)]
        if len(hits) >= 2 and (best is None or len(hits) > best[1]):
            best = (c, len(hits))
    return best[0] if best else None


def parse(text: str, df: pd.DataFrame, kinds: dict[str, str] | None = None, target: str | None = None,
          previous: ChartSpec | None = None) -> ParseResult:
    kinds = kinds or infer_kinds(df)
    low = " " + text.lower().strip() + " "
    cols = resolve_columns(text, list(df.columns))
    filters, assumptions = _value_filters(text, df, kinds, cols)
    filter_cols = {f["column"] for f in filters}
    cols = [c for c in cols if c not in filter_cols or sum(1 for x in cols if x == c) > 1]
    top = None
    m = re.search(r"\btop\s+(\d{1,2})\b", low)
    if m:
        top = max(1, min(50, int(m.group(1))))
    chart = next((v for k, v in _CHART_WORDS.items() if f" {k}" in low), None)
    freq = next((v for k, v in _FREQ_WORDS.items() if k in low), None)
    agg = None
    for word, a in (("average", "mean"), ("mean", "mean"), ("median", "median"), ("total", "sum"), ("sum", "sum"),
                    ("maximum", "max"), ("minimum", "min"), (" rate", "rate"), ("share", "rate"), ("how many", "count"),
                    ("number of", "count"), ("count", "count")):
        if word in low:
            agg = a
            break
    res = ParseResult(None, assumptions, columns=cols)

    # follow-up: "now by region", "same but for movies"
    if previous is not None and re.match(r"^\s*(now|same|and|also|instead|what about|then)\b", text.lower()):
        spec = ChartSpec(**previous.to_dict())
        if cols:
            if " by " in low or "split" in low:
                if spec.chart in ("bar", "box", "violin", "stacked_bar"):
                    spec.x = cols[-1]
                else:
                    spec.color = cols[-1]
            else:
                spec.x = cols[0]
        if filters:
            spec.filters = filters
        spec.title = None
        res.assumptions.append("Modified the previous chart.")
        return _finish(res, spec, df, kinds)

    if re.search(r"missing", low) and not cols:
        return _finish(res, ChartSpec("missing"), df, kinds)
    if chart == "heatmap" or re.search(r"correlation (heat ?map|matrix)|correlations\b", low):
        num_cols = [c for c in (cols or df.columns) if kinds.get(c) == "numeric"]
        if len(cols) < 2:
            num_cols = [c for c, k in kinds.items() if k == "numeric" and df[c].nunique() > 1][:15]
            res.assumptions.append("Used all numeric columns (up to 15).")
        return _finish(res, ChartSpec("heatmap", columns=num_cols), df, kinds)

    if not cols:
        close = difflib.get_close_matches(text.lower(), [c.lower() for c in df.columns], n=3, cutoff=0.3)
        res.errors.append("I could not find a column from this dataset in the request.")
        res.clarification = ("Which column should the chart use? Available columns include: "
                             + ", ".join(list(df.columns)[:12]) + ("…" if df.shape[1] > 12 else "") + "."
                             + (f" Did you mean {', '.join(close)}?" if close else ""))
        return res

    def kind(c):
        k = kinds[c]
        return "time" if k == "datetime" or (k == "numeric" and is_yearlike(df[c]) and ("year" in c.lower())) else k

    split_col = _mentioned_value_column(text, df, kinds, exclude=filter_cols)
    if split_col and split_col not in cols:
        cols.append(split_col)
        res.columns = cols
    over_time = bool(re.search(r"over time|trend|monthly|weekly|daily|yearly|per (day|week|month|quarter|year)|by (month|quarter|year)|timeline", low))
    time_cols = [c for c in cols if kind(c) == "time"]
    if over_time and not time_cols:
        cand = [c for c, k in kinds.items() if k == "datetime"] or [c for c in df.columns if "year" in c.lower() and is_yearlike(df[c])]
        if cand:
            time_cols = [cand[0]]
            res.assumptions.append(f"Used {cand[0]} as the time axis.")
    first, second = cols[0], (cols[1] if len(cols) > 1 else None)

    spec: ChartSpec
    wants_dist = bool(re.search(r"distribution|histogram|spread", low))
    if split_col and len(cols) >= 2 and chart in (None, "bar", "stacked_bar"):
        other = next(c for c in cols if c != split_col)
        if kinds[other] in ("categorical", "text"):
            spec = ChartSpec("stacked_bar", x=other, color=split_col, top_n=top or 12, normalize=False,
                             explode=bool(multi_value_separator(df[other])))
            res.assumptions.append(f"Split the counts by {split_col}.")
            if filters:
                spec.filters = filters
            return _finish(res, spec, df, kinds)
    if chart == "line" or over_time or (not wants_dist and time_cols and len(cols) <= 2 and (chart in (None, "line"))
                                        and kind(first) == "time" and (second is None or kinds.get(second) == "numeric")):
        t = time_cols[0] if time_cols else first
        value = next((c for c in cols if c != t and kinds[c] == "numeric"), None)
        a = agg if agg in ("mean", "median", "sum", "min", "max") else ("sum" if value else "count")
        if value and agg is None:
            res.assumptions.append(f"Summed {value} per period (say “average” for the mean).")
        f = freq or ("MS" if kinds[t] == "datetime" else None)
        if kinds[t] == "datetime" and freq is None:
            res.assumptions.append("Grouped by month.")
        spec = ChartSpec("line", x=t, y=value, agg=a if value else "count", freq=f)
    elif chart == "scatter" or (second and kinds[first] == "numeric" and kinds[second] == "numeric" and chart is None):
        if not second:
            res.errors.append("A scatter plot needs two numeric columns.")
            return res
        spec = ChartSpec("scatter", x=first, y=second, color=cols[2] if len(cols) > 2 and kinds[cols[2]] == "categorical" else None)
    elif chart in ("box", "violin") or (second and {kinds[first], kinds[second]} == {"numeric", "categorical"}
                                        and re.search(r"relationship|distribution|spread|vs|versus|against", low) and agg is None):
        numc = next((c for c in cols if kinds[c] == "numeric"), first)
        catc = next((c for c in cols if kinds[c] in ("categorical",) and c != numc), None)
        if catc and df[catc].nunique() == 2 and agg is None and re.search(r"relationship|vs|versus|against", low) and kinds[numc] == "numeric":
            res.assumptions.append(f"Compared the distribution of {numc} across the two {catc} groups.")
        spec = ChartSpec(chart if chart in ("box", "violin") else "box", x=catc, y=numc)
    elif chart == "stacked_bar" or (second and kinds[first] == "categorical" and kinds[second] == "categorical" and agg != "rate"):
        a, b = (first, second)
        if df[a].nunique() < df[b].nunique():
            a, b = b, a
        spec = ChartSpec("stacked_bar", x=a, color=b, top_n=top or 12)
    elif second or agg in ("mean", "median", "sum", "min", "max", "rate"):
        # "compare sales by region", "attrition rate across departments"
        group = None
        by = re.search(r"\b(?:by|across|per|for each|in each|within)\s+([a-z0-9_ ]+)", low)
        if by:
            g = resolve_columns(by.group(1), list(df.columns))
            group = g[0] if g else None
        if group is not None and kinds[group] == "numeric" and any(kinds[c] in ("categorical", "text") for c in cols if c != group):
            group = None
        if group is None:
            group = next((c for c in cols if kinds[c] in ("categorical", "text") or kind(c) == "time"), None)
        value = next((c for c in cols if c != group), None)
        if group is None or value is None:
            res.errors.append("Name both the value and the column to group by, e.g. “average salary by department”.")
            return res
        if agg == "rate" or (kinds[value] != "numeric" and df[value].nunique() == 2):
            a = "rate"
        elif kinds[value] == "numeric":
            a = agg if agg in ("mean", "median", "sum", "min", "max") else ("sum" if any(w in value.lower() for w in ("revenue", "sales", "amount", "quantity")) else "mean")
            if agg is None:
                res.assumptions.append(f"Used the {a} of {value} (say “total”, “average” or “median” to change it).")
        else:
            res.errors.append(f"'{value}' is not numeric, so it cannot be averaged or summed. Try a count or a stacked share chart.")
            return res
        spec = ChartSpec("bar", x=group, y=value, agg=a, top_n=top or 15, explode=bool(multi_value_separator(df[group])))
    else:
        c = first
        if kinds[c] == "numeric" and not (("count" in low or "number of" in low) and df[c].nunique() <= 30):
            spec = ChartSpec(chart if chart in ("histogram", "box", "violin") else "histogram", x=c if chart not in ("box", "violin") else None,
                             y=c if chart in ("box", "violin") else None)
        elif kinds[c] == "datetime":
            spec = ChartSpec("line", x=c, agg="count", freq=freq or "MS")
            res.assumptions.append("Counted rows per month.")
        else:
            nn = df[c].dropna()
            if len(nn) > 50 and nn.nunique() > 0.9 * len(nn) and not multi_value_separator(df[c]):
                res.errors.append(f"'{c}' is unique (or almost unique) per row — {nn.nunique():,} distinct values — so a frequency "
                                  "chart would show one bar per row. It looks like an identifier or free text.")
                return res
            spec = ChartSpec("bar", x=c, agg="count", top_n=top or 15, explode=bool(multi_value_separator(df[c])))
    if top:
        spec.top_n = top
    if filters:
        spec.filters = filters
    return _finish(res, spec, df, kinds)


def _finish(res: ParseResult, spec: ChartSpec, df: pd.DataFrame, kinds: dict[str, str]) -> ParseResult:
    errs = validate(df, spec, kinds)
    if errs:
        res.errors.extend(errs)
        return res
    res.spec = spec
    return res
