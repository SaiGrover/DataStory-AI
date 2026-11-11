"""Reproducible Jupyter notebook export.

``build_notebook`` turns the recorded analysis (dataset provenance, cleaning
history, saved chart specifications, training configuration, permutation
importance and clustering configuration) into a notebook that re-runs every
step with plain pandas / scikit-learn / plotly code. The notebook never imports
DataStory modules. Values recorded by the app appear only as "recorded in the
app" comparison targets; the notebook computes its own results and reports
whether they match.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import nbformat
from nbformat import v4

PACKAGES = ["pandas", "numpy", "scikit-learn", "imbalanced-learn", "plotly", "scipy"]

# Model constructors mirror core/modeling.py (_classifiers / _regressors).
_CLASSIFIERS = {
    "Logistic Regression": ("LogisticRegression(max_iter=1000, random_state=SEED, class_weight=CLASS_WEIGHT)", {"C": [0.1, 1.0, 10.0]}),
    "Decision Tree": ("DecisionTreeClassifier(random_state=SEED, class_weight=CLASS_WEIGHT)",
                      {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20]}),
    "Random Forest": ("RandomForestClassifier(n_estimators=150, random_state=SEED, class_weight=CLASS_WEIGHT, n_jobs=1)",
                      {"n_estimators": [100, 200], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 5]}),
    "Gradient Boosting": ("GradientBoostingClassifier(random_state=SEED)",
                          {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    "K-Nearest Neighbors": ("KNeighborsClassifier()", {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}),
}
_REGRESSORS = {
    "Linear Regression": ("LinearRegression()", {}),
    "Ridge Regression": ("Ridge(random_state=SEED)", {"alpha": [0.1, 1.0, 10.0, 100.0]}),
    "Lasso Regression": ("Lasso(random_state=SEED, max_iter=10000)", {"alpha": [0.001, 0.01, 0.1, 1.0]}),
    "Decision Tree Regressor": ("DecisionTreeRegressor(random_state=SEED)", {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20]}),
    "Random Forest Regressor": ("RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1)",
                                {"n_estimators": [100, 200], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 5]}),
    "Gradient Boosting Regressor": ("GradientBoostingRegressor(random_state=SEED)",
                                    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
}
_SCORING_REG = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}


def package_versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def _r(x: Any) -> str:
    """Python literal for a JSON-like value."""
    return repr(x)


def _stem(ds: dict[str, Any]) -> str:
    name = ds.get("original_filename") or ds.get("name") or "dataset"
    return Path(str(name)).stem.replace(" ", "_").lower() or "dataset"


def default_data_path(ctx: dict[str, Any]) -> str:
    return ctx.get("data_path") or f"data/{_stem(ctx.get('dataset') or {})}.csv"


def _scoring(metric: str, task: str, n_classes: int) -> str:
    if task == "regression":
        return _SCORING_REG[metric]
    if metric == "accuracy":
        return "accuracy"
    if metric == "roc_auc":
        return "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    return metric if n_classes == 2 else f"{metric}_weighted"


# ------------------------------------------------------------- cleaning
def _cleaning_code(op: str, p: dict[str, Any]) -> str:
    cols = list(p.get("columns") or [])
    head = "out = df.copy()\n"
    if op == "impute":
        strategy = p.get("strategy", "median")
        if strategy == "knn":
            return (f"# KNN imputation: neighbours measured on every numeric column that has at least one value\n"
                    f"num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])\n"
                    f"            and df[c].notna().any()]\n"
                    f"cols = [c for c in {_r(cols)} if c in num_cols]\n"
                    f"k = max(1, min(int({_r(p.get('n_neighbors', 5))}), len(df) - 1))\n"
                    f"imputed = pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(df[num_cols].astype(float)), columns=num_cols, index=df.index)\n"
                    + head + "for c in cols:\n    out[c] = imputed[c]\ndf = out")
        body = head + f"for c in {_r(cols)}:\n    s = df[c]\n    if s.isna().sum() == 0:\n        continue\n"
        if strategy in ("mean", "median"):
            body += f"    out[c] = s.astype(float).fillna(float(s.{strategy}()))\n"
        elif strategy == "mode":
            body += "    out[c] = s.fillna(s.mode(dropna=True).iloc[0])\n"
        elif strategy == "constant":
            body += (f"    value = {_r(p.get('value'))}\n"
                     "    if pd.api.types.is_numeric_dtype(s):\n        out[c] = s.astype(float).fillna(float(value))\n"
                     "    else:\n        out[c] = s.astype(object).fillna(str(value))\n")
        else:
            raise ValueError(f"Unknown imputation strategy {strategy!r}")
        return body + "df = out"
    if op == "drop_missing_rows":
        subset = cols or None
        return f"df = df.dropna(subset={_r(subset)} if {_r(subset)} else list(df.columns), how={_r(p.get('how', 'any'))})  # index kept"
    if op == "drop_columns":
        return f"df = df.drop(columns={_r(cols)})"
    if op == "drop_sparse_columns":
        return (f"share = df.isna().mean() * 100\nsparse = [c for c in df.columns if share[c] > {_r(float(p.get('threshold_pct', 50.0)))}]\n"
                "if sparse:\n    df = df.drop(columns=sparse)\nprint('Dropped:', sparse)")
    if op == "drop_duplicates":
        return f"df = df.drop_duplicates(subset={_r(cols or None)}, keep={_r(p.get('keep', 'first'))})  # index kept"
    if op == "convert_type":
        to = p.get("to", "numeric")
        conv = {
            "numeric": ("cleaned = s.astype(str).str.replace(r'[,\\s$€£%]', '', regex=True) if not pd.api.types.is_numeric_dtype(s) else s\n"
                        "    new = pd.to_numeric(cleaned, errors='coerce')\n    new[s.isna()] = np.nan"),
            "integer": "new = pd.to_numeric(s, errors='coerce').round().astype('Int64')",
            "string": "new = s.astype('string')",
            "category": "new = s.astype('category')",
            "datetime": "new = pd.to_datetime(s, errors='coerce', format='mixed')",
            "boolean": ("m = {'true': True, 'yes': True, 'y': True, '1': True, 't': True,\n"
                        "         'false': False, 'no': False, 'n': False, '0': False, 'f': False}\n"
                        "    new = s.astype(str).str.strip().str.lower().map(m).astype('boolean')\n    new[s.isna()] = pd.NA"),
        }[to]
        return head + f"for c in {_r(cols)}:\n    s = df[c]\n    {conv}\n    out[c] = new\ndf = out"
    if op == "parse_dates":
        return head + (f"for c in {_r(cols)}:\n    out[c] = pd.to_datetime(df[c], errors='coerce', dayfirst={_r(bool(p.get('dayfirst', False)))}, "
                       "format='mixed')\ndf = out")
    if op == "normalize_categories":
        case = p.get("case", "lower")
        mapping = {str(k): str(v) for k, v in (p.get("mapping") or {}).items()}
        lines = [head + f"for c in {_r(cols)}:", "    s = df[c]", "    mask = s.notna()", "    new = s.astype(object).copy()",
                 "    vals = s[mask].astype(str)"]
        if p.get("strip", True):
            lines.append("    vals = vals.str.strip().str.replace(r'\\s+', ' ', regex=True)")
        if case in ("lower", "upper", "title"):
            lines.append(f"    vals = vals.str.{case}()")
        if mapping:
            lines.append(f"    vals = vals.replace({_r(mapping)})")
        lines += ["    new[mask] = vals", "    out[c] = new if s.dtype == object else new.astype('string')", "df = out"]
        return "\n".join(lines)
    if op == "clean_text":
        lines = [head + f"for c in {_r(cols)}:", "    s = df[c]", "    mask = s.notna()", "    vals = s[mask].astype(str)"]
        if p.get("strip", True):
            lines.append("    vals = vals.str.strip()")
        if p.get("collapse_spaces", True):
            lines.append("    vals = vals.str.replace(r'\\s+', ' ', regex=True)")
        if p.get("lower", False):
            lines.append("    vals = vals.str.lower()")
        if p.get("remove_punctuation", False):
            lines.append("    vals = vals.str.replace(r'[^\\w\\s]', '', regex=True)")
        lines += ["    new = s.astype(object).copy()", "    new[mask] = vals",
                  "    out[c] = new if s.dtype == object else new.astype('string')", "df = out"]
        return "\n".join(lines)
    if op == "handle_outliers":
        method, action, thr = p.get("method", "iqr"), p.get("action", "cap"), p.get("threshold")
        if method == "iqr":
            bounds = (f"    k = {_r(1.5 if thr is None else float(thr))}\n    q1, q3 = x.quantile(0.25), x.quantile(0.75)\n"
                      "    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)\n")
        else:
            bounds = (f"    k = {_r(3.0 if thr is None else float(thr))}\n    mu, sd = x.mean(), x.std()\n"
                      "    if not sd or np.isnan(sd):\n        continue\n    lo, hi = mu - k * sd, mu + k * sd\n")
        act = {"cap": "    out[c] = df[c].astype(float).clip(lo, hi)\n",
               "null": "    out[c] = df[c].astype(float).mask(mask)\n",
               "remove": "    drop |= mask\n"}[action]
        return (head + "drop = pd.Series(False, index=df.index)\n"
                f"for c in {_r(cols)}:\n    x = pd.to_numeric(df[c], errors='coerce').astype(float)\n" + bounds +
                "    mask = (x < lo) | (x > hi)\n    print(f'{c}: {int(mask.sum()):,} values outside [{lo:.4g}, {hi:.4g}]')\n"
                "    if int(mask.sum()) == 0:\n        continue\n" + act +
                ("out = out[~drop]  # index kept\n" if action == "remove" else "") + "df = out")
    if op == "scale":
        method = p.get("method", "standard")
        stat = {"standard": "centre, spread = x.mean(), x.std(ddof=0)",
                "minmax": "centre, spread = x.min(), x.max() - x.min()",
                "robust": "centre, spread = x.median(), x.quantile(0.75) - x.quantile(0.25)"}[method]
        return head + f"for c in {_r(cols)}:\n    x = df[c].astype(float)\n    {stat}\n    out[c] = (x - centre) / spread\ndf = out"
    if op == "encode":
        if p.get("method", "onehot") == "onehot":
            return (head + f"for c in {_r(cols)}:\n"
                    f"    dummies = pd.get_dummies(df[c], prefix=c, drop_first={_r(bool(p.get('drop_first', False)))}, dtype=int)\n"
                    "    pos = out.columns.get_loc(c)\n    out = pd.concat([out.iloc[:, :pos], dummies, out.iloc[:, pos + 1:]], axis=1)\ndf = out")
        return (head + f"for c in {_r(cols)}:\n    cats = sorted(df[c].dropna().astype(str).unique())\n"
                "    codes = df[c].astype(str).map({v: i for i, v in enumerate(cats)})\n"
                "    out[c] = codes.where(df[c].notna()).astype('Int64')\ndf = out")
    if op == "rename_columns":
        mapping = {k: str(v).strip() for k, v in (p.get("mapping") or {}).items() if str(v).strip() and str(v).strip() != k}
        return f"df = df.rename(columns={_r(mapping)})"
    raise ValueError(f"Unsupported cleaning operation {op!r}")


# ---------------------------------------------------------------- charts
_FILTER_FN = '''def apply_filters(frame, filters):
    """Same semantics as the app's chart filters."""
    out = frame
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


def categories(s, explode, dropna):
    """Category labels; comma-separated lists are split and counted at most once per row."""
    x = s.astype("object")
    if explode:
        x = x.dropna().astype(str).str.split(",").explode().str.strip()
        x = x[x != ""]
        tmp = x.to_frame("v")
        tmp["_row"] = tmp.index
        x = tmp.drop_duplicates(["_row", "v"])["v"]
    if dropna:
        return x.dropna().astype(str)
    return x.where(x.notna(), "(missing)").astype(str)


def positive_class(s):
    vals = list(pd.Series(s.dropna().unique()))
    for cand in vals:
        if str(cand).strip().lower() in ("yes", "true", "1", "1.0", "y", "churned", "left", "positive"):
            return cand
    return sorted(vals, key=str)[-1]'''


def _chart_code(spec: dict[str, Any], title: str) -> str:
    c = spec.get("chart")
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    agg, top = spec.get("agg", "count"), spec.get("top_n")
    lines = [f"spec = {_r(spec)}", "data = apply_filters(df, spec['filters']) if spec['filters'] else df",
             "print(f'{len(data):,} rows used')"]
    t = _r(title)
    if c == "histogram":
        lines += [f"d = data.dropna(subset=[{_r(x)}])",
                  f"fig = px.histogram(d, x={_r(x)}, color={_r(color)}, nbins={int(spec.get('bins', 30))}, title={t})",
                  f"display(pd.to_numeric(d[{_r(x)}], errors='coerce').describe().to_frame().T)"]
    elif c in ("box", "violin"):
        lines += [f"d = data[[c for c in [{_r(y)}, {_r(x)}] if c]].dropna(subset=[{_r(y)}])"]
        if x:
            lines += [f"keep = d[{_r(x)}].astype(str).value_counts().head({_r(top or 15)}).index",
                      f"d = d[d[{_r(x)}].astype(str).isin(keep)].assign(**{{{_r(x)}: lambda f: f[{_r(x)}].astype(str)}})",
                      f"display(d.groupby({_r(x)})[{_r(y)}].agg(['median', 'mean', 'count']).sort_values('median', ascending=False))"]
        fn = "px.violin" if c == "violin" else "px.box"
        lines += [f"fig = {fn}(d, y={_r(y)}, x={_r(x)}, color={_r(x)}, points='outliers', title={t})"]
    elif c == "bar":
        lines += [f"cats = categories(data[{_r(x)}], {_r(bool(spec.get('explode')))}, {_r(bool(spec.get('dropna', True)))})"]
        if agg == "count":
            lines += [f"table = cats.value_counts().rename_axis({_r(x)}).reset_index(name='count')", "val = 'count'"]
        elif agg == "rate":
            lines += [f"pos = positive_class(data[{_r(y)}])",
                      f"frame = pd.DataFrame({{{_r(x)}: cats.values, 'v': data.loc[cats.index, {_r(y)}].values}})",
                      "frame['v'] = (frame['v'].astype(str) == str(pos)).astype(float).where(frame['v'].notna())",
                      "frame = frame.dropna()",
                      f"table = frame.groupby({_r(x)})['v'].agg(['mean', 'count']).rename(columns={{'mean': 'rate', 'count': 'n'}})",
                      "table['rate'] = table['rate'] * 100", "table = table.reset_index()", "val = 'rate'",
                      f"print(f\"Rate = % of rows where {y} is '{{pos}}'\")"]
        else:
            lines += [f"frame = pd.DataFrame({{{_r(x)}: cats.values, 'v': data.loc[cats.index, {_r(y)}].values}})",
                      "frame['v'] = pd.to_numeric(frame['v'], errors='coerce')", "frame = frame.dropna()",
                      f"val = {_r(f'{agg}_{y}')}",
                      f"table = frame.groupby({_r(x)})['v'].agg([{_r(agg)}, 'count']).rename(columns={{{_r(agg)}: val, 'count': 'n'}}).reset_index()"]
        if spec.get("sort", "desc") != "none":
            lines.append(f"table = table.sort_values(val, ascending={_r(spec.get('sort') == 'asc')}, kind='stable')")
        if top:
            other = bool(spec.get("other", True)) and agg in ("count", "sum")
            lines += [f"if len(table) > {int(top)}:", f"    rest = table.iloc[{int(top)}:]", f"    table = table.iloc[:{int(top)}]"]
            if other:
                lines.append(f"    table = pd.concat([table, pd.DataFrame({{{_r(x)}: [f'Other ({{len(rest):,}} categories)'], val: [rest[val].sum()]}})], ignore_index=True)")
        lines += [f"fig = px.bar(table, x=val, y={_r(x)}, orientation='h', title={t})", "fig.update_yaxes(type='category', autorange='reversed')",
                  "display(table)"]
    elif c == "line":
        lines += [f"is_dt = not pd.api.types.is_numeric_dtype(data[{_r(x)}])",
                  f"t = pd.to_datetime(data[{_r(x)}], errors='coerce', format='mixed') if is_dt else pd.to_numeric(data[{_r(x)}], errors='coerce')",
                  ("v = pd.Series(1.0, index=data.index)" if agg == "count" else f"v = pd.to_numeric(data[{_r(y)}], errors='coerce')"),
                  "d = pd.DataFrame({'t': t, 'v': v}).dropna()",
                  f"how = {_r('sum' if agg == 'count' else ('mean' if agg == 'rate' else agg))}",
                  "if is_dt:",
                  f"    table = d.set_index('t')['v'].resample({_r(spec.get('freq') or 'MS')}).agg(how).reset_index()",
                  ("    table = table.dropna()" if agg != "count" else "    pass"),
                  "else:", "    table = d.groupby('t')['v'].agg(how).reset_index().sort_values('t')",
                  f"table.columns = [{_r(x)}, 'value']",
                  f"fig = px.line(table, x={_r(x)}, y='value', markers=len(table) <= 60, title={t})", "display(table.tail())"]
    elif c == "scatter":
        lines += [f"d = data[[c for c in dict.fromkeys([{_r(x)}, {_r(y)}, {_r(color)}]) if c]].dropna(subset=[{_r(x)}, {_r(y)}])",
                  "plot = d.sample(5000, random_state=42) if len(d) > 5000 else d",
                  f"fig = px.scatter(plot, x={_r(x)}, y={_r(y)}, color={_r(color)}, opacity=0.7, title={t})",
                  f"full = data[[{_r(x)}, {_r(y)}]].apply(pd.to_numeric, errors='coerce').dropna()",
                  f"print(f'Pearson r = {{full[{_r(x)}].corr(full[{_r(y)}]):.4f}} over {{len(full):,}} rows')"]
    elif c == "heatmap":
        lines += [f"cols = {_r(list(spec.get('columns') or []))} or [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]",
                  "cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 1][:25]",
                  "corr = data[cols].corr(numeric_only=True)",
                  f"fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale='RdBu'))",
                  f"fig.update_layout(title={t})", "fig.update_yaxes(autorange='reversed')", "display(corr.round(3))"]
    elif c == "stacked_bar":
        lines += [f"d = data[[{_r(x)}, {_r(color)}]].dropna()"]
        if spec.get("explode"):
            lines += [f"xs = categories(d[{_r(x)}], True, True)",
                      f"d = pd.DataFrame({{{_r(x)}: xs.values, {_r(color)}: d.loc[xs.index, {_r(color)}].astype(str).values}})"]
        else:
            lines += ["d = d.astype(str)"]
        lines += [f"keep = d[{_r(x)}].value_counts().head({_r(top or 12)}).index", f"d = d[d[{_r(x)}].isin(keep)]",
                  f"counts = pd.crosstab(d[{_r(x)}], d[{_r(color)}]).loc[list(keep)]"]
        if spec.get("normalize", True):
            lines += ["table = counts.div(counts.sum(axis=1), axis=0).mul(100)", "vname = 'share_pct'"]
        else:
            lines += ["table = counts", "vname = 'rows'"]
        lines += [f"long = table.reset_index().melt(id_vars={_r(x)}, var_name={_r(color)}, value_name=vname)",
                  f"fig = px.bar(long, x={_r(x)}, y=vname, color={_r(color)}, barmode={_r('stack' if spec.get('normalize', True) else 'group')}, title={t})",
                  "display(table.round(1))"]
    elif c == "missing":
        lines += ["miss = data.isna().sum()", "miss = miss[miss > 0].sort_values(ascending=False)",
                  "table = pd.DataFrame({'column': miss.index, 'missing': miss.values, 'missing_pct': 100 * miss.values / len(data)})",
                  f"fig = px.bar(table, x='missing_pct', y='column', orientation='h', title={t})", "display(table)"]
    else:
        raise ValueError(f"Unsupported chart type {c!r}")
    lines.append("fig.show()")
    return "\n".join(lines)


# -------------------------------------------------------------- modeling
def _modeling_cells(ctx: dict[str, Any]) -> list:
    tr = ctx.get("training") or {}
    cfg = ctx.get("train_config") or {}
    if not tr:
        return []
    task = tr["task"]
    feats = tr["features"]
    target = tr["target"]
    models = [m for m in tr["models"] if "error" not in m]
    names = [m["name"] for m in models]
    registry = _CLASSIFIERS if task == "classification" else _REGRESSORS
    n_classes = len(tr.get("classes") or []) or 2
    primary = tr["primary_metric"]
    scoring = _scoring(primary, task, n_classes)
    folds, tuning, imbalance, smote_k = int(tr.get("cv_folds") or 0), tr.get("tuning", "none"), tr.get("imbalance", "none"), tr.get("smote_k")
    n_iter = int(cfg.get("n_iter", 8))
    cells = [v4.new_markdown_cell(
        "## 5. Model training\n\n"
        f"Target **{target}** · task **{task}** · {len(names)} model(s) · test share {tr['test_size']} · seed {tr['seed']} · "
        f"{folds or 'no'}-fold cross-validation · tuning: {tuning} · imbalance handling: {imbalance}.\n\n"
        "The split happens first; imputation, scaling, encoding (and SMOTE, when used) are pipeline steps fitted on the "
        "training rows only. The feature lists below are the ones the app's feature preparation produced.")]
    prep = [
        f"TARGET = {_r(target)}", f"SEED = {_r(int(tr['seed']))}", f"TEST_SIZE = {_r(float(tr['test_size']))}",
        f"INPUT_COLUMNS = {_r(feats['input_columns'])}", f"NUMERIC = {_r(feats['numeric'])}", f"CATEGORICAL = {_r(feats['categorical'])}",
        f"DATETIME_EXPANDED = {_r(feats.get('datetime_expanded') or {})}",
        "data = df.loc[df[TARGET].notna()]",
        "X = pd.DataFrame(index=data.index)",
        "dt_parts = {part: base for base, parts in DATETIME_EXPANDED.items() for part in parts}",
        "for c in INPUT_COLUMNS:",
        "    if c in dt_parts:",
        "        dt = pd.to_datetime(data[dt_parts[c]], errors='coerce', format='mixed')",
        "        X[c] = getattr(dt.dt, c.rsplit('__', 1)[1]).astype(float)",
        "    elif c in CATEGORICAL:",
        "        s = data[c]",
        "        X[c] = s.astype(object).where(s.notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))",
        "    else:",
        "        X[c] = pd.to_numeric(data[c], errors='coerce').astype(float)",
        "y_raw = data[TARGET]",
    ]
    if task == "classification":
        prep += ["y_str = y_raw.astype(str)", "counts = y_str.value_counts()", "rare = counts[counts < 2].index.tolist()",
                 "if rare:", "    print('Excluded single-row classes:', rare)", "    keep = ~y_str.isin(rare)",
                 "    X, y_str = X[keep], y_str[keep]",
                 "enc = LabelEncoder().fit(y_str)", "CLASSES = [str(c) for c in enc.classes_]",
                 "y = pd.Series(enc.transform(y_str), index=y_str.index)", "print('Classes:', CLASSES)",
                 "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)"]
    else:
        prep += ["y = pd.to_numeric(y_raw, errors='coerce').astype(float)",
                 "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)"]
    prep += [f"print(f'{{len(X_train):,}} training rows, {{len(X_test):,}} test rows (app recorded {tr['n_train']:,} / {tr['n_test']:,})')"]
    cells.append(v4.new_code_cell("\n".join(prep)))

    pipe = [
        f"SCALE = {_r(bool(tr.get('scale', True)))}", f"SMOTE_K = {_r(smote_k)}",
        f"CLASS_WEIGHT = {_r('balanced' if imbalance == 'class_weight' else None)}", f"SCORING = {_r(scoring)}",
        f"FOLDS = {_r(folds)}", f"TUNING = {_r(tuning)}", f"N_ITER = {_r(n_iter)}",
        "",
        "def build_pipeline(estimator):",
        "    num_steps = [('impute', SimpleImputer(strategy='median'))]",
        "    if SCALE:",
        "        num_steps.append(('scale', StandardScaler()))",
        "    transformers = []",
        "    if NUMERIC:",
        "        transformers.append(('num', Pipeline(num_steps), NUMERIC))",
        "    if CATEGORICAL:",
        "        transformers.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),",
        "                                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL))",
        "    pre = ColumnTransformer(transformers, remainder='drop')",
        "    if SMOTE_K:",
        "        return ImbPipeline([('preprocess', pre), ('smote', SMOTE(random_state=SEED, k_neighbors=SMOTE_K)), ('model', estimator)])",
        "    return Pipeline([('preprocess', pre), ('model', estimator)])",
        "",
        "MODELS = {",
    ]
    for n in names:
        ctor, grid = registry[n]
        pipe.append(f"    {_r(n)}: ({ctor}, {_r(grid)}),")
    pipe += ["}",
             "if FOLDS >= 2:",
             ("    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)" if task == "classification"
              else "    cv = KFold(FOLDS, shuffle=True, random_state=SEED)"),
             "else:", "    cv = None",
             "fitted, cv_scores = {}, {}",
             "for name, (est, grid) in MODELS.items():",
             "    pipe = build_pipeline(est)",
             "    with warnings.catch_warnings():",
             "        warnings.simplefilter('ignore')",
             "        if TUNING != 'none' and grid and cv is not None:",
             "            params = {f'model__{k}': v for k, v in grid.items()}",
             "            if TUNING == 'grid':",
             "                search = GridSearchCV(pipe, params, scoring=SCORING, cv=cv, n_jobs=1, refit=True)",
             "            else:",
             "                total = int(np.prod([len(v) for v in grid.values()]))",
             "                search = RandomizedSearchCV(pipe, params, n_iter=min(N_ITER, total), scoring=SCORING, cv=cv, n_jobs=1,",
             "                                            refit=True, random_state=SEED)",
             "            search.fit(X_train, y_train)",
             "            pipe = search.best_estimator_",
             "            cv_scores[name] = search.cv_results_['mean_test_score'][search.best_index_]",
             "            print(name, 'best params:', search.best_params_)",
             "        else:",
             "            if cv is not None:",
             "                cv_scores[name] = cross_validate(pipe, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=1)['test_score'].mean()",
             "            pipe.fit(X_train, y_train)",
             "    fitted[name] = pipe",
             "print('Mean CV score (training data, scikit-learn sign convention):', {k: round(float(v), 6) for k, v in cv_scores.items()})"]
    cells.append(v4.new_code_cell("\n".join(pipe)))

    recorded = {m["name"]: m["metrics"] for m in models}
    ev = [f"RECORDED = {_r(recorded)}  # test metrics recorded in the app at export time", "rows = []",
          "for name, pipe in fitted.items():", "    pred = pipe.predict(X_test)"]
    if task == "classification":
        ev += ["    binary = len(CLASSES) == 2", "    avg = 'binary' if binary else 'weighted'",
               "    m = {'accuracy': accuracy_score(y_test, pred),",
               "         'precision': precision_score(y_test, pred, average=avg, zero_division=0),",
               "         'recall': recall_score(y_test, pred, average=avg, zero_division=0),",
               "         'f1': f1_score(y_test, pred, average=avg, zero_division=0), 'roc_auc': None}",
               "    if hasattr(pipe, 'predict_proba') and len(np.unique(y_test)) == len(CLASSES):",
               "        proba = pipe.predict_proba(X_test)",
               "        m['roc_auc'] = roc_auc_score(y_test, proba[:, 1]) if binary else roc_auc_score(",
               "            y_test, proba, multi_class='ovr', average='weighted', labels=list(range(len(CLASSES))))"]
    else:
        ev += ["    mse = mean_squared_error(y_test, pred)",
               "    m = {'mae': mean_absolute_error(y_test, pred), 'mse': mse, 'rmse': float(np.sqrt(mse)),",
               "         'r2': r2_score(y_test, pred) if len(y_test) > 1 else None}"]
    ev += ["    for k, v in m.items():",
           "        rec = RECORDED.get(name, {}).get(k)",
           "        diff = None if v is None or rec is None else abs(float(v) - float(rec))",
           "        rows.append({'model': name, 'metric': k, 'reproduced': v, 'recorded_in_app': rec, 'abs_diff': diff})",
           "comparison = pd.DataFrame(rows)", "display(comparison)",
           "diffs = comparison['abs_diff'].dropna()",
           "max_diff = float(diffs.max()) if len(diffs) else 0.0",
           "print('metrics match' if max_diff <= 1e-9 else f'metrics differ (max abs difference {max_diff:.3g})')",
           f"print('Best model recorded in the app: {tr.get('best_model')}')"]
    cells.append(v4.new_markdown_cell("## 6. Evaluation on the held-out test set\n\nEach metric is recomputed here and shown next to the value the app recorded."))
    cells.append(v4.new_code_cell("\n".join(ev)))

    imp = ctx.get("importance")
    if imp and imp.get("model") in names:
        cells.append(v4.new_markdown_cell(
            "## 7. Explainability — held-out permutation importance\n\n"
            f"Each feature column of the test set is shuffled {imp['n_repeats']} times; the drop in {imp.get('metric_label', imp['metric'])} "
            "is its importance. Computed on the held-out test set only."))
        rec_top = {f["feature"]: f["mean"] for f in imp.get("features", [])}
        code = [f"RECORDED_IMPORTANCE = {_r(rec_top)}",
                f"res = permutation_importance(fitted[{_r(imp['model'])}], X_test, y_test, scoring={_r(imp['scoring'])},",
                f"                             n_repeats={int(imp['n_repeats'])}, random_state={_r(imp['seed'])}, n_jobs=1)",
                "imp = pd.DataFrame({'feature': X_test.columns, 'mean': res.importances_mean, 'std': res.importances_std})",
                "imp['recorded_in_app'] = imp['feature'].map(RECORDED_IMPORTANCE)",
                "imp = imp.sort_values('mean', ascending=False)", "display(imp)",
                "d = (imp['mean'] - imp['recorded_in_app']).abs().max()",
                "print('importance match' if d <= 1e-9 else f'importance differs (max abs difference {d:.3g})')"]
        cells.append(v4.new_code_cell("\n".join(code)))
    return cells


def _clustering_cells(ctx: dict[str, Any]) -> list:
    cl = ctx.get("clustering")
    if not cl:
        return []
    cfg = cl.get("config", cl)
    exp_n, exp_sizes = cl.get("n_clusters"), cl.get("sizes")
    p = cfg.get("params") or {}
    algo, seed, scale = cfg["algorithm"], int(cfg.get("seed", 42)), cfg.get("scale", "standard")
    fit = {
        "kmeans": f"KMeans(n_clusters={int(p.get('n_clusters', 3))}, n_init=10, random_state={seed}).fit_predict(Xc)",
        "gmm": (f"GaussianMixture(n_components={int(p.get('n_components', p.get('n_clusters', 3)))}, "
                f"covariance_type={_r(p.get('covariance_type', 'full'))}, random_state={seed}).fit_predict(Xc)"),
        "agglomerative": f"AgglomerativeClustering(n_clusters={int(p.get('n_clusters', 3))}, linkage={_r(p.get('linkage', 'ward'))}).fit_predict(Xc)",
        "dbscan": f"DBSCAN(eps={float(p.get('eps', 0.5))}, min_samples={int(p.get('min_samples', 5))}).fit_predict(Xc)",
    }[algo]
    scaler = {"standard": "[('scale', StandardScaler())]", "robust": "[('scale', RobustScaler())]", "none": "[]"}[scale]
    code = [f"CL_FEATURES = {_r(cfg['features'])}", f"CL_NUMERIC = {_r(cfg['numeric'])}", f"CL_CATEGORICAL = {_r(cfg['categorical'])}",
            "cdata = df[CL_FEATURES].copy()",
            "for c in CL_CATEGORICAL:",
            "    cdata[c] = cdata[c].astype(object).where(cdata[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))",
            "tr = []",
            f"if CL_NUMERIC:\n    tr.append(('num', Pipeline([('impute', SimpleImputer(strategy='median'))] + {scaler}), CL_NUMERIC))",
            "if CL_CATEGORICAL:",
            "    tr.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),",
            "                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CL_CATEGORICAL))",
            "Xc = np.asarray(ColumnTransformer(tr, remainder='drop').fit_transform(cdata), dtype=float)"]
    if cfg.get("reduce") == "pca":
        code.append(f"Xc = PCA(n_components={int(cfg['n_components'])}, random_state={seed}).fit_transform(Xc)")
    code += [f"labels = {fit}",
             "sizes = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}",
             "n_clusters = len([k for k in sizes if k != '-1'])",
             f"print('Clusters:', n_clusters, '(recorded in the app: {exp_n})')",
             "print('Sizes:', sizes)"]
    if exp_sizes is not None:
        code += [f"RECORDED_SIZES = {_r(exp_sizes)}", "print('cluster sizes match' if sizes == RECORDED_SIZES else 'cluster sizes differ from the app')"]
    code += ["if n_clusters >= 2 and len(set(labels)) < len(labels):",
             "    keep = labels != -1",
             "    print('Silhouette (all clustered rows):', round(float(silhouette_score(Xc[keep], labels[keep])), 4) if len(set(labels[keep])) >= 2 else 'n/a')"]
    return [v4.new_markdown_cell(
        f"## 8. Clustering — {cfg.get('algorithm_label', algo)}\n\nPreprocessing: {cfg.get('preprocessing', '')} Seed {seed}. "
        "Cluster labels describe structure in the chosen features only; they are not business segments by themselves."),
        v4.new_code_cell("\n".join(code))]


# ------------------------------------------------------------- notebook
def build_notebook(ctx: dict[str, Any]) -> nbformat.NotebookNode:
    ds = ctx.get("dataset") or {}
    history = ctx.get("history") or []
    charts = ctx.get("charts") or []
    versions = package_versions()
    data_path = default_data_path(ctx)
    nb = v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["datastory"] = {"exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                "dataset_id": ds.get("dataset_id"), "dataset_version": f"v{len(history)}"}
    cells = nb.cells

    # title + provenance
    cat = ds.get("provenance_category")
    prov = [f"# DataStory AI — reproducible analysis of {ds.get('name', 'dataset')}", "",
            f"Exported {nb.metadata['datastory']['exported_at']} · dataset version v{len(history)} ({len(history)} cleaning step(s)).", "",
            "## Data provenance", ""]
    if ds.get("synthetic"):
        prov += ["> **Synthetic Dataset — For Demonstration Purposes.** The values were generated for a demo; results must not be "
                 "cited as findings about the real world.", ""]
    elif cat == "user_upload_unverified":
        prov += ["> **User-uploaded dataset — provenance unverified.** DataStory AI has not verified where this data comes from.", ""]
    prov += [f"- Category: {ds.get('provenance_label', cat or 'unknown')}", f"- Source note: {ds.get('provenance', '')}",
             f"- Original file: {ds.get('original_filename') or 'n/a'} · SHA-256 {ds.get('file_hash') or 'n/a'}",
             f"- Licence: {ds.get('license') or 'not recorded'} · URL: {ds.get('source_url') or 'n/a'}",
             f"- Loaded in the app: {ds.get('loaded_at', 'n/a')}", "",
             "## How to run", "",
             f"Put the dataset CSV at `{data_path}` (relative to this notebook) or change `DATA_PATH` below, install the "
             "packages listed under *Environment*, then run all cells top to bottom. Values labelled *recorded in the app* "
             "are what DataStory AI computed at export time; the notebook recomputes everything itself and reports whether it matches."]
    cells.append(v4.new_markdown_cell("\n".join(prov)))
    cells.append(v4.new_markdown_cell("## Environment\n\nPackage versions used when this notebook was exported:\n\n"
                                      + "\n".join(f"- {k} {v}" for k, v in versions.items())))
    cells.append(v4.new_code_cell("\n".join([
        "import hashlib, warnings",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "import sklearn",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer, KNNImputer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate, GridSearchCV, RandomizedSearchCV",
        "from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso",
        "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor",
        "from sklearn.neighbors import KNeighborsClassifier",
        "from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error,",
        "                             mean_squared_error, r2_score, silhouette_score)",
        "from sklearn.inspection import permutation_importance",
        "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering",
        "from sklearn.mixture import GaussianMixture",
        "from sklearn.decomposition import PCA",
        "from imblearn.pipeline import Pipeline as ImbPipeline",
        "from imblearn.over_sampling import SMOTE",
        "try:",
        "    from IPython.display import display",
        "except ImportError:  # running as a plain script",
        "    display = print",
        "",
        "print('pandas', pd.__version__, '| numpy', np.__version__, '| scikit-learn', sklearn.__version__)",
    ])))

    # data load
    cells.append(v4.new_markdown_cell("## 1. Load the data"))
    cells.append(v4.new_code_cell("\n".join([
        f"DATA_PATH = Path({_r(data_path)})  # change this if the CSV lives elsewhere",
        f"ORIGINAL_SHA256 = {_r(ds.get('file_hash'))}  # hash of the file originally loaded into the app",
        f"EXPORTED_SHA256 = {_r(ctx.get('data_sha256'))}  # hash of the CSV exported with this notebook",
        "raw_bytes = DATA_PATH.read_bytes()",
        "digest = hashlib.sha256(raw_bytes).hexdigest()",
        "if digest == ORIGINAL_SHA256:",
        "    print('Data file matches the originally loaded file (SHA-256).')",
        "elif digest == EXPORTED_SHA256:",
        "    print('Data file matches the CSV exported by DataStory AI (SHA-256).')",
        "else:",
        "    warnings.warn('The data file does not match the recorded SHA-256 hashes; results may differ from the app.')",
        "df = pd.read_csv(DATA_PATH, low_memory=False)",
        "raw = df.copy()",
        f"print(df.shape, '(app recorded rows/columns: {ds.get('rows')} × {ds.get('columns')})')",
    ])))
    cells.append(v4.new_markdown_cell("## 2. Profile"))
    cells.append(v4.new_code_cell("\n".join([
        "print(f'{len(df):,} rows × {df.shape[1]} columns; {int(df.isna().sum().sum()):,} missing cells; '",
        "      f'{int(df.duplicated().sum()):,} duplicate rows')",
        "display(pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'distinct': df.nunique()}))",
        "display(df.describe(include='all').T)",
    ])))

    # cleaning
    cells.append(v4.new_markdown_cell(
        "## 3. Cleaning\n\n" + ("No cleaning steps were applied in the app." if not history else
                                f"{len(history)} step(s), applied in order. Row labels (the index) are never reset, so rows keep their identity.")))
    for i, h in enumerate(history, 1):
        after = h.get("after") or {}
        cells.append(v4.new_markdown_cell(f"**Step {i} — {h.get('label', h['operation'])}.** Recorded in the app: {h.get('detail', '')} "
                                          f"(after: {after.get('rows', '?')} rows × {after.get('columns', '?')} columns)."))
        cells.append(v4.new_code_cell(_cleaning_code(h["operation"], h.get("params") or {}) + "\nprint(df.shape)"))
    hash_code = [
        "def frame_hash(frame):",
        "    \"\"\"Same content hash as the app: column names + dtypes, then values and index.\"\"\"",
        "    h = hashlib.sha256()",
        "    h.update('|'.join(f'{c}:{t}' for c, t in zip(frame.columns, frame.dtypes.astype(str))).encode())",
        "    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())",
        "    return h.hexdigest()",
        "",
        f"EXPECTED_SHAPE = {_r(tuple(ctx['expected_shape']) if ctx.get('expected_shape') else None)}",
        f"EXPECTED_HASH = {_r(ctx.get('expected_hash'))}",
        "if EXPECTED_SHAPE is not None:",
        "    assert df.shape == EXPECTED_SHAPE, f'Shape {df.shape} differs from the app ({EXPECTED_SHAPE})'",
        "if EXPECTED_HASH is not None:",
        "    print('hash match' if frame_hash(df) == EXPECTED_HASH else 'hash MISMATCH: the cleaned data differs from the app')",
    ]
    cells.append(v4.new_code_cell("\n".join(hash_code)))

    # charts
    cells.append(v4.new_markdown_cell("## 4. Exploratory charts\n\n" + (
        "No charts were saved in the app." if not charts else
        "Each chart re-executes the saved chart specification (filters, aggregation, top-N) on the cleaned data.")))
    if charts:
        cells.append(v4.new_code_cell(_FILTER_FN))
        for ch in charts:
            spec = ch["spec"]
            title = ch.get("title") or spec.get("title") or spec.get("chart")
            cells.append(v4.new_markdown_cell(f"### {title}"))
            cells.append(v4.new_code_cell(_chart_code(spec, title)))

    cells.extend(_modeling_cells(ctx))
    cells.extend(_clustering_cells(ctx))

    # conclusions
    findings = ctx.get("findings") or []
    concl = ["## Conclusions and limitations", "",
             "_The findings below were computed by DataStory AI at export time from the same data and are reproduced here as text; "
             "re-running the cells above recomputes the underlying numbers._", ""]
    for f in findings:
        concl.append(f"- **{f.get('title', '')}** — {f.get('observation', '')}" + (f" *Limitation:* {f['limitation']}" if f.get("limitation") else ""))
    if not findings:
        concl.append("- No findings were recorded.")
    concl += ["", "General limitations: associations shown here are not causal; metrics on small test sets are uncertain; "
              "results apply to this dataset version only."]
    if ds.get("synthetic"):
        concl.append("This is a synthetic demonstration dataset, so none of these results describe the real world.")
    cells.append(v4.new_markdown_cell("\n".join(concl)))
    nbformat.validate(nb)
    return nb


def notebook_bytes(ctx: dict[str, Any]) -> bytes:
    return nbformat.writes(build_notebook(ctx)).encode("utf-8")


def bundle_zip(ctx: dict[str, Any], dataset_csv_bytes: bytes) -> bytes:
    """Notebook + dataset + pinned requirements + README in one zip."""
    ctx = dict(ctx)
    path = default_data_path(ctx)
    ctx["data_path"] = path
    ctx["data_sha256"] = hashlib.sha256(dataset_csv_bytes).hexdigest()
    reqs = "\n".join(f"{k}=={v}" for k, v in package_versions().items() if v != "not installed") + "\nnbformat\njupyter\n"
    readme = (
        "DataStory AI — reproducible analysis bundle\n\n"
        "Contents:\n  analysis.ipynb   the notebook\n"
        f"  {path}   the original dataset as loaded in the app\n  requirements.txt   package versions used at export\n\n"
        "Run:\n  python -m venv .venv && . .venv/bin/activate   (Windows: .venv\\Scripts\\activate)\n"
        "  pip install -r requirements.txt\n  jupyter notebook analysis.ipynb   then Run All\n\n"
        "The notebook checks the dataset's SHA-256, re-applies every cleaning step, compares the cleaned data's content hash and "
        "the model metrics with the values recorded in the app, and reports any difference.\n"
        "The bundle includes your data. Share it only with people allowed to see that data.\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis.ipynb", notebook_bytes(ctx))
        z.writestr(path, dataset_csv_bytes)
        z.writestr("requirements.txt", reqs)
        z.writestr("README.txt", readme)
        z.writestr("manifest.json", json.dumps({"exported_at": datetime.now(timezone.utc).isoformat(), "data_path": path,
                                                "data_sha256": ctx["data_sha256"]}, indent=2))
    return buf.getvalue()
