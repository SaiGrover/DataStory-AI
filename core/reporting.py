"""Report generation (Markdown, HTML, PDF).

``build_sections`` turns the shared results structure into format-neutral
blocks; the three renderers only lay those blocks out. Because all formats are
rendered from the same blocks, they cannot disagree with each other or with the
dashboard, and a value that was not computed appears as "Unavailable" or is
omitted — it is never filled in.
"""
from __future__ import annotations

import base64
import html
import io
import re
from typing import Any

from core.data_loader import SYNTHETIC_LABEL
from core.modeling import METRIC_LABELS
from core.quality_metrics import format_dimension
from core.results import UNAVAILABLE, fmt

SECTION_TITLES = [
    "Dataset overview", "Data provenance", "Data-quality assessment", "Cleaning operations",
    "Exploratory analysis", "Machine-learning methodology", "Model evaluation", "Feature importance",
    "Analytical narrative", "Limitations", "Recommendations and next steps",
]


def P(text): return {"type": "p", "text": text}
def NOTE(text): return {"type": "note", "text": text}
def BULLETS(items): return {"type": "bullets", "items": list(items)}
def TABLE(columns, rows): return {"type": "table", "columns": list(columns), "rows": [[_cell(c) for c in r] for r in rows]}
def CHART(title, labels, values, errors=None, xlabel=""): return {"type": "chart", "title": title, "labels": list(labels), "values": list(values), "errors": errors, "xlabel": xlabel}


def _cell(v: Any) -> str:
    return v if isinstance(v, str) else fmt(v)


def build_sections(results: dict[str, Any], narrative: str | None = None, include: list[str] | None = None) -> list[dict[str, Any]]:
    p, q, ds, cl = results["profile"], results["quality"], results["dataset"], results["cleaning"]
    m, imp = results.get("modeling"), results.get("importance")
    dims = q["dimensions"]
    S: list[dict[str, Any]] = []

    # 1 overview
    kc = p["kind_counts"]
    S.append({"title": "Dataset overview", "blocks": [
        TABLE(["Property", "Value"], [
            ["Dataset", ds["name"]], ["Rows", p["rows"]], ["Columns", p["columns"]],
            ["Numeric / categorical / datetime / text columns", f"{kc['numeric']} / {kc['categorical']} / {kc['datetime']} / {kc['text']}"],
            ["Missing cells", f"{p['missing_cells']:,} ({fmt(p['missing_pct'], 2, pct=True)})"],
            ["Exact duplicate rows", f"{p['duplicate_rows']:,} ({fmt(p['duplicate_pct'], 2, pct=True)})"],
            ["Memory footprint", f"{fmt(p['memory_mb'], 2)} MB"],
            ["Target column", results["target"] or "Not selected"],
        ]),
        TABLE(["Column", "Type", "Missing", "Missing %", "Distinct"],
              [[c["name"], c["kind"], c["missing"], fmt(c["missing_pct"], 1, pct=True), c["unique"]] for c in p["column_details"][:40]]),
    ] + ([NOTE(f"Column table truncated to the first 40 of {p['columns']} columns.")] if p["columns"] > 40 else [])})

    # 2 provenance
    blocks = [P(f"Source: {'built-in sample dataset' if ds['source'] == 'sample' else 'user upload'}."), P(ds["provenance"])]
    if ds.get("synthetic"):
        blocks.insert(0, NOTE(f"{SYNTHETIC_LABEL}. All figures in this report describe generated values, not real-world observations."))
    blocks += [P(n) for n in ds.get("notes", [])]
    S.append({"title": "Data provenance", "blocks": blocks})

    # 3 quality
    rows = [[k.title(), format_dimension(d), d.get("definition") or q["definitions"][k]] for k, d in dims.items()]
    blocks = [TABLE(["Dimension", "Result", "Definition"], rows)]
    if dims["completeness"].get("excluded_columns"):
        blocks.append(P("Columns excluded from completeness: " + ", ".join(dims["completeness"]["excluded_columns"]) + "."))
    rule_rows = [[r["dimension"].title(), r["rule"], r.get("error") or f"{r['passed']:,} of {r['applicable']:,} ({fmt(r['value'], 1, pct=True)})"]
                 for k in ("validity", "consistency") for r in dims[k].get("rules", [])]
    if rule_rows:
        blocks.append(TABLE(["Dimension", "Rule", "Passed"], rule_rows))
    h = q["health"]
    if h.get("value") is not None:
        blocks.append(P(f"Data health score: {h['value']:.1f} / 100, based on {', '.join(h['included'])}"
                        + (f" ({', '.join(h['excluded'])} not evaluated)." if h["excluded"] else ".")))
        blocks.append(NOTE(h["formula"]))
    else:
        blocks.append(P(f"Data health score: {h.get('message', UNAVAILABLE)}"))
    if p["outlier_columns"]:
        blocks.append(P(p["outlier_method"]))
        blocks.append(TABLE(["Column", "Outliers", "% of non-missing"], [[r["column"], r["outliers"], fmt(r["outlier_pct"], 1, pct=True)] for r in p["outlier_columns"][:15]]))
    for label, key in (("Constant columns", "constant_columns"), ("Near-constant columns", "near_constant_columns"), ("Identifier-like columns", "id_candidates")):
        if p[key]:
            blocks.append(P(f"{label}: {', '.join(p[key])}."))
    S.append({"title": "Data-quality assessment", "blocks": blocks})

    # 4 cleaning
    if cl["history"]:
        blocks = [TABLE(["#", "Operation", "What changed", "Rows", "Missing cells"],
                        [[i, hrec["label"], hrec["detail"], f"{hrec['before']['rows']:,} → {hrec['after']['rows']:,}",
                          f"{hrec['before']['missing_cells']:,} → {hrec['after']['missing_cells']:,}"] for i, hrec in enumerate(cl["history"], 1)]),
                  TABLE(["", "Rows", "Columns", "Missing cells", "Duplicate rows"],
                        [["Original"] + [cl["original"][k] for k in ("rows", "columns", "missing_cells", "duplicate_rows")],
                         ["Current"] + [cl["current"][k] for k in ("rows", "columns", "missing_cells", "duplicate_rows")]]),
                  NOTE("These figures describe changes to the data. No effect on model performance is claimed unless models were retrained and compared.")]
    else:
        blocks = [P("No cleaning operations were applied; the analysis uses the dataset as loaded.")]
    S.append({"title": "Cleaning operations", "blocks": blocks})

    # 5 EDA
    blocks = []
    miss = sorted([c for c in p["column_details"] if c["missing"]], key=lambda c: -c["missing"])[:15]
    if miss:
        blocks.append(CHART("Missing values by column (% of rows)", [c["name"] for c in miss], [c["missing_pct"] for c in miss], xlabel="% missing"))
    num = [c for c in p["column_details"] if c.get("mean") is not None][:25]
    if num:
        blocks.append(TABLE(["Numeric column", "Mean", "Std", "Min", "Median", "Max", "Skew"],
                            [[c["name"]] + [fmt(c[k], 3) for k in ("mean", "std", "min", "median", "max", "skew")] for c in num]))
    cat = [c for c in p["column_details"] if c["kind"] == "categorical" and c.get("top_value") is not None][:20]
    if cat:
        blocks.append(TABLE(["Categorical column", "Distinct", "Most frequent", "Share"],
                            [[c["name"], c["unique"], c["top_value"], fmt(c["top_share_pct"], 1, pct=True)] for c in cat]))
    pairs = results["correlations"]["pairs"][:10]
    if pairs:
        blocks.append(CHART("Strongest Pearson correlations", [f"{x['a']} ↔ {x['b']}" for x in pairs], [x["r"] for x in pairs], xlabel="r"))
        blocks.append(NOTE("Correlation measures linear association in this dataset; it does not establish causation."))
    t = p.get("target")
    if t and t.get("class_counts"):
        items = list(t["class_counts"].items())[:20]
        blocks.append(CHART(f"Class distribution of {t['name']}", [k for k, _ in items], [v for _, v in items], xlabel="rows"))
    S.append({"title": "Exploratory analysis", "blocks": blocks or [P("No exploratory summaries are available for this dataset.")]})

    # 6-7 modelling
    if m:
        f = m["features"]
        blocks = [TABLE(["Setting", "Value"], [
            ["Task", m["task"]], ["Target", m["target"]],
            ["Rows used", m["n_rows"]], ["Training rows", m["n_train"]], ["Held-out test rows", m["n_test"]],
            ["Split", f"{1 - m['test_size']:.0%} train / {m['test_size']:.0%} test" + (", stratified by class" if m["stratified"] else "")],
            ["Random seed", str(m["seed"])], ["Cross-validation", f"{m['cv_folds']}-fold" if m["cv_folds"] else "Not used"],
            ["Hyperparameter tuning", {"none": "Not used", "grid": "Grid search", "random": "Randomized search"}[m["tuning"]]],
            ["Imbalance handling", {"none": "None", "class_weight": "Class weighting (where supported)", "smote": f"SMOTE on training folds only (k = {m['smote_k']})"}[m["imbalance"]]],
            ["Features", f"{len(f['numeric'])} numeric, {len(f['categorical'])} categorical"],
            ["Model selection", m["ranking_basis"]],
        ]), P(m["preprocessing"])]
        if f["excluded"]:
            blocks.append(P("Excluded features: " + "; ".join(f"{e['column']} ({e['reason']})" for e in f["excluded"]) + "."))
        blocks += [NOTE(w) for w in m["warnings"]]
        S.append({"title": "Machine-learning methodology", "blocks": blocks})

        ok = [x for x in m["models"] if "error" not in x]
        cols = ["Model"] + ([f"CV {METRIC_LABELS[m['primary_metric']]}"] if any(x["cv"] for x in ok) else []) + [f"Test {METRIC_LABELS[k]}" for k in m["metric_names"]]
        rows = []
        for x in ok:
            r = [x["name"]]
            if len(cols) > 1 + len(m["metric_names"]):
                r.append(f"{x['cv']['mean']:.4f} ± {x['cv']['std']:.4f}" if x["cv"] else UNAVAILABLE)
            rows.append(r + [fmt(x["metrics"].get(k)) for k in m["metric_names"]])
        best = ok[0]
        blocks = [P(f"Best model: {best['name']} — selected by {m['ranking_basis']}."), TABLE(cols, rows)]
        if best.get("best_params"):
            blocks.append(P("Tuned hyperparameters of the best model: " + ", ".join(f"{k} = {v}" for k, v in best["best_params"].items()) + "."))
        if m["task"] == "classification":
            blocks.append(P(f"Precision, recall and F1 use {best['averaging']} averaging."))
            cm = best["confusion_matrix"]
            blocks.append(TABLE(["Actual \\ Predicted"] + m["classes"], [[c] + row for c, row in zip(m["classes"], cm)]) if len(m["classes"]) <= 12
                          else P(f"Confusion matrix omitted ({len(m['classes'])} classes); see the application."))
            blocks.append(TABLE(["Class", "Precision", "Recall", "F1-score", "Support"],
                                [[c, fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1-score"]), int(v["support"])] for c, v in best["class_report"].items()]))
            if best.get("roc_auc_note"):
                blocks.append(NOTE(best["roc_auc_note"]))
        failed = [x for x in m["models"] if "error" in x]
        if failed:
            blocks.append(P("Models that failed to train: " + "; ".join(f"{x['name']} ({x['error']})" for x in failed) + "."))
        S.append({"title": "Model evaluation", "blocks": blocks})
    else:
        S.append({"title": "Machine-learning methodology", "blocks": [P("No model has been trained for this dataset.")]})
        S.append({"title": "Model evaluation", "blocks": [P("Unavailable — no model has been trained.")]})

    # 8 importance
    if imp:
        feats = imp["features"][:20]
        S.append({"title": "Feature importance", "blocks": [
            P(f"Method: {imp['method']} for '{imp['model']}', evaluated on the {imp['evaluated_on']}. Scoring metric: {imp['metric_label']}. "
              f"Repeats: {imp['n_repeats']}. Seed: {imp['seed']}."),
            CHART(f"Mean change in {imp['metric_label']} when a feature is shuffled", [x["feature"] for x in feats], [x["mean"] for x in feats],
                  errors=[x["std"] for x in feats], xlabel=f"Δ {imp['metric_label']}"),
            TABLE(["Feature", "Mean importance", "Std"], [[x["feature"], fmt(x["mean"]), fmt(x["std"])] for x in feats]),
            NOTE(imp["explanation"])] + [NOTE(n) for n in imp["notes"]]})
    else:
        S.append({"title": "Feature importance", "blocks": [P("Feature importance unavailable — a valid fitted model and evaluation dataset are required.")]})

    # 9 narrative
    blocks = []
    if narrative:
        blocks += [P(par) for par in narrative.split("\n\n") if par.strip()]
    for ins in results["insights"]:
        blocks.append({"type": "h3", "text": ins["title"]})
        blocks.append(P(f"Observation: {ins['observation']}"))
        blocks.append(BULLETS(ins["evidence"]))
        blocks.append(P(f"Interpretation: {ins['interpretation']}"))
        blocks.append(P(f"Limitation: {ins['limitation']}"))
    S.append({"title": "Analytical narrative", "blocks": blocks})

    # 10 limitations
    lim = ["All statistics describe this dataset only; they are not estimates for a wider population unless the data is a representative sample.",
           "Correlations and feature importances describe association and model reliance, not causation.",
           "Column types and identifier columns are inferred heuristically and may need manual correction."]
    if ds.get("synthetic"):
        lim.insert(0, f"{SYNTHETIC_LABEL}: findings have no real-world meaning.")
    if dims["validity"]["value"] is None or dims["consistency"]["value"] is None:
        lim.append("Validity and/or consistency were not evaluated because no validation rules were configured.")
    if m:
        lim.append(f"Model scores come from one train/test split (seed {m['seed']}) of {m['n_rows']:,} rows; expect variation on new data.")
        if not m["ranked_by_cv"]:
            lim.append("The best model was chosen using the same test set that reports its score, which makes that score slightly optimistic.")
    S.append({"title": "Limitations", "blocks": [BULLETS(lim)]})

    # 11 next steps
    steps = [f"{ins['title']}: {ins['next_step']}" for ins in results["insights"] if ins["severity"] != "info"]
    steps += [s for s in cl["suggestions"][:6]]
    if not m:
        steps.append("Select a target and train models to obtain evaluation metrics and feature importance.")
    elif not imp:
        steps.append("Compute permutation importance on the Model Results page to see which features the best model relies on.")
    S.append({"title": "Recommendations and next steps",
              "blocks": [BULLETS(list(dict.fromkeys(steps)) or ["No specific follow-up actions were identified from the computed analysis."]),
                         NOTE("Recommendations are prompts for further analysis derived from the computed profile; no outcome or improvement is implied.")]})
    if include:
        S = [s for s in S if s["title"] in include]
    return S


def _title(results) -> tuple[str, str]:
    ds = results["dataset"]
    return f"DataStory AI Report — {ds['name']}", f"Generated {results['generated_at']}"


def _strip_md(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*|\*(.+?)\*", lambda mo: mo.group(1) or mo.group(2), text)


# ----------------------------------------------------------------- markdown
def to_markdown(results: dict[str, Any], narrative: str | None = None, include: list[str] | None = None) -> str:
    title, sub = _title(results)
    out = [f"# {title}", "", f"_{sub}_", ""]
    if results["dataset"].get("synthetic"):
        out += [f"> **{SYNTHETIC_LABEL}**", ""]
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    for i, sec in enumerate(build_sections(results, narrative, include), 1):
        out += [f"## {i}. {sec['title']}", ""]
        for b in sec["blocks"]:
            if b["type"] == "p":
                out += [b["text"], ""]
            elif b["type"] == "h3":
                out += [f"### {b['text']}", ""]
            elif b["type"] == "note":
                out += [f"> {b['text']}", ""]
            elif b["type"] == "bullets":
                out += [f"- {x}" for x in b["items"]] + [""]
            elif b["type"] == "table":
                out += ["| " + " | ".join(esc(c) for c in b["columns"]) + " |", "|" + " --- |" * len(b["columns"])]
                out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in b["rows"]] + [""]
            elif b["type"] == "chart":
                out += [f"**{b['title']}**", "", "| Item | Value |" + (" ± |" if b["errors"] else ""), "| --- | ---: |" + (" ---: |" if b["errors"] else "")]
                for j, (l, v) in enumerate(zip(b["labels"], b["values"])):
                    out.append(f"| {esc(l)} | {fmt(v)} |" + (f" {fmt(b['errors'][j])} |" if b["errors"] else ""))
                out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------- charts
def chart_png(block: dict[str, Any], dpi: int = 150) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, values = block["labels"][::-1], [v if v is not None else 0 for v in block["values"]][::-1]
    errors = block["errors"][::-1] if block["errors"] else None
    fig, ax = plt.subplots(figsize=(7.2, max(1.6, 0.3 * len(labels) + 0.9)))
    colors = ["#667F4D" if v >= 0 else "#A65D57" for v in values]
    ax.barh([str(l)[:38] for l in labels], values, xerr=errors, color=colors, ecolor="#39463A", capsize=2, height=0.62)
    ax.axvline(0, color="#39463A", linewidth=0.6)
    ax.set_title(block["title"], fontsize=10, loc="left", color="#172018")
    ax.set_xlabel(block.get("xlabel", ""), fontsize=8)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#E1E8D8", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------- html
_CSS = """body{font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#172018;background:#fbfbf5;max-width:960px;margin:32px auto;padding:0 24px;line-height:1.55}
h1{font-size:28px;margin-bottom:4px}h2{font-size:20px;margin-top:36px;padding-bottom:6px;border-bottom:2px solid #e1e8d8;color:#3f5b2f}h3{font-size:15px;margin:20px 0 4px}
.sub{color:#697468;font-size:13px}.banner{background:#fbf3df;border:1px solid #e6cf9a;border-radius:8px;padding:10px 14px;font-weight:700;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th{background:#eef3e6;text-align:left}th,td{border:1px solid #e1e8d8;padding:6px 9px;vertical-align:top}
.note{background:#f4f7ef;border-left:3px solid #93ad74;padding:8px 12px;font-size:13px;color:#39463a;margin:10px 0}img{max-width:100%}"""


def to_html(results: dict[str, Any], narrative: str | None = None, include: list[str] | None = None) -> str:
    e = html.escape
    title, sub = _title(results)
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{e(title)}</title><style>{_CSS}</style></head><body>",
           f"<h1>{e(title)}</h1><div class='sub'>{e(sub)}</div>"]
    if results["dataset"].get("synthetic"):
        out.append(f"<div class='banner'>{e(SYNTHETIC_LABEL)}</div>")
    for i, sec in enumerate(build_sections(results, narrative, include), 1):
        out.append(f"<h2>{i}. {e(sec['title'])}</h2>")
        for b in sec["blocks"]:
            if b["type"] == "p":
                out.append(f"<p>{e(_strip_md(b['text']))}</p>")
            elif b["type"] == "h3":
                out.append(f"<h3>{e(b['text'])}</h3>")
            elif b["type"] == "note":
                out.append(f"<div class='note'>{e(b['text'])}</div>")
            elif b["type"] == "bullets":
                out.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in b["items"]) + "</ul>")
            elif b["type"] == "table":
                out.append("<table><thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in b["columns"]) + "</tr></thead><tbody>"
                           + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in b["rows"]) + "</tbody></table>")
            elif b["type"] == "chart":
                out.append(f"<img alt='{e(b['title'])}' src='data:image/png;base64,{base64.b64encode(chart_png(b, 110)).decode()}'>")
    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------- pdf
def _fonts() -> tuple[str, str]:
    """Register a Unicode font (DejaVu, shipped with matplotlib) so symbols such as ≥, ↔ and ² render."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    try:
        import matplotlib
        from pathlib import Path

        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if "DSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DSans", str(d / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DSans-Bold", str(d / "DejaVuSans-Bold.ttf")))
        return "DSans", "DSans-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def to_pdf(results: dict[str, Any], narrative: str | None = None, include: list[str] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold = _fonts()
    ink, sage, line = colors.HexColor("#172018"), colors.HexColor("#4E6B3A"), colors.HexColor("#D9E2CE")
    base = ParagraphStyle("base", fontName=regular, fontSize=9.5, leading=13.5, textColor=ink, spaceAfter=6)
    st = {
        "title": ParagraphStyle("title", parent=base, fontName=bold, fontSize=20, leading=24, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9, textColor=colors.HexColor("#697468"), spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base, fontName=bold, fontSize=13.5, leading=17, textColor=sage, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base, fontName=bold, fontSize=10.5, spaceBefore=8, spaceAfter=3),
        "note": ParagraphStyle("note", parent=base, fontSize=8.5, leading=12, textColor=colors.HexColor("#39463A"),
                               backColor=colors.HexColor("#F1F5EA"), borderPadding=(5, 6, 5, 6), leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=10),
        "banner": ParagraphStyle("banner", parent=base, fontName=bold, fontSize=10, backColor=colors.HexColor("#FBF3DF"),
                                 borderPadding=(6, 8, 6, 8), spaceBefore=4, spaceAfter=12),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10.5, spaceAfter=0),
        "head": ParagraphStyle("head", parent=base, fontName=bold, fontSize=8, leading=10.5, spaceAfter=0),
    }
    e = lambda s: html.escape(_strip_md(str(s)), quote=False)
    title, sub = _title(results)
    synthetic = results["dataset"].get("synthetic")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                            title=title, author="DataStory AI")
    width = A4[0] - 36 * mm

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(colors.HexColor("#697468"))
        left = "DataStory AI" + (f"  ·  {SYNTHETIC_LABEL}" if synthetic else "")
        canvas.drawString(18 * mm, 10 * mm, left)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        canvas.restoreState()

    flow: list[Any] = [Paragraph(e(title), st["title"]), Paragraph(e(sub), st["sub"])]
    if synthetic:
        flow.append(Paragraph(e(SYNTHETIC_LABEL), st["banner"]))
    for i, sec in enumerate(build_sections(results, narrative, include), 1):
        flow.append(Paragraph(f"{i}. {e(sec['title'])}", st["h2"]))
        for b in sec["blocks"]:
            if b["type"] == "p":
                flow.append(Paragraph(e(b["text"]), base))
            elif b["type"] == "h3":
                flow.append(Paragraph(e(b["text"]), st["h3"]))
            elif b["type"] == "note":
                flow.append(Paragraph(e(b["text"]), st["note"]))
            elif b["type"] == "bullets":
                flow.append(ListFlowable([ListItem(Paragraph(e(x), base), leftIndent=10) for x in b["items"]],
                                         bulletType="bullet", bulletFontSize=6, leftIndent=12))
                flow.append(Spacer(1, 4))
            elif b["type"] == "table":
                n = len(b["columns"])
                lens = [max([len(str(b["columns"][j]))] + [min(len(str(r[j])), 60) for r in b["rows"]]) + 2 for j in range(n)]
                widths = [width * l / sum(lens) for l in lens]
                data = [[Paragraph(e(c), st["head"]) for c in b["columns"]]] + [[Paragraph(e(c), st["cell"]) for c in r] for r in b["rows"]]
                tbl = Table(data, colWidths=widths, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3E6")), ("GRID", (0, 0), (-1, -1), 0.4, line),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBF6")]),
                ]))
                flow += [tbl, Spacer(1, 8)]
            elif b["type"] == "chart":
                png = chart_png(b)
                img = Image(io.BytesIO(png))
                scale = min(1.0, width / img.imageWidth, (150 * mm) / img.imageHeight)
                img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
                flow += [KeepTogether([img]), Spacer(1, 6)]
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
