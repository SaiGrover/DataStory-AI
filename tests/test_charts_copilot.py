"""Chart specs, recommendations, natural-language charts and the tool-based copilot."""
import pandas as pd
import pytest

from ai import copilot
from core import charts as ch
from core import data_loader as dl
from core import nl_chart, profiler
from core.results import build_results
from fixtures import employee_attrition, netflix_like, sales


@pytest.fixture(scope="module")
def netflix():
    df = netflix_like(3000)
    return df, profiler.profile_dataset(df)


# --------------------------------------------------------------- specs
def test_spec_schema_rejects_unknown_fields_and_values():
    with pytest.raises(ch.ChartSpecError):
        ch.ChartSpec.from_dict({"chart": "bar", "x": "a", "code": "import os"})
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert ch.validate(df, ch.ChartSpec("pie", x="a"))
    assert ch.validate(df, ch.ChartSpec("bar", x="a", y="b", agg="median_of_squares"))
    assert ch.validate(df, ch.ChartSpec("bar", x="missing_col"))
    assert ch.validate(df, ch.ChartSpec("scatter", x="a", y="b"))  # categorical x in a scatter
    assert ch.validate(df, ch.ChartSpec("bar", x="a", filters=[{"column": "a", "op": ">", "value": 1}]))


def test_aggregation_values_are_correct():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0, None]})
    out = ch.render(df, ch.ChartSpec("bar", x="g", y="v", agg="mean"))
    t = out.table.set_index("g")
    assert t.loc["a", "mean_v"] == 2.0 and t.loc["b", "mean_v"] == 15.0 and t.loc["b", "n"] == 2
    out = ch.render(df, ch.ChartSpec("bar", x="g", y="v", agg="sum"))
    assert out.table.set_index("g").loc["b", "sum_v"] == 30.0


def test_top_n_with_other_bucket_and_missing_handling():
    df = pd.DataFrame({"c": list("aaaabbbcdefg") + [None]})
    out = ch.render(df, ch.ChartSpec("bar", x="c", top_n=2, other=True))
    assert out.table["c"].tolist()[:2] == ["a", "b"] and out.table["c"].iloc[-1].startswith("Other")
    assert int(out.table["count"].sum()) == 12  # missing excluded by default
    out = ch.render(df, ch.ChartSpec("bar", x="c", top_n=50, dropna=False))
    assert "(missing)" in out.table["c"].tolist()


def test_rate_aggregation():
    df = employee_attrition()
    out = ch.render(df, ch.ChartSpec("bar", x="Department", y="Attrition", agg="rate"))
    t = out.table.set_index("Department")
    expected = 100 * (df[df.Department == "Sales"].Attrition == "Yes").mean()
    assert abs(t.loc["Sales", "rate"] - expected) < 1e-9


def test_empty_after_filter_is_reported():
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    with pytest.raises(ch.ChartSpecError, match="No rows"):
        ch.render(df, ch.ChartSpec("bar", x="a", filters=[{"column": "a", "op": "==", "value": "zzz"}]))


def test_identifier_and_unique_columns_are_not_recommended(netflix):
    df, p = netflix
    assert "show_id" in p["id_candidates"]
    recs = ch.recommend(df, p, limit=10)
    used = {c for r in recs for c in r["columns"]}
    assert "show_id" not in used and "title" not in used and "description" not in used
    titles = " ".join(r["title"] for r in recs)
    assert "type" in titles and "release_year" in titles and "date_added" in titles
    assert ch.chart_suitability(df, "show_id", p)


def test_multi_valued_country_is_split(netflix):
    df, p = netflix
    out = ch.render(df, ch.ChartSpec("bar", x="country", explode=True), p["kinds"])
    assert "United States" in out.table["country"].tolist()
    assert not any("," in v for v in out.table["country"])
    assert any("split" in n for n in out.notes)


def test_month_name_dates_detected(netflix):
    _, p = netflix
    assert p["kinds"]["date_added"] == "datetime"


def test_long_labels_are_truncated_with_full_hover():
    df = pd.DataFrame({"c": ["A very long category label that goes on and on " + str(i % 3) for i in range(30)]})
    out = ch.render(df, ch.ChartSpec("bar", x="c"))
    bar = out.figure.data[0]
    assert all(len(str(y)) <= ch.LABEL_MAX for y in bar.y) and all(len(cd) > ch.LABEL_MAX for cd in bar.customdata)


# ---------------------------------------------------------- NL → chart
@pytest.mark.parametrize("text,chart,x,agg", [
    ("Show the distribution of release years.", "histogram", "release_year", None),
    ("Plot the top 10 ratings", "bar", "rating", "count"),
    ("Titles added per month", "line", "date_added", "count"),
])
def test_nl_netflix(netflix, text, chart, x, agg):
    df, p = netflix
    r = nl_chart.parse(text, df, p["kinds"])
    assert r.spec is not None, r.errors
    assert r.spec.chart == chart and r.spec.x == x
    if agg:
        assert r.spec.agg == agg


def test_nl_movie_tv_by_country_splits_by_type(netflix):
    df, p = netflix
    r = nl_chart.parse("Compare movie and TV show counts by country.", df, p["kinds"])
    assert r.spec.chart == "stacked_bar" and r.spec.x == "country" and r.spec.color == "type" and r.spec.explode


def test_nl_identifier_request_is_refused(netflix):
    df, p = netflix
    r = nl_chart.parse("show_id counts", df, p["kinds"])
    assert r.spec is None and "identifier" in " ".join(r.errors)


def test_nl_sales_and_hr_requests():
    s = sales()
    k = profiler.infer_kinds(s)
    r = nl_chart.parse("Plot monthly revenue.", s, k)
    assert (r.spec.chart, r.spec.x, r.spec.y, r.spec.agg, r.spec.freq) == ("line", "order_date", "revenue", "sum", "MS")
    r = nl_chart.parse("Plot the top 10 categories by sales.", s, k)
    assert (r.spec.chart, r.spec.x, r.spec.y, r.spec.top_n) == ("bar", "category", "revenue", 10)
    r = nl_chart.parse("Create a correlation heatmap.", s, k)
    assert r.spec.chart == "heatmap"
    h = employee_attrition()
    kh = profiler.infer_kinds(h)
    r = nl_chart.parse("Show attrition rates across departments.", h, kh)
    assert (r.spec.x, r.spec.y, r.spec.agg) == ("Department", "Attrition", "rate")
    r = nl_chart.parse("Show the relationship between age and salary.", h, kh)
    assert r.spec.chart == "scatter" and {r.spec.x, r.spec.y} == {"Age", "MonthlyIncome"}
    r = nl_chart.parse("Generate a chart showing the relationship between salary and attrition", h, kh)
    assert r.spec.chart == "box" and r.spec.y == "MonthlyIncome" and r.spec.x == "Attrition"


def test_nl_unknown_column_asks_for_clarification():
    h = employee_attrition()
    r = nl_chart.parse("plot frobnicate levels", h, profiler.infer_kinds(h))
    assert r.spec is None and r.clarification and "Department" in r.clarification


def test_nl_follow_up_modifies_previous_chart():
    h = employee_attrition()
    k = profiler.infer_kinds(h)
    first = nl_chart.parse("histogram of age", h, k).spec
    r = nl_chart.parse("now by Department", h, k, previous=first)
    assert r.spec.x == "Age" and r.spec.color == "Department"


# --------------------------------------------------------------- copilot
@pytest.fixture(scope="module")
def hr_ctx():
    df = employee_attrition()
    _, meta = dl.load_upload("hr.csv", dl.to_csv_bytes(df))
    r = build_results(df, meta, target="Attrition")
    return copilot.Ctx(df, r, r["profile"]["kinds"])


@pytest.mark.parametrize("q,tool", [
    ("What are the biggest issues in this dataset?", "quality_issues"),
    ("Which features are most strongly associated with the target?", "target_associations"),
    ("Show missing values by column", "missing_by_column"),
    ("Generate a histogram of age", "make_chart"),
    ("Which model performed best?", "model_summary"),
    ("Explain the confusion matrix", "confusion_matrix"),
    ("What cleaning steps should I perform?", "cleaning_suggestions"),
    ("Which customer segments are present?", "segments"),
    ("Summarize the main findings", "insights"),
    ("average MonthlyIncome by Department", "group_aggregate"),
    ("how many rows where Department is Sales", "filter_count"),
])
def test_copilot_routes_to_tools(hr_ctx, q, tool):
    a = copilot.ask(q, hr_ctx, use_llm=False)
    assert a["tool"] == tool and a["answer"]


def test_copilot_numbers_come_from_data(hr_ctx):
    df = hr_ctx.df
    a = copilot.ask("how many rows where Department is Sales", hr_ctx, use_llm=False)
    assert f"{int((df.Department == 'Sales').sum()):,}" in a["answer"]
    a = copilot.ask("average MonthlyIncome by Department", hr_ctx, use_llm=False)
    top = df.groupby("Department").MonthlyIncome.mean().sort_values(ascending=False)
    assert a["table"].iloc[0]["Department"] == top.index[0]


def test_copilot_refuses_code_execution(hr_ctx):
    for q in ["import os; os.system('rm -rf /')", "run this python code: print(1)", "SELECT * FROM users", "eval(open('x').read())"]:
        a = copilot.ask(q, hr_ctx, use_llm=False)
        assert a["tool"] == "refused" and "don't execute code" in a["answer"]


def test_tool_argument_validation():
    with pytest.raises(ValueError):
        copilot.validate_args("group_aggregate", {"group": "a"})  # missing value
    with pytest.raises(ValueError):
        copilot.validate_args("group_aggregate", {"group": "a", "value": "b", "agg": "exec"})
    with pytest.raises(ValueError):
        copilot.validate_args("column_stats", {"column": "a", "shell": "ls"})
    with pytest.raises(ValueError):
        copilot.validate_args("not_a_tool", {})


def test_invalid_column_returns_readable_error(hr_ctx):
    res = copilot.run_tool("column_stats", {"column": "zzz_nope"}, hr_ctx)
    assert not res.ok and "no column" in res.text.lower()


def test_llm_tool_proposal_is_validated(hr_ctx, monkeypatch):
    from ai import llm

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm, "chat_tools", lambda *a, **k: {"tool_calls": [{"function": {"name": "group_aggregate",
                                                                                           "arguments": '{"group": "Department", "value": "MonthlyIncome", "agg": "DROP TABLE"}'}}]})
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Rephrased.")
    a = copilot.ask("average MonthlyIncome by Department", hr_ctx)
    assert a["routed_by"] == "rules" and a["tool"] == "group_aggregate"  # invalid LLM call rejected, rules used


def test_llm_rephrasing_with_invented_numbers_is_discarded(hr_ctx, monkeypatch):
    from ai import llm

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm, "chat_tools", lambda *a, **k: {"tool_calls": []})
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Exactly 98,765 employees left, a 77.7% rate.")
    a = copilot.ask("how many rows where Department is Sales", hr_ctx)
    assert a["mode"] == "deterministic" and "98,765" not in a["answer"] and a["note"]


def test_no_llm_means_rule_based(hr_ctx, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = copilot.ask("Show missing values by column", hr_ctx)
    assert a["mode"] == "deterministic"
