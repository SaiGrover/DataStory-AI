import numpy as np
import pandas as pd

from core import profiler
from core import quality_metrics as qm


def frame():
    return pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "y", None], "c": [1, 1, 1, 1]})


def test_missing_value_counts():
    p = profiler.profile_dataset(frame())
    assert p["missing_cells"] == 2 and abs(p["missing_pct"] - 100 * 2 / 12) < 1e-9
    by = {c["name"]: c["missing"] for c in p["column_details"]}
    assert by == {"a": 1, "b": 1, "c": 0}
    assert p["constant_columns"] == ["c"]


def test_duplicate_statistics():
    df = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    p = profiler.profile_dataset(df)
    assert p["duplicate_rows"] == 2 and p["duplicate_pct"] == 50.0
    assert qm.uniqueness(df)["value"] == 50.0


def test_completeness_and_exclusions():
    df = frame()
    assert abs(qm.completeness(df)["value"] - 100 * 10 / 12) < 1e-9
    c = qm.completeness(df, exclude=["b"])
    assert abs(c["value"] - 100 * 7 / 8) < 1e-9 and c["excluded_columns"] == ["b"]


# Credibility test 2 — metrics move when the data moves
def test_quality_metrics_change_with_input():
    clean = pd.DataFrame({"a": range(10), "b": list("abcdefghij")})
    q0 = qm.assess_quality(clean)
    assert q0["dimensions"]["completeness"]["value"] == 100.0 and q0["dimensions"]["uniqueness"]["value"] == 100.0
    dirty = clean.astype({"a": float}).copy()
    dirty.loc[:4, "a"] = np.nan
    dirty = pd.concat([dirty, dirty.iloc[[9]]], ignore_index=True)
    q1 = qm.assess_quality(dirty)
    assert abs(q1["dimensions"]["completeness"]["value"] - 100 * 17 / 22) < 1e-9
    assert abs(q1["dimensions"]["uniqueness"]["value"] - 100 * 10 / 11) < 1e-9
    assert q1["health"]["value"] < q0["health"]["value"]


# Credibility test 3 — no rules means "not evaluated", never a number
def test_validity_and_consistency_unavailable_without_rules():
    q = qm.assess_quality(frame())
    for k in ("validity", "consistency"):
        d = q["dimensions"][k]
        assert d["value"] is None and d["status"] == "not_evaluated"
        assert qm.format_dimension(d) == "Not evaluated — validation rules not configured."
    assert q["health"]["included"] == ["completeness", "uniqueness"]
    assert set(q["health"]["excluded"]) == {"validity", "consistency"}


def test_rule_based_validity():
    df = pd.DataFrame({"age": [5, 150, -1, None, 40]})
    q = qm.assess_quality(df, [{"dimension": "validity", "type": "range", "column": "age", "min": 0, "max": 120}])
    v = q["dimensions"]["validity"]
    assert v["passed"] == 2 and v["applicable"] == 4 and v["value"] == 50.0  # missing value is not "applicable"


def test_rule_based_consistency():
    df = pd.DataFrame({"start": [1, 5, 3], "end": [2, 4, 3], "sex": ["Male", "male", "F"]})
    rules = [{"dimension": "consistency", "type": "compare", "column": "start", "op": "<=", "other": "end"},
             {"dimension": "consistency", "type": "case", "column": "sex"}]
    c = qm.assess_quality(df, rules)["dimensions"]["consistency"]
    assert [r["passed"] for r in c["rules"]] == [2, 1]
    assert abs(c["value"] - 100 * 3 / 6) < 1e-9


def test_rule_on_removed_column_reports_error_not_score():
    q = qm.assess_quality(frame(), [{"dimension": "validity", "type": "numeric", "column": "gone"}])
    v = q["dimensions"]["validity"]
    assert v["value"] is None and "error" in v["rules"][0]


def test_health_score_is_the_documented_weighted_average():
    df = pd.DataFrame({"age": [5, 150, None, 40]})
    q = qm.assess_quality(df, [{"dimension": "validity", "type": "range", "column": "age", "min": 0, "max": 120}])
    d = q["dimensions"]
    expected = (0.4 * d["completeness"]["value"] + 0.2 * d["uniqueness"]["value"] + 0.2 * d["validity"]["value"]) / 0.8
    assert abs(q["health"]["value"] - expected) < 1e-9


def test_empty_frame_is_unavailable_not_a_crash():
    q = qm.assess_quality(pd.DataFrame({"a": []}))
    assert q["dimensions"]["uniqueness"]["value"] is None and q["dimensions"]["completeness"]["value"] is None
    assert q["health"]["value"] is None


def test_outliers_and_type_inference():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 500], "when": pd.date_range("2024-01-01", periods=10).astype(str),
                       "flag": [0, 1] * 5})
    p = profiler.profile_dataset(df, target="flag")
    assert p["outlier_columns"] == [{"column": "v", "outliers": 1, "outlier_pct": 10.0}]
    assert p["kinds"]["when"] == "datetime" and p["kinds"]["v"] == "numeric"
    assert p["target"]["class_counts"] == {"0": 5, "1": 5} and p["target"]["imbalance_ratio"] == 1.0
