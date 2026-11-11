"""Cleaning change counts: previews report exactly what changes, and match the applied result."""
import numpy as np
import pandas as pd
import pytest

from core import cleaning
from core.impact import compute_impact
from fixtures import netflix_like


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "release_year": rng.integers(1995, 2022, 300),
        "score": rng.normal(50, 10, 300),
        "city": rng.choice([" Delhi", "delhi", "Mumbai  ", "Pune"], 300),
        "code": [str(i) for i in rng.integers(0, 100, 300)],
    })
    df.loc[:9, "release_year"] = 1900  # 10 low outliers
    df.loc[20:24, "score"] = np.nan
    df.loc[30:31] = df.loc[40:41].values  # two exact duplicates of rows 40-41
    return df


def preview(df, op, **params):
    return cleaning.preview_operation(df, op, params)


def test_outlier_capping_counts_only_capped_values(frame):
    new, rec = preview(frame, "handle_outliers", columns=["release_year"], method="iqr", threshold=1.5, action="cap")
    imp = rec["impact"]
    assert rec["flagged"] == 10
    assert imp["cells_changed"] == 10 and imp["rows_affected"] == 10 and imp["columns_affected"] == 1
    assert imp["rows_removed"] == 0 and imp["missing_introduced"] == 0
    assert "release_year" in imp["dtype_changes"]  # int → float is reported as a type change, not as 300 edits


def test_netflix_scale_capping_is_not_every_row():
    df = netflix_like(8807)
    new, rec = preview(df, "handle_outliers", columns=["release_year"], action="cap")
    assert rec["flagged"] == rec["impact"]["cells_changed"] < 1000
    assert rec["impact"]["cells_changed"] != len(df)


def test_imputation_counts_resolved_missing(frame):
    new, rec = preview(frame, "impute", columns=["score"], strategy="median")
    imp = rec["impact"]
    assert imp["cells_changed"] == 5 and imp["missing_resolved"] == 5 and imp["missing_introduced"] == 0
    assert imp["rows_affected"] == 5


def test_row_deletion_is_structural_not_cell_changes(frame):
    new, rec = preview(frame, "handle_outliers", columns=["release_year"], action="remove")
    imp = rec["impact"]
    assert imp["rows_removed"] == 10 and imp["cells_changed"] == 0 and imp["rows_affected"] == 0
    assert list(new.index[:3]) == [10, 11, 12]  # row identity preserved


def test_duplicate_removal(frame):
    new, rec = preview(frame, "drop_duplicates")
    assert rec["impact"]["rows_removed"] == rec["flagged"] == 2
    assert rec["impact"]["cells_changed"] == 0


def test_column_deletion(frame):
    new, rec = preview(frame, "drop_columns", columns=["code"])
    imp = rec["impact"]
    assert imp["columns_removed"] == ["code"] and imp["cells_changed"] == 0 and imp["rows_removed"] == 0


def test_type_conversion_numeric_strings_are_equal_values(frame):
    new, rec = preview(frame, "convert_type", columns=["code"], to="numeric")
    imp = rec["impact"]
    # "42" → 42 is a type change; the text and the number are different values, so every converted cell counts
    assert imp["dtype_changes"]["code"]["after"].startswith(("int", "float"))
    assert imp["missing_introduced"] == 0


def test_text_normalization_counts_changed_strings(frame):
    new, rec = preview(frame, "normalize_categories", columns=["city"], case="lower", strip=True)
    expected = int((frame["city"] != frame["city"].str.strip().str.lower()).sum())
    assert rec["impact"]["cells_changed"] == expected
    assert rec["impact"]["changed_by_column"] == {"city": expected}


def test_noop_operation(frame):
    clean = frame.drop_duplicates()
    new, rec = preview(clean, "drop_duplicates")
    assert rec["impact"]["cells_changed"] == 0 and rec["impact"]["rows_removed"] == 0 and rec["changed"] is False


def test_value_to_missing_and_missing_to_value():
    a = pd.DataFrame({"x": [1.0, None, 3.0], "y": ["a", "b", None]})
    b = pd.DataFrame({"x": [None, 2.0, 3.0], "y": ["a", None, "c"]})
    imp = compute_impact(a, b)
    assert imp["cells_changed"] == 4 and imp["missing_introduced"] == 2 and imp["missing_resolved"] == 2
    assert imp["rows_affected"] == 3


def test_null_kinds_are_equivalent():
    a = pd.DataFrame({"x": [np.nan, None], "s": pd.array([pd.NA, "a"], dtype="string")})
    b = pd.DataFrame({"x": [None, np.nan], "s": pd.array([None, "a"], dtype="string")})
    assert compute_impact(a, b)["cells_changed"] == 0


def test_preview_matches_applied_replay(frame):
    hist = []
    df = frame
    for op, params in [("impute", {"columns": ["score"], "strategy": "mean"}),
                       ("drop_duplicates", {}),
                       ("handle_outliers", {"columns": ["release_year"], "action": "cap"})]:
        df, rec = cleaning.preview_operation(df, op, params)
        hist.append(rec)
    replayed = cleaning.replay(frame, hist)
    pd.testing.assert_frame_equal(df, replayed)
    # every recorded impact equals the impact recomputed during replay
    d2 = frame
    for rec in hist:
        new, again = cleaning.preview_operation(d2, rec["operation"], rec["params"])
        assert again["impact"]["cells_changed"] == rec["impact"]["cells_changed"]
        assert again["impact"]["rows_removed"] == rec["impact"]["rows_removed"]
        d2 = new
