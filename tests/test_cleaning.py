import numpy as np
import pandas as pd
import pytest

from core import cleaning


def base():
    return pd.DataFrame({"n": [1.0, None, 3.0, 100.0, 3.0], "c": [" A", "a", None, "b ", "a"],
                         "d": ["2024-01-01", "bad", "2024-03-01", None, "2024-03-01"]})


def test_median_imputation_is_correct_and_does_not_mutate_input():
    df = base()
    before = df.copy()
    out, rec = cleaning.preview_operation(df, "impute", {"columns": ["n"], "strategy": "median"})
    assert out["n"].tolist() == [1.0, 3.0, 3.0, 100.0, 3.0]
    pd.testing.assert_frame_equal(df, before)
    assert rec["before"]["missing_cells"] - rec["after"]["missing_cells"] == 1 and rec["cells_changed"] == 1


def test_mean_mode_constant_and_knn():
    df = base()
    assert cleaning.impute(df, ["n"], "mean")[0]["n"][1] == pytest.approx(np.mean([1, 3, 100, 3]))
    assert cleaning.impute(df, ["c"], "mode")[0]["c"][2] == "a"
    assert cleaning.impute(df, ["c"], "constant", value="?")[0]["c"][2] == "?"
    num = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, None, 4.0]})
    assert cleaning.impute(num, ["y"], "knn", n_neighbors=2)[0]["y"].notna().all()


def test_mean_on_text_column_is_rejected():
    with pytest.raises(cleaning.CleaningError):
        cleaning.preview_operation(base(), "impute", {"columns": ["c"], "strategy": "mean"})


def test_duplicates_dates_types_and_categories():
    df = base()
    out, _ = cleaning.preview_operation(df, "normalize_categories", {"columns": ["c"], "case": "lower", "strip": True})
    assert sorted(out["c"].dropna().unique()) == ["a", "b"] and out["c"].isna().sum() == 1
    out, rec = cleaning.preview_operation(df, "parse_dates", {"columns": ["d"]})
    assert str(out["d"].dtype).startswith("datetime64") and out["d"].isna().sum() == 2 and "unparseable" in rec["detail"]
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out, rec = cleaning.preview_operation(dup, "drop_duplicates", {})
    assert len(out) == 5 and rec["after"]["duplicate_rows"] == 0
    out, _ = cleaning.preview_operation(pd.DataFrame({"p": ["1,200", "$30", "x"]}), "convert_type", {"columns": ["p"], "to": "numeric"})
    assert out["p"].tolist()[:2] == [1200.0, 30.0] and np.isnan(out["p"][2])


def test_outlier_capping_and_removal():
    df = pd.DataFrame({"v": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 500]})
    capped, _ = cleaning.preview_operation(df, "handle_outliers", {"columns": ["v"], "action": "cap"})
    assert capped["v"].max() < 500 and len(capped) == 10
    removed, _ = cleaning.preview_operation(df, "handle_outliers", {"columns": ["v"], "action": "remove"})
    assert len(removed) == 9


def test_encode_scale_rename_drop():
    df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1.0, 2.0, 3.0]})
    assert list(cleaning.encode(df, ["k"])[0].columns) == ["k_a", "k_b", "v"]
    assert cleaning.scale(df, ["v"], "minmax")[0]["v"].tolist() == [0.0, 0.5, 1.0]
    assert list(cleaning.rename_columns(df, {"v": "value"})[0].columns) == ["k", "value"]
    with pytest.raises(cleaning.CleaningError):
        cleaning.drop_columns(df, ["k", "v"])
    with pytest.raises(cleaning.CleaningError):
        cleaning.scale(pd.DataFrame({"v": [1.0, 1.0]}), ["v"])


def test_operation_that_would_empty_the_dataset_is_refused():
    with pytest.raises(cleaning.CleaningError):
        cleaning.preview_operation(pd.DataFrame({"a": [None, None]}), "drop_missing_rows", {})


def test_history_replay_reproduces_the_working_dataset():
    df = base()
    hist, cur = [], df
    for op, params in [("impute", {"columns": ["n"], "strategy": "median"}), ("drop_duplicates", {}),
                       ("handle_outliers", {"columns": ["n"], "action": "cap"})]:
        cur, rec = cleaning.preview_operation(cur, op, params)
        hist.append(rec)
    pd.testing.assert_frame_equal(cleaning.replay(df, hist), cur)
    pd.testing.assert_frame_equal(cleaning.replay(df, []), df)  # reset
