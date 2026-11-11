import io

import numpy as np
import pandas as pd
import pytest

from core import data_loader as dl

SYNTHETIC_IDS = ["titanic", "churn", "bike", "student"]


def test_csv_upload_roundtrip():
    df, meta = dl.load_upload("people.csv", b"name,age\nAnn,31\nBob,\n")
    assert df.shape == (2, 2) and meta.source == "upload" and not meta.synthetic
    assert df["age"].isna().sum() == 1


def test_delimiter_and_encoding_detection():
    df, meta = dl.load_upload("eu.csv", "ville;prix\nOrléans;3\nNîmes;4\n".encode("cp1252"))
    assert list(df.columns) == ["ville", "prix"] and df.loc[0, "ville"] == "Orléans"
    assert any("delimiter" in n for n in meta.notes)


def test_excel_upload():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
    df, _ = dl.load_upload("book.xlsx", buf.getvalue())
    assert df.shape == (2, 2)


def test_duplicate_and_blank_headers_are_made_unique():
    df, _ = dl.load_upload("d.csv", b"a,a,\n1,2,3\n")
    assert len(set(df.columns)) == 3


@pytest.mark.parametrize("name,data", [("x.json", b"{}"), ("x.csv", b""), ("x.csv", b"a,b\n"), ("x.xlsx", b"not a workbook")])
def test_bad_files_raise_readable_errors(name, data):
    with pytest.raises(dl.DataLoadError):
        dl.load_upload(name, data)


def test_all_samples_load():
    for sid in dl.SAMPLES:
        df, meta = dl.load_sample(sid)
        assert len(df) > 0 and meta.suggested_target in df.columns


# Credibility test 4 — synthetic samples are labelled wherever they are represented
@pytest.mark.parametrize("sid", SYNTHETIC_IDS)
def test_synthetic_samples_are_labelled(sid):
    _, meta = dl.load_sample(sid)
    assert meta.synthetic and meta.label == dl.SYNTHETIC_LABEL == "Synthetic Dataset — For Demonstration Purposes"
    assert dl.SYNTHETIC_LABEL in dl.sample_display_name(sid)
    assert meta.to_dict()["label"] == dl.SYNTHETIC_LABEL


def test_only_the_four_demo_samples_are_synthetic_and_others_document_provenance():
    assert sorted(s for s, v in dl.SAMPLES.items() if v.synthetic) == sorted(SYNTHETIC_IDS)
    for sid, s in dl.SAMPLES.items():
        if not s.synthetic:
            assert "scikit-learn" in s.provenance and dl.SYNTHETIC_LABEL not in dl.sample_display_name(sid)


def test_sklearn_samples_match_their_documented_source():
    from sklearn import datasets

    for sid, loader in [("iris", datasets.load_iris), ("wine", datasets.load_wine), ("breast_cancer", datasets.load_breast_cancer)]:
        df, _ = dl.load_sample(sid)
        X = loader().data
        assert np.allclose(df.iloc[:, : X.shape[1]].to_numpy(float), X)
