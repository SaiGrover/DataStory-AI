"""Target suitability and pre-training readiness."""
import uuid

import numpy as np
import pandas as pd
import pytest

from core import data_loader as dl
from core import ml_readiness as mr
from core import targets as tg


def netflix_like(n=500, seed=0):
    rng = np.random.default_rng(seed)
    months = ["January", "March", "June", "September", "December"]
    return pd.DataFrame({
        "show_id": [f"s{i + 1}" for i in range(n)],
        "type": rng.choice(["Movie", "TV Show"], n, p=[0.7, 0.3]),
        "title": [f"Title number {i} of the catalogue" for i in range(n)],
        "director": [f"Director {i % 300}" for i in range(n)],
        "cast": [", ".join(f"Actor {j}" for j in rng.integers(0, 2000, 4)) for _ in range(n)],
        "country": rng.choice(["United States", "India", "United Kingdom", "Japan", "Spain"], n),
        "date_added": [f"{rng.choice(months)} {rng.integers(1, 28)}, {rng.integers(2015, 2022)}" for _ in range(n)],
        "release_year": rng.integers(1990, 2022, n),
        "rating": rng.choice(["TV-MA", "TV-14", "PG-13", "R", "TV-PG"], n),
        "duration": [f"{rng.integers(60, 180)} min" for _ in range(n)],
        "listed_in": rng.choice(["Dramas", "Comedies", "Documentaries", "International TV Shows"], n),
        "description": [f"A long description of show {i} with a plot about something happening somewhere far away." for i in range(n)],
    })


@pytest.mark.parametrize("name,values", [
    ("show_id", [f"s{i}" for i in range(1, 201)]),
    ("customer_id", list(range(1000, 1200))),
    ("record", [str(uuid.UUID(int=i)) for i in range(200)]),
    ("row", list(range(200))),
    ("CustomerID", [f"CUST-{i:05d}" for i in range(200)]),
])
def test_identifier_detection(name, values):
    s = pd.Series(values)
    assert tg.is_identifier(name, s, len(s))[0]


def test_non_identifiers_not_flagged():
    rng = np.random.default_rng(1)
    assert not tg.is_identifier("cnt", pd.Series(rng.integers(0, 10_000, 300)), 300)[0]
    assert not tg.is_identifier("Churn", pd.Series(rng.choice(["Yes", "No"], 300)), 300)[0]
    assert not tg.is_identifier("price", pd.Series(rng.normal(100, 10, 300)), 300)[0]


def test_netflix_like_has_no_identifier_target():
    df = netflix_like()
    r = tg.assess_targets(df)
    by = {c["column"]: c for c in r["candidates"]}
    for col in ("show_id", "title", "description"):
        assert by[col]["verdict"] == "blocked", col
    assert by["type"]["verdict"] in ("possible", "discouraged")
    assert r["best"] is None  # nothing is *recommended*
    assert "clustering" in r["recommendation"].lower()
    v = tg.validate_target(df, "show_id", "classification")
    assert not v["ok"] and not v["overridable"]


@pytest.mark.parametrize("sample,target,task", [
    ("churn", "Churn", "classification"), ("bike", "cnt", "regression"), ("student", "G3", "regression"),
    ("iris", "species", "classification"), ("titanic", "Survived", "classification"),
])
def test_samples_recommend_their_outcome(sample, target, task):
    df, _ = dl.load_sample(sample)
    r = tg.assess_targets(df)
    assert r["best"] == target
    c = next(c for c in r["candidates"] if c["column"] == target)
    assert c["verdict"] == "recommended" and c["task"] == task
    ids = [c for c in r["candidates"] if c["column"] in ("CustomerID", "PassengerId")]
    assert all(c["verdict"] == "blocked" for c in ids)


def test_housing_and_attrition_frames():
    rng = np.random.default_rng(3)
    n = 300
    house = pd.DataFrame({"sqft": rng.integers(500, 4000, n), "bedrooms": rng.integers(1, 6, n),
                          "price": rng.normal(300_000, 50_000, n)})
    r = tg.assess_targets(house)
    assert r["best"] == "price" and r["candidates"][0]["task"] == "regression"
    emp = pd.DataFrame({"EmployeeNumber": range(1, n + 1), "Age": rng.integers(20, 60, n),
                        "Department": rng.choice(["Sales", "R&D", "HR"], n), "Attrition": rng.choice(["Yes", "No"], n, p=[0.2, 0.8])})
    r = tg.assess_targets(emp)
    assert r["best"] == "Attrition"
    assert next(c for c in r["candidates"] if c["column"] == "EmployeeNumber")["verdict"] == "blocked"


def test_no_target_dataset():
    df = pd.DataFrame({"id": range(100), "comment": [f"free text comment number {i} written by a person about stuff" for i in range(100)]})
    r = tg.assess_targets(df)
    assert not r["supervised_viable"] and r["best"] is None


def test_many_classes_blocked_and_high_cardinality_discouraged():
    rng = np.random.default_rng(0)
    many = pd.DataFrame({"x": rng.normal(size=600), "label": [f"c{i % 60}" for i in range(600)]})
    v = tg.validate_target(many, "label", "classification")
    assert not v["ok"] and any("exceeds" in b for b in v["blocking"])
    mid = pd.DataFrame({"x": rng.normal(size=600), "label": [f"c{i % 25}" for i in range(600)]})
    v = tg.validate_target(mid, "label", "classification")
    assert v["ok"] and v["overridable"] and v["discouraged"]


def test_readiness_flags_leakage_and_blocks_invalid_target():
    df, _ = dl.load_sample("churn")
    df = df.assign(churn_copy=df["Churn"])
    r = mr.assess_readiness(df, "Churn", [c for c in df.columns if c != "Churn"], "classification")
    assert any(f["title"] == "Possible target leakage" and "churn_copy" in f["detail"] for f in r["findings"])
    assert any(e["column"] == "CustomerID" for e in r["recommendations"]["exclude"])
    assert r["status"] in ("caution", "ready")
    nf = netflix_like()
    r = mr.assess_readiness(nf, "show_id", ["type", "country"], "classification")
    assert r["status"] == "blocked"


def test_readiness_numeric_leak_and_recommendations():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    df = pd.DataFrame({"a": x, "b": rng.normal(size=400), "leak": x * 2 + 0.001 * rng.normal(size=400), "price": x * 3 + 1})
    r = mr.assess_readiness(df, "price", ["a", "b", "leak"], "regression")
    assert {e["column"] for e in r["recommendations"]["exclude"]} >= {"a", "leak"}
    assert r["recommendations"]["metric"] == "rmse"


def test_imbalanced_binary_recommends_f1():
    df, _ = dl.load_sample("churn")
    r = mr.assess_readiness(df, "Churn", ["Tenure", "MonthlyCharges", "Contract"], "classification")
    counts = df["Churn"].value_counts()
    if counts.max() / counts.min() >= mr.IMBALANCE_RATIO:
        assert r["recommendations"]["metric"] == "f1"
    else:
        assert r["recommendations"]["metric"] in ("accuracy", "f1")
    assert all(f["severity"] in ("blocker", "warning", "info") for f in r["findings"])
