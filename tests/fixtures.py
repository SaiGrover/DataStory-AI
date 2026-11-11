"""Synthetic test fixtures that mimic the *shape* of well-known public datasets.

The real Netflix titles dataset is not bundled (licensing). ``netflix_like``
generates random values in the same column layout so tests can exercise
identifier detection, multi-valued categories and month-name dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["United States", "India", "United Kingdom", "Japan", "South Korea", "Canada", "Spain", "France", "Mexico", "Egypt"]
GENRES = ["Dramas", "Comedies", "International Movies", "Documentaries", "Action & Adventure", "Kids' TV", "Thrillers", "Reality TV"]
RATINGS = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "PG", "TV-G"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def netflix_like(n: int = 8807, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_movie = rng.random(n) < 0.7
    years = np.clip(np.round(rng.normal(2015, 6, n)), 1942, 2021).astype(int)
    years[:217] = rng.integers(1942, 1980, 217)  # a tail of old titles

    def multi(pool, k_max):
        return [", ".join(rng.choice(pool, rng.integers(1, k_max + 1), replace=False)) for _ in range(n)]

    added = [f"{MONTHS[m]} {d}, {y}" for m, d, y in zip(rng.integers(0, 12, n), rng.integers(1, 29, n), rng.integers(2016, 2022, n))]
    df = pd.DataFrame({
        "show_id": [f"s{i}" for i in range(1, n + 1)],
        "type": np.where(is_movie, "Movie", "TV Show"),
        "title": [f"Title {i} {rng.integers(1e6)}" for i in range(n)],
        "director": [None if rng.random() < 0.3 else f"Director {rng.integers(0, 4000)}" for _ in range(n)],
        "cast": [None if rng.random() < 0.1 else ", ".join(f"Actor {a}" for a in rng.integers(0, 20000, 4)) for _ in range(n)],
        "country": multi(COUNTRIES, 3),
        "date_added": added,
        "release_year": years,
        "rating": rng.choice(RATINGS, n),
        "duration": [f"{rng.integers(60, 180)} min" if m else f"{rng.integers(1, 6)} Season{'s' if rng.random() < .5 else ''}" for m in is_movie],
        "listed_in": multi(GENRES, 3),
        "description": [f"A story number {i} about people, places and a long sentence of description text." for i in range(n)],
    })
    df.loc[rng.choice(n, 800, replace=False), "country"] = None
    return df


def employee_attrition(n: int = 600, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "R&D", "HR"], n, p=[.35, .55, .10])
    income = np.round(rng.lognormal(8.6, 0.4, n))
    years = rng.integers(0, 30, n)
    p = 1 / (1 + np.exp(-(-1.2 - 0.00012 * (income - 5000) - 0.08 * years + (dept == "Sales") * 0.6)))
    return pd.DataFrame({"EmployeeNumber": np.arange(1001, 1001 + n), "Age": rng.integers(18, 60, n), "Department": dept,
                         "MonthlyIncome": income, "YearsAtCompany": years,
                         "Attrition": np.where(rng.random(n) < p, "Yes", "No")})


def housing(n: int = 500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 400, n).clip(400)
    beds = rng.integers(1, 6, n)
    region = rng.choice(["North", "South", "East", "West"], n)
    price = 50000 + 120 * area + 8000 * beds + (region == "West") * 30000 + rng.normal(0, 20000, n)
    return pd.DataFrame({"id": np.arange(n), "area_sqft": area.round(), "bedrooms": beds, "region": region, "price": price.round(-2)})


def sales(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    return pd.DataFrame({"order_id": [f"ORD-{i:05d}" for i in range(n)], "order_date": dates.strftime("%Y-%m-%d"),
                         "region": rng.choice(["North", "South", "East", "West"], n),
                         "category": rng.choice(["Furniture", "Technology", "Office Supplies"], n),
                         "product": rng.choice([f"Product {i}" for i in range(40)], n),
                         "quantity": rng.integers(1, 10, n), "revenue": rng.gamma(2, 150, n).round(2)})
