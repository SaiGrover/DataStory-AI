import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def clf_df():
    """Binary target driven by `signal`; `noise` is irrelevant; some missing values."""
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    df = pd.DataFrame({
        "signal": signal,
        "noise": rng.normal(size=n),
        "group": rng.choice(["a", "b", "c"], size=n),
        "label": np.where(signal + rng.normal(scale=0.3, size=n) > 0.8, "yes", "no"),
    })
    df.loc[rng.choice(n, 30, replace=False), "noise"] = np.nan
    return df


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(1)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "junk": rng.normal(size=n), "y": 3 * x1 - 2 * x2 + rng.normal(scale=0.2, size=n)})
