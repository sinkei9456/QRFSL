"""Data sets of the interval and decision experiments.

Every loader returns a dict with X_train, y_train, X_cal, y_cal, X_test,
y_test (numpy arrays) and, for the heteroscedastic synthetic data, the true
noise scale of the test rows (``sigma_test``) to check conditional coverage.

Synthetic
  friedman_hetero : Friedman #1 signal, normal noise whose sd grows with x0:
                    sd = 0.2 + 4 * x0 (0.2 to 4.2)
  lognormal       : y = exp(f / 6 + 0.8 * N(0, 1)), a right-skewed target
Public
  california      : California housing (sklearn), target in 100k USD
  bike            : Bike sharing hourly demand (OpenML 'Bike_Sharing_Demand' v2),
                    target = rentals per hour
  claims          : French motor claims (OpenML freMTPL2freq + freMTPL2sev):
                    total claim amount per policy with at least one claim,
                    capped at 50,000 EUR (0.6% of policies), features from the
                    policy table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, make_friedman1

NAMES = ["friedman_hetero", "lognormal", "california", "bike", "claims"]
_CACHE = {}


def _friedman(n, rng):
    X, f = make_friedman1(n_samples=n, n_features=10, noise=0.0, random_state=int(rng.integers(2**31 - 1)))
    return X, f


def _split(X, y, rng, n_train, n_cal, n_test, extra=None):
    n = len(y)
    idx = rng.permutation(n)
    if n_train + n_cal + n_test > n:                    # smaller data: keep the proportions
        f = n / (n_train + n_cal + n_test)
        n_train, n_cal = int(n_train * f), int(n_cal * f)
        n_test = n - n_train - n_cal
    a, b = idx[:n_train], idx[n_train:n_train + n_cal]
    c = idx[n_train + n_cal:n_train + n_cal + n_test]
    out = dict(X_train=X[a], y_train=y[a], X_cal=X[b], y_cal=y[b], X_test=X[c], y_test=y[c])
    if extra is not None:
        out["sigma_test"] = extra[c]
    return out


def _real(name):
    if name in _CACHE:
        return _CACHE[name]
    if name == "california":
        d = fetch_california_housing()
        X, y = d.data.astype(float), d.target.astype(float)
    elif name == "bike":
        df = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True, parser="auto").frame
        y = df.pop("count").to_numpy(dtype=float)
        for c in df.columns:
            if str(df[c].dtype) in ("category", "object", "str"):
                df[c] = df[c].astype("category").cat.codes
        X = df.to_numpy(dtype=float)
    elif name == "claims":
        freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
        sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True, parser="auto").frame
        amount = sev.groupby("IDpol")["ClaimAmount"].sum()
        df = freq.set_index("IDpol").join(amount, how="inner")
        y = np.minimum(df.pop("ClaimAmount").to_numpy(dtype=float), 50_000.0)
        df = df.drop(columns=["ClaimNb"])
        for c in df.columns:
            if str(df[c].dtype) in ("category", "object", "str"):
                df[c] = df[c].astype("category").cat.codes
        X = df.to_numpy(dtype=float)
    else:
        raise ValueError(name)
    _CACHE[name] = (X, y)
    return X, y


def load(name, seed, n_train=2000, n_cal=1000, n_test=3000):
    rng = np.random.default_rng(seed)
    if name == "friedman_hetero":
        X, f = _friedman(n_train + n_cal + n_test, rng)
        sigma = 0.2 + 4 * X[:, 0]
        return _split(X, f + sigma * rng.normal(size=len(f)), rng, n_train, n_cal, n_test, extra=sigma)
    if name == "lognormal":
        X, f = _friedman(n_train + n_cal + n_test, rng)
        return _split(X, np.exp(f / 6 + 0.8 * rng.normal(size=len(f))), rng, n_train, n_cal, n_test)
    X, y = _real(name)
    return _split(X, y, rng, n_train, n_cal, n_test)
