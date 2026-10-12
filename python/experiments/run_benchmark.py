"""Benchmark of the original QRFSL as a point predictor, plus an ablation of its second stage.

Question 1: when does QRFSL beat a plain random forest (mean prediction)?
Question 2: what does the second stage use? Same stacking with fewer inputs.

Data (1,000 training rows, 2,000 test rows, 3 seeds): Friedman #1 with normal,
heavy-tailed (t with 2 df) and x-dependent noise; a log-normal target;
California housing; diabetes (n = 442); the synthetic heavy rain damage data
of the companion repository if present.

Outputs experiments/results/benchmark_rmse.csv and benchmark_ablation.csv.
Usage: python -m experiments.run_benchmark --seeds 3 --n-jobs 16
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qrfsl import QRFSL, ConvexWeights, RForest  # noqa: E402

warnings.filterwarnings("ignore")
HEAVY_RAIN = Path(__file__).resolve().parents[3] / "heavyrain" / "python" / "data" / "Dat_train" / "tdat.csv"


def friedman(n, rng, noise):
    X, f = make_friedman1(n_samples=n, n_features=10, noise=0.0, random_state=int(rng.integers(2**31 - 1)))
    return X, f + noise(rng, n, X)


SYNTHETIC = {
    "friedman, normal noise": lambda n, r: friedman(n, r, lambda r, n, X: r.normal(0, 1, n)),
    "friedman, heavy-tailed noise": lambda n, r: friedman(n, r, lambda r, n, X: r.standard_t(2, n)),
    "friedman, noise grows with x": lambda n, r: friedman(n, r, lambda r, n, X: r.normal(0, 1, n) * (0.2 + 4 * X[:, 0])),
    "skewed (lognormal) target": lambda n, r: (lambda X, f: (X, np.exp(f / 6 + r.normal(0, 0.8, n))))(
        *make_friedman1(n, 10, noise=0, random_state=int(r.integers(2**31 - 1)))),
}


def real(name, rng, n_train):
    if name == "california housing":
        d = fetch_california_housing(); X, y = d.data, d.target
    elif name == "diabetes":
        d = load_diabetes(); X, y = d.data, d.target
    else:
        t = pd.read_csv(HEAVY_RAIN, encoding="euc-kr")
        y = t["y"].to_numpy(float)
        X = t[[c for c in t.columns if c[0] in "dx" or c in ("tot", "region", "sido_code")]].to_numpy(float)
    idx = rng.permutation(len(y)); ntr = min(n_train, int(0.6 * len(y)))
    return X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:ntr + 2000]], y[idx[ntr:ntr + 2000]]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def summary_features(q, kind):
    lo, md, hi = q[:, 0], q[:, 10], q[:, -1]
    if kind == "mean only":
        return q.mean(1, keepdims=True)
    if kind == "median+spread+skew":
        return np.column_stack([md, hi - lo, (hi + lo) / 2 - md])
    return q


def run(Xtr, ytr, Xte, yte, seed, n_jobs):
    """Returns (benchmark RMSEs, ablation RMSEs)."""
    rf = RForest(500, 5, seed, n_jobs=n_jobs).fit(Xtr, ytr)
    m = QRFSL(k=10, seed=seed, full_seed=seed + 1, n_jobs=min(n_jobs, 10)).fit(Xtr, ytr)
    pred, q = m.predict(Xte, return_quantiles=True)
    Q, yq, Qte = m.cv_quantiles_.to_numpy(), m.cv_y_.to_numpy(), q.to_numpy()
    bench = {"random forest (mean)": rf.predict(Xte), "QRF median": Qte[:, 10],
             "QRF mean of 21 quantiles": Qte.mean(1), "QRFSL (rf)": pred,
             "QRFSL (weights)": ConvexWeights().fit(Q, yq).predict(Qte),
             "GBM (HistGradientBoosting)": HistGradientBoostingRegressor(random_state=seed).fit(Xtr, ytr).predict(Xte)}
    if (ytr > 0).all():
        bench["random forest on log y, exp back"] = np.exp(RForest(500, 5, seed, n_jobs=n_jobs).fit(Xtr, np.log(ytr)).predict(Xte))
    abl = {"random forest (mean)": bench["random forest (mean)"]}
    for kind in ["mean only", "median+spread+skew", "all 21 quantiles (= QRFSL)"]:
        sl = RForest(1000, 5, seed, n_jobs=n_jobs).fit(summary_features(Q, kind), yq)
        abl[f"2nd stage on {kind}"] = sl.predict(summary_features(Qte, kind))
    oof = np.empty(len(ytr))
    for i, te in enumerate(m.folds_):
        tr = np.setdiff1d(np.arange(len(ytr)), te)
        oof[te] = RForest(500, 5, i + 1, n_jobs=n_jobs).fit(Xtr[tr], ytr[tr]).predict(Xtr[te])
    order = np.concatenate(m.folds_)
    sl = RForest(1000, 5, seed, n_jobs=n_jobs).fit(oof[order][:, None], ytr[order])
    abl["2nd stage on OOF RF mean (no quantiles)"] = sl.predict(rf.predict(Xte)[:, None])
    return {k: rmse(v, yte) for k, v in bench.items()}, {k: rmse(v, yte) for k, v in abl.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=16)
    a = p.parse_args()
    out = Path(__file__).with_name("results")
    bench_rows, abl_rows = [], []
    reals = ["california housing", "diabetes"] + (["heavy rain synthetic damage"] if HEAVY_RAIN.exists() else [])
    for seed in range(1, a.seeds + 1):
        r = np.random.default_rng(seed)
        cases = [(n, *(lambda X, y: (X[:1000], y[:1000], X[1000:], y[1000:]))(*g(3000, r))) for n, g in SYNTHETIC.items()]
        cases += [(n, *real(n, r, 1000)) for n in reals]
        for name, Xtr, ytr, Xte, yte in cases:
            b, ab = run(Xtr, ytr, Xte, yte, seed, a.n_jobs)
            bench_rows += [dict(data=name, seed=seed, model=k, rmse=v) for k, v in b.items()]
            abl_rows += [dict(data=name, seed=seed, model=k, rmse=v) for k, v in ab.items()]
        print(f"seed {seed} done", flush=True)
    for rows, fname in [(bench_rows, "benchmark_rmse.csv"), (abl_rows, "benchmark_ablation.csv")]:
        d = pd.DataFrame(rows); d.to_csv(out / fname, index=False)
        base = d[d.model == "random forest (mean)"].set_index(["data", "seed"])["rmse"].rename("base")
        d = d.join(base, on=["data", "seed"]); d["ratio"] = d.rmse / d.base
        pd.set_option("display.width", 250)
        print(f"\n{fname}: RMSE relative to the plain random forest (mean over seeds)")
        print(d.pivot_table(index="model", columns="data", values="ratio").round(3).to_string())


if __name__ == "__main__":
    main()
