"""Interval experiment: does QRF + CQR give valid, adaptive 90% intervals?

Methods (nominal coverage 1 - alpha = 90%)
  QRF raw                 : QRF 5% and 95% quantiles as they are
  QRF + CQR split         : shift from a held-out calibration set
  QRF + CQR split asym    : separate lower / upper shifts
  QRF + CQR cv            : shift from the out-of-fold quantiles (no calibration set)
  RF residual conformal   : RF point +- Q, Q from |residuals| on the calibration set
  LightGBM raw            : two LightGBM quantile models (5%, 95%) as they are
  LightGBM + CQR split    : the same, conformalized on the calibration set

Outputs experiments/results/intervals.csv (one row per data set, seed, method)
and prints a summary. Usage: python -m experiments.run_intervals --seeds 10 --n-jobs 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformal import CQR, ResidualConformal, coverage, coverage_by_bins, mean_width  # noqa: E402
from experiments.datasets import NAMES, load  # noqa: E402
from qrfsl import QRFSL  # noqa: E402

ALPHA = 0.1
LO, HI = 0.05, 0.95


def lgb_quantile(Xtr, ytr, X, tau, seed):
    import lightgbm as lgb
    cols = [f"f{j}" for j in range(np.asarray(Xtr).shape[1])]
    m = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=400, learning_rate=0.05,
                          num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    return m.fit(pd.DataFrame(Xtr, columns=cols), ytr).predict(pd.DataFrame(X, columns=cols))


def run_one(name, seed, n_jobs):
    d = load(name, seed)
    Xtr, ytr, Xc, yc, Xte, yte = (d[k] for k in ("X_train", "y_train", "X_cal", "y_cal", "X_test", "y_test"))
    sd = float(np.std(ytr))
    m = QRFSL(k=10, seed=seed, full_seed=seed + 1000, n_jobs=n_jobs, sl="none", leaf_sample="all").fit(Xtr, ytr)
    probs = list(m.probs)
    ilo, ihi = probs.index(LO), probs.index(HI)
    q_te, q_cal = m.predict_quantiles(Xte).to_numpy(), m.predict_quantiles(Xc).to_numpy()
    q_oof, y_oof = m.cv_quantiles_.to_numpy(), m.cv_y_.to_numpy()
    rf_te, rf_cal = m.qmod_.forest_.predict(Xte), m.qmod_.forest_.predict(Xc)

    intervals = {"QRF raw": (q_te[:, ilo], q_te[:, ihi])}
    intervals["QRF + CQR split"] = CQR(ALPHA).fit(q_cal[:, ilo], q_cal[:, ihi], yc).predict(q_te[:, ilo], q_te[:, ihi])
    intervals["QRF + CQR split asym"] = CQR(ALPHA, asymmetric=True).fit(q_cal[:, ilo], q_cal[:, ihi], yc).predict(q_te[:, ilo], q_te[:, ihi])
    intervals["QRF + CQR cv"] = CQR(ALPHA).fit(q_oof[:, ilo], q_oof[:, ihi], y_oof).predict(q_te[:, ilo], q_te[:, ihi])
    intervals["RF residual conformal"] = ResidualConformal(ALPHA).fit(rf_cal, yc).predict(rf_te)
    g_lo = lgb_quantile(Xtr, ytr, np.vstack([Xc, Xte]), LO, seed)
    g_hi = lgb_quantile(Xtr, ytr, np.vstack([Xc, Xte]), HI, seed)
    nc = len(yc)
    intervals["LightGBM raw"] = (g_lo[nc:], g_hi[nc:])
    intervals["LightGBM + CQR split"] = CQR(ALPHA).fit(g_lo[:nc], g_hi[:nc], yc).predict(g_lo[nc:], g_hi[nc:])

    rows = []
    for meth, (lo, hi) in intervals.items():
        width = hi - lo
        if np.ptp(width) > 1e-8 * sd:                               # coverage in width quintiles
            by_w = coverage_by_bins(lo, hi, yte, width)
        else:                                                       # constant width: fifths are meaningless
            by_w = np.full(5, np.nan)
        row = dict(data=name, seed=seed, method=meth, coverage=coverage(lo, hi, yte),
                   width=mean_width(lo, hi), width_rel=mean_width(lo, hi) / sd,
                   cov_narrowest=by_w[0], cov_widest=by_w[-1],
                   width_sd_rel=float(np.std(width)) / sd)
        if "sigma_test" in d:                                        # coverage in noise-scale quintiles
            by_s = coverage_by_bins(lo, hi, yte, d["sigma_test"])
            row.update(cov_low_noise=by_s[0], cov_high_noise=by_s[-1])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--data", nargs="*", default=NAMES)
    p.add_argument("--out", default=str(Path(__file__).with_name("results") / "intervals.csv"))
    a = p.parse_args()
    rows, t0 = [], time.time()
    for name in a.data:
        for seed in range(1, a.seeds + 1):
            rows += run_one(name, seed, a.n_jobs)
            print(f"{name} seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nmean over seeds: coverage (nominal 0.90)")
    print(df.pivot_table(index="method", columns="data", values="coverage").round(3).to_string())
    print("\nmean over seeds: width / sd(y_train)")
    print(df.pivot_table(index="method", columns="data", values="width_rel").round(3).to_string())
    print("\nmean over seeds: coverage in the narrowest / widest fifth of the intervals")
    print(df.pivot_table(index="method", columns="data", values=["cov_narrowest", "cov_widest"]).round(3).to_string())
    if "cov_low_noise" in df:
        sub = df.dropna(subset=["cov_low_noise"])
        print("\nfriedman_hetero: coverage in the lowest / highest fifth of the true noise scale")
        print(sub.groupby("method")[["cov_low_noise", "cov_high_noise"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
