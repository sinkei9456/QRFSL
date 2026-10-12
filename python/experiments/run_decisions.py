"""Decision experiment: which way from QRF output to a decision has the lowest cost?

Costs (per data set, penalty = 0.5 * sd of the training target)
  linear 1:1 (tau 0.5), linear 3:1 (tau 0.75), linear 9:1 (tau 0.9),
  linear 1:1 + penalty if y > d
Methods
  RF mean                       : ignores the cost
  QRF tau-quantile raw          : the natural answer for a linear cost          [linear]
  QRF tau-quantile + conf shift : one-sided conformal shift from out-of-fold rows [linear]
  LightGBM direct tau-quantile  : boosting with pinball loss on x               [linear]
  QRF plug-in (all atoms)       : expected-cost minimiser under the full QRF distribution
  QRF plug-in (21 quantiles)    : the same with the 21 quantiles as equal atoms
  LightGBM plug-in (21 quant.)  : 21 LightGBM quantile models, then expected cost
  QRFSL pinball stage           : boosting with pinball loss on the 21 out-of-fold quantiles [linear]
  QRFSL cost-weight stage       : convex weights on the 21 quantiles minimising the cost
  QRF median + shift            : baseline, one constant shift tuned out-of-fold

Outputs experiments/results/decisions.csv. Usage: python -m experiments.run_decisions --seeds 10 --n-jobs 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformal import conformal_shift  # noqa: E402
from decision import (CostWeightStage, LinearCost, PenaltyCost, PinballStage, ShiftStage,  # noqa: E402
                      expected_cost_decision, mean_cost, quantiles_as_distribution)
from experiments.datasets import NAMES, load  # noqa: E402
from experiments.run_intervals import lgb_quantile  # noqa: E402
from qrfsl import QRFSL  # noqa: E402


def costs_for(sd):
    return {"linear 1:1": LinearCost(1, 1), "linear 3:1": LinearCost(3, 1), "linear 9:1": LinearCost(9, 1),
            "linear 1:1 + penalty": PenaltyCost(1, 1, penalty=0.5 * sd)}


def run_one(name, seed, n_jobs):
    d = load(name, seed)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    sd = float(np.std(ytr))
    m = QRFSL(k=10, seed=seed, full_seed=seed + 1000, n_jobs=n_jobs, sl="none", leaf_sample="all").fit(Xtr, ytr)
    probs = np.array(m.probs)
    q_te = m.predict_quantiles(Xte).to_numpy()
    q_oof, y_oof = m.cv_quantiles_.to_numpy(), m.cv_y_.to_numpy()
    atoms, weights = m.predict_distribution(Xte)
    rf_te = m.qmod_.forest_.predict(Xte)
    i50 = int(np.argmin(np.abs(probs - 0.5)))
    g21 = np.column_stack([lgb_quantile(Xtr, ytr, Xte, p, seed) for p in probs])   # 21 LightGBM quantile models
    g21_atoms, g21_w = quantiles_as_distribution(g21)
    q21_atoms, q21_w = quantiles_as_distribution(q_te)

    rows = []
    for cname, cost in costs_for(sd).items():
        dec = {"RF mean": rf_te}
        if isinstance(cost, PenaltyCost) is False:
            it = int(np.argmin(np.abs(probs - cost.tau)))
            dec["QRF tau-quantile raw"] = q_te[:, it]
            dec["QRF tau-quantile + conf shift"] = q_te[:, it] + conformal_shift(q_oof[:, it], y_oof, cost.tau)
            dec["LightGBM direct tau-quantile"] = g21[:, it]
            dec["QRFSL pinball stage"] = PinballStage(cost.tau, random_state=seed).fit(q_oof, y_oof).predict(q_te)
        dec["QRF plug-in (all atoms)"] = expected_cost_decision(atoms, weights, cost)
        dec["QRF plug-in (21 quantiles)"] = expected_cost_decision(q21_atoms, q21_w, cost)
        dec["LightGBM plug-in (21 quant.)"] = expected_cost_decision(g21_atoms, g21_w, cost)
        dec["QRFSL cost-weight stage"] = CostWeightStage(cost, start_tau=cost.tau).fit(q_oof, y_oof, probs).predict(q_te)
        dec["QRF median + shift"] = ShiftStage(cost).fit(q_oof[:, i50], y_oof).predict(q_te[:, i50])
        for meth, dd in dec.items():
            rows.append(dict(data=name, seed=seed, cost=cname, method=meth,
                             mean_cost=mean_cost(dd, yte, cost), mean_cost_rel_sd=mean_cost(dd, yte, cost) / sd,
                             under_rate=float(np.mean(yte > dd))))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--data", nargs="*", default=NAMES)
    p.add_argument("--out", default=str(Path(__file__).with_name("results") / "decisions.csv"))
    a = p.parse_args()
    rows, t0 = [], time.time()
    for name in a.data:
        for seed in range(1, a.seeds + 1):
            rows += run_one(name, seed, a.n_jobs)
            print(f"{name} seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    ref = df[df.method == "QRF plug-in (all atoms)"].set_index(["data", "seed", "cost"])["mean_cost"].rename("ref")
    df = df.join(ref, on=["data", "seed", "cost"])
    df["ratio"] = df.mean_cost / df.ref
    pd.set_option("display.width", 250)
    for cname in df.cost.unique():
        print(f"\nmean cost relative to 'QRF plug-in (all atoms)', cost = {cname} (mean over seeds)")
        print(df[df.cost == cname].pivot_table(index="method", columns="data", values="ratio").round(3).to_string())


if __name__ == "__main__":
    main()
