"""Train a QRFSL model on a CSV file and save it (R: ``QRFSL()``).

Example (the mtcars example of Rcode_QRFSL.R):
    python train.py --data data/mtcars_train.csv --target mpg --drop car \
        --k 10 --seed 1 --full-seed 2019 --model model/qrfsl_mtcars.joblib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from qrfsl import DEFAULT_PROBS, QRFSL


def parse_args():
    p = argparse.ArgumentParser(description="Train a QRFSL model and save it with joblib.")
    p.add_argument("--data", required=True, help="training CSV file")
    p.add_argument("--target", required=True, help="name of the target column (y)")
    p.add_argument("--drop", nargs="*", default=[],
                   help="columns that are not features, e.g. an id column")
    p.add_argument("--model", required=True, help="output path of the model (.joblib)")
    p.add_argument("--k", type=int, default=10, help="number of CV folds (R k, default 10)")
    p.add_argument("--probs", type=float, nargs="+", default=list(DEFAULT_PROBS),
                   help="quantile levels (R prob, default 0.025, 0.05, ..., 0.95, 0.975)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the fold split and the super learner (R seed of QRFSL)")
    p.add_argument("--full-seed", type=int, default=0,
                   help="seed for the QRF on all training rows (R seed of predict.QRFSL)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="folds fitted in parallel (R mc, default 1)")
    p.add_argument("--qrf-trees", type=int, default=500, help="trees per QRF (default 500)")
    p.add_argument("--sl", choices=["rf", "weights"], default="rf",
                   help="super learner: rf = random forest as in the R code (default), "
                        "weights = weighted average of the quantiles (R sl = 0)")
    p.add_argument("--sl-trees", type=int, default=1000,
                   help="trees of the random forest super learner (default 1000)")
    p.add_argument("--nodesize", type=int, default=5,
                   help="R randomForest nodesize (default 5)")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target, *args.drop])

    print(f"data     : {args.data} ({len(df)} rows)")
    print(f"target   : {args.target}")
    print(f"features : {len(X.columns)} -> {', '.join(X.columns)}")
    print(f"settings : k={args.k}, quantiles={len(args.probs)}, sl={args.sl}, "
          f"seed={args.seed}, full_seed={args.full_seed}, n_jobs={args.n_jobs}")

    start = time.time()
    model = QRFSL(
        k=args.k,
        probs=args.probs,
        seed=args.seed,
        full_seed=args.full_seed,
        n_jobs=args.n_jobs,
        qrf_trees=args.qrf_trees,
        sl=args.sl,
        sl_trees=args.sl_trees,
        nodesize=args.nodesize,
    ).fit(X, y)

    if args.sl == "weights":
        w = pd.Series(model.sl_.coef_, index=model.quantile_names_)
        print("weights  : " + ", ".join(f"{q}: {v:.3f}" for q, v in w[w > 0].items()))

    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"saved    : {out} ({time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
