"""Predict with a saved QRFSL model (R: ``predict.QRFSL()``).

Example (the mtcars example of Rcode_QRFSL.R):
    python predict.py --model model/qrfsl_mtcars.joblib --data data/mtcars_test.csv \
        --out output/pred_mtcars.csv --keep car --target mpg --quantiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Predict with a saved QRFSL model.")
    p.add_argument("--model", required=True, help="model saved by train.py (.joblib)")
    p.add_argument("--data", required=True,
                   help="CSV with the same feature columns as the training data")
    p.add_argument("--out", required=True, help="output CSV file")
    p.add_argument("--keep", nargs="*", default=[],
                   help="input columns copied to the output, e.g. an id column")
    p.add_argument("--target", default=None,
                   help="if the CSV has the true y, print RMSE and squared correlation")
    p.add_argument("--quantiles", action="store_true",
                   help="also write the quantiles predicted by the full-data QRF")
    return p.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)  # needs qrfsl.py next to this script
    df = pd.read_csv(args.data)

    pred, qpred = model.predict(df, return_quantiles=True)

    out = df[args.keep].copy()
    if args.target:
        out[args.target] = df[args.target]
    out["pred"] = pred
    if args.quantiles:
        out = pd.concat([out, qpred], axis=1)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"predicted {len(out)} rows -> {path}")

    if args.target:
        yact = df[args.target].to_numpy(dtype=float)
        rmse = np.sqrt(np.mean((yact - pred) ** 2))
        r2 = np.corrcoef(yact, pred)[0, 1] ** 2
        print(f"RMSE = {rmse:.4f}")
        print(f"squared correlation (corr in the R example) = {r2:.3f}")


if __name__ == "__main__":
    main()
