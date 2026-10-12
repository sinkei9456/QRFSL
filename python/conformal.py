"""Conformal calibration of QRF quantiles: CQR intervals and one-sided quantile shifts.

CQR (Romano, Patterson and Candes, 2019): fit lower and upper quantile models
on training data, measure on calibration data how far each y falls outside
[q_lo(x), q_hi(x)], take the (1 - alpha)(1 + 1/n) quantile Q of those
distances, and widen every interval by Q. The new interval covers a new y
with probability at least 1 - alpha whatever the quantile model is, as long
as calibration and new rows are exchangeable.

Two ways to get calibration scores from a fitted ``QRFSL``:
  split : a held-out calibration set (exact finite-sample guarantee)
  cv    : the out-of-fold quantiles of the training rows (no extra data).
          No formal guarantee: the scores come from the fold models but the
          shift is applied to the QRF refitted on all rows, so calibration
          and new scores are not exchangeable. CV+ (Barber et al., 2021;
          Gupta et al., 2022 for quantile scores) would restore a 1 - 2 alpha
          guarantee by applying the fold models
          to the new row. In the experiments the coverage was close to nominal.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores, alpha) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score (= the (1-alpha)(1+1/n) empirical quantile)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


class CQR:
    """Conformalized quantile regression interval.

    symmetric (default): one shift Q for both ends, from scores
        max(q_lo - y, y - q_hi) at level alpha.
    asymmetric: separate shifts for the lower and upper end, each at level
        alpha / 2 (useful when the errors are skewed).
    """

    def __init__(self, alpha=0.1, asymmetric=False):
        self.alpha = alpha
        self.asymmetric = asymmetric

    def fit(self, q_lo_cal, q_hi_cal, y_cal):
        q_lo_cal, q_hi_cal, y_cal = (np.asarray(a, dtype=float) for a in (q_lo_cal, q_hi_cal, y_cal))
        if self.asymmetric:
            self.shift_lo_ = conformal_quantile(q_lo_cal - y_cal, self.alpha / 2)
            self.shift_hi_ = conformal_quantile(y_cal - q_hi_cal, self.alpha / 2)
        else:
            q = conformal_quantile(np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal), self.alpha)
            self.shift_lo_ = self.shift_hi_ = q
        return self

    def predict(self, q_lo, q_hi):
        return np.asarray(q_lo, dtype=float) - self.shift_lo_, np.asarray(q_hi, dtype=float) + self.shift_hi_


def conformal_shift(q_tau_cal, y_cal, tau) -> float:
    """One-sided calibration of a tau-quantile: shift s such that P(y <= q_tau(x) + s) >= tau."""
    return conformal_quantile(np.asarray(y_cal, dtype=float) - np.asarray(q_tau_cal, dtype=float), 1 - tau)


class ResidualConformal:
    """Split conformal interval around a point prediction: point +- Q, Q from |y - point| (constant width)."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, point_cal, y_cal):
        self.q_ = conformal_quantile(np.abs(np.asarray(y_cal, dtype=float) - np.asarray(point_cal, dtype=float)), self.alpha)
        return self

    def predict(self, point):
        point = np.asarray(point, dtype=float)
        return point - self.q_, point + self.q_


def coverage(lo, hi, y) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(lo) <= y) & (y <= np.asarray(hi))))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def coverage_by_bins(lo, hi, y, by, n_bins=5) -> np.ndarray:
    """Coverage inside n_bins equal-count bins of the variable ``by`` (e.g. the interval width)."""
    by = np.asarray(by, dtype=float)
    edges = np.quantile(by, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, by, side="right") - 1, 0, n_bins - 1)
    inside = (np.asarray(lo) <= y) & (y <= np.asarray(hi))
    return np.array([inside[bins == b].mean() if np.any(bins == b) else np.nan for b in range(n_bins)])
