"""Decision values from a predictive distribution under asymmetric costs.

Cost functions
  LinearCost(c_under, c_over): c_under * max(y - d, 0) + c_over * max(d - y, 0).
      The decision with the smallest expected cost is the tau-quantile of y
      with tau = c_under / (c_under + c_over).
  PenaltyCost(c_under, c_over, penalty): LinearCost plus a fixed penalty
      whenever y > d (for example a budget overrun). The best decision is no
      longer a fixed quantile; it depends on the shape of the distribution.

Ways to turn QRF output into a decision
  expected_cost_decision : plug-in. Evaluate the expected cost of every atom
      of the predictive distribution and take the smallest (no learning).
  PinballStage           : QRFSL-style learned stage for LinearCost: gradient
      boosting with the pinball loss at tau on the 21 out-of-fold quantiles.
  CostWeightStage        : QRFSL-style learned stage for any cost: a weighted
      average of the quantiles (weights on the simplex) that minimises the
      empirical cost on the out-of-fold quantiles.
  ShiftStage             : baseline, a location (e.g. the median) plus one
      constant shift chosen on the out-of-fold rows.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor


class LinearCost:
    def __init__(self, c_under=1.0, c_over=1.0):
        self.c_under = float(c_under)
        self.c_over = float(c_over)

    @property
    def tau(self) -> float:
        return self.c_under / (self.c_under + self.c_over)

    def __call__(self, d, y):
        d, y = np.asarray(d, dtype=float), np.asarray(y, dtype=float)
        return self.c_under * np.maximum(y - d, 0) + self.c_over * np.maximum(d - y, 0)

    def expected(self, atoms, weights):
        """Expected cost of choosing each atom, shape like atoms (atoms sorted per row)."""
        F = np.cumsum(weights, axis=1)
        S1 = np.cumsum(weights * atoms, axis=1)
        total = S1[:, -1:]
        e_over = atoms * F - S1                       # E[(a - Y)+]
        e_under = (total - S1) - atoms * (1 - F)      # E[(Y - a)+]
        return self.c_under * e_under + self.c_over * e_over

    def __repr__(self):
        return f"linear {self.c_under:g}:{self.c_over:g} (tau={self.tau:.3f})"


class PenaltyCost(LinearCost):
    def __init__(self, c_under=1.0, c_over=1.0, penalty=1.0):
        super().__init__(c_under, c_over)
        self.penalty = float(penalty)

    def __call__(self, d, y):
        d, y = np.asarray(d, dtype=float), np.asarray(y, dtype=float)
        return super().__call__(d, y) + self.penalty * (y > d)

    def expected(self, atoms, weights):
        F = np.cumsum(weights, axis=1)
        return super().expected(atoms, weights) + self.penalty * (1 - F)   # P(Y > a)

    def __repr__(self):
        return f"linear {self.c_under:g}:{self.c_over:g} + penalty {self.penalty:g} if y > d"


def expected_cost_decision(atoms, weights, cost) -> np.ndarray:
    """Plug-in decision: the atom with the smallest expected cost in every row.

    For piecewise-linear costs with downward jumps at the atoms, the minimum
    over all real d is attained at an atom, so the atoms are the only
    candidates that need checking.
    """
    atoms = np.asarray(atoms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    best = np.argmin(cost.expected(atoms, weights), axis=1)
    return atoms[np.arange(len(atoms)), best]


def quantiles_as_distribution(Q):
    """Treat a row of K predicted quantiles as K equally weighted atoms (sorted)."""
    Q = np.sort(np.asarray(Q, dtype=float), axis=1)
    return Q, np.full(Q.shape, 1.0 / Q.shape[1])


class PinballStage:
    """Learned stage for LinearCost: boosting with pinball loss at tau, inputs = predicted quantiles."""

    def __init__(self, tau, n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0):
        self.tau = tau
        self.model = GradientBoostingRegressor(loss="quantile", alpha=tau, n_estimators=n_estimators,
                                               learning_rate=learning_rate, max_depth=max_depth,
                                               subsample=0.8, random_state=random_state)

    def fit(self, Q, y):
        self.model.fit(np.asarray(Q, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, Q):
        return self.model.predict(np.asarray(Q, dtype=float))


class CostWeightStage:
    """Learned stage for any cost: d = Q @ w, w >= 0, sum w = 1, minimising the mean cost on (Q, y)."""

    def __init__(self, cost, start_tau=None, maxiter=20000):
        self.cost = cost
        self.start_tau = start_tau
        self.maxiter = maxiter

    @staticmethod
    def _softmax(theta):
        e = np.exp(theta - theta.max())
        return e / e.sum()

    def fit(self, Q, y, probs=None):
        Q, y = np.asarray(Q, dtype=float), np.asarray(y, dtype=float)
        k = Q.shape[1]
        theta0 = np.zeros(k)
        if probs is not None and self.start_tau is not None:      # start near the tau-quantile column
            theta0[np.argmin(np.abs(np.asarray(probs) - self.start_tau))] = 3.0
        obj = lambda th: float(np.mean(self.cost(Q @ self._softmax(th), y)))
        res = minimize(obj, theta0, method="Powell", options=dict(maxiter=self.maxiter, xtol=1e-4, ftol=1e-8))
        self.coef_ = self._softmax(res.x)
        self.objective_ = res.fun
        return self

    def predict(self, Q):
        return np.asarray(Q, dtype=float) @ self.coef_


class ShiftStage:
    """Baseline: location + one constant shift; the shift minimises the mean cost on the fitting rows."""

    def __init__(self, cost):
        self.cost = cost

    def fit(self, location, y):
        location, y = np.asarray(location, dtype=float), np.asarray(y, dtype=float)
        resid = np.sort(y - location)
        cands = np.unique(np.quantile(resid, np.linspace(0, 1, 401)))
        costs = [np.mean(self.cost(location + s, y)) for s in cands]
        self.shift_ = float(cands[int(np.argmin(costs))])
        return self

    def predict(self, location):
        return np.asarray(location, dtype=float) + self.shift_


def mean_cost(d, y, cost) -> float:
    return float(np.mean(cost(d, y)))
