"""QRFSL: a super learner built on quantile regression forests.

Python port of ``Rcode_QRFSL.R``. This module is shared by ``train.py`` and
``predict.py`` (a saved model is a pickled ``QRFSL`` object, so both scripts
must be able to import this module).

Algorithm (same as the R code)
  1. Split the training rows into k folds.
  2. For each fold, fit a quantile regression forest (QRF) on the other k-1
     folds and predict the quantiles ``probs`` of the held-out fold.
  3. Fit the super learner that maps those cross-validated quantiles to the
     observed y: a random forest (``sl="rf"``, what the R code does) or a
     weighted average of the quantiles (``sl="weights"``, the R ``sl = 0``
     option, which the R code declares but does not implement).
  4. Fit one more QRF on all training rows. For new data, its predicted
     quantiles are fed to the super learner to give the final prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.tree import DecisionTreeRegressor

# 0.025, 0.05, 0.10, ..., 0.95, 0.975 (21 levels), the R default of `prob`
DEFAULT_PROBS = (0.025, *(round(0.05 * i, 2) for i in range(1, 20)), 0.975)


def r_mtry(n_features: int) -> int:
    """Default mtry of R randomForest for regression: max(floor(p / 3), 1)."""
    return max(n_features // 3, 1)


class RForest:
    """Regression random forest grown like R ``randomForest`` (defaults).

    - Each tree is fitted on n rows drawn with replacement (``replace = TRUE,
      sampsize = n``). The drawn rows are passed as real copies, so node sizes
      count repeated rows the same way R does.
    - ``mtry = max(floor(p / 3), 1)`` candidate features at each split.
    - R makes a node terminal when it holds ``nodesize`` rows or fewer, i.e. a
      node is split only while it has more than ``nodesize`` rows:
      ``min_samples_split = nodesize + 1`` with ``min_samples_leaf = 1``.
    - The prediction is the mean of the trees.
    """

    def __init__(self, n_estimators=500, nodesize=5, random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.nodesize = nodesize
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        # one seed per tree, drawn up front, so the result does not depend on n_jobs
        seeds = np.random.default_rng(self.random_state).integers(2**31 - 1, size=self.n_estimators)

        def grow(seed):
            rng = np.random.default_rng(seed)
            boot = rng.integers(0, n, size=n)
            tree = DecisionTreeRegressor(max_features=r_mtry(p),
                                         min_samples_split=self.nodesize + 1,
                                         min_samples_leaf=1,
                                         random_state=int(seed))
            return tree.fit(X[boot], y[boot])

        self.estimators_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(grow)(s) for s in seeds
        )
        return self

    def apply(self, X) -> np.ndarray:
        """Leaf id of every row in every tree, shape (n_rows, n_trees)."""
        X = np.asarray(X, dtype=float)
        return np.column_stack([tree.apply(X) for tree in self.estimators_])

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean([tree.predict(X) for tree in self.estimators_], axis=0)


def quantile_names(probs) -> list[str]:
    """Column names of the predicted quantiles, as in R after gsub("\\s", ""): quantile=0.025."""
    return [f"quantile={p:g}" for p in probs]


def cv_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Same rule as the R ``cv.folds``: split(sample(1:n), rep(1:k, length = n)).

    The rows are shuffled, then the j-th shuffled row goes to fold (j mod k).
    """
    perm = np.random.default_rng(seed).permutation(n)
    labels = np.arange(n) % k
    return [perm[labels == j] for j in range(k)]


class QuantileRegressionForest:
    """Quantile regression forest (Meinshausen, 2006) in two variants.

    leaf_sample="one" (default, port of R ``quantregForest``): after growing
        the forest, keep for every tree and every leaf the y of ONE training
        row picked at random among the rows in that leaf. A prediction
        collects these values over the trees (ntree values per row) and takes
        their quantiles (R ``quantile`` type 7, numpy method "linear").
    leaf_sample="all" (Meinshausen's definition): the predictive distribution
        of a row is the weighted empirical distribution of ALL training y,
        where the weight of training row i is the average over trees of
        1 / (size of the leaf) when i shares the leaf with the new row. The
        mean of this distribution is the forest prediction in Meinshausen's
        definition (all rows of a leaf weighted equally); it differs slightly
        from randomForest's prediction, whose leaf values are means of the
        bootstrap draws only. Quantiles are the generalised inverse of the
        weighted CDF.
    ``predict_distribution`` returns the atoms and weights of either variant,
    which ``decision.py`` uses to compute expected costs.
    """

    def __init__(self, n_estimators=500, nodesize=5, random_state=None, n_jobs=None,
                 leaf_sample="one"):
        self.n_estimators = n_estimators
        self.nodesize = nodesize
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.leaf_sample = leaf_sample

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.leaf_sample not in ("one", "all"):
            raise ValueError("leaf_sample must be 'one' or 'all'")
        # separate random streams for growing the trees and for picking leaf values
        forest_seed, leaf_seed = np.random.SeedSequence(self.random_state).spawn(2)
        self.forest_ = RForest(self.n_estimators, self.nodesize,
                               forest_seed, self.n_jobs).fit(X, y)

        nodes = self.forest_.apply(X)  # (n, ntree) leaf id of every training row
        n, ntree = nodes.shape
        if self.leaf_sample == "all":
            # training rows sorted by y; per tree, the leaf id of each sorted row
            self.order_ = np.argsort(y, kind="mergesort")
            self.y_sorted_ = y[self.order_]
            self.train_nodes_ = nodes[self.order_]
            # 1 / leaf size for every (training row, tree)
            self.inv_leaf_size_ = np.empty_like(nodes, dtype=float)
            for tree in range(ntree):
                _, inv, counts = np.unique(nodes[:, tree], return_inverse=True, return_counts=True)
                self.inv_leaf_size_[:, tree] = 1.0 / counts[inv]
            self.inv_leaf_size_ = self.inv_leaf_size_[self.order_]
            return self

        n_nodes = max(est.tree_.node_count for est in self.forest_.estimators_)
        rng = np.random.default_rng(leaf_seed)
        # values_nodes_[leaf, tree] = y of one random training row in that leaf
        self.values_nodes_ = np.full((n_nodes, ntree), np.nan)
        for tree in range(ntree):
            shuffled = rng.permutation(n)
            nodes_tree = nodes[shuffled, tree]
            leaves, first = np.unique(nodes_tree, return_index=True)
            self.values_nodes_[leaves, tree] = y[shuffled[first]]
        return self

    def _weights(self, nodes, chunk=500) -> np.ndarray:
        """leaf_sample="all": weights of the training rows (sorted by y) for each new row."""
        ntree = nodes.shape[1]
        out = np.empty((nodes.shape[0], len(self.y_sorted_)))
        for start in range(0, nodes.shape[0], chunk):
            block = nodes[start:start + chunk]
            W = np.zeros((len(block), len(self.y_sorted_)))
            for tree in range(ntree):
                same = block[:, tree][:, None] == self.train_nodes_[:, tree][None, :]
                W += same * self.inv_leaf_size_[:, tree][None, :]
            out[start:start + chunk] = W / ntree
        return out

    def predict_distribution(self, X):
        """Predictive distribution of every row as (atoms, weights), both (n_rows, n_atoms).

        Atoms are sorted along axis 1; weights are non-negative and sum to 1
        per row. For leaf_sample="one" the atoms are the ntree stored leaf
        values; for "all" they are the sorted training y.
        """
        nodes = self.forest_.apply(np.asarray(X, dtype=float))
        if self.leaf_sample == "all":
            W = self._weights(nodes)
            return np.broadcast_to(self.y_sorted_, W.shape), W
        values = self.values_nodes_[nodes, np.arange(nodes.shape[1])]
        values = np.sort(values, axis=1)                       # NaN sorts last
        ok = ~np.isnan(values)
        w = ok / ok.sum(axis=1, keepdims=True)
        return np.where(ok, values, 0.0), w

    def predict(self, X, probs) -> np.ndarray:
        """Predicted quantiles, shape (n_rows, len(probs))."""
        probs = np.atleast_1d(np.asarray(probs, dtype=float))
        nodes = self.forest_.apply(np.asarray(X, dtype=float))
        if self.leaf_sample == "all":
            cdf = np.cumsum(self._weights(nodes), axis=1)
            cdf[:, -1] = 1.0
            idx = np.stack([np.argmax(cdf >= p - 1e-12, axis=1) for p in probs], axis=1)
            return self.y_sorted_[idx]
        values = self.values_nodes_[nodes, np.arange(nodes.shape[1])]  # (m, ntree)
        return np.nanquantile(values, probs, axis=1, method="linear").T


class ConvexWeights:
    """Super learner of the "weights" type (R ``sl = 0``, not implemented in R).

    The prediction is a weighted average of the quantiles: the weights are
    non-negative, sum to 1, and minimise the squared error on the
    cross-validated quantiles (``method.CC_LS`` of the R SuperLearner package).

    The constraint "sum = 1" is added to non-negative least squares as an extra
    row with a very large weight, then the tiny remaining gap is divided out.
    Because the weights sum to 1, subtracting the mean of y from y and from all
    quantiles does not change the solution; it only makes the numbers better
    conditioned. Fitting with the constraint (rather than scaling NNLS weights
    to sum 1 afterwards, as ``method.NNLS`` does) means the weights do not
    change when the same constant is added to y and to the quantiles.
    """

    def fit(self, Q, y):
        y = np.asarray(y, dtype=float)
        center = y.mean()
        Q = np.asarray(Q, dtype=float) - center
        big = 1e3 * np.sqrt(len(y)) * max(np.abs(Q).max(), 1.0)
        w, _ = nnls(np.vstack([Q, np.full(Q.shape[1], big)]), np.append(y - center, big))
        self.coef_ = w / w.sum()
        return self

    def predict(self, Q) -> np.ndarray:
        return np.asarray(Q, dtype=float) @ self.coef_


SL_TYPES = ("rf", "weights", "none")


def _fold_quantiles(X, y, test_idx, probs, fold_seed, qrf_trees, nodesize, leaf_sample):
    """Fit a QRF without one fold and predict the quantiles of that fold."""
    train = np.ones(len(y), dtype=bool)
    train[test_idx] = False
    qrf = QuantileRegressionForest(qrf_trees, nodesize, random_state=fold_seed,
                                   leaf_sample=leaf_sample)
    qrf.fit(X[train], y[train])
    return qrf.predict(X[test_idx], probs)


class QRFSL:
    """Quantile regression forest super learner.

    ``fit`` corresponds to R ``QRFSL()`` and ``predict`` to R
    ``predict.QRFSL()``.

    Parameters
    ----------
    k : number of cross-validation folds (R ``k``).
    probs : quantile levels used as super learner inputs (R ``prob``).
    seed : seed for the fold split and the super learner (R ``seed`` of
        ``QRFSL()``). As in R, the QRF of fold i always uses seed i.
    full_seed : seed for the QRF fitted on all training rows (R ``seed`` of
        ``predict.QRFSL()``). R refits that QRF in every prediction call;
        here it is fitted once in ``fit`` and saved with the model.
    n_jobs : number of folds fitted in parallel (R ``mc``); also used for the
        trees of the full-data QRF and the super learner.
    qrf_trees, nodesize : QRF settings (quantregForest / randomForest
        defaults: 500 trees, nodesize 5).
    sl : type of super learner. "rf" (default) is the random forest that the
        R code always fits. "weights" is a weighted average of the
        quantiles (``ConvexWeights``); R lists it as ``sl = 0`` but the R code
        never uses it. "none" fits no super learner (only the cross-validated
        quantiles and the full-data QRF, for ``conformal.py`` / ``decision.py``).
    sl_trees : number of trees of the random forest super learner
        (R ``ntree = 1000``).
    leaf_sample : "one" (R quantregForest) or "all" (Meinshausen), see
        ``QuantileRegressionForest``.
    """

    def __init__(self, k=10, probs=DEFAULT_PROBS, seed=0, full_seed=0, n_jobs=1,
                 qrf_trees=500, sl="rf", sl_trees=1000, nodesize=5, leaf_sample="one"):
        self.k = k
        self.probs = tuple(probs)
        self.seed = seed
        self.full_seed = full_seed
        self.n_jobs = n_jobs
        self.qrf_trees = qrf_trees
        self.sl = sl
        self.sl_trees = sl_trees
        self.nodesize = nodesize
        self.leaf_sample = leaf_sample

    def fit(self, X, y):
        X = _as_frame(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
        if self.sl not in SL_TYPES:
            raise ValueError(f"sl must be one of {SL_TYPES}, got {self.sl!r}")
        if not 2 <= self.k <= len(y):
            raise ValueError(f"k must be between 2 and the number of rows ({len(y)})")
        if np.isnan(y).any() or X.isna().any().any():
            raise ValueError("missing values are not allowed (as in quantregForest)")

        self.feature_names_ = list(X.columns)
        self.quantile_names_ = quantile_names(self.probs)
        Xa = X.to_numpy(dtype=float)

        # 1) cross-validated quantiles: fold i (1-based) uses seed i, as set.seed(i) in R
        self.folds_ = cv_folds(len(y), self.k, self.seed)
        jobs = (
            delayed(_fold_quantiles)(Xa, y, idx, self.probs, i + 1, self.qrf_trees,
                                     self.nodesize, self.leaf_sample)
            for i, idx in enumerate(self.folds_)
        )
        pred_cv = np.vstack(Parallel(n_jobs=self.n_jobs)(jobs))
        order = np.concatenate(self.folds_)
        yact_cv = y[order]

        # 2) super learner from the quantiles to y
        if self.sl == "rf":
            self.sl_ = RForest(self.sl_trees, self.nodesize, self.seed,
                               self.n_jobs).fit(pred_cv, yact_cv)
        elif self.sl == "weights":
            self.sl_ = ConvexWeights().fit(pred_cv, yact_cv)
        else:
            self.sl_ = None

        # 3) QRF on all training rows, used at prediction time
        self.qmod_ = QuantileRegressionForest(self.qrf_trees, self.nodesize, self.full_seed,
                                              self.n_jobs, self.leaf_sample).fit(Xa, y)

        self.cv_quantiles_ = pd.DataFrame(pred_cv, columns=self.quantile_names_,
                                          index=X.index[order])
        self.cv_y_ = pd.Series(yact_cv, index=X.index[order], name="y")
        return self

    def predict_quantiles(self, X) -> pd.DataFrame:
        """Quantiles predicted by the full-data QRF (R ``qpred``)."""
        X = _as_frame(X, self.feature_names_)
        q = self.qmod_.predict(X.to_numpy(dtype=float), self.probs)
        return pd.DataFrame(q, columns=self.quantile_names_, index=X.index)

    def predict_distribution(self, X):
        """Atoms and weights of the full-data QRF's predictive distribution per row."""
        X = _as_frame(X, self.feature_names_)
        return self.qmod_.predict_distribution(X.to_numpy(dtype=float))

    def predict(self, X, return_quantiles=False):
        """Final prediction of the super learner (R ``pred``)."""
        if self.sl_ is None:
            raise ValueError('no super learner was fitted (sl="none"); use predict_quantiles')
        qpred = self.predict_quantiles(X)
        pred = self.sl_.predict(qpred.to_numpy())
        return (pred, qpred) if return_quantiles else pred


def _as_frame(X, feature_names=None) -> pd.DataFrame:
    """Turn X into a numeric DataFrame; at prediction time pick the training columns in order."""
    if not isinstance(X, pd.DataFrame):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = pd.DataFrame(X, columns=feature_names or [f"x{j}" for j in range(X.shape[1])])
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"columns missing from the input: {missing}")
        X = X[feature_names]
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"non-numeric columns (encode them first): {non_numeric}")
    return X
