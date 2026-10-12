# QRFSL: Quantile Regression Forest Super Learner

[한국어 문서](README_ko.md)

QRFSL is a regression method that predicts a numeric target in two stages.

1. A **quantile regression forest (QRF)** predicts 21 quantiles of the target for each row: the 2.5 %, 5 %, 10 %, ..., 95 % and 97.5 % points of the predicted distribution.
2. A **super learner** takes those 21 quantiles as inputs and returns the final prediction. In the R code it is a random forest; the Python port also offers a weighted average of the quantiles (section 5.4).

The super learner is trained on *cross-validated* quantiles: every training row is predicted by a QRF that did not see that row. The super learner therefore learns how the quantiles relate to the true value on data the QRF has not memorised.

This repository holds a Python port ([python/](python/)) of the original R code, split into a training script and an inference script. The R code (`Rcode_QRFSL.R`) has been removed from the current files; it is kept in the history at commit [1fea660](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R), and the line numbers in sections 3 and 4 link to it there.

## Contents

1. [Repository layout](#1-repository-layout)
2. [How the algorithm works](#2-how-the-algorithm-works)
3. [Walk-through of Rcode_QRFSL.R](#3-walk-through-of-rcode_qrfslr)
4. [Details that are easy to miss in the R code](#4-details-that-are-easy-to-miss-in-the-r-code)
5. [Python port](#5-python-port)
6. [Running the Python version](#6-running-the-python-version)
7. [R and Python results compared](#7-r-and-python-results-compared)
8. [Does QRFSL help, and for what?](#8-does-qrfsl-help-and-for-what-added-after-the-port)
9. [References](#9-references)

## 1. Repository layout

```
QRFSL/
├── python/
│   ├── data/
│   │   ├── mtcars_test.csv     5 test cars of the R example
│   │   └── mtcars_train.csv    27 training cars of the R example
│   ├── predict.py              inference: load a saved model, predict a CSV
│   ├── qrfsl.py                shared code: random forest, QRF, QRFSL class
│   ├── requirements.txt
│   └── train.py                training: fit QRFSL on a CSV, save the model
├── README.md                   this document (English)
└── README_ko.md                Korean version
```

The example commands in section 6 create `python/model/` (saved models) and `python/output/` (prediction files).

## 2. How the algorithm works

```
Training (n rows, p features)
  1. shuffle the rows and deal them into k folds (default k = 10)
  2. for each fold i = 1, ..., k
       fit a QRF on the other k-1 folds
       predict the 21 quantiles of the rows in fold i
     -> n x 21 table of cross-validated quantiles
  3. fit a random forest (1,000 trees): 21 quantiles -> y   (super learner)
  4. fit one more QRF on all n rows                         (full-data QRF)

Prediction (new rows)
  full-data QRF -> 21 quantiles per row -> super learner -> final prediction
```

**Quantile regression forest** (Meinshausen, 2006). A random forest is grown as usual, but the leaves keep the training targets instead of only their mean. For a new row, the targets in the leaves that the row reaches are pooled over all trees and their quantiles are read off. The R package used here, quantregForest (version 1.4-0), uses a fast approximation of this: for every tree and every leaf it stores the target of **one** training row picked at random among the rows in that leaf. The prediction for a new row is then the sample quantile (R `quantile` type 7) of the stored values, one value per tree, 500 values in total.

**Super learner** (van der Laan et al., 2007) is a form of stacking: a second model learns how to combine first-stage predictions, using predictions made by cross-validation so that the combination is learned on unseen data. In QRFSL the first-stage predictions are the 21 quantiles from one QRF, and the combiner in the R code is a random forest rather than a weighted average. The Python port adds the weighted average as an option (`sl="weights"`, section 5.4).

Feeding all 21 quantiles to the super learner lets it use the whole predicted distribution, for example its spread or skewness, instead of only one summary such as the mean or the median.

## 3. Walk-through of Rcode_QRFSL.R

The line numbers below refer to [Rcode_QRFSL.R at commit 1fea660](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R).

### 3.1 Packages ([lines 5–8](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L5-L8))

`quantregForest` (which also attaches `randomForest`), `doParallel` and `foreach`.

### 3.2 Training: `QRFSL()` ([lines 12–107](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L12-L107))

| Argument | Default | Meaning |
|:--|:--|:--|
| `x` | none | features (matrix or data frame) |
| `y` | none | numeric target vector |
| `k` | 10 | number of cross-validation folds |
| `prob` | 0.025, 0.05, 0.10, ..., 0.95, 0.975 (21 values) | quantile levels |
| `seed` | 0 | random seed for the fold split and the super learner |
| `mc` | 1 | number of parallel workers; folds run in parallel when above 1 |
| `sl` | 0 | documented as "0: weights, 1: rf" but never used; the code always fits a random forest (section 4) |

1. **Fold split ([lines 32–39](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L32-L39)).** `cv.folds()` shuffles the row numbers 1, ..., n and deals them to folds 1, 2, ..., k, 1, 2, ... in turn (`split(sample(1:n), rep(1:folds, length = n))`), so fold sizes differ by at most one row. With the 27 training rows of the example and k = 10, seven folds have 3 rows and three folds have 2.
2. **Fold data ([lines 42–46](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L42-L46)).** For fold i, `tdat_x_cv[[i]]` and `tdat_y_cv[[i]]` hold every row except fold i (used for fitting); `idat_x_cv[[i]]` and `idat_y_cv[[i]]` hold the rows of fold i (used for predicting).
3. **Cross-validated quantiles ([lines 49–79](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L49-L79)).** For each fold: `set.seed(i)`, fit `quantregForest()` on the fitting part with the package defaults, then predict the 21 quantiles of the held-out rows. With `mc > 1` the folds run on a `doParallel` cluster. Because every fold sets its own seed, `mc = 1` and `mc = 2` give identical results (checked).
4. **Stacking ([lines 80–86](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L80-L86)).** The fold predictions are stacked into an n x 21 matrix `pred_cv`. The column names returned by quantregForest, such as `quantile= 0.025`, lose their spaces and become `quantile=0.025`. The true targets are stacked in the same fold order into `yact_cv`.
5. **Super learner ([lines 89–96](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L89-L96)).** `set.seed(seed)`, then `randomForest(x = pred_cv, y = yact_cv, ntree = 1000)`. All other settings are the package defaults: mtry = floor(21 / 3) = 7 candidate inputs per split, nodesize = 5.
6. **Return value ([lines 99–106](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L99-L106)).** `list(sl, x, y, prob)`. The full-data QRF is not fitted here; the training data is kept so that `predict.QRFSL()` can fit it.

### 3.3 Prediction: `predict.QRFSL()` ([lines 111–154](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L111-L154))

Arguments: `object` (the list returned by `QRFSL()`), `newx` (new features), `seed` (default 0).

1. `set.seed(seed)`, then fit `quantregForest()` on all training rows `object$x`, `object$y`. This is `qmod`.
2. Predict the quantiles `object$prob` of `newx` with `qmod`, and remove the spaces from the column names. This is `qpred`.
3. Feed `qpred` to the super learner `object$sl`. This is `pred`.
4. Return `list(qmod, qpred, pred)`.

### 3.4 Example ([lines 160–206](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L160-L206))

- Data: `mtcars`, 32 cars and 11 columns. Target: `mpg` (fuel economy, miles per gallon). Features: the other 10 columns (`cyl`, `disp`, `hp`, `drat`, `wt`, `qsec`, `vs`, `am`, `gear`, `carb`).
- `set.seed(2)` and `sample()` pick 5 test rows (row numbers 21, 15, 6, 30, 8); the other 27 rows are the training data.
- Training: `QRFSL(k = 10, seed = 1, mc = 1)`. Prediction: `predict.QRFSL(seed = 2019)`.
- The script prints the RMSE (root mean squared error) and a column named `corr`, which is the **squared** Pearson correlation between the true and predicted values.

Output of the unchanged script in our environment (R 4.5.3, quantregForest 1.4-0, randomForest 4.7-1.2):

Table 1. R example result (unit: mpg)

| Test car | True | Predicted |
|:--|--:|--:|
| Toyota Corona | 21.5 | 24.02 |
| Cadillac Fleetwood | 10.4 | 13.34 |
| Valiant | 18.1 | 19.97 |
| Ferrari Dino | 19.7 | 19.37 |
| Merc 240D | 24.4 | 23.86 |

RMSE is 1.944 mpg and `corr` (squared correlation) is 0.922. Other R versions can give other numbers because the random number generator has changed over time (for example, `sample()` changed in R 3.6.0).

## 4. Details that are easy to miss in the R code

1. **The `sl` argument does nothing.** [Line 90](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L90) (`sl <- suppressWarnings(randomForest(...))`) overwrites it. Only the random forest super learner exists; the "weights" option is not implemented. The Python port adds it as an option (section 5.4).
2. **Fold models always use seeds 1, ..., k.** `set.seed(i)` on lines [52](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L52) and [67](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L67) does not depend on `seed`. `seed` controls only the fold split ([line 35](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L35)) and the super learner ([line 89](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L89)). The full-data QRF is controlled by the `seed` of `predict.QRFSL()`.
3. **QRF settings are the package defaults** because the code does not set them: 500 trees, mtry = max(floor(p / 3), 1) candidate features per split (3 for the 10 mtcars features), nodesize = 5 (a node with 5 rows or fewer is not split any further), and a bootstrap sample of n rows drawn with replacement for each tree.
4. **One stored target per leaf.** As described in section 2, quantregForest keeps one randomly chosen target per leaf and tree. The candidates are all training rows that reach the leaf, including rows that were not in that tree's bootstrap sample.
5. **The full-data QRF is refitted in every prediction call.** The prediction therefore depends on the `seed` given to `predict.QRFSL()`, and the trained object carries the training data `x` and `y` (754 KB for the mtcars example).
6. **What `suppressWarnings()` hides.** The super learner call raises "Recycling array of length 1 in vector-array arithmetic is deprecated" inside `randomForest()`, because `yact_cv` is a one-column matrix rather than a vector. It does not affect the result.
7. **`randomForest()` is used without `library(randomForest)`.** It is available because `quantregForest` attaches `randomForest` when it is loaded.
8. **Column names.** The spaces are removed from both `pred_cv` (training) and `qpred` (prediction), so the super learner always sees the same column names.

## 5. Python port

### 5.1 Files

- [python/qrfsl.py](python/qrfsl.py): shared code used by both scripts.
  - `RForest`: regression random forest grown with the rules of R `randomForest`, built from scikit-learn `DecisionTreeRegressor`.
  - `QuantileRegressionForest`: port of `quantregForest()` and its `predict(what = ...)`.
  - `ConvexWeights`: the "weights" super learner, which the R code does not implement (section 5.4).
  - `cv_folds()`: port of `cv.folds()`.
  - `QRFSL`: `fit()` corresponds to `QRFSL()`, `predict()` to `predict.QRFSL()`.
- [python/train.py](python/train.py): reads a CSV, fits QRFSL and saves the model with joblib.
- [python/predict.py](python/predict.py): loads a saved model, predicts a CSV and writes a CSV. If the true target is in the input, it also prints the RMSE and the squared correlation, like the R example.

### 5.2 R to Python

| R | Python |
|:--|:--|
| `QRFSL(x, y, k, prob, seed, mc)` | `QRFSL(k, probs, seed, full_seed, n_jobs).fit(X, y)`, or `train.py` |
| `predict.QRFSL(object, newx, seed)` | `model.predict(X, return_quantiles=True)`, or `predict.py` |
| `seed` of `predict.QRFSL()` | `full_seed`, given at training time |
| `mc` | `n_jobs` |
| `sl` (0 = weights, 1 = rf; the code always fits rf) | `sl="rf"` (default) or `sl="weights"` |
| `quantregForest()` and `predict(what = prob)` | `QuantileRegressionForest.fit()` and `.predict()` |
| `randomForest()` | `RForest` |
| `cv.folds()` | `cv_folds()` |
| `$sl` of the training result | `model.sl_` |
| `$qmod`, `$qpred`, `$pred` of the prediction result | `model.qmod_`, `model.predict_quantiles(X)`, `model.predict(X)` |

### 5.3 Differences from the R code

| Item | R code | Python port |
|:--|:--|:--|
| When the full-data QRF is fitted | in every `predict.QRFSL()` call | once, during training; saved in the model file |
| Data kept in the model | training data `x`, `y` | no feature data; only the cross-validated quantiles and targets used by the super learner (`cv_quantiles_`, `cv_y_`) |
| Random numbers | R generator | numpy generator; the same seed gives different draws than R |
| Tree growing | randomForest C code | scikit-learn `DecisionTreeRegressor` on copies of the bootstrap rows, with the same bootstrap, mtry and nodesize rules |
| Node where none of the mtry candidate features can split | becomes a leaf | scikit-learn tries further features |
| Categorical features | factors are handled directly | numeric columns only; encode categories first |
| `sl` argument | declared but unused; always a random forest | `"rf"` (default, same as the R code) or `"weights"` (added, section 5.4) |
| Parallel run | `doParallel` cluster over the folds | joblib: processes over the folds, threads over the trees of the full-data QRF and the super learner |

Why the trees are built from `DecisionTreeRegressor` instead of scikit-learn's `RandomForestRegressor`: `RandomForestRegressor` represents repeated bootstrap rows as weights, so its node size counts distinct rows, while R counts every drawn row. Under the same nodesize rule this makes the scikit-learn trees slightly smaller. A first version built on `RandomForestRegressor` gave a mean mtcars RMSE of 2.368 (R: 2.196) and a mean Toyota Corona prediction of 24.63 (R: 24.16). Fitting each tree on actual copies of the drawn rows brought these to 2.227 and 24.08 (section 7).

### 5.4 Weights super learner (`sl="weights"`, added in Python)

The R code lists `sl = 0` ("weights") but does not implement it. The Python port implements it as follows.

- The prediction is a weighted average of the 21 quantiles: w1 x q1 + ... + w21 x q21, where every weight is 0 or more and the weights sum to 1.
- The weights minimise the squared error between the cross-validated quantiles and the true values under those two conditions. This is the `method.CC_LS` rule of the R SuperLearner package.
- They are found with scipy non-negative least squares plus one extra row for "sum = 1". A general optimiser (scipy `trust-constr`) gave the same weights, with the largest difference 2.5e-8.
- The weights are stored in `model.sl_.coef_`, and `train.py` prints the non-zero ones.

Why not take non-negative least squares weights and divide them by their sum afterwards (`method.NNLS`, the default of SuperLearner)? QRF quantiles are pulled toward the middle of the data, so when the true values are spread wider, the undivided weights sum to more than 1 (1.381 on average on Friedman #1). Dividing by that sum shrinks every prediction toward zero: the Friedman #1 test RMSE was 4.606 with that rule against 2.302 with the rule above. Fitting with the condition "sum = 1" also means the weights stay the same when the same constant is added to the target and to the quantiles.

The default stays `"rf"`: it is what the R code actually runs (even though its argument list shows `sl = 0`), and it was more accurate in our tests (section 7, Tables 6 and 7).

## 6. Running the Python version

Tested with Python 3.12.3, numpy 2.5.2, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1 and joblib 1.5.2.

```bash
pip install -r python/requirements.txt
cd python
```

### 6.1 Training

```bash
python train.py --data data/mtcars_train.csv --target mpg --drop car \
    --k 10 --seed 1 --full-seed 2019 --model model/qrfsl_mtcars.joblib
```

| Option | Default | Meaning |
|:--|:--|:--|
| `--data` | required | training CSV (with a header row) |
| `--target` | required | target column |
| `--drop` | none | columns that are not features, such as an id column |
| `--model` | required | output path of the model file |
| `--k` | 10 | number of folds (R `k`) |
| `--probs` | 21 levels, as in R | quantile levels (R `prob`) |
| `--seed` | 0 | seed for the fold split and the super learner (R `seed` of `QRFSL()`) |
| `--full-seed` | 0 | seed for the full-data QRF (R `seed` of `predict.QRFSL()`) |
| `--n-jobs` | 1 | folds fitted in parallel (R `mc`) |
| `--qrf-trees` | 500 | trees per QRF |
| `--sl` | rf | super learner: `rf` (random forest, as in the R code) or `weights` (weighted average of the quantiles, section 5.4) |
| `--sl-trees` | 1000 | trees of the random forest super learner |
| `--nodesize` | 5 | R `nodesize` for both forests |

All columns except the target and `--drop` are used as features. They must be numeric and contain no missing values (quantregForest has the same rule).

### 6.2 Inference

```bash
python predict.py --model model/qrfsl_mtcars.joblib --data data/mtcars_test.csv \
    --out output/pred_mtcars.csv --keep car --target mpg --quantiles
```

| Option | Default | Meaning |
|:--|:--|:--|
| `--model` | required | model file saved by `train.py` |
| `--data` | required | CSV with the training feature columns (any order; other columns are ignored) |
| `--out` | required | output CSV |
| `--keep` | none | input columns copied to the output, such as an id column |
| `--target` | none | if the input has the true target, copy it to the output and print RMSE and squared correlation |
| `--quantiles` | off | also write the 21 quantiles of the full-data QRF |

Output columns: the `--keep` columns, the `--target` column (if given), `pred` (final prediction) and, with `--quantiles`, `quantile=0.025`, ..., `quantile=0.975`.

The example above printed RMSE 2.0598 and squared correlation 0.948 in our run. With only 5 test cars this number moves noticeably with the seeds; section 7 compares R and Python over many seeds. Adding `--sl weights` to the training command gave RMSE 1.9097 and squared correlation 0.925 in our run, but over 30 seeds the random forest was slightly better (section 7, Table 6).

### 6.3 Using it from Python

```python
import pandas as pd
from qrfsl import QRFSL

train = pd.read_csv("data/mtcars_train.csv")
test = pd.read_csv("data/mtcars_test.csv")

model = QRFSL(k=10, seed=1, full_seed=2019)  # sl="weights" for the weighted average super learner
model.fit(train.drop(columns=["car", "mpg"]), train["mpg"])

pred, qpred = model.predict(test, return_quantiles=True)  # extra columns in test are ignored
```

### 6.4 Notes

- A saved model is a pickle file (joblib). Load only model files you trust, and load them with the same scikit-learn version that was used for training. `predict.py` needs `qrfsl.py` in the same folder.
- The same seeds give the same model and predictions, whatever `n_jobs` is.
- Model size grows with the number of training rows (500 QRF trees plus 1,000 super learner trees; with `sl="weights"` the super learner is only 21 numbers). The mtcars model is 1.5 MB.

## 7. R and Python results compared

R and numpy draw different random numbers, so the same seed does not give the same model in the two languages. We therefore ran both many times and compared averages. Each run used `QRFSL(k = 10, seed = s)` and prediction seed `1000 + s`. Versions: R 4.5.3, quantregForest 1.4-0, randomForest 4.7-1.2, and the Python versions listed in section 6.

**mtcars**, the training and test split of the R example, 30 runs (s = 1, ..., 30):

Table 2. Prediction per test car, mean over 30 runs with standard deviation in brackets (unit: mpg)

| Test car | True | R | Python |
|:--|--:|--:|--:|
| Toyota Corona | 21.5 | 24.16 (0.75) | 24.08 (0.84) |
| Cadillac Fleetwood | 10.4 | 13.80 (0.25) | 13.87 (0.32) |
| Valiant | 18.1 | 19.81 (0.50) | 19.99 (0.47) |
| Ferrari Dino | 19.7 | 18.87 (0.41) | 18.93 (0.41) |
| Merc 240D | 24.4 | 23.58 (0.67) | 23.57 (0.55) |

Table 3. mtcars test RMSE over 30 runs (unit: mpg)

| | Mean | Standard deviation | Minimum | Maximum |
|:--|--:|--:|--:|--:|
| R | 2.196 | 0.234 | 1.728 | 2.949 |
| Python | 2.227 | 0.219 | 1.933 | 2.902 |

**Friedman #1**, a standard synthetic regression problem: scikit-learn `make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=0)`, first 300 rows for training and last 200 for testing (target standard deviation 4.89), 5 runs (s = 1, ..., 5):

Table 4. Friedman #1 test RMSE over 5 runs (target has no unit)

| | Mean | Standard deviation |
|:--|--:|--:|
| R | 2.040 | 0.041 |
| Python | 2.053 | 0.047 |

The run-averaged predictions of R and Python have a correlation of 0.997, and their mean absolute difference is 0.32.

**QRF alone** on Friedman #1 (quantregForest in R, `QuantileRegressionForest` in Python, default settings, seeds 1 to 5), predicting the 10 %, 50 % and 90 % quantiles:

Table 5. Share of the 200 test rows whose true value lies below each predicted quantile (mean over 5 runs, 0 to 1)

| Predicted quantile | R | Python |
|:--|--:|--:|
| 10 % | 0.026 | 0.024 |
| 50 % | 0.479 | 0.481 |
| 90 % | 0.989 | 0.990 |

The mean distance between the 10 % and 90 % quantiles is 9.765 in both. The intervals are wider than their nominal level on this data in both implementations, so this comes from the method and its settings, not from the port.

In short, the Python port is not identical to the R code run by run, but its predictions and errors match R within the spread caused by the random seeds.

**Random forest and weights super learners** (Python only; the same data, seeds and test rows as above; `sl="weights"` has no R counterpart):

Table 6. mtcars test RMSE over 30 runs (unit: mpg)

| Super learner | Mean | Standard deviation | Minimum | Maximum |
|:--|--:|--:|--:|--:|
| rf | 2.227 | 0.219 | 1.933 | 2.902 |
| weights | 2.275 | 0.229 | 1.893 | 2.846 |

Table 7. Friedman #1 test RMSE over 5 runs (target has no unit)

| Super learner | Mean | Standard deviation | Minimum | Maximum |
|:--|--:|--:|--:|--:|
| rf | 2.053 | 0.047 | 1.981 | 2.098 |
| weights | 2.302 | 0.014 | 2.285 | 2.319 |

The weights super learner gave non-zero weight to 5 of the 21 quantiles on average (3 to 8 per run on mtcars, 4 to 6 on Friedman #1). Using the 50 % quantile (the median) alone gave a mean RMSE of 2.512 on mtcars and 2.346 on Friedman #1, so the weighted average beats the median alone but not the random forest.

## 8. Does QRFSL help, and for what? (added after the port)

The R code was written in 2019 with the intuition that feeding a model's whole predictive distribution to a second stage must carry more information than one point prediction. After porting it we tested that intuition. This section records the results, the reasoning, and the version of the idea that survives: **a decision value under an asymmetric cost, with an interval that has a coverage guarantee**. All code is in [python/conformal.py](python/conformal.py), [python/decision.py](python/decision.py) and [python/experiments/](python/experiments/); every table below can be regenerated with the commands in 8.7.

### 8.1 What the second stage can and cannot do

A QRF builds, for a new row, a weighted empirical distribution of the training targets, with the weights of the forest. The mean of that distribution is the forest's own prediction (in Meinshausen's definition of the forest; randomForest's bootstrap-leaf means differ slightly). So when the second stage is trained with squared error, its target is the conditional mean, and the forest already provides that mean. The only thing left for the second stage to learn is a correction of the forest's bias. This explains the benchmark below: QRFSL helps where the forest is biased (a smooth signal, few rows), and there a better base model helps more.

### 8.2 Benchmark of the original method as a point predictor

1,000 training rows, 2,000 test rows, 3 seeds; `python -m experiments.run_benchmark`. Numbers are the test RMSE divided by that of a plain random forest (mean prediction); below 1 is better. Differences under about 2% are within the seed-to-seed spread.

Table 8. Test RMSE relative to the plain random forest (no unit)

| Model | Friedman, normal noise | Friedman, heavy-tailed noise | Friedman, noise grows with x | Log-normal target | California housing | Diabetes (n = 265) | Heavy rain synthetic damage |
|:--|--:|--:|--:|--:|--:|--:|--:|
| QRFSL (rf) | 0.91 | 1.00 | 0.98 | 1.05 | 0.99 | 1.06 | 1.01 |
| QRFSL (weights) | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.98 |
| QRF median | 1.02 | 1.00 | 1.01 | 1.05 | 1.00 | 1.03 | 1.01 |
| Random forest on log y, exp back | 1.03 | 1.01 | 1.04 | 1.04 | 1.05 | 1.05 | 1.00 |
| GBM (HistGradientBoosting) | 0.74 | 0.97 | 0.93 | 1.03 | 0.98 | 1.04 | 1.05 |

Table 9. What the second stage uses: the same stacking with fewer inputs (RMSE relative to the plain random forest)

| Second-stage input | Friedman, normal noise | California housing | Diabetes | Heavy rain synthetic damage |
|:--|--:|--:|--:|--:|
| Out-of-fold RF mean only (1 input) | 1.03 | 1.13 | 1.17 | 1.17 |
| Median, spread, skew (3 inputs) | 0.97 | 1.02 | 1.07 | 1.04 |
| All 21 quantiles (= QRFSL) | 0.91 | 0.99 | 1.06 | 1.01 |

Reading: QRFSL (rf) beats the forest clearly only on the smooth Friedman signal, where GBM gains far more; it loses on the skewed target and on small data, where the second-stage forest overfits. The "weights" variant is safe but gains at most 2 to 3%. Using all 21 quantiles works better than three summaries and much better than one number, mainly because a forest on one input is a poor smoother. As a general accuracy booster the method is not worth its cost.

### 8.3 The version that survives: decisions and guaranteed intervals

Two things a predictive distribution can give that a mean cannot:

**A decision value under an asymmetric cost.** If under-predicting costs c_under per unit and over-predicting costs c_over per unit, the decision with the lowest expected cost is the quantile at level tau = c_under / (c_under + c_over). With a fixed penalty on top (for example when a budget is exceeded), the best decision is no longer one quantile; it depends on the shape of the distribution. [decision.py](python/decision.py) offers four routes from QRF output to a decision:
- *plug-in*: compute the expected cost of every atom of the QRF distribution and take the smallest (no learning);
- *QRFSL pinball stage*: boosting with the pinball loss at tau on the 21 out-of-fold quantiles (the original idea with the loss changed);
- *QRFSL cost-weight stage*: a weighted average of the 21 quantiles whose weights minimise the empirical cost on the out-of-fold rows (works for any cost);
- baselines: the raw tau-quantile of the QRF, the same quantile after a one-sided conformal shift, LightGBM fitted directly with the pinball loss, and the median plus one constant shift.

**An interval with a coverage guarantee (CQR).** Conformalized quantile regression (Romano, Patterson and Candès, 2019) takes the QRF's 5% and 95% quantiles, measures on calibration rows how far each y falls outside [q_lo(x), q_hi(x)], and widens (or narrows) every interval by the (1 - alpha)(1 + 1/n) quantile of those distances. The resulting interval covers a new y with probability at least 1 - alpha whatever the quantile model is, as long as the rows are exchangeable; the model's quality only decides the width. Because the shift is one constant, the width still varies with x as the QRF quantiles do, unlike a conformal interval built from residuals of a point prediction, which has the same width everywhere. [conformal.py](python/conformal.py) implements the split version (held-out calibration rows, exact guarantee), an asymmetric version (separate lower and upper shifts) and a version that reuses QRFSL's out-of-fold quantiles as calibration scores (no extra data; no formal guarantee, see 8.4.3).

### 8.4 Theory behind the pieces

This section gives the reasoning behind every method in 8.3 and the results in 8.6. Notation: training rows $(x_i, y_i)$, $i = 1, \dots, n$; a new row $x$ with unknown $y$; $F(\cdot \mid x)$ the true conditional distribution function of $y$ given $x$ and $q_\tau(x)$ its $\tau$-quantile.

**8.4.1 The QRF predictive distribution.** Each of the $T$ trees puts the new row in one leaf $L_t(x)$. Training row $i$ gets the weight

$$w_i(x) = \frac{1}{T} \sum_{t=1}^{T} \frac{\mathbf{1}\{x_i \in L_t(x)\}}{|L_t(x)|}, \qquad w_i(x) \ge 0, \quad \sum_i w_i(x) = 1,$$

and the QRF estimates the conditional distribution and its quantiles by

$$\hat F(y \mid x) = \sum_i w_i(x)\, \mathbf{1}\{y_i \le y\}, \qquad \hat q_\tau(x) = \inf\{\, y : \hat F(y \mid x) \ge \tau \,\}.$$

Meinshausen (2006) shows that $\hat F$ is consistent under regularity conditions. The mean of $\hat F$, $\sum_i w_i(x) y_i$, is the forest's own prediction (in this definition of the forest; R's randomForest averages only the bootstrap draws of each leaf, so it differs slightly). `leaf_sample="all"` computes these weights exactly; `leaf_sample="one"` is the approximation of R's quantregForest, which stores one random $y$ per leaf and tree.

*Consequence for the original QRFSL (8.1).* A second stage $g$ trained with squared error on the quantile vector $\hat{\mathbf q}(x)$ estimates $E[y \mid \hat{\mathbf q}(x)]$. If $\hat F$ were exact, this would equal the mean of $\hat F$, which the forest already gives; $g$ therefore only corrects the forest's bias. This is why it helps where the forest is biased and nowhere else.

*Why the second stage uses out-of-fold quantiles.* For a training row, its own $y_i$ sits in the leaves that build $\hat F(\cdot \mid x_i)$, so its in-sample quantiles are too close to $y_i$. A second stage trained on them would learn a correction that does not hold for new rows. Quantiles from a QRF fitted without that row (10-fold cross-validation) behave like quantiles of new rows; this is the usual rule of stacking (Wolpert, 1992).

**8.4.2 Decisions under a cost.** Choosing a value $d$ before seeing $y$ costs $c(d, y)$; the best decision for row $x$ minimises the expected cost $R(d) = E[c(d, y) \mid x]$.

*Linear asymmetric cost.* With $c(d, y) = c_u \max(y - d, 0) + c_o \max(d - y, 0)$, where $c_u$ is the cost per unit of under-prediction and $c_o$ per unit of over-prediction,

$$R'(d) = -c_u\,\bigl(1 - F(d \mid x)\bigr) + c_o\, F(d \mid x) = 0 \;\Longleftrightarrow\; F(d \mid x) = \frac{c_u}{c_u + c_o} = \tau,$$

so the best decision is the conditional quantile $q_\tau(x)$. Up to the factor $c_u + c_o$, this cost is the pinball (check) loss $\rho_\tau(u) = u\,(\tau - \mathbf{1}\{u < 0\})$ with $u = y - d$. Minimising the average pinball loss over a class of functions estimates the conditional $\tau$-quantile (Koenker and Bassett, 1978). A cost of 3:1 gives $\tau = 0.75$, 9:1 gives $\tau = 0.9$.

*Penalty cost.* Adding a fixed penalty $p$ whenever $y > d$ gives $R(d) = R_{\text{linear}}(d) + p\,\bigl(1 - F(d \mid x)\bigr)$. The condition for the minimum now contains the density of $y$ at $d$, so the best decision is not a fixed quantile; it depends on the shape of the whole distribution. For a discrete predictive distribution with atoms $a_j$ and weights $w_j$ (the QRF distribution, or the 21 quantiles taken as equal atoms)

$$\hat R(d) = c_u \sum_j w_j \max(a_j - d, 0) + c_o \sum_j w_j \max(d - a_j, 0) + p \sum_{j:\, a_j > d} w_j .$$

This function is piecewise linear and only jumps down at the atoms, so its minimum is at one of the atoms; `expected_cost_decision` evaluates $\hat R$ at every atom and takes the smallest (the plug-in decision).

*The second stages.* The pinball stage fits $g(\hat{\mathbf q}(x))$ by gradient boosting with $\rho_\tau$ on the out-of-fold quantile vectors, so it estimates the $\tau$-quantile of $y$ given the 21 QRF quantiles. If the QRF were calibrated, this would return $\hat q_\tau(x)$ itself; when it is not (8.6: on bike, california and friedman_hetero only 4 to 5% of the rows exceed the raw 0.9-quantile instead of 10%), $g$ moves each row by its own amount. The conformal shift of 8.4.3 instead moves every row by the same constant, which fixes the average level but not the row-to-row errors; this is why the pinball stage lowers the cost more. The cost-weight stage uses $d = \sum_k v_k\, \hat q_{p_k}(x)$ with weights $v_k \ge 0$, $\sum_k v_k = 1$, chosen to minimise the average cost on the out-of-fold rows. A weighted average of quantiles works for any cost and moves with the data: adding a constant to $y$ adds the same constant to $d$.

**8.4.3 Conformal calibration (CQR).** Conformal prediction needs one assumption: the calibration rows and the new row are *exchangeable*, that is, their joint distribution does not change when they are put in another order (true for independent rows from one distribution). The quantile model $\hat q_{lo}, \hat q_{hi}$ is fitted on other rows and then held fixed.

1. For the $n$ calibration rows compute the scores $E_i = \max\bigl(\hat q_{lo}(x_i) - y_i,\; y_i - \hat q_{hi}(x_i)\bigr)$: negative inside the interval, positive outside. The new row's score $E$ is defined the same way.
2. By exchangeability, the rank of $E$ among the $n + 1$ scores is equally likely to be any of $1, \dots, n+1$. Hence $E$ is at most the $k$-th smallest calibration score with probability at least $k / (n + 1)$.
3. Take $k = \lceil (n+1)(1-\alpha) \rceil$ and let $Q$ be the $k$-th smallest calibration score. Then $P(E \le Q) \ge 1 - \alpha$.
4. $E \le Q$ is the same as $\hat q_{lo}(x) - Q \le y \le \hat q_{hi}(x) + Q$. So the interval $[\hat q_{lo}(x) - Q,\; \hat q_{hi}(x) + Q]$ covers the new $y$ with probability at least $1 - \alpha$; if the scores have no ties, also at most $1 - \alpha + 1/(n+1)$ (Romano, Patterson and Candès, 2019).

Nothing in this argument requires $\hat q_{lo}$ and $\hat q_{hi}$ to be accurate. A poor quantile model still gives the promised coverage; a good one gives a smaller $|Q|$, that is, a narrower interval. When the QRF intervals are too wide, $Q$ is negative and the interval shrinks (8.6: 2.27 to 1.80 standard deviations on friedman_hetero).

*Other versions in conformal.py.*
- Asymmetric: the lower scores $\hat q_{lo}(x_i) - y_i$ and the upper scores $y_i - \hat q_{hi}(x_i)$ are calibrated separately, each at level $\alpha/2$. Each end is missed with probability at most $\alpha/2$, so both together at most $\alpha$.
- One-sided shift for a decision: $s$ is the $\lceil (n+1)\tau \rceil$-th smallest of $y_i - \hat q_\tau(x_i)$, which gives $P\bigl(y \le \hat q_\tau(x) + s\bigr) \ge \tau$.
- Residual conformal (baseline): scores $|y_i - \hat\mu(x_i)|$ around a point prediction $\hat\mu$; same guarantee, but the interval has the same width $2Q$ for every $x$.

*What the guarantee does not cover.* It is a guarantee on average over new rows (marginal coverage). Coverage for each value of $x$ (conditional coverage) cannot be guaranteed without assumptions on the distribution: any method that achieves it for all distributions must give intervals of infinite expected length (Vovk, 2012; Lei and Wasserman, 2014; Barber, Candès, Ramdas and Tibshirani, 2021b). In 8.6 this shows as 93% coverage in the quietest fifth of friedman_hetero and 82% in the noisiest.

*The cv version has no formal guarantee.* It takes the scores from the out-of-fold quantiles (each from a QRF fitted on 90% of the training rows) but applies $Q$ to the QRF refitted on all training rows. The new row's score and the calibration scores then come from different models, so they are not exchangeable and step 2 fails. CV+ (Barber, Candès, Ramdas and Tibshirani, 2021a), extended to quantile-based scores by Gupta, Kuchibhotla and Ramdas (2022), restores a guarantee of at least $1 - 2\alpha$ by applying the ten fold models, not the refitted one, to the new row; it is not implemented here. In the experiments the cv version still covered 0.897 to 0.911 of the test rows, the same as the split version.

### 8.5 Experiment design

Five data sets, 10 random splits each: 2,000 training rows, 1,000 calibration rows, 3,000 test rows (fewer when the data set is smaller). QRF with 500 trees and the full weighted distribution (`leaf_sample="all"`), 10 folds.

| Data | Rows | Why it is here |
|:--|--:|:--|
| friedman_hetero | synthetic | noise sd grows from 0.2 to 4.2 with x0; tests whether the interval width follows the noise |
| lognormal | synthetic | right-skewed target |
| california | 20,640 | standard tabular regression, house value in 100k USD |
| bike | 17,379 | hourly bike rentals; shortage and surplus have different costs |
| claims | 24,944 | French motor claim amounts per policy, capped at 50,000 EUR; heavy right tail |

Roles of the rows:

| Rows | Used for | Role |
|:--|:--|:--|
| training rows, out-of-fold quantiles | second stages, conformal shift of the "cv" variant, median shift | validation inside training |
| calibration rows | shifts of the split conformal variants | calibration |
| test rows | coverage, width, mean cost; never used for any choice | evaluation |

Costs: linear 1:1 (tau 0.5), 3:1 (tau 0.75), 9:1 (tau 0.9), and linear 1:1 plus a fixed penalty of half the training standard deviation of y whenever y exceeds the decision.

### 8.6 Results

**Intervals (nominal coverage 90%).** `experiments/results/intervals.csv`. Means over 10 splits; the spread between splits is at most 0.02 in coverage.

Table 10. Coverage of the 90% intervals (share of test rows inside, 0 to 1)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF raw (5%, 95%) | 0.963 | 0.966 | 0.895 | 0.956 | 0.921 |
| QRF + CQR split | 0.907 | 0.898 | 0.896 | 0.895 | 0.894 |
| QRF + CQR split, asymmetric | 0.912 | 0.932 | 0.895 | 0.896 | 0.895 |
| QRF + CQR cv (no calibration set) | 0.911 | 0.904 | 0.905 | 0.897 | 0.900 |
| RF point +- residual conformal | 0.900 | 0.892 | 0.898 | 0.892 | 0.892 |
| LightGBM quantiles raw | 0.715 | 0.702 | 0.729 | 0.693 | 0.710 |
| LightGBM + CQR split | 0.896 | 0.893 | 0.896 | 0.894 | 0.894 |

Table 11. Mean interval width divided by the standard deviation of the training target (no unit)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF raw (5%, 95%) | 1.67 | 1.93 | 1.53 | 2.27 | 2.02 |
| QRF + CQR split | 1.64 | 1.83 | 1.53 | 1.80 | 1.99 |
| QRF + CQR split, asymmetric | 1.54 | 1.86 | 1.57 | 1.80 | 1.85 |
| QRF + CQR cv (no calibration set) | 1.64 | 1.85 | 1.53 | 1.81 | 1.99 |
| RF point +- residual conformal | 1.45 | 1.49 | 1.39 | 1.73 | 1.87 |
| LightGBM quantiles raw | 0.77 | 0.97 | 1.12 | 1.06 | 1.16 |
| LightGBM + CQR split | 0.96 | 1.33 | 1.34 | 1.58 | 1.40 |

Table 12. friedman_hetero: coverage in the fifth of test rows with the smallest and with the largest true noise scale (0 to 1)

| Method | Quietest fifth | Noisiest fifth |
|:--|--:|--:|
| QRF raw (5%, 95%) | 0.988 | 0.894 |
| QRF + CQR split | 0.928 | 0.817 |
| QRF + CQR cv | 0.934 | 0.817 |
| RF point +- residual conformal | 0.998 | 0.723 |
| LightGBM + CQR split | 0.968 | 0.818 |

Reading:
- Raw QRF intervals are too wide (coverage 0.92 to 0.97 instead of 0.90, except on claims); raw LightGBM quantiles are far too narrow (0.69 to 0.73). CQR brings every method to 0.89 to 0.91, as the guarantee says, and for QRF it also shrinks the width (friedman_hetero: 2.27 to 1.80 standard deviations).
- At the same coverage, QRF + CQR gives the widest intervals: 1.1 to 1.7 times those of LightGBM + CQR and 1.04 to 1.23 times those of the constant-width interval around the forest's point prediction. The forest's quantiles are noisy; CQR fixes their level, not their sharpness.
- The width follows the noise only partly. Where the noise varies (friedman_hetero), the constant-width interval covers 99.8% of the quietest fifth and 72% of the noisiest; QRF + CQR covers 93% and 82%. On the heavy-tailed claims data the narrowest fifth of the QRF + CQR intervals still covers 87% of its rows, LightGBM + CQR 77%. No method reaches 90% in every part of the data; CQR guarantees the average only.
- The cv variant, which reuses QRFSL's out-of-fold quantiles instead of a calibration set, gives the same coverage (0.90 to 0.91) and width as the split variant. This is the one practical advantage QRFSL's structure brings to intervals: no data has to be set aside.

In short: always conformalize; QRF + CQR is valid, simple and needs no calibration set, but a conformalized boosting quantile model is sharper at the same coverage.

**Decisions under asymmetric costs.** `experiments/results/decisions.csv`. Means over 10 splits. Numbers are the mean test cost divided by that of the raw QRF tau-quantile (the natural answer for a linear cost); below 1 is better. The spread between splits is at most 0.02 at 3:1 and 0.04 at 9:1 for the learned methods.

Table 13. Mean cost relative to the raw QRF tau-quantile, linear cost 3:1 (tau 0.75, no unit)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF mean (ignores the cost) | 1.27 | 1.17 | 1.06 | 1.19 | 1.02 |
| QRF median + one shift | 1.18 | 1.06 | 0.99 | 1.01 | 1.07 |
| QRF tau-quantile raw | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF tau-quantile + conformal shift | 0.99 | 0.99 | 1.00 | 0.98 | 1.00 |
| QRFSL cost-weight stage | 0.97 | 0.98 | 1.00 | 0.98 | 1.00 |
| QRFSL pinball stage | 0.94 | 0.96 | 1.02 | 0.92 | 1.00 |
| LightGBM direct tau-quantile | 0.63 | 0.94 | 1.03 | 0.88 | 1.00 |
| LightGBM plug-in (21 quantiles) | 0.61 | 0.92 | 1.02 | 0.86 | 0.99 |

Table 14. Mean cost relative to the raw QRF tau-quantile, linear cost 9:1 (tau 0.9, no unit)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF mean (ignores the cost) | 1.82 | 1.75 | 1.02 | 1.97 | 1.33 |
| QRF median + one shift | 1.23 | 1.11 | 0.97 | 0.99 | 1.17 |
| QRF tau-quantile raw | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF tau-quantile + conformal shift | 0.96 | 0.99 | 1.00 | 0.95 | 1.00 |
| QRFSL cost-weight stage | 0.87 | 0.93 | 0.98 | 0.94 | 0.99 |
| QRFSL pinball stage | 0.82 | 0.92 | 0.98 | 0.88 | 0.99 |
| LightGBM direct tau-quantile | 0.63 | 1.00 | 1.03 | 0.93 | 1.05 |
| LightGBM plug-in (21 quantiles) | 0.58 | 0.97 | 1.00 | 0.89 | 1.03 |

Table 15. Share of test rows whose y exceeds the decision, linear cost 9:1 (the cost-optimal share is 0.10)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF tau-quantile raw | 0.043 | 0.043 | 0.099 | 0.053 | 0.083 |
| QRF tau-quantile + conformal shift | 0.099 | 0.095 | 0.097 | 0.101 | 0.101 |
| QRFSL pinball stage | 0.114 | 0.120 | 0.118 | 0.121 | 0.119 |
| LightGBM direct tau-quantile | 0.188 | 0.191 | 0.166 | 0.213 | 0.187 |

Table 16. Mean cost relative to the plug-in decision under the full QRF distribution, linear 1:1 plus a penalty of 0.5 sd when y exceeds the decision (no unit)

| Method | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF mean (ignores the cost) | 1.10 | 1.09 | 1.21 | 1.04 | 1.01 |
| QRF median + one shift | 1.02 | 1.02 | 1.00 | 0.99 | 1.00 |
| QRF plug-in (all atoms) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF plug-in (21 quantiles) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRFSL cost-weight stage | 1.00 | 1.02 | 1.03 | 0.98 | 1.00 |
| LightGBM plug-in (21 quantiles) | 0.73 | 0.96 | 1.07 | 0.89 | 1.00 |

Reading:
- The raw QRF quantile is too cautious: at tau 0.9 only 4 to 5% of the test rows exceed it instead of 10% (except on claims), the same miscalibration that made the raw intervals too wide. The QRFSL pinball stage corrects this row by row and lowers the cost by 4 to 8% at 3:1 and by 8 to 18% at 9:1 on bike, california and friedman_hetero. The one-constant conformal shift sets the exceedance share exactly to 10% but lowers the cost by only 1 to 5%: the second stage gains because it moves different rows by different amounts. On claims and lognormal, where the raw quantile already sits near its level, nothing changes the cost by more than 2%.
- The cost-weight stage (a weighted average of the 21 quantiles) lies in between (3 to 13% at 9:1) and needs no boosting.
- A boosting quantile model trained directly on x is far better on bike (0.63) and friedman_hetero (0.88 to 0.93) although its exceedance share is off (17 to 21%); it ties on california and is slightly worse on claims and lognormal. The choice of base model matters more than the second stage. The second stage repairs the forest's quantile; it does not make the forest the best model. We did not conformalize or stack the boosting model; combining the two is the obvious next step.
- For the penalty cost, the plug-in decision under the QRF distribution is as good as any learned stage (cost-weight stage 0.98 to 1.03), and the 21 quantiles carry the same information as the full distribution (differences under 0.1%).
- Ignoring the cost altogether (the forest mean) costs 1.3 to 2.0 times as much at 9:1. Using a cost-aware decision at all is the largest effect in these tables.

**Overall.** Of the 2019 idea, this is what holds up: a second stage on the out-of-fold quantiles is a useful conditional recalibration of a forest's tau-quantile when the cost is strongly asymmetric (5 to 18% lower cost than the raw quantile at 9:1, more than a conformal shift), and the out-of-fold quantiles double as free calibration data for valid intervals. What does not hold up: the second stage as a general accuracy booster, and the forest as the best base model. If the task allows a boosting quantile model, train it directly and conformalize it; if a forest is wanted (small data, robustness, an existing pipeline), add the pinball stage for decisions and CQR for intervals.

### 8.7 Running the experiments

```bash
cd python
python -m experiments.run_benchmark --seeds 3 --n-jobs 16      # Tables 8-9
python -m experiments.run_intervals --seeds 10 --n-jobs 16     # intervals
python -m experiments.run_decisions --seeds 10 --n-jobs 16     # decisions
```

Results are written to `experiments/results/*.csv`. The public data sets are downloaded through scikit-learn (California housing) and OpenML (bike sharing, French motor claims) on first use. Using the pieces directly:

```python
from qrfsl import QRFSL
from conformal import CQR
from decision import LinearCost, expected_cost_decision

m = QRFSL(k=10, seed=1, sl="none", leaf_sample="all").fit(X_train, y_train)
q = m.predict_quantiles(X_test)                         # 21 quantiles per row
cqr = CQR(alpha=0.1).fit(m.cv_quantiles_["quantile=0.05"], m.cv_quantiles_["quantile=0.95"], m.cv_y_)
lo, hi = cqr.predict(q["quantile=0.05"], q["quantile=0.95"])   # 90% interval, no calibration set
atoms, weights = m.predict_distribution(X_test)
d = expected_cost_decision(atoms, weights, LinearCost(c_under=3, c_over=1))   # decision for a 3:1 cost
```

## 9. References

- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021a). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486–507.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021b). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2), 455–482.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Gupta, C., Kuchibhotla, A. K., & Ramdas, A. (2022). Nested conformal prediction and quantile out-of-bag ensemble methods. *Pattern Recognition*, 127, 108496.
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
- Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric regression. *Journal of the Royal Statistical Society: Series B*, 76(1), 71–96.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.
- van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1), Article 25.
- Vovk, V. (2012). Conditional validity of inductive conformal predictors. *Proceedings of the Asian Conference on Machine Learning*, PMLR 25, 475–490.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- R packages: quantregForest (Meinshausen and Michel), randomForest (Liaw and Wiener).
