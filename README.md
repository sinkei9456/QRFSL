# QRFSL: Quantile Regression Forest Super Learner

[한국어 문서](README_ko.md)

QRFSL is a regression method that predicts a numeric target in two stages.

1. A **quantile regression forest (QRF)** predicts 21 quantiles of the target for each row: the 2.5 %, 5 %, 10 %, ..., 95 % and 97.5 % points of the predicted distribution.
2. A **super learner** takes those 21 quantiles as inputs and returns the final prediction. In the R code it is a random forest; the Python port also offers a weighted average of the quantiles (section 5.4).

The super learner is trained on *cross-validated* quantiles: every training row is predicted by a QRF that did not see that row. The super learner therefore learns how the quantiles relate to the true value on data the QRF has not memorised.

This repository holds the original R code ([Rcode_QRFSL.R](Rcode_QRFSL.R)) and a Python port ([python/](python/)) split into a training script and an inference script.

## Contents

1. [Repository layout](#1-repository-layout)
2. [How the algorithm works](#2-how-the-algorithm-works)
3. [Walk-through of Rcode_QRFSL.R](#3-walk-through-of-rcode_qrfslr)
4. [Details that are easy to miss in the R code](#4-details-that-are-easy-to-miss-in-the-r-code)
5. [Python port](#5-python-port)
6. [Running the Python version](#6-running-the-python-version)
7. [R and Python results compared](#7-r-and-python-results-compared)
8. [References](#8-references)

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
├── README_ko.md                Korean version
└── Rcode_QRFSL.R               original R code
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

### 3.1 Packages (lines 5–8)

`quantregForest` (which also attaches `randomForest`), `doParallel` and `foreach`.

### 3.2 Training: `QRFSL()` (lines 12–107)

| Argument | Default | Meaning |
|:--|:--|:--|
| `x` | none | features (matrix or data frame) |
| `y` | none | numeric target vector |
| `k` | 10 | number of cross-validation folds |
| `prob` | 0.025, 0.05, 0.10, ..., 0.95, 0.975 (21 values) | quantile levels |
| `seed` | 0 | random seed for the fold split and the super learner |
| `mc` | 1 | number of parallel workers; folds run in parallel when above 1 |
| `sl` | 0 | documented as "0: weights, 1: rf" but never used; the code always fits a random forest (section 4) |

1. **Fold split (lines 32–39).** `cv.folds()` shuffles the row numbers 1, ..., n and deals them to folds 1, 2, ..., k, 1, 2, ... in turn (`split(sample(1:n), rep(1:folds, length = n))`), so fold sizes differ by at most one row. With the 27 training rows of the example and k = 10, seven folds have 3 rows and three folds have 2.
2. **Fold data (lines 42–46).** For fold i, `tdat_x_cv[[i]]` and `tdat_y_cv[[i]]` hold every row except fold i (used for fitting); `idat_x_cv[[i]]` and `idat_y_cv[[i]]` hold the rows of fold i (used for predicting).
3. **Cross-validated quantiles (lines 49–79).** For each fold: `set.seed(i)`, fit `quantregForest()` on the fitting part with the package defaults, then predict the 21 quantiles of the held-out rows. With `mc > 1` the folds run on a `doParallel` cluster. Because every fold sets its own seed, `mc = 1` and `mc = 2` give identical results (checked).
4. **Stacking (lines 80–86).** The fold predictions are stacked into an n x 21 matrix `pred_cv`. The column names returned by quantregForest, such as `quantile= 0.025`, lose their spaces and become `quantile=0.025`. The true targets are stacked in the same fold order into `yact_cv`.
5. **Super learner (lines 89–96).** `set.seed(seed)`, then `randomForest(x = pred_cv, y = yact_cv, ntree = 1000)`. All other settings are the package defaults: mtry = floor(21 / 3) = 7 candidate inputs per split, nodesize = 5.
6. **Return value (lines 99–106).** `list(sl, x, y, prob)`. The full-data QRF is not fitted here; the training data is kept so that `predict.QRFSL()` can fit it.

### 3.3 Prediction: `predict.QRFSL()` (lines 111–154)

Arguments: `object` (the list returned by `QRFSL()`), `newx` (new features), `seed` (default 0).

1. `set.seed(seed)`, then fit `quantregForest()` on all training rows `object$x`, `object$y`. This is `qmod`.
2. Predict the quantiles `object$prob` of `newx` with `qmod`, and remove the spaces from the column names. This is `qpred`.
3. Feed `qpred` to the super learner `object$sl`. This is `pred`.
4. Return `list(qmod, qpred, pred)`.

### 3.4 Example (lines 160–206)

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

1. **The `sl` argument does nothing.** Line 90 (`sl <- suppressWarnings(randomForest(...))`) overwrites it. Only the random forest super learner exists; the "weights" option is not implemented. The Python port adds it as an option (section 5.4).
2. **Fold models always use seeds 1, ..., k.** `set.seed(i)` on lines 52 and 67 does not depend on `seed`. `seed` controls only the fold split (line 35) and the super learner (line 89). The full-data QRF is controlled by the `seed` of `predict.QRFSL()`.
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

## 8. References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1), Article 25.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- R packages: quantregForest (Meinshausen and Michel), randomForest (Liaw and Wiener).
