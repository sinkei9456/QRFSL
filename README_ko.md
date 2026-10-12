# QRFSL: 분위수 회귀 포레스트 기반 슈퍼 러너

[English](README.md)

QRFSL 은 숫자로 된 목표값을 두 단계로 예측하는 회귀 방법입니다.

1. **분위수 회귀 포레스트**(Quantile Regression Forest, 이하 QRF)가 행마다 목표값의 분위수 21개를 예측합니다. 예측한 분포의 2.5 %, 5 %, 10 %, ..., 95 %, 97.5 % 지점입니다.
2. **슈퍼 러너**가 이 분위수 21개를 입력으로 받아 최종 예측값을 냅니다. R 코드에서는 랜덤 포레스트이고, 파이썬 버전에서는 분위수의 가중 평균도 고를 수 있습니다(5.4절).

슈퍼 러너는 *교차검증으로 만든* 분위수로 학습합니다. 즉 학습 데이터의 각 행을, 그 행을 학습에 쓰지 않은 QRF 로 예측한 값입니다. 그래서 슈퍼 러너는 QRF 가 외워 버린 데이터가 아니라 처음 보는 데이터에서 분위수와 실제값이 어떻게 이어지는지를 배웁니다.

이 저장소에는 원본 R 코드를 학습 코드와 추론 코드로 나누어 옮긴 파이썬 버전([python/](python/))이 있습니다. R 코드(`Rcode_QRFSL.R`)는 현재 파일에서는 지웠고 커밋 [1fea660](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R) 기록에 남아 있으며, 3장과 4장의 행 번호는 그 파일로 연결됩니다.

## 목차

1. [폴더 구성](#1-폴더-구성)
2. [알고리즘 흐름](#2-알고리즘-흐름)
3. [Rcode_QRFSL.R 코드 설명](#3-rcode_qrfslr-코드-설명)
4. [R 코드에서 놓치기 쉬운 점](#4-r-코드에서-놓치기-쉬운-점)
5. [파이썬 버전](#5-파이썬-버전)
6. [파이썬 버전 실행 방법](#6-파이썬-버전-실행-방법)
7. [R 과 파이썬 결과 비교](#7-r-과-파이썬-결과-비교)
8. [QRFSL 은 쓸모가 있는가, 있다면 어디에](#8-qrfsl-은-쓸모가-있는가-있다면-어디에-이식-후-추가)
9. [참고 문헌](#9-참고-문헌)

## 1. 폴더 구성

```
QRFSL/
├── python/
│   ├── data/
│   │   ├── mtcars_test.csv     R 예제의 테스트용 자동차 5대
│   │   └── mtcars_train.csv    R 예제의 학습용 자동차 27대
│   ├── predict.py              추론: 저장된 모델을 불러 CSV 를 예측
│   ├── qrfsl.py                공통 코드: 랜덤 포레스트, QRF, QRFSL 클래스
│   ├── requirements.txt
│   └── train.py                학습: CSV 로 QRFSL 을 학습하고 모델 저장
├── README.md                   영문 문서
└── README_ko.md                이 문서(한국어)
```

아래 6장의 예시 명령을 실행하면 `python/model/`(저장된 모델)과 `python/output/`(예측 결과 파일) 폴더가 생깁니다.

## 2. 알고리즘 흐름

```
학습 (행 n 개, 입력 변수 p 개)
  1. 행을 섞은 뒤 차례로 k 개 폴드에 나눠 담음 (기본 k = 10)
  2. 폴드 i = 1, ..., k 마다
       나머지 k-1 개 폴드로 QRF 를 학습
       폴드 i 에 든 행의 분위수 21개를 예측
     -> 교차검증 분위수 표 (n 행 x 21 열)
  3. 랜덤 포레스트(나무 1,000 그루) 학습: 분위수 21개 -> y   (슈퍼 러너)
  4. 전체 n 행으로 QRF 를 하나 더 학습                     (전체 데이터 QRF)

예측 (새 행)
  전체 데이터 QRF -> 행마다 분위수 21개 -> 슈퍼 러너 -> 최종 예측값
```

여기서 폴드(fold)는 k-겹 교차검증에서 데이터를 k 묶음으로 나눈 한 묶음을 말합니다.

**분위수 회귀 포레스트**(Meinshausen, 2006). 랜덤 포레스트를 평소처럼 키우되, 끝 노드(leaf)에 목표값의 평균만 두지 않고 학습 행의 목표값 자체를 남겨 둡니다. 새 행이 들어오면 모든 나무에서 그 행이 도착한 끝 노드의 목표값을 모아 분위수를 읽습니다. 이 코드가 쓰는 R 패키지 quantregForest(버전 1.4-0)는 이를 빠르게 근사합니다. 나무마다, 끝 노드마다 그 노드에 도착한 학습 행 가운데 **하나**를 무작위로 골라 그 목표값만 저장합니다. 새 행의 예측은 나무마다 하나씩 모은 저장값 500개의 표본 분위수(R `quantile` 의 type 7 방식)입니다.

**슈퍼 러너**(van der Laan 외, 2007)는 스태킹(stacking)의 한 방식입니다. 1단계 모델들의 예측을 어떻게 합칠지를 2단계 모델이 배우는데, 이때 교차검증으로 만든 예측을 써서 처음 보는 데이터 기준으로 합치는 법을 배웁니다. QRFSL 에서 1단계 예측은 QRF 하나가 낸 분위수 21개이고, R 코드에서 합치는 모델은 가중 평균이 아니라 랜덤 포레스트입니다. 파이썬 버전은 가중 평균을 선택 옵션으로 더했습니다(`sl="weights"`, 5.4절).

분위수 21개를 모두 슈퍼 러너에 넣으면 평균이나 중앙값 같은 값 하나만 쓰는 대신, 예측 분포 전체(퍼진 정도, 한쪽으로 치우친 정도 등)를 활용할 수 있습니다.

## 3. Rcode_QRFSL.R 코드 설명

아래 행 번호는 [커밋 1fea660 의 Rcode_QRFSL.R](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R) 기준입니다.

### 3.1 패키지 불러오기 ([5–8행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L5-L8))

`quantregForest`(불러올 때 `randomForest` 도 함께 붙음), `doParallel`, `foreach`.

### 3.2 학습 함수 `QRFSL()` ([12–107행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L12-L107))

| 인자 | 기본값 | 뜻 |
|:--|:--|:--|
| `x` | 없음 | 입력 변수(행렬 또는 데이터 프레임) |
| `y` | 없음 | 숫자로 된 목표값 벡터 |
| `k` | 10 | 교차검증 폴드 수 |
| `prob` | 0.025, 0.05, 0.10, ..., 0.95, 0.975 (21개) | 예측할 분위수 |
| `seed` | 0 | 폴드 나누기와 슈퍼 러너에 쓰는 난수 시드 |
| `mc` | 1 | 병렬 작업 수. 1보다 크면 폴드를 동시에 학습 |
| `sl` | 0 | 설명은 "0: 가중치, 1: 랜덤 포레스트" 이지만 실제로 쓰이지 않고, 언제나 랜덤 포레스트를 학습함(4장) |

1. **폴드 나누기 ([32–39행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L32-L39)).** `cv.folds()` 는 행 번호 1, ..., n 을 섞은 뒤 폴드 1, 2, ..., k, 1, 2, ... 순서로 돌아가며 나눠 줍니다(`split(sample(1:n), rep(1:folds, length = n))`). 그래서 폴드끼리 크기 차이는 많아야 한 행입니다. 예제처럼 학습 행이 27개이고 k = 10 이면 7개 폴드는 3행, 3개 폴드는 2행입니다.
2. **폴드별 데이터 ([42–46행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L42-L46)).** 폴드 i 에 대해 `tdat_x_cv[[i]]`, `tdat_y_cv[[i]]` 는 폴드 i 를 뺀 나머지 행(학습용), `idat_x_cv[[i]]`, `idat_y_cv[[i]]` 는 폴드 i 의 행(예측용)입니다.
3. **교차검증 분위수 만들기 ([49–79행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L49-L79)).** 폴드마다 `set.seed(i)` 후 학습용 행으로 `quantregForest()` 를 패키지 기본 설정 그대로 학습하고, 빼 둔 행의 분위수 21개를 예측합니다. `mc > 1` 이면 `doParallel` 로 폴드를 동시에 돌립니다. 폴드마다 시드를 따로 정하므로 `mc = 1` 과 `mc = 2` 의 결과는 똑같습니다(직접 확인함).
4. **결과 쌓기 ([80–86행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L80-L86)).** 폴드별 예측을 위아래로 이어 n 행 x 21 열 행렬 `pred_cv` 를 만듭니다. quantregForest 가 붙이는 열 이름 `quantile= 0.025` 에서 빈칸을 지워 `quantile=0.025` 로 바꿉니다. 실제값도 같은 폴드 순서로 이어 `yact_cv` 를 만듭니다.
5. **슈퍼 러너 학습 ([89–96행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L89-L96)).** `set.seed(seed)` 후 `randomForest(x = pred_cv, y = yact_cv, ntree = 1000)`. 나머지 설정은 패키지 기본값입니다. 분기마다 살펴보는 입력 수 mtry = floor(21 / 3) = 7, nodesize = 5.
6. **돌려주는 값 ([99–106행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L99-L106)).** `list(sl, x, y, prob)`. 전체 데이터 QRF 는 여기서 학습하지 않고, `predict.QRFSL()` 이 학습할 수 있도록 학습 데이터를 그대로 담아 둡니다.

### 3.3 예측 함수 `predict.QRFSL()` ([111–154행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L111-L154))

인자: `object`(`QRFSL()` 이 돌려준 목록), `newx`(새 입력 변수), `seed`(기본 0).

1. `set.seed(seed)` 후 전체 학습 행 `object$x`, `object$y` 로 `quantregForest()` 를 학습합니다. 이것이 `qmod` 입니다.
2. `qmod` 로 `newx` 의 분위수 `object$prob` 를 예측하고 열 이름의 빈칸을 지웁니다. 이것이 `qpred` 입니다.
3. `qpred` 를 슈퍼 러너 `object$sl` 에 넣어 최종 예측값 `pred` 를 얻습니다.
4. `list(qmod, qpred, pred)` 를 돌려줍니다.

### 3.4 예제 ([160–206행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L160-L206))

- 데이터: `mtcars`, 자동차 32대, 11개 열. 목표값은 `mpg`(연비, 갤런당 마일), 입력 변수는 나머지 10개 열(`cyl`, `disp`, `hp`, `drat`, `wt`, `qsec`, `vs`, `am`, `gear`, `carb`).
- `set.seed(2)` 와 `sample()` 로 테스트용 5행(행 번호 21, 15, 6, 30, 8)을 고르고, 나머지 27행을 학습 데이터로 씁니다.
- 학습: `QRFSL(k = 10, seed = 1, mc = 1)`. 예측: `predict.QRFSL(seed = 2019)`.
- 평균 제곱근 오차(RMSE)와 `corr` 라는 열을 출력합니다. `corr` 은 실제값과 예측값의 피어슨 상관계수를 **제곱한** 값입니다.

원본 코드를 고치지 않고 우리 환경(R 4.5.3, quantregForest 1.4-0, randomForest 4.7-1.2)에서 돌린 결과입니다.

표 1. R 예제 결과 (단위: mpg)

| 테스트용 자동차 | 실제값 | 예측값 |
|:--|--:|--:|
| Toyota Corona | 21.5 | 24.02 |
| Cadillac Fleetwood | 10.4 | 13.34 |
| Valiant | 18.1 | 19.97 |
| Ferrari Dino | 19.7 | 19.37 |
| Merc 240D | 24.4 | 23.86 |

RMSE 는 1.944 mpg, `corr`(상관계수 제곱)은 0.922 입니다. R 버전에 따라 난수 생성 방식이 바뀐 적이 있어서(예: R 3.6.0 에서 `sample()` 변경) 다른 버전에서는 숫자가 다를 수 있습니다.

## 4. R 코드에서 놓치기 쉬운 점

1. **`sl` 인자는 아무 일도 하지 않습니다.** [90행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L90) `sl <- suppressWarnings(randomForest(...))` 가 이 인자를 덮어씁니다. 슈퍼 러너는 랜덤 포레스트 하나뿐이고, "가중치" 방식은 구현되어 있지 않습니다. 파이썬 버전은 이를 선택 옵션으로 더했습니다(5.4절).
2. **폴드별 모델의 시드는 언제나 1, ..., k 입니다.** [52행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L52)과 [67행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L67)의 `set.seed(i)` 는 `seed` 와 상관이 없습니다. `seed` 는 폴드 나누기([35행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L35))와 슈퍼 러너([89행](https://github.com/sinkei9456/QRFSL/blob/1fea660854e70d047ac88389a87849fa2caefef5/Rcode_QRFSL.R#L89))에만 쓰입니다. 전체 데이터 QRF 의 시드는 `predict.QRFSL()` 의 `seed` 입니다.
3. **QRF 설정은 코드에 적혀 있지 않아 패키지 기본값을 씁니다.** 나무 500 그루, 분기마다 살펴보는 변수 수 mtry = max(floor(p / 3), 1) (mtcars 의 10개 변수면 3), nodesize = 5 (행이 5개 이하인 노드는 더 나누지 않음), 나무마다 n 행을 복원 추출한 부트스트랩 표본.
4. **끝 노드마다 목표값 하나만 저장합니다.** 2장에서 설명한 대로 quantregForest 는 나무마다, 끝 노드마다 목표값 하나를 무작위로 골라 둡니다. 고르는 대상은 그 노드에 도착하는 모든 학습 행이며, 그 나무의 부트스트랩 표본에 뽑히지 않은 행도 포함됩니다.
5. **전체 데이터 QRF 는 예측할 때마다 새로 학습합니다.** 그래서 예측값이 `predict.QRFSL()` 의 `seed` 에 따라 달라지고, 학습 결과 객체에 학습 데이터 `x`, `y` 가 들어 있습니다(mtcars 예제에서 754 KB).
6. **`suppressWarnings()` 가 숨기는 경고.** 슈퍼 러너를 학습할 때 `randomForest()` 안에서 "Recycling array of length 1 in vector-array arithmetic is deprecated" 경고가 납니다. `yact_cv` 가 벡터가 아니라 열이 하나인 행렬이기 때문이며, 결과에는 영향이 없습니다.
7. **`library(randomForest)` 없이 `randomForest()` 를 씁니다.** `quantregForest` 를 불러올 때 `randomForest` 가 함께 붙기 때문에 동작합니다.
8. **열 이름.** `pred_cv`(학습)와 `qpred`(예측) 모두 같은 규칙으로 빈칸을 지우므로, 슈퍼 러너는 학습과 예측에서 같은 열 이름을 봅니다.

## 5. 파이썬 버전

### 5.1 파일

- [python/qrfsl.py](python/qrfsl.py): 두 스크립트가 함께 쓰는 공통 코드.
  - `RForest`: R `randomForest` 와 같은 규칙으로 키우는 회귀용 랜덤 포레스트. scikit-learn 의 `DecisionTreeRegressor` 로 만듦.
  - `QuantileRegressionForest`: `quantregForest()` 와 `predict(what = ...)` 를 옮긴 것.
  - `ConvexWeights`: R 코드에 구현되지 않은 "가중치" 방식 슈퍼 러너(5.4절).
  - `cv_folds()`: `cv.folds()` 를 옮긴 것.
  - `QRFSL`: `fit()` 이 `QRFSL()`, `predict()` 가 `predict.QRFSL()` 에 해당.
- [python/train.py](python/train.py): CSV 를 읽어 QRFSL 을 학습하고 joblib 으로 모델을 저장.
- [python/predict.py](python/predict.py): 저장된 모델을 불러 CSV 를 예측하고 CSV 로 저장. 입력에 실제값이 있으면 R 예제처럼 RMSE 와 상관계수 제곱도 출력.

### 5.2 R 과 파이썬 대응

| R | 파이썬 |
|:--|:--|
| `QRFSL(x, y, k, prob, seed, mc)` | `QRFSL(k, probs, seed, full_seed, n_jobs).fit(X, y)` 또는 `train.py` |
| `predict.QRFSL(object, newx, seed)` | `model.predict(X, return_quantiles=True)` 또는 `predict.py` |
| `predict.QRFSL()` 의 `seed` | `full_seed` (학습할 때 지정) |
| `mc` | `n_jobs` |
| `sl` (0 = 가중치, 1 = 랜덤 포레스트, 코드는 언제나 랜덤 포레스트) | `sl="rf"` (기본값) 또는 `sl="weights"` |
| `quantregForest()` 와 `predict(what = prob)` | `QuantileRegressionForest.fit()` 와 `.predict()` |
| `randomForest()` | `RForest` |
| `cv.folds()` | `cv_folds()` |
| 학습 결과의 `$sl` | `model.sl_` |
| 예측 결과의 `$qmod`, `$qpred`, `$pred` | `model.qmod_`, `model.predict_quantiles(X)`, `model.predict(X)` |

### 5.3 R 코드와 다른 점

| 항목 | R 코드 | 파이썬 버전 |
|:--|:--|:--|
| 전체 데이터 QRF 를 학습하는 때 | `predict.QRFSL()` 을 부를 때마다 | 학습할 때 한 번, 모델 파일에 함께 저장 |
| 모델에 담기는 데이터 | 학습 데이터 `x`, `y` | 입력 변수는 담지 않음. 슈퍼 러너 학습에 쓴 교차검증 분위수와 실제값만 담음(`cv_quantiles_`, `cv_y_`) |
| 난수 | R 난수 생성기 | numpy 난수 생성기. 같은 시드라도 R 과 다른 난수가 나옴 |
| 나무 키우기 | randomForest 의 C 코드 | 부트스트랩으로 뽑은 행을 실제로 복사해 scikit-learn `DecisionTreeRegressor` 로 학습. 부트스트랩, mtry, nodesize 규칙은 같음 |
| 살펴본 mtry 개 변수로 노드를 나눌 수 없을 때 | 그 노드를 끝 노드로 둠 | scikit-learn 은 다른 변수를 더 살펴봄 |
| 범주형 변수 | factor 를 그대로 처리 | 숫자 열만 받음. 범주형은 먼저 숫자로 바꿔야 함 |
| `sl` 인자 | 선언만 있고 쓰이지 않음. 언제나 랜덤 포레스트 | `"rf"` (기본값, R 코드와 같음) 또는 `"weights"` (새로 더함, 5.4절) |
| 병렬 실행 | `doParallel` 로 폴드를 동시에 학습 | joblib: 폴드는 여러 프로세스로, 전체 데이터 QRF 와 슈퍼 러너의 나무는 여러 스레드로 |

scikit-learn 의 `RandomForestRegressor` 대신 `DecisionTreeRegressor` 로 나무를 직접 만든 이유: `RandomForestRegressor` 는 부트스트랩에서 여러 번 뽑힌 행을 가중치로 처리하기 때문에 노드 크기를 셀 때 서로 다른 행의 수만 셉니다. R 은 뽑힌 횟수만큼 모두 셉니다. 같은 nodesize 규칙에서도 scikit-learn 쪽 나무가 조금 작아집니다. 처음에 `RandomForestRegressor` 로 만든 버전은 mtcars RMSE 평균이 2.368(R 2.196), Toyota Corona 예측 평균이 24.63(R 24.16)이었습니다. 나무마다 뽑힌 행을 실제로 복사해 학습하도록 바꾸자 각각 2.227, 24.08 로 R 에 가까워졌습니다(7장).

### 5.4 가중치 방식 슈퍼 러너 (`sl="weights"`, 파이썬에서 새로 더함)

R 코드는 `sl = 0`("가중치")을 인자 설명에만 적어 두고 구현하지 않았습니다. 파이썬 버전은 이렇게 구현했습니다.

- 예측값은 분위수 21개의 가중 평균입니다: w1 x q1 + ... + w21 x q21. 가중치는 모두 0 이상이고 합이 1 입니다.
- 이 두 조건 아래에서, 교차검증 분위수로 만든 예측과 실제값의 제곱 오차가 가장 작아지는 가중치를 고릅니다. R SuperLearner 패키지의 `method.CC_LS` 와 같은 규칙입니다.
- scipy 의 음수 없는 최소제곱(NNLS)에 "가중치 합 = 1" 을 나타내는 행 하나를 더해 풉니다. 일반 최적화 함수(scipy `trust-constr`)로 푼 가중치와 비교했을 때 가장 큰 차이는 2.5e-8 이었습니다.
- 가중치는 `model.sl_.coef_` 에 저장되고, `train.py` 가 0이 아닌 가중치를 출력합니다.

음수 없는 최소제곱으로 가중치를 먼저 구한 뒤 합으로 나누는 방식(SuperLearner 의 기본값 `method.NNLS`)을 쓰지 않은 이유: QRF 분위수는 데이터의 가운데 쪽으로 몰리는 경향이 있어서, 실제값이 더 넓게 퍼져 있으면 나누기 전 가중치의 합이 1보다 커집니다(Friedman #1 에서 평균 1.381). 이를 합으로 나누면 모든 예측값이 0 쪽으로 줄어듭니다. Friedman #1 테스트 RMSE 가 이 방식은 4.606, 위 방식은 2.302 였습니다. 또 "합 = 1" 조건을 걸고 풀면, 목표값과 분위수에 같은 상수를 더해도 가중치가 그대로입니다.

기본값은 `"rf"` 로 두었습니다. R 코드가 실제로 돌리는 방식이고(인자 목록에는 `sl = 0` 으로 적혀 있지만), 우리 비교에서도 더 정확했기 때문입니다(7장 표 6, 표 7).

## 6. 파이썬 버전 실행 방법

확인한 환경: Python 3.12.3, numpy 2.5.2, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.5.2.

```bash
pip install -r python/requirements.txt
cd python
```

### 6.1 학습

```bash
python train.py --data data/mtcars_train.csv --target mpg --drop car \
    --k 10 --seed 1 --full-seed 2019 --model model/qrfsl_mtcars.joblib
```

| 옵션 | 기본값 | 뜻 |
|:--|:--|:--|
| `--data` | 필수 | 학습용 CSV (첫 줄에 열 이름) |
| `--target` | 필수 | 목표값 열 |
| `--drop` | 없음 | 입력 변수가 아닌 열(예: 이름이나 번호 열) |
| `--model` | 필수 | 모델 파일을 저장할 경로 |
| `--k` | 10 | 폴드 수 (R `k`) |
| `--probs` | R 과 같은 21개 | 예측할 분위수 (R `prob`) |
| `--seed` | 0 | 폴드 나누기와 슈퍼 러너의 시드 (R `QRFSL()` 의 `seed`) |
| `--full-seed` | 0 | 전체 데이터 QRF 의 시드 (R `predict.QRFSL()` 의 `seed`) |
| `--n-jobs` | 1 | 동시에 학습할 폴드 수 (R `mc`) |
| `--qrf-trees` | 500 | QRF 하나의 나무 수 |
| `--sl` | rf | 슈퍼 러너 종류: `rf`(랜덤 포레스트, R 코드와 같음) 또는 `weights`(분위수의 가중 평균, 5.4절) |
| `--sl-trees` | 1000 | 랜덤 포레스트 슈퍼 러너의 나무 수 |
| `--nodesize` | 5 | 두 포레스트에 쓰는 R `nodesize` |

목표값 열과 `--drop` 으로 지정한 열을 뺀 나머지 열이 모두 입력 변수가 됩니다. 입력 변수는 숫자여야 하고 빈 값이 없어야 합니다(quantregForest 와 같은 규칙).

### 6.2 추론

```bash
python predict.py --model model/qrfsl_mtcars.joblib --data data/mtcars_test.csv \
    --out output/pred_mtcars.csv --keep car --target mpg --quantiles
```

| 옵션 | 기본값 | 뜻 |
|:--|:--|:--|
| `--model` | 필수 | `train.py` 가 저장한 모델 파일 |
| `--data` | 필수 | 학습 때와 같은 입력 변수 열이 있는 CSV (열 순서는 상관없고 다른 열은 무시) |
| `--out` | 필수 | 결과 CSV |
| `--keep` | 없음 | 결과에 그대로 옮길 입력 열(예: 이름 열) |
| `--target` | 없음 | 입력에 실제값이 있으면 결과에 함께 적고 RMSE 와 상관계수 제곱을 출력 |
| `--quantiles` | 꺼짐 | 전체 데이터 QRF 가 예측한 분위수 21개도 함께 저장 |

결과 열 순서: `--keep` 으로 지정한 열, `--target` 열(지정한 경우), `pred`(최종 예측값), 그리고 `--quantiles` 를 켜면 `quantile=0.025`, ..., `quantile=0.975`.

위 예시를 한 번 돌렸을 때 RMSE 2.0598, 상관계수 제곱 0.948 이 나왔습니다. 테스트용 자동차가 5대뿐이라 시드에 따라 이 값이 꽤 달라집니다. 여러 시드로 R 과 비교한 결과는 7장에 있습니다. 학습 명령에 `--sl weights` 를 더하면 이번 실행에서는 RMSE 1.9097, 상관계수 제곱 0.925 가 나왔지만, 시드 30개 평균으로는 랜덤 포레스트가 조금 더 나았습니다(7장 표 6).

### 6.3 파이썬 코드에서 쓰기

```python
import pandas as pd
from qrfsl import QRFSL

train = pd.read_csv("data/mtcars_train.csv")
test = pd.read_csv("data/mtcars_test.csv")

model = QRFSL(k=10, seed=1, full_seed=2019)  # 가중 평균 슈퍼 러너는 sl="weights"
model.fit(train.drop(columns=["car", "mpg"]), train["mpg"])

pred, qpred = model.predict(test, return_quantiles=True)  # test 의 다른 열은 무시됨
```

### 6.4 참고 사항

- 저장된 모델은 pickle 파일(joblib)입니다. 믿을 수 있는 파일만 불러오고, 학습할 때와 같은 scikit-learn 버전으로 불러와야 합니다. `predict.py` 는 같은 폴더에 `qrfsl.py` 가 있어야 동작합니다.
- 시드가 같으면 `n_jobs` 값과 관계없이 같은 모델과 같은 예측값이 나옵니다.
- 모델 파일 크기는 학습 행 수에 따라 커집니다(QRF 나무 500 그루와 슈퍼 러너 나무 1,000 그루. `sl="weights"` 이면 슈퍼 러너는 숫자 21개뿐). mtcars 모델은 1.5 MB 입니다.

## 7. R 과 파이썬 결과 비교

R 과 numpy 는 서로 다른 난수를 만들기 때문에, 같은 시드를 줘도 두 언어의 모델이 똑같아지지는 않습니다. 그래서 양쪽을 여러 번 돌려 평균을 비교했습니다. 매번 `QRFSL(k = 10, seed = s)` 로 학습하고 예측 시드는 `1000 + s` 를 썼습니다. 버전: R 4.5.3, quantregForest 1.4-0, randomForest 4.7-1.2, 파이썬 쪽은 6장의 버전.

**mtcars**: R 예제와 같은 학습·테스트 행 나누기, 30번 실행(s = 1, ..., 30).

표 2. 테스트용 자동차별 예측값, 30번 실행의 평균과 괄호 안 표준편차 (단위: mpg)

| 테스트용 자동차 | 실제값 | R | 파이썬 |
|:--|--:|--:|--:|
| Toyota Corona | 21.5 | 24.16 (0.75) | 24.08 (0.84) |
| Cadillac Fleetwood | 10.4 | 13.80 (0.25) | 13.87 (0.32) |
| Valiant | 18.1 | 19.81 (0.50) | 19.99 (0.47) |
| Ferrari Dino | 19.7 | 18.87 (0.41) | 18.93 (0.41) |
| Merc 240D | 24.4 | 23.58 (0.67) | 23.57 (0.55) |

표 3. mtcars 테스트 RMSE, 30번 실행 (단위: mpg)

| | 평균 | 표준편차 | 최솟값 | 최댓값 |
|:--|--:|--:|--:|--:|
| R | 2.196 | 0.234 | 1.728 | 2.949 |
| 파이썬 | 2.227 | 0.219 | 1.933 | 2.902 |

**Friedman #1**: 회귀 방법 비교에 흔히 쓰는 인공 데이터입니다. scikit-learn `make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=0)` 로 만든 뒤 앞 300행을 학습, 뒤 200행을 테스트에 썼습니다(목표값 표준편차 4.89). 5번 실행(s = 1, ..., 5).

표 4. Friedman #1 테스트 RMSE, 5번 실행 (목표값에 단위 없음)

| | 평균 | 표준편차 |
|:--|--:|--:|
| R | 2.040 | 0.041 |
| 파이썬 | 2.053 | 0.047 |

5번 실행을 평균한 R 예측값과 파이썬 예측값의 상관계수는 0.997, 평균 절대 차이는 0.32 입니다.

**QRF 만 따로** Friedman #1 에서 비교했습니다(R 은 quantregForest, 파이썬은 `QuantileRegressionForest`, 기본 설정, 시드 1부터 5). 10 %, 50 %, 90 % 분위수를 예측했습니다.

표 5. 테스트 200행 가운데 실제값이 예측 분위수보다 작은 행의 비율 (5번 실행 평균, 0에서 1 사이)

| 예측 분위수 | R | 파이썬 |
|:--|--:|--:|
| 10 % | 0.026 | 0.024 |
| 50 % | 0.479 | 0.481 |
| 90 % | 0.989 | 0.990 |

10 % 분위수와 90 % 분위수 사이의 평균 간격은 양쪽 모두 9.765 입니다. 이 데이터에서는 두 구현 모두 구간이 명목 수준보다 넓게 나오므로, 이는 옮기는 과정의 문제가 아니라 방법과 설정에서 오는 성질입니다.

정리하면, 파이썬 버전은 실행 한 번 한 번이 R 과 똑같지는 않지만 예측값과 오차가 시드에 따른 흔들림 범위 안에서 R 과 맞습니다.

**랜덤 포레스트와 가중치 방식 슈퍼 러너 비교** (파이썬만, 위와 같은 데이터·시드·테스트 행. `sl="weights"` 는 R 에 대응하는 코드가 없음):

표 6. mtcars 테스트 RMSE, 30번 실행 (단위: mpg)

| 슈퍼 러너 | 평균 | 표준편차 | 최솟값 | 최댓값 |
|:--|--:|--:|--:|--:|
| rf | 2.227 | 0.219 | 1.933 | 2.902 |
| weights | 2.275 | 0.229 | 1.893 | 2.846 |

표 7. Friedman #1 테스트 RMSE, 5번 실행 (목표값에 단위 없음)

| 슈퍼 러너 | 평균 | 표준편차 | 최솟값 | 최댓값 |
|:--|--:|--:|--:|--:|
| rf | 2.053 | 0.047 | 1.981 | 2.098 |
| weights | 2.302 | 0.014 | 2.285 | 2.319 |

가중치 방식은 분위수 21개 가운데 평균 5개에만 0이 아닌 가중치를 주었습니다(실행마다 mtcars 3개에서 8개, Friedman #1 4개에서 6개). 50 % 분위수(중앙값) 하나만 쓰면 평균 RMSE 가 mtcars 2.512, Friedman #1 2.346 이었으므로, 가중 평균은 중앙값 하나보다는 낫지만 랜덤 포레스트보다는 못했습니다.

## 8. QRFSL 은 쓸모가 있는가, 있다면 어디에 (이식 후 추가)

R 코드는 2019년에 "모델의 예측분포 전체를 2단계에 넣으면 점 예측 하나보다 정보가 많을 것"이라는 직관으로 만들어졌습니다. 이식을 마친 뒤 그 직관을 확인했습니다. 이 장은 그 결과와 이유, 그리고 살아남은 형태인 **비대칭 비용 아래의 결정값 + 덮는 비율이 보증된 구간**을 기록합니다. 코드는 [python/conformal.py](python/conformal.py), [python/decision.py](python/decision.py), [python/experiments/](python/experiments/)에 있고, 아래 표는 모두 8.7절의 명령으로 다시 만들 수 있습니다.

### 8.1 2단계가 할 수 있는 것과 없는 것

QRF 는 새 행에 대해 포레스트의 가중치로 학습 목표값들의 가중 경험분포를 만듭니다. 이 분포의 평균은 포레스트 자신의 예측값입니다(Meinshausen 의 포레스트 정의 기준. randomForest 처럼 부트스트랩 표본의 잎 평균을 쓰면 조금 다릅니다). 그래서 2단계를 제곱오차로 학습하면 목표는 조건부 평균인데, 그 평균은 포레스트가 이미 내고 있습니다. 2단계에 남는 일은 포레스트의 편향을 되돌리는 것뿐입니다. 아래 벤치마크가 그대로 보여 줍니다. QRFSL 은 포레스트가 편향되는 곳(매끄러운 신호, 적은 행)에서만 도움이 되고, 그런 곳에서는 더 좋은 기본 모델이 더 크게 도움이 됩니다.

### 8.2 원래 방법의 점 예측 벤치마크

학습 1,000행, 테스트 2,000행, 시드 3개. `python -m experiments.run_benchmark`. 숫자는 일반 랜덤 포레스트(평균 예측)의 테스트 RMSE 로 나눈 값이며 1보다 작으면 더 좋습니다. 약 2 % 안쪽 차이는 시드 사이 흔들림 범위입니다.

표 8. 일반 랜덤 포레스트 대비 테스트 RMSE (단위 없음)

| 모델 | Friedman 정규 잡음 | Friedman 두꺼운 꼬리 잡음 | Friedman 잡음이 x 에 따라 커짐 | 로그정규 목표값 | 캘리포니아 집값 | 당뇨 (n = 265) | 호우 피해 합성 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| QRFSL (rf) | 0.91 | 1.00 | 0.98 | 1.05 | 0.99 | 1.06 | 1.01 |
| QRFSL (weights) | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.98 |
| QRF 중앙값 | 1.02 | 1.00 | 1.01 | 1.05 | 1.00 | 1.03 | 1.01 |
| log y 로 학습한 랜덤 포레스트 | 1.03 | 1.01 | 1.04 | 1.04 | 1.05 | 1.05 | 1.00 |
| GBM (HistGradientBoosting) | 0.74 | 0.97 | 0.93 | 1.03 | 0.98 | 1.04 | 1.05 |

표 9. 2단계가 무엇을 쓰는가: 입력을 줄인 같은 stacking (랜덤 포레스트 대비 RMSE)

| 2단계 입력 | Friedman 정규 잡음 | 캘리포니아 집값 | 당뇨 | 호우 피해 합성 |
|:--|--:|--:|--:|--:|
| 교차검증 RF 평균값 하나 | 1.03 | 1.13 | 1.17 | 1.17 |
| 중앙값, 폭, 치우침 (3개) | 0.97 | 1.02 | 1.07 | 1.04 |
| 분위수 21개 (= QRFSL) | 0.91 | 0.99 | 1.06 | 1.01 |

읽는 법: QRFSL (rf) 이 포레스트를 뚜렷이 이긴 곳은 매끄러운 Friedman 신호뿐이고, 거기서는 GBM 이 훨씬 더 큽니다. 치우친 목표값과 작은 데이터에서는 2단계 포레스트가 과적합해 오히려 나빠집니다. weights 방식은 안전하지만 이득이 2에서 3 % 이내입니다. 분위수 21개를 다 넣는 것이 요약 3개보다 낫고 숫자 하나보다 훨씬 나은데, 이는 입력 하나 위의 포레스트가 매끄럽지 못하기 때문입니다. 일반적인 정확도 향상 수단으로는 품값을 못 합니다.

### 8.3 살아남은 형태: 결정값과 보증된 구간

예측분포는 평균이 줄 수 없는 것 두 가지를 줍니다.

**비대칭 비용 아래의 결정값.** 과소 예측이 단위당 c_under, 과대 예측이 단위당 c_over 의 비용을 내면, 기대 비용이 가장 작은 결정값은 tau = c_under / (c_under + c_over) 분위수입니다. 여기에 고정 벌금(예: 예산 초과 시)이 붙으면 최적 결정값은 더 이상 분위수 하나가 아니라 분포 모양에 따라 달라집니다. [decision.py](python/decision.py)는 QRF 출력에서 결정값으로 가는 길 네 가지를 제공합니다.
- *plug-in*: QRF 분포의 원자(atom)마다 기대 비용을 계산해 가장 작은 것을 고름(학습 없음)
- *QRFSL pinball stage*: 교차검증 분위수 21개 위에서 tau 의 pinball 손실로 부스팅(원래 아이디어에서 손실만 바꾼 것)
- *QRFSL cost-weight stage*: 교차검증 행에서 경험 비용을 최소화하는 가중치로 분위수 21개를 가중 평균(어떤 비용에도 적용 가능)
- 기준선: QRF 의 tau 분위수 그대로, 그 분위수에 한쪽 conformal 보정을 더한 것, pinball 손실로 직접 학습한 LightGBM, 중앙값 + 상수 이동

**덮는 비율이 보증된 구간 (CQR).** Conformalized quantile regression(Romano, Patterson, Candès, 2019)은 QRF 의 5 %·95 % 분위수를 받아, 보정용 행마다 y 가 [q_lo(x), q_hi(x)] 밖으로 얼마나 벗어났는지를 재고, 그 거리들의 (1 − alpha)(1 + 1/n) 분위수만큼 모든 구간을 넓히거나 줄입니다. 그 결과 구간은 분위수 모델이 무엇이든, 행들이 서로 바꿔도 분포가 같기만 하면 새 y 를 1 − alpha 이상의 확률로 덮습니다. 모델의 품질은 폭에만 영향을 줍니다. 보정값이 상수 하나라서 폭은 QRF 분위수를 따라 x 마다 달라지며, 점 예측의 잔차로 만든 conformal 구간(모든 x 에서 같은 폭)과 이 점이 다릅니다. [conformal.py](python/conformal.py)는 split 방식(따로 뗀 보정 행, 정확한 보증), 비대칭 방식(아래·위 보정값 분리), QRFSL 의 교차검증 분위수를 보정 점수로 재사용하는 방식(추가 데이터 없음, 형식적 보증 없음, 8.4.3 참고)을 구현합니다.

### 8.4 각 방법의 이론

이 절은 8.3절의 방법들과 8.6절 결과의 근거를 설명합니다. 기호: 학습 행 $(x_i, y_i)$, $i = 1, \dots, n$, 새 행 $x$ 와 아직 모르는 $y$, $x$ 가 주어졌을 때 $y$ 의 실제 조건부 누적분포 $F(\cdot \mid x)$, 그 $\tau$ 분위수 $q_\tau(x)$.

**8.4.1 QRF 의 예측분포.** $T$ 그루의 나무는 각각 새 행을 끝 노드 하나 $L_t(x)$ 에 넣습니다. 학습 행 $i$ 의 가중치는

$$w_i(x) = \frac{1}{T} \sum_{t=1}^{T} \frac{\mathbf{1}\{x_i \in L_t(x)\}}{|L_t(x)|}, \qquad w_i(x) \ge 0, \quad \sum_i w_i(x) = 1$$

이고, QRF 는 조건부 분포와 분위수를 다음과 같이 추정합니다.

$$\hat F(y \mid x) = \sum_i w_i(x)\, \mathbf{1}\{y_i \le y\}, \qquad \hat q_\tau(x) = \inf\{\, y : \hat F(y \mid x) \ge \tau \,\}$$

Meinshausen(2006)은 일정한 조건에서 $\hat F$ 가 참 분포로 수렴함을 보였습니다. $\hat F$ 의 평균 $\sum_i w_i(x) y_i$ 는 포레스트 자신의 예측값입니다(이 정의의 포레스트 기준. R 의 randomForest 는 끝 노드마다 부트스트랩으로 뽑힌 행만 평균하므로 조금 다름). `leaf_sample="all"` 은 이 가중치를 그대로 계산하고, `leaf_sample="one"` 은 끝 노드·나무마다 y 하나만 무작위로 저장하는 R quantregForest 의 근사입니다.

*원래 QRFSL 에 대한 결론 (8.1절).* 분위수 벡터 $\hat{\mathbf q}(x)$ 위에서 제곱오차로 학습한 2단계 $g$ 는 $E[y \mid \hat{\mathbf q}(x)]$ 를 추정합니다. $\hat F$ 가 정확하다면 이 값은 $\hat F$ 의 평균, 즉 포레스트가 이미 내는 값과 같습니다. 그래서 $g$ 가 하는 일은 포레스트의 편향을 고치는 것뿐이고, 포레스트가 편향된 곳에서만 도움이 됩니다.

*2단계가 교차검증 분위수를 쓰는 이유.* 학습 행은 자기 $y_i$ 가 $\hat F(\cdot \mid x_i)$ 를 만드는 끝 노드 안에 들어 있어서, 학습 데이터에서 낸 분위수가 $y_i$ 에 지나치게 가깝습니다. 그것으로 2단계를 학습하면 새 행에는 맞지 않는 보정을 배웁니다. 그 행을 빼고 학습한 QRF 의 분위수(10겹 교차검증)는 새 행의 분위수처럼 행동합니다. stacking 의 일반 원칙입니다(Wolpert, 1992).

**8.4.2 비용 아래의 결정값.** $y$ 를 보기 전에 값 $d$ 를 정하면 비용 $c(d, y)$ 가 듭니다. 행 $x$ 의 최적 결정값은 기대 비용 $R(d) = E[c(d, y) \mid x]$ 를 최소로 하는 값입니다.

*선형 비대칭 비용.* $c(d, y) = c_u \max(y - d, 0) + c_o \max(d - y, 0)$ ($c_u$ 는 과소 예측 단위당 비용, $c_o$ 는 과대 예측 단위당 비용)이면

$$R'(d) = -c_u\,\bigl(1 - F(d \mid x)\bigr) + c_o\, F(d \mid x) = 0 \;\Longleftrightarrow\; F(d \mid x) = \frac{c_u}{c_u + c_o} = \tau$$

이므로 최적 결정값은 조건부 분위수 $q_\tau(x)$ 입니다. 이 비용은 $c_u + c_o$ 배를 빼면 pinball 손실 $\rho_\tau(u) = u\,(\tau - \mathbf{1}\{u < 0\})$ ($u = y - d$)과 같습니다. 함수들 가운데 평균 pinball 손실을 최소로 하는 것을 고르면 조건부 $\tau$ 분위수를 추정하게 됩니다(Koenker, Bassett, 1978). 비용 3:1 은 $\tau = 0.75$, 9:1 은 $\tau = 0.9$ 입니다.

*벌금형 비용.* $y > d$ 일 때마다 고정 벌금 $p$ 를 더하면 $R(d) = R_{\text{선형}}(d) + p\,\bigl(1 - F(d \mid x)\bigr)$ 입니다. 이제 최솟값 조건에 $d$ 에서의 $y$ 의 밀도가 들어가므로, 최적 결정값은 정해진 분위수 하나가 아니라 분포 전체의 모양에 따라 달라집니다. 원자(atom) $a_j$ 와 가중치 $w_j$ 로 된 이산 예측분포(QRF 분포, 또는 분위수 21개를 같은 가중치의 원자로 본 것)에서는

$$\hat R(d) = c_u \sum_j w_j \max(a_j - d, 0) + c_o \sum_j w_j \max(d - a_j, 0) + p \sum_{j:\, a_j > d} w_j$$

입니다. 이 함수는 조각마다 직선이고 원자에서만 아래로 뛰므로 최솟값은 원자 가운데 하나에 있습니다. `expected_cost_decision` 은 모든 원자에서 $\hat R$ 를 계산해 가장 작은 것을 고릅니다(plug-in 결정값).

*2단계들.* pinball 2단계는 교차검증 분위수 벡터 위에서 $\rho_\tau$ 로 부스팅해 $g(\hat{\mathbf q}(x))$ 를 학습하므로, QRF 분위수 21개가 주어졌을 때 $y$ 의 $\tau$ 분위수를 추정합니다. QRF 가 잘 맞아 있으면 $\hat q_\tau(x)$ 를 그대로 돌려주고, 어긋나 있으면(8.6절: bike, california, friedman_hetero 에서 0.9 분위수를 넘는 행이 10 %가 아니라 4에서 5 %) 행마다 다른 양을 옮깁니다. 8.4.3의 conformal 이동은 모든 행을 같은 상수만큼 옮겨 평균 수준만 맞추고 행마다의 오차는 고치지 못하기 때문에, pinball 2단계가 비용을 더 많이 줄입니다. cost-weight 2단계는 $d = \sum_k v_k\, \hat q_{p_k}(x)$ ($v_k \ge 0$, $\sum_k v_k = 1$)로 두고 교차검증 행의 평균 비용이 최소가 되게 가중치를 고릅니다. 분위수의 가중 평균은 어떤 비용에도 쓸 수 있고, $y$ 에 상수를 더하면 $d$ 에도 같은 상수가 더해집니다.

**8.4.3 Conformal 보정 (CQR).** Conformal 예측에 필요한 가정은 하나입니다. 보정 행과 새 행이 *교환 가능*해야 합니다. 즉 순서를 바꿔도 결합분포가 같아야 하며, 서로 독립이고 같은 분포에서 나온 행이면 성립합니다. 분위수 모델 $\hat q_{lo}, \hat q_{hi}$ 는 다른 행으로 학습한 뒤 고정해 둡니다.

1. 보정 행 $n$ 개의 점수 $E_i = \max\bigl(\hat q_{lo}(x_i) - y_i,\; y_i - \hat q_{hi}(x_i)\bigr)$ 를 계산합니다. 구간 안이면 음수, 밖이면 양수입니다. 새 행의 점수 $E$ 도 같은 식입니다.
2. 교환 가능하므로 $n + 1$ 개 점수 가운데 $E$ 의 순위는 $1, \dots, n+1$ 이 모두 같은 확률입니다. 따라서 $E$ 가 $k$ 번째로 작은 보정 점수 이하일 확률은 $k / (n + 1)$ 이상입니다.
3. $k = \lceil (n+1)(1-\alpha) \rceil$ 로 잡고 $Q$ 를 $k$ 번째로 작은 보정 점수라 하면 $P(E \le Q) \ge 1 - \alpha$ 입니다.
4. $E \le Q$ 는 $\hat q_{lo}(x) - Q \le y \le \hat q_{hi}(x) + Q$ 와 같은 말입니다. 따라서 구간 $[\hat q_{lo}(x) - Q,\; \hat q_{hi}(x) + Q]$ 는 새 $y$ 를 $1 - \alpha$ 이상의 확률로 덮고, 점수에 동점이 없으면 $1 - \alpha + 1/(n+1)$ 이하로 덮습니다(Romano, Patterson, Candès, 2019).

이 논증 어디에도 $\hat q_{lo}, \hat q_{hi}$ 가 정확해야 한다는 조건이 없습니다. 분위수 모델이 나빠도 약속한 덮는 비율은 지켜지고, 좋으면 $|Q|$ 가 작아져 구간이 좁아집니다. QRF 구간이 너무 넓으면 $Q$ 가 음수가 되어 구간이 줄어듭니다(8.6절 friedman_hetero: 표준편차의 2.27배에서 1.80배).

*conformal.py 의 다른 방식.*
- 비대칭: 아래쪽 점수 $\hat q_{lo}(x_i) - y_i$ 와 위쪽 점수 $y_i - \hat q_{hi}(x_i)$ 를 각각 $\alpha/2$ 수준으로 따로 보정합니다. 각 끝을 벗어날 확률이 $\alpha/2$ 이하이므로 둘을 합쳐도 $\alpha$ 이하입니다.
- 결정값용 한쪽 이동: $s$ 를 $y_i - \hat q_\tau(x_i)$ 가운데 $\lceil (n+1)\tau \rceil$ 번째로 작은 값으로 잡으면 $P\bigl(y \le \hat q_\tau(x) + s\bigr) \ge \tau$ 입니다.
- 잔차 conformal (기준선): 점 예측 $\hat\mu$ 둘레의 점수 $|y_i - \hat\mu(x_i)|$ 를 씁니다. 보증은 같지만 모든 $x$ 에서 폭이 $2Q$ 로 같습니다.

*보증이 다루지 못하는 것.* 이 보증은 새 행들에 대한 평균(전체) 덮는 비율입니다. $x$ 값마다의 덮는 비율(조건부 덮는 비율)은 분포에 대한 가정 없이는 보증할 수 없습니다. 모든 분포에서 이를 보증하는 방법은 기대 폭이 무한한 구간을 낼 수밖에 없음이 증명되어 있습니다(Vovk, 2012; Lei, Wasserman, 2014; Barber, Candès, Ramdas, Tibshirani, 2021b). 8.6절에서 friedman_hetero 의 가장 조용한 1/5 은 93 %, 가장 시끄러운 1/5 은 82 %를 덮은 것이 그 모습입니다.

*cv 방식에는 형식적 보증이 없습니다.* 점수는 교차검증 분위수(학습 행의 90 %로 학습한 QRF 각각)에서 계산하지만, $Q$ 는 학습 행 전체로 다시 학습한 QRF 에 적용합니다. 새 행의 점수와 보정 점수가 서로 다른 모델에서 나오므로 교환 가능하지 않고, 2단계가 성립하지 않습니다. CV+(Barber, Candès, Ramdas, Tibshirani, 2021a)와 이를 분위수 점수로 넓힌 Gupta, Kuchibhotla, Ramdas(2022)의 방법은 새 행에도 다시 학습한 모델 대신 폴드별 모델 10개를 적용해 $1 - 2\alpha$ 이상의 보증을 되살리며, 여기에는 구현하지 않았습니다. 실험에서 cv 방식은 테스트 행의 0.897에서 0.911을 덮어 split 방식과 같았습니다.

### 8.5 실험 설계

데이터 5종, 각각 무작위 나누기 10번: 학습 2,000행, 보정 1,000행, 테스트 3,000행(데이터가 작으면 비례해서 줄임). QRF 는 나무 500그루, 전체 가중 분포(`leaf_sample="all"`), 10겹.

| 데이터 | 행 수 | 넣은 이유 |
|:--|--:|:--|
| friedman_hetero | 합성 | 잡음 표준편차가 x0 에 따라 0.2에서 4.2까지 커짐. 구간 폭이 잡음을 따라가는지 봄 |
| lognormal | 합성 | 오른쪽으로 치우친 목표값 |
| california | 20,640 | 표 형식 회귀의 표준, 주택 가격(10만 달러 단위) |
| bike | 17,379 | 시간별 자전거 대여 수. 부족과 남음의 비용이 다름 |
| claims | 24,944 | 프랑스 자동차보험 계약별 청구액, 50,000유로에서 자름. 오른쪽 꼬리가 두꺼움 |

행의 역할:

| 행 | 쓰는 곳 | 역할 |
|:--|:--|:--|
| 학습 행의 교차검증 분위수 | 2단계, "cv" 방식의 conformal 보정값, 중앙값 이동 | 학습 안 검증 |
| 보정 행 | split conformal 방식들의 보정값 | 보정 |
| 테스트 행 | 덮는 비율, 폭, 평균 비용. 어떤 선택에도 쓰지 않음 | 평가 |

비용: 선형 1:1 (tau 0.5), 3:1 (tau 0.75), 9:1 (tau 0.9), 그리고 선형 1:1 에 y 가 결정값을 넘을 때마다 학습 y 표준편차의 절반을 벌금으로 더한 것.

### 8.6 결과

**구간 (이름값 덮는 비율 90 %).** `experiments/results/intervals.csv`. 10번 나누기의 평균이며, 나누기 사이 덮는 비율의 흔들림은 최대 0.02 입니다.

표 10. 90 % 구간의 덮는 비율 (테스트 행 가운데 구간 안에 든 비율, 0에서 1)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF 그대로 (5 %, 95 %) | 0.963 | 0.966 | 0.895 | 0.956 | 0.921 |
| QRF + CQR split | 0.907 | 0.898 | 0.896 | 0.895 | 0.894 |
| QRF + CQR split, 비대칭 | 0.912 | 0.932 | 0.895 | 0.896 | 0.895 |
| QRF + CQR cv (보정 데이터 없음) | 0.911 | 0.904 | 0.905 | 0.897 | 0.900 |
| RF 점 예측 ± 잔차 conformal | 0.900 | 0.892 | 0.898 | 0.892 | 0.892 |
| LightGBM 분위수 그대로 | 0.715 | 0.702 | 0.729 | 0.693 | 0.710 |
| LightGBM + CQR split | 0.896 | 0.893 | 0.896 | 0.894 | 0.894 |

표 11. 평균 구간 폭 ÷ 학습 목표값 표준편차 (단위 없음)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF 그대로 (5 %, 95 %) | 1.67 | 1.93 | 1.53 | 2.27 | 2.02 |
| QRF + CQR split | 1.64 | 1.83 | 1.53 | 1.80 | 1.99 |
| QRF + CQR split, 비대칭 | 1.54 | 1.86 | 1.57 | 1.80 | 1.85 |
| QRF + CQR cv (보정 데이터 없음) | 1.64 | 1.85 | 1.53 | 1.81 | 1.99 |
| RF 점 예측 ± 잔차 conformal | 1.45 | 1.49 | 1.39 | 1.73 | 1.87 |
| LightGBM 분위수 그대로 | 0.77 | 0.97 | 1.12 | 1.06 | 1.16 |
| LightGBM + CQR split | 0.96 | 1.33 | 1.34 | 1.58 | 1.40 |

표 12. friedman_hetero: 실제 잡음 크기가 가장 작은 1/5 과 가장 큰 1/5 의 테스트 행에서 덮는 비율 (0에서 1)

| 방법 | 잡음 가장 작은 1/5 | 잡음 가장 큰 1/5 |
|:--|--:|--:|
| QRF 그대로 (5 %, 95 %) | 0.988 | 0.894 |
| QRF + CQR split | 0.928 | 0.817 |
| QRF + CQR cv | 0.934 | 0.817 |
| RF 점 예측 ± 잔차 conformal | 0.998 | 0.723 |
| LightGBM + CQR split | 0.968 | 0.818 |

읽는 법:
- QRF 구간을 그대로 쓰면 너무 넓고(claims 를 빼면 덮는 비율 0.92에서 0.97, 목표는 0.90), LightGBM 분위수를 그대로 쓰면 너무 좁습니다(0.69에서 0.73). CQR 을 거치면 보증대로 모든 방법이 0.89에서 0.91 이 되고, QRF 는 폭도 줄어듭니다(friedman_hetero: 표준편차의 2.27배에서 1.80배로).
- 같은 덮는 비율에서 QRF + CQR 의 구간이 가장 넓습니다. LightGBM + CQR 의 1.1에서 1.7배, 포레스트 점 예측 둘레의 일정 폭 구간의 1.04에서 1.23배입니다. 포레스트의 분위수가 시끄럽기 때문이며, CQR 은 수준을 맞출 뿐 날카로움은 못 고칩니다.
- 폭이 잡음을 따라가는 정도는 부분적입니다. 잡음이 변하는 friedman_hetero 에서 일정 폭 구간은 가장 조용한 1/5 의 99.8 %, 가장 시끄러운 1/5 의 72 %를 덮고, QRF + CQR 은 93 %와 82 %를 덮습니다. 꼬리가 두꺼운 claims 에서는 QRF + CQR 의 가장 좁은 1/5 구간도 그 행의 87 %를 덮고 LightGBM + CQR 은 77 %입니다. 어떤 방법도 데이터의 모든 부분에서 90 %에 이르지 못합니다. CQR 이 보증하는 것은 평균뿐입니다.
- 보정 데이터 대신 QRFSL 의 교차검증 분위수를 재사용하는 cv 방식은 split 방식과 같은 덮는 비율(0.90에서 0.91)과 폭을 냅니다. 구간에 관해 QRFSL 의 구조가 주는 실질적 이점은 이것 하나입니다. 데이터를 따로 떼어 둘 필요가 없습니다.

정리하면, conformal 보정은 언제나 해야 하고, QRF + CQR 은 타당하고 단순하며 보정 데이터가 필요 없지만, 같은 덮는 비율에서 더 날카로운 것은 conformal 보정한 부스팅 분위수 모델입니다.

**비대칭 비용 아래의 결정값.** `experiments/results/decisions.csv`. 10번 나누기의 평균. 숫자는 테스트 평균 비용을 "QRF τ 분위수 그대로"(선형 비용의 자연스러운 답)의 평균 비용으로 나눈 값이며 1보다 작으면 더 좋습니다. 학습하는 방법들의 나누기 사이 흔들림은 3:1 에서 최대 0.02, 9:1 에서 최대 0.04 입니다.

표 13. QRF τ 분위수 대비 평균 비용, 선형 비용 3:1 (τ 0.75, 단위 없음)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF 평균 (비용 무시) | 1.27 | 1.17 | 1.06 | 1.19 | 1.02 |
| QRF 중앙값 + 상수 이동 | 1.18 | 1.06 | 0.99 | 1.01 | 1.07 |
| QRF τ 분위수 그대로 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF τ 분위수 + conformal 이동 | 0.99 | 0.99 | 1.00 | 0.98 | 1.00 |
| QRFSL cost-weight 2단계 | 0.97 | 0.98 | 1.00 | 0.98 | 1.00 |
| QRFSL pinball 2단계 | 0.94 | 0.96 | 1.02 | 0.92 | 1.00 |
| LightGBM 직접 τ 분위수 | 0.63 | 0.94 | 1.03 | 0.88 | 1.00 |
| LightGBM plug-in (분위수 21개) | 0.61 | 0.92 | 1.02 | 0.86 | 0.99 |

표 14. QRF τ 분위수 대비 평균 비용, 선형 비용 9:1 (τ 0.9, 단위 없음)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF 평균 (비용 무시) | 1.82 | 1.75 | 1.02 | 1.97 | 1.33 |
| QRF 중앙값 + 상수 이동 | 1.23 | 1.11 | 0.97 | 0.99 | 1.17 |
| QRF τ 분위수 그대로 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF τ 분위수 + conformal 이동 | 0.96 | 0.99 | 1.00 | 0.95 | 1.00 |
| QRFSL cost-weight 2단계 | 0.87 | 0.93 | 0.98 | 0.94 | 0.99 |
| QRFSL pinball 2단계 | 0.82 | 0.92 | 0.98 | 0.88 | 0.99 |
| LightGBM 직접 τ 분위수 | 0.63 | 1.00 | 1.03 | 0.93 | 1.05 |
| LightGBM plug-in (분위수 21개) | 0.58 | 0.97 | 1.00 | 0.89 | 1.03 |

표 15. 테스트 행 가운데 y 가 결정값을 넘은 비율, 선형 비용 9:1 (비용이 최소가 되는 비율은 0.10)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| QRF τ 분위수 그대로 | 0.043 | 0.043 | 0.099 | 0.053 | 0.083 |
| QRF τ 분위수 + conformal 이동 | 0.099 | 0.095 | 0.097 | 0.101 | 0.101 |
| QRFSL pinball 2단계 | 0.114 | 0.120 | 0.118 | 0.121 | 0.119 |
| LightGBM 직접 τ 분위수 | 0.188 | 0.191 | 0.166 | 0.213 | 0.187 |

표 16. QRF 전체 분포 plug-in 결정값 대비 평균 비용, 선형 1:1 + y 가 결정값을 넘으면 표준편차 0.5배 벌금 (단위 없음)

| 방법 | bike | california | claims | friedman_hetero | lognormal |
|:--|--:|--:|--:|--:|--:|
| RF 평균 (비용 무시) | 1.10 | 1.09 | 1.21 | 1.04 | 1.01 |
| QRF 중앙값 + 상수 이동 | 1.02 | 1.02 | 1.00 | 0.99 | 1.00 |
| QRF plug-in (전체 분포) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRF plug-in (분위수 21개) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| QRFSL cost-weight 2단계 | 1.00 | 1.02 | 1.03 | 0.98 | 1.00 |
| LightGBM plug-in (분위수 21개) | 0.73 | 0.96 | 1.07 | 0.89 | 1.00 |

읽는 법:
- QRF 분위수를 그대로 쓰면 너무 조심스럽습니다. τ 0.9 에서 결정값을 넘는 테스트 행이 10 %여야 하는데 4에서 5 %뿐입니다(claims 제외). 구간이 너무 넓었던 것과 같은 어긋남입니다. QRFSL pinball 2단계는 이를 행마다 다르게 고쳐서 bike, california, friedman_hetero 에서 비용을 3:1 은 4에서 8 %, 9:1 은 8에서 18 % 줄입니다. 상수 하나로 옮기는 conformal 이동은 넘는 비율을 정확히 10 %로 맞추지만 비용은 1에서 5 %만 줄입니다. 2단계의 이득은 행마다 다른 양을 옮기는 데서 옵니다. 원래 분위수가 이미 제 수준에 있는 claims 와 lognormal 에서는 어떤 방법도 비용을 2 % 넘게 바꾸지 못합니다.
- cost-weight 2단계(분위수 21개의 가중 평균)는 그 중간(9:1 에서 3에서 13 %)이고 부스팅이 필요 없습니다.
- x 로 직접 학습한 부스팅 분위수 모델은 넘는 비율이 어긋나 있는데도(17에서 21 %) bike(0.63)와 friedman_hetero(0.88에서 0.93)에서 훨씬 낫고, california 에서는 비기며, claims 와 lognormal 에서는 조금 나쁩니다. 2단계보다 기본 모델의 선택이 더 중요합니다. 2단계는 포레스트의 분위수를 고칠 뿐, 포레스트를 가장 좋은 모델로 만들지는 못합니다. 부스팅 모델에는 conformal 보정이나 stacking 을 적용하지 않았으므로, 둘을 합치는 것이 다음 할 일입니다.
- 벌금형 비용에서는 QRF 분포의 plug-in 결정값이 학습한 어떤 2단계와도 같거나 낫고(cost-weight 2단계 0.98에서 1.03), 분위수 21개가 전체 분포와 같은 정보를 담습니다(차이 0.1 % 미만).
- 비용을 아예 무시하면(포레스트 평균) 9:1 에서 비용이 1.3에서 2.0배입니다. 이 표들에서 가장 큰 효과는 비용을 반영한 결정값을 쓰는 것 자체입니다.

**정리.** 2019년 아이디어 가운데 살아남은 것은 이것입니다. 교차검증 분위수 위의 2단계는 비용이 크게 비대칭일 때 포레스트 τ 분위수의 유용한 조건부 재보정이고(9:1 에서 분위수 그대로보다 비용 5에서 18 % 절감, conformal 이동보다 큼), 교차검증 분위수는 보증된 구간을 위한 공짜 보정 데이터가 됩니다. 살아남지 못한 것은 일반 정확도 향상 수단으로서의 2단계와, 가장 좋은 기본 모델로서의 포레스트입니다. 부스팅 분위수 모델을 쓸 수 있는 문제라면 그것을 직접 학습해 conformal 보정하고, 포레스트를 쓸 이유가 있다면(작은 데이터, 안정성, 기존 흐름) 결정값에는 pinball 2단계를, 구간에는 CQR 을 더하면 됩니다.

### 8.7 실험 실행

```bash
cd python
python -m experiments.run_benchmark --seeds 3 --n-jobs 16      # 표 8, 9
python -m experiments.run_intervals --seeds 10 --n-jobs 16     # 구간
python -m experiments.run_decisions --seeds 10 --n-jobs 16     # 결정값
```

결과는 `experiments/results/*.csv` 에 저장됩니다. 공개 데이터는 처음 쓸 때 scikit-learn(캘리포니아 집값)과 OpenML(자전거 대여, 프랑스 자동차보험)에서 내려받습니다. 부품을 직접 쓰는 예:

```python
from qrfsl import QRFSL
from conformal import CQR
from decision import LinearCost, expected_cost_decision

m = QRFSL(k=10, seed=1, sl="none", leaf_sample="all").fit(X_train, y_train)
q = m.predict_quantiles(X_test)                         # 행마다 분위수 21개
cqr = CQR(alpha=0.1).fit(m.cv_quantiles_["quantile=0.05"], m.cv_quantiles_["quantile=0.95"], m.cv_y_)
lo, hi = cqr.predict(q["quantile=0.05"], q["quantile=0.95"])   # 90 % 구간, 보정 데이터 없이
atoms, weights = m.predict_distribution(X_test)
d = expected_cost_decision(atoms, weights, LinearCost(c_under=3, c_over=1))   # 3:1 비용의 결정값
```

## 9. 참고 문헌

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
- R 패키지: quantregForest (Meinshausen, Michel), randomForest (Liaw, Wiener).
