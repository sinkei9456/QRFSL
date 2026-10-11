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
8. [참고 문헌](#8-참고-문헌)

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

## 8. 참고 문헌

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1), Article 25.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- R 패키지: quantregForest (Meinshausen, Michel), randomForest (Liaw, Wiener).
