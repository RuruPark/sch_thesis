# 머신러닝 기반 전력 설비 상태 진단 및 이상 발생 조기 탐지

> AI-Hub 전력설비 센서 데이터(펌프/일반모터, 공조설비 105대)로 정상 운전 중인 설비가 30분 안에 이상 상태로 바뀔지 예측하는 졸업논문 코드

## 연구 질문

- 지금 정상으로 운전 중인 설비가 앞으로 30분 안에 전류고조파 이상(주의/경고)으로 바뀌는지 미리 알 수 있는가
- 어떤 측정 항목의 변화가 이상 발생을 예고하는가

## 핵심 결과

- 전처리 완료: 설비 105대, 약 394만 행을 1분 단위 "시각 × 측정항목 35개 + 라벨 5개" 표로 변환하고, 결측·이상값을 정리해 209개 파일 401만 행으로 확정
- SOH 라벨은 같은 시각 평균값에 고정 경계(역률 0.6/0.8, 전류고조파 30/60, 전압고조파 3/5)를 적용한 규칙 라벨이다. 운전 구간에서 경계만으로 97~99%가 재현된다
- 역률 SOH "경고"는 정지 구간의 93%에 붙어 있어 설비 상태가 아니라 정지 여부를 반영한다
- 조기 경보 데이터셋 구성: 예측 대상 시점 Training 134만 / Validation 18만, 30분 내 이상 발생 양성 비율 2.3% / 2.9%, 특징 148개
- 30분 내 이상 발생 예측 PR-AUC: 전류고조파 임계값 규칙 0.326, LightGBM 0.722(같은 설비의 뒤 기간), 0.479(학습에 없던 설비)
- LightGBM은 사건의 85.9%를 이상 발생 약 10분 전에 잡았고, 설비 1대당 하루 오경보는 규칙 12.4회에서 2.6회로 줄었다

## 데이터

| 항목 | 내용 |
| --- | --- |
| 출처 | AI-Hub 전력설비 라벨링 데이터 |
| 기간 | 2020-09-16 ~ 2021-02-08 |
| 규모 | 설비 105대 (펌프/일반모터 73대, 공조설비 32대), 약 394만 행 |
| 분할 | 같은 설비를 기간으로 나눈 Training / Validation |
| 주요 컬럼 | 상별 전류·전압·유효/무효전력·역률·고조파, 평균값, 온도, 주파수 |
| 라벨 | 기동패턴 2종(값: Stop/Loading/Unloading), SOH진단 3종(값: 정상/주의/경고) |
| 샘플링 주기 | 1분 |

원본과 변환 결과는 용량 문제로 저장소에 올리지 않는다.

## 전처리

1차로 설비 1대의 JSON 5개를 합쳐 설비별 표 1개로 만들고, 2차로 EDA에서 찾은 결측과 이상값을 정리했다. 자세한 내용은 [docs/preprocessing.md](docs/preprocessing.md)에 정리했다.

```mermaid
flowchart LR
  A[원본 JSON<br>설비당 5개] --> B[설비별 parquet<br>시각 × 35항목 + 라벨 5개] --> C[EDA<br>결측·이상값·라벨 점검] --> D[2차 전처리<br>1분 격자, 이상값 결측화] --> E[data/cleaned<br>209개 파일]
```

1차 결과를 EDA로 점검한 뒤 기준을 정해 2차 전처리를 했다.

- 결측치: 1분 격자에 맞추고 빈 분은 행으로 추가했다. 5분 이하 끊김만 선형 보간하고 더 긴 끊김(1시간 이상 73곳)은 결측으로 남겼다.
- 이상치: 전류 측정 상한값(512, 4095.94 A), 상전압 500 V 초과, 전원이 들어온 상태의 주파수 55~65 Hz 밖은 결측으로 바꿨다. 음수 전력·역률은 특정 설비에서 일관되게 나와 그대로 뒀다.
- 제외 구간: 전원이 95% 이상 꺼져 있던 Validation 80번 파일을 뺐다. 정지 구간은 지우지 않고 `flag_power_off`와 기동 라벨로 구분한다.
- 스케일링: 모델 단계에서 Training 기준으로 한다.

## 분석 방법

1. EDA: 210개 파일 전체에서 시간 간격, 결측, 0값·음수값, 측정 상한값, 라벨 분포와 라벨끼리의 관계를 집계했다. ([docs/eda.md](docs/eda.md))
2. 2차 전처리: EDA 결과로 기준을 정해 잘못된 값만 결측으로 바꾸고 표시 컬럼을 남겼다. ([docs/preprocessing.md](docs/preprocessing.md#2차-전처리))
3. 문제 정의와 정답 라벨: 현재 라벨 분류 대신 "정상 운전 중 30분 내 이상 발생" 이진 분류로 정하고 정답을 만들었다. ([docs/problem-definition.md](docs/problem-definition.md))
4. 특징 생성: 그 시점까지의 값으로 최근 10/30/60분 통계, 상간 불균형, 최근 이상 이력 등 148개를 만들었다.

조기 경보 기준과 근거는 아래와 같다.

| 기준 | 정한 값 | 근거 |
| --- | --- | --- |
| 운전 구간 | 기동 라벨이 Stop이 아닌 행 | 전류 > 0 기준이면 Stop 대기 전류(중앙값 2.1 A) 구간 6.4%가 섞이고, 이 구간은 고조파 '주의'가 28.8%로 부풀려져 있다 |
| 이상 발생 | 5분 이상 이어진 주의/경고의 시작 | 이상 구간의 48%가 5분 미만 깜빡임이다 |
| 예측 구간 | 30분 (10분, 60분은 추가 실험) | 양성 비율 10분 1.1%, 30분 2.3%, 60분 3.5%. 대응 시간과 예측 가능성의 균형 |
| 라벨 | 전류고조파 SOH (전압고조파는 보조) | 전압고조파는 정지 중에도 경고 29%, 주의 24%라 설비 자체 상태를 덜 반영한다 |

| 선택 | 대안 | 고른 이유 | 포기한 점 |
| --- | --- | --- | --- |
| 이상값을 결측으로 바꾸고 표시 | 이상값이 있는 행 삭제 | 1분 격자가 유지되어 윈도우를 자를 때 시간이 어긋나지 않는다 | 결측이 섞인 행을 모델 단계에서 따로 걸러야 한다 |
| 측정 범위·정격 기준 규칙 | 설비별 IQR | 정지/운전이 섞여 분포가 두 봉우리라 IQR이면 운전 구간 전체가 이상치로 잡힌다 | 범위 안에서 튀는 값은 못 잡는다 |
| 5분 이하만 보간 | 모든 결측 보간 | 긴 끊김을 채우면 없는 운전 패턴이 생긴다 | 결측 행 95,967행(2.4%)이 남는다 |
| 정지 구간 유지 | 정지 구간 삭제 | 기동 패턴 분석에 정지 → 운전 전환이 필요하다 | SOH 분석 때는 운전 구간만 따로 골라야 한다 |

## 실험 기록

### 1. 베이스라인 비교
[case-studies/baseline-comparison.md](case-studies/baseline-comparison.md)

"전류고조파가 일정 값을 넘으면 경보"하는 단순 규칙과 특징 148개를 쓴 Logistic Regression을 같은 기준으로 비교했다. 규칙만으로도 PR-AUC 0.326(무작위 0.029)이 나왔고, Logistic Regression은 0.493으로 더 높았다. 이후 모델은 규칙 대비 개선 폭으로 보고한다.

핵심 질문:
- 단순 임계값 규칙은 어디까지 맞히는가
- 여러 특징을 함께 쓰면 규칙보다 나아지는가
- 이상 발생 몇 분 전에, 오경보 몇 번으로 잡는가

### 2. 모델 비교
[case-studies/model-comparison.md](case-studies/model-comparison.md)

Random Forest와 LightGBM을 규칙, Logistic Regression과 같은 조건에서 비교했다. LightGBM이 PR-AUC 0.722로 가장 높았고, 중요도 상위는 최근 30/60분 전류고조파 최댓값과 마지막 이상 후 지난 분이었다. 이상 이력이 없는 시점을 따로 표시하는 방식도 함께 정했다.

핵심 질문:
- 트리 모델이 선형 모델보다 얼마나 나은가
- 모델은 어떤 신호를 주로 보는가
- 이상 이력이 없는 시점은 어떻게 다루는가

### 3. 설비 단위 분할
[case-studies/unit-split.md](case-studies/unit-split.md)

설비를 5묶음으로 나눠, 같은 평가 행에서 그 설비를 학습에 넣었을 때와 뺐을 때를 비교했다. 처음 보는 설비에서 LightGBM PR-AUC는 0.479로 떨어졌지만 규칙(0.326)보다는 높았고 5묶음 중 4묶음에서 우세했다.

핵심 질문:
- 같은 설비 기간 분할 성능 중 설비를 외운 효과는 얼마인가
- 처음 보는 설비에서도 규칙보다 나은가
- 묶음마다 결과가 얼마나 흔들리는가

### 4. 연속 경보 규칙
[case-studies/alarm-rule.md](case-studies/alarm-rule.md)

경보가 N분 이어질 때만 알리는 방식으로 오경보를 줄였다. LightGBM은 2분 대기에서 사건 재현율 74.9%, 하루 오경보 1.6회였고, 비슷한 오경보 수준에서 규칙보다 사건을 28.7%p 더 잡았다.

핵심 질문:
- 대기 시간을 늘리면 오경보와 놓친 사건이 어떻게 바뀌는가
- 오경보 수준을 맞추면 모델 간 차이는 얼마인가

## 결과와 해석

![모델별 PR-AUC](charts/model_pr_auc.png)

LightGBM은 같은 설비의 뒤 기간 평가에서 PR-AUC 0.722로 전류고조파 규칙(0.326)의 2.2배였다.

![본 적 있는 설비와 처음 보는 설비의 PR-AUC](charts/unit_split_pr_auc.png)

학습에 없던 설비에서는 0.479로 떨어졌다. 앞의 성능 중 상당 부분이 설비별 패턴을 외운 효과이지만, 규칙보다는 여전히 높았다.

![경보 대기 시간별 오경보와 사건 재현율](charts/alarm_rule_tradeoff.png)

같은 오경보 수준에서 LightGBM이 규칙보다 많은 사건을 미리 잡았다. 대기 시간은 현장이 오경보와 놓친 사건 중 무엇을 더 꺼리는지에 따라 고른다.

EDA 그래프와 해석(역률 라벨이 정지 여부를 반영하는 문제, SOH 라벨이 고정 경계 규칙 라벨인 근거, 전류 측정 상한)은 [docs/eda.md](docs/eda.md)에 정리했다.

## 한계와 다음 계획

- 공조설비는 30분 내 이상 발생이 Training 19건, Validation 0건뿐이라 사실상 펌프/일반모터 설비의 문제이다.
- 이상 발생 사건의 47%가 상위 5대 설비에 몰려 있다. 설비 단위 분할은 양성 수 순서로 묶음에 배정해 이 쏠림을 나눴다.
- 음수 유효전력 설비 4대(363, 406, 467, 548번)를 부호 반전할지 정하지 않았다.
- 1시간 넘게 끊긴 73곳은 2021-01-23 ~ 02-08에 몰려 있다. 이 기간을 쓰는 윈도우는 결측 처리 방식을 따로 정해야 한다.
- 처음 보는 설비에서는 PR-AUC가 0.722에서 0.479로 떨어지고 하루 오경보가 2.6회에서 6.3회로 늘었다. 묶음별 편차도 크다(0.370~0.796).
- LightGBM 중요도의 54.6%가 전류고조파 관련 특징 3개(최근 60/30분 최댓값, 최근 30분 평균)에 몰려 있다. 라벨이 전류고조파 경계로 정해지므로 다른 측정 항목에서 배운 것은 아직 작다.
- 다음 단계: 처음 보는 설비 성능을 높이는 설비별 표준화 특징, 하이퍼파라미터 탐색, SHAP 해석, 10/60분 예측 구간 비교

## 실행 방법

```bash
git clone https://github.com/RuruPark/sch_thesis.git
cd sch_thesis
pip install -r requirements.txt
python scripts/01_preprocess.py --raw-dir "<원본 데이터 폴더>"
python scripts/01b_export_csv.py
python scripts/02_eda.py
python scripts/03_clean.py
python scripts/04_make_target.py
python scripts/05_make_features.py
python scripts/06_baseline.py
python scripts/07_model_compare.py
python scripts/08_unit_split.py
python scripts/09_alarm_rule.py
python scripts/10_charts.py
```

## 폴더 구조

```text
sch_thesis/
+-- README.md
+-- CLAUDE.md              # Claude 작업 지침
+-- case-studies/
|   +-- baseline-comparison.md # 규칙 vs Logistic Regression
|   +-- model-comparison.md # Random Forest, LightGBM 비교
|   +-- unit-split.md      # 처음 보는 설비 평가
|   +-- alarm-rule.md      # 연속 경보 대기 시간
+-- docs/
|   +-- CONVENTIONS.md     # 정리 규칙
|   +-- preprocessing.md   # 전처리 기록 (1차, 2차)
|   +-- eda.md             # EDA 기록
|   +-- troubleshooting.md # 막혔던 문제 (STAR)
|   +-- problem-definition.md # 조기 경보 문제 정의와 기준 근거
|   +-- metrics.md         # 지표 정리
+-- scripts/
|   +-- 01_preprocess.py   # 원본 JSON → 설비별 parquet
|   +-- 01b_export_csv.py  # parquet → csv.gz 사본
|   +-- 02_eda.py          # 결측·이상값·라벨 점검, 그래프
|   +-- 03_clean.py        # 2차 전처리
|   +-- 04_make_target.py  # 30분 내 이상 발생 정답 라벨
|   +-- 05_make_features.py # 모델 입력 특징 148개
|   +-- 06_baseline.py     # 베이스라인 비교
|   +-- 07_model_compare.py # Random Forest, LightGBM 비교
|   +-- 08_unit_split.py   # 설비 단위 5묶음 분할 평가
|   +-- 09_alarm_rule.py   # 연속 경보 대기 시간 비교
|   +-- 10_charts.py       # 07~09번 결과 그래프
|   +-- evaluate.py        # 06번 이후 공통 함수 (데이터 읽기, 임계값, 지표)
+-- charts/                # README와 문서에 넣는 그래프
+-- data/                  # git에 올리지 않음
|   +-- raw/               # 원본
|   +-- processed/         # 1차 결과 parquet + meta.csv
|   +-- processed_csv/     # csv.gz
|   +-- eda/               # EDA 집계표
|   +-- cleaned/           # 2차 결과 parquet + clean_log.csv
|   +-- target/            # 설비별 정답 라벨 + target_summary.csv
|   +-- features/          # 설비별 특징 parquet (예측 대상 시점만)
|   +-- results/           # 실험 결과표
+-- requirements.txt
+-- .gitignore
```

## 작업 기록

| 날짜 | 한 일 | 관련 문서 |
| --- | --- | --- |
| 2026-10-10 | 원본 JSON을 설비별 parquet로 변환, 정리 규칙 작성 | [preprocessing.md](docs/preprocessing.md), [CONVENTIONS.md](docs/CONVENTIONS.md) |
| 2026-10-10 | 전체 설비 EDA, SOH 라벨이 규칙 라벨임을 확인 | [eda.md](docs/eda.md), `[기능/EDA]` |
| 2026-10-10 | 2차 전처리(1분 격자, 측정 상한값·전압 튐·주파수 이상 결측화, 5분 이하 보간) | [preprocessing.md](docs/preprocessing.md#2차-전처리), [troubleshooting.md](docs/troubleshooting.md), `[기능/전처리]` |
| 2026-10-10 | 조기 경보 문제 정의 확정, 정답 라벨과 특징 148개 생성 | [problem-definition.md](docs/problem-definition.md), `[기능/전처리]`, `[기능/특징]` |
| 2026-10-10 | 베이스라인 비교(항상 음성, 전류고조파 규칙, Logistic Regression) | [baseline-comparison.md](case-studies/baseline-comparison.md), `[실험/모델]` |
| 2026-10-10 | Random Forest·LightGBM 비교, 설비 단위 분할 평가, 연속 경보 규칙 비교 | [model-comparison.md](case-studies/model-comparison.md), [unit-split.md](case-studies/unit-split.md), [alarm-rule.md](case-studies/alarm-rule.md), `[실험/모델]` |
