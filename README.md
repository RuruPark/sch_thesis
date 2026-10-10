# 전력설비 센서 데이터 기반 에너지 패턴 및 고장 분석

> AI-Hub 전력설비 센서 데이터(펌프/일반모터, 공조설비 105대)로 기동 패턴과 설비 상태(SOH)를 분석하는 졸업논문 코드

## 연구 질문

- 설비의 전력 사용 패턴으로 기동 상태와 이상 징후를 구분할 수 있는가
- (분석 진행에 따라 구체화 예정)

## 핵심 결과

- 전처리 완료: 설비 105대, 약 394만 행을 1분 단위 "시각 × 측정항목 35개 + 라벨 5개" 표로 변환
- 분석 결과는 진행 후 추가

## 데이터

| 항목 | 내용 |
| --- | --- |
| 출처 | AI-Hub 전력설비 라벨링 데이터 |
| 기간 | 2020-09-16 ~ 2021-02-08 |
| 규모 | 설비 105대 (펌프/일반모터 73대, 공조설비 32대), 약 394만 행 |
| 분할 | 같은 설비를 기간으로 나눈 Training / Validation |
| 주요 컬럼 | 상별 전류·전압·유효/무효전력·역률·고조파, 평균값, 온도, 주파수 |
| 라벨 | 기동패턴 2종(Stop/Loading), SOH진단 3종(정상/주의/경고) |
| 샘플링 주기 | 1분 |

원본과 변환 결과는 용량 문제로 저장소에 올리지 않는다.

## 전처리

설비 1대의 JSON 5개를 합쳐 설비별 표 1개로 만들었다. 자세한 내용은 [docs/preprocessing.md](docs/preprocessing.md)에 정리했다.

```mermaid
flowchart LR
  A[원본 JSON<br>설비당 5개] --> B[센서값 1회 읽기<br>중복 제거] --> C[가로형 변환<br>측정항목 35개] --> D[라벨 5개 결합] --> E[설비별 parquet<br>+ meta.csv]
```

센서값은 5개 파일에서 같아서 한 번만 읽고, 나머지 파일에서는 라벨만 가져왔다.

- 결측치: 아직 처리하지 않음
- 이상치: 아직 처리하지 않음
- 제외 구간: 없음 (설비 정지 구간 포함 여부는 다음 단계에서 결정)

## 분석 방법

진행 후 추가

## 결과와 해석

진행 후 추가

## 한계와 다음 계획

- 결측치, 이상치, 정지 구간 처리 기준 정하기
- 라벨 분포를 설비 전체 기준으로 확인하기

## 실행 방법

```bash
git clone https://github.com/RuruPark/sch_thesis.git
cd sch_thesis
pip install -r requirements.txt
python scripts/01_preprocess.py --raw-dir "<원본 데이터 폴더>"
python scripts/01b_export_csv.py
```

## 폴더 구조

```text
sch_thesis/
+-- README.md
+-- CLAUDE.md              # Claude 작업 지침
+-- docs/
|   +-- CONVENTIONS.md     # 정리 규칙
|   +-- preprocessing.md   # 전처리 기록
+-- scripts/
|   +-- 01_preprocess.py   # 원본 JSON → 설비별 parquet
|   +-- 01b_export_csv.py  # parquet → csv.gz 사본
+-- data/                  # git에 올리지 않음
|   +-- raw/               # 원본
|   +-- processed/         # parquet + meta.csv
|   +-- processed_csv/     # csv.gz
+-- requirements.txt
+-- .gitignore
```

## 작업 기록

| 날짜 | 한 일 | 관련 문서 |
| --- | --- | --- |
| 2026-10-10 | 원본 JSON을 설비별 parquet로 변환, 정리 규칙 작성 | [preprocessing.md](docs/preprocessing.md), [CONVENTIONS.md](docs/CONVENTIONS.md) |
