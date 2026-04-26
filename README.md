# NIA KOTSA Traffic Safety

> 2025 행정안전부·NIA 주최 / 한국교통안전공단 주관 **"운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회"** 참여 프로젝트입니다. 운수종사자 신규 자격검사(A) 와 자격유지검사(B) 의 인지·행동 시퀀스 데이터로부터 다음 검사에서의 위험(부적합) 확률을 예측합니다.

## Overview

| 항목 | 내용 |
| --- | --- |
| 대회 | 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회 |
| 기간 | 2025.10.13 ~ 2025.11.14 |
| 주최 | 행정안전부, 한국지능정보사회진흥원(NIA) |
| 주관 | 한국교통안전공단(KOTSA) |
| 팀 구성 | 개인 |
| 결과 | 34등 / 437팀 |
| 과제 | 인지검사 결과 시퀀스 → 다음 검사 위험 확률 예측 (이진 분류, AUC) |

## Approach

- **도메인 분리 + 단일 모델**: A 검사(신규 자격) 와 B 검사(자격 유지) 는 컬럼 셋이 완전히 다르므로 각각 도메인 피처를 만든 뒤, 한 데이터프레임으로 합쳐 단일 모델이 도메인 지표(`is_assessment_b`) 와 함께 학습하도록 했습니다.
- **시퀀스 → 통계 피처**: `A1-3`, `B4-1` 등 콤마 csv 문자열 시퀀스를 numpy 행렬로 행렬화한 뒤 평균·표준편차·변동계수·조건부 평균 등을 일괄 계산하는 헬퍼(`features/primitives.py`)를 마련했습니다.
- **PrimaryKey 단위 과거 이력**: 같은 운수종사자가 여러 번 검사를 받는 구조라 누적 합격/불합격 카운트와 마지막 검사 이후 개월 수를 누설 없이 (현재 행 미만 시점만 사용) 계산해 강한 신호로 추가했습니다.
- **상호작용 피처**: 정확도-반응시간 트레이드오프, Stroop gap, 변동계수 등을 결합한 휴리스틱 risk score 두 종(A/B) 을 추가했습니다.
- **모델 / 검증**: LightGBM + CatBoost 5-fold `StratifiedGroupKFold` (그룹: `PrimaryKey`) — 동일 종사자가 train/valid 양쪽에 노출되지 않게 막아 일반화 성능을 보수적으로 측정합니다. 두 모델의 OOF AUC 를 weighted blend(0.55:0.45) 해 최종 제출을 만듭니다.

## Repository Structure

```text
.
├── configs/
│   └── default.yaml           # 데이터/스플릿/모델/앙상블/출력 경로 설정
├── scripts/
│   ├── train.py               # python scripts/train.py --config configs/default.yaml
│   └── infer.py               # python scripts/infer.py --config configs/default.yaml
├── src/
│   └── traffic_safety/
│       ├── __init__.py
│       ├── config.py          # TrainingConfig (yaml 로딩)
│       ├── io.py              # 메타 + A/B csv 로딩
│       ├── pipeline.py        # build_dataset (도메인 피처 + 이력 + 상호작용 결합)
│       ├── models.py          # LGBM/CatBoost KFold 학습·블렌딩·추론
│       ├── cli.py             # train / infer entry point
│       └── features/
│           ├── primitives.py  # 시퀀스 통계 헬퍼 (numpy 기반)
│           ├── assessment_a.py
│           ├── assessment_b.py
│           ├── history.py     # PrimaryKey 누적 이력 피처
│           └── interactions.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Run

대회 데이터(`train.csv`, `test.csv`, `sample_submission.csv`, `train/A.csv`, `train/B.csv`, `test/A.csv`, `test/B.csv`) 를 `data/` 경로에 배치합니다.

```bash
pip install -e .

# 학습 + OOF + 제출
python -m traffic_safety.cli train --config configs/default.yaml

# 저장된 fold 모델로 재추론만
python -m traffic_safety.cli infer --config configs/default.yaml
```

학습이 끝나면 다음 산출물이 생성됩니다.

- `models/lightgbm_fold{1..5}.joblib`, `models/catboost_fold{1..5}.joblib`, `models/history.json`
- `artifacts/oof.csv` — fold OOF 확률 + 라벨
- `submissions/submission.csv` — sample_submission 형식의 최종 제출

## Public Scope

이 저장소는 포트폴리오 공개용으로 정리한 버전입니다.

- 대회 제공 데이터, 학습된 모델, 제출 CSV, 로그는 포함하지 않습니다 (`.gitignore` 로 차단).
- 코드는 패키지 형태(`src/traffic_safety/`) 로 정리해 학습/추론을 동일 인터페이스로 재현할 수 있게 했습니다.
- 노트북 위주 흐름 대신 modular 패키지 + CLI 진입점 구조로 구성했습니다.

## Links

- [DACON competition page](https://dacon.io/competitions/official/236501/overview/description)
