# 운수종사자 인지 특성 위험 예측

운수종사자 신규검사(A)와 자격유지검사(B)의 인지·행동 시퀀스 데이터를 이용해 다음 검사에서의 위험 확률을 예측한 DACON 경진대회 프로젝트입니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 대회 | 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회 |
| 기간 | 2025.10.13 - 2025.11.14 |
| 주최 / 주관 | 행정안전부, 한국지능정보사회진흥원 / 한국교통안전공단 |
| 팀 구성 | 개인 |
| 결과 | 34등 / 437팀, Top 7.8% |
| 과제 | 검사 시퀀스와 과거 이력 기반 위험 확률 예측 |
| 모델 | LightGBM, CatBoost, weighted blend |

## 접근

- A/B 검사의 컬럼 체계가 달라 도메인별 feature builder를 분리했습니다.
- 콤마로 저장된 검사 시퀀스를 행렬로 변환하고 평균, 표준편차, 변동계수, 조건부 평균 등을 만들었습니다.
- 동일 운수종사자의 과거 검사 이력을 현재 시점 이전 데이터만 사용해 누설 없이 구성했습니다.
- `StratifiedGroupKFold`를 사용해 같은 `PrimaryKey`가 train/valid 양쪽에 동시에 들어가지 않도록 했습니다.
- LightGBM과 CatBoost OOF prediction을 weighted blend해 최종 제출을 생성했습니다.

## 저장소 구성

```text
.
├── configs/default.yaml
├── scripts/
│   ├── train.py
│   └── infer.py
├── src/traffic_safety/
│   ├── features/
│   ├── io.py
│   ├── pipeline.py
│   ├── models.py
│   └── cli.py
├── pyproject.toml
└── requirements.txt
```

## 실행

대회 파일을 `data/` 아래에 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/A.csv
├── train/B.csv
├── test/A.csv
└── test/B.csv
```

```bash
pip install -e .
python -m traffic_safety.cli train --config configs/default.yaml
python -m traffic_safety.cli infer --config configs/default.yaml
```

## 공개 범위

대회 원본 데이터, 학습된 fold 모델, 로그, 제출 파일은 포함하지 않았습니다. 공개 저장소에는 패키지 형태의 학습·추론 파이프라인만 남겼습니다.
