"""검사 결과 시퀀스로부터 피처를 만드는 모듈 모음.

- `primitives` : csv 문자열 시퀀스에 대한 평균/표준편차/조건부 통계 헬퍼.
- `assessment_a` / `assessment_b` : A 검사·B 검사 도메인 피처.
- `history` : `PrimaryKey` 기준 과거 검사 이력 파생.
- `interactions` : 도메인 간 상호작용·속도-정확도 트레이드오프 등 derived feature.
"""

from .assessment_a import build_assessment_a_features
from .assessment_b import build_assessment_b_features
from .history import build_history_features
from .interactions import add_interaction_features

__all__ = [
    "build_assessment_a_features",
    "build_assessment_b_features",
    "build_history_features",
    "add_interaction_features",
]
