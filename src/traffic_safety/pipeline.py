"""학습/추론에 공통으로 쓰이는 dataset 빌더.

A 검사·B 검사 피처를 각각 만든 뒤 메타와 join 하고, `PrimaryKey` 단위 과거 이력
파생까지 붙인 단일 모델 입력 프레임을 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .features import (
    add_interaction_features,
    build_assessment_a_features,
    build_assessment_b_features,
    build_history_features,
)
from .io import CompetitionFrames

META_COLUMNS = ("Test_id", "Test", "PrimaryKey", "Age", "TestDate")
NON_FEATURE_COLUMNS = (*META_COLUMNS, "Label")


def _encode_age(series: pd.Series) -> pd.Series:
    """`30a` → 30, `30b` → 35 형태의 5살 단위 인코딩."""

    raw = series.astype(str).str.strip()
    decade = pd.to_numeric(raw.str[:-1], errors="coerce")
    band = raw.str[-1:].map({"a": 0, "b": 5}).astype("float32")
    return (decade + band).astype("float32")


def _build_side_frame(
    detail_a: pd.DataFrame,
    detail_b: pd.DataFrame,
    *,
    history_path: Path,
    update_history: bool,
) -> pd.DataFrame:
    feature_a = build_assessment_a_features(detail_a)
    feature_b = build_assessment_b_features(detail_b)

    meta_a = detail_a[list(META_COLUMNS) + ([] if "Label" not in detail_a.columns else ["Label"])]
    meta_b = detail_b[list(META_COLUMNS) + ([] if "Label" not in detail_b.columns else ["Label"])]

    enriched_a = meta_a.merge(feature_a, on="Test_id", how="left")
    enriched_b = meta_b.merge(feature_b, on="Test_id", how="left")

    combined = pd.concat([enriched_a, enriched_b], ignore_index=True, sort=False)
    combined = combined.sort_values(["PrimaryKey", "TestDate", "Test_id"]).reset_index(drop=True)

    combined["age_band"] = _encode_age(combined["Age"])
    combined["test_year"] = pd.to_numeric(combined["TestDate"].astype(str).str[:-2], errors="coerce").astype("float32")
    combined["test_month"] = pd.to_numeric(combined["TestDate"].astype(str).str[-2:], errors="coerce").astype("float32")
    combined["is_assessment_b"] = (combined["Test"] == "B").astype("int8")

    combined = build_history_features(
        combined,
        history_path=history_path,
        update_history=update_history,
    )
    combined = add_interaction_features(combined)
    return combined


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]


def build_dataset(frames: CompetitionFrames, config: TrainingConfig) -> DatasetBundle:
    """학습/테스트 입력 프레임을 한 번에 만든다."""

    history_path = config.output.model_dir / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = _build_side_frame(
        frames.a_train,
        frames.b_train,
        history_path=history_path,
        update_history=True,
    )
    test_df = _build_side_frame(
        frames.a_test,
        frames.b_test,
        history_path=history_path,
        update_history=False,
    )

    feature_columns = sorted(
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    )
    train_df = train_df[[*NON_FEATURE_COLUMNS, *feature_columns]]
    test_df = test_df[[*META_COLUMNS, *feature_columns]]

    return DatasetBundle(train=train_df, test=test_df, feature_columns=feature_columns)


__all__ = ["DatasetBundle", "build_dataset"]
