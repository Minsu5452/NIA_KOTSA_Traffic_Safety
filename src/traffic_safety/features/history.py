"""`PrimaryKey` 기반 과거 검사 이력 피처.

같은 운수종사자가 여러 번 검사를 받는 구조라, 과거 합격/불합격 이력은 다음 검사의
강한 신호다. 학습 셋은 train 자체의 시간순 누적값으로, 테스트 셋은 학습에서
미리 만든 (`PrimaryKey` → 이력) 사전을 바탕으로 계산한다.

타깃 누설을 막기 위해 학습 단계에서는 "현재 행 미만" 시점의 이력만 사용한다.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import json
import numpy as np
import pandas as pd


def _to_year_month(date_value: int | float | str) -> int:
    """`YYMM` 또는 `YYYYMM` 모두 받아 절대 month index 로 변환."""

    raw = str(date_value).strip()
    if len(raw) == 4:  # YYMM
        year = 2000 + int(raw[:2])
        month = int(raw[2:])
    elif len(raw) == 6:  # YYYYMM
        year = int(raw[:4])
        month = int(raw[4:])
    else:
        raise ValueError(f"unexpected TestDate format: {date_value!r}")
    return year * 12 + (month - 1)


def _add_year_month(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_year_month"] = out["TestDate"].map(_to_year_month).astype(np.int32)
    return out


def build_history_features(
    frame: pd.DataFrame,
    *,
    history_path: Path | None = None,
    update_history: bool = False,
) -> pd.DataFrame:
    """과거 이력 기반 누적 피처를 추가한 데이터프레임을 반환.

    Parameters
    ----------
    frame
        `PrimaryKey`, `TestDate`, (`Label` if train) 컬럼을 포함한 메타+상세 셋.
    history_path
        json 형태로 저장한 (`PrimaryKey` → 이력 리스트). 추론 시 학습 단계에서
        만든 파일을 그대로 로드한다.
    update_history
        True 면 현재 frame 의 행을 이력에 합쳐 `history_path` 에 저장한다.
    """

    work = _add_year_month(frame)
    work = work.sort_values(["PrimaryKey", "_year_month", "Test_id"], kind="stable").reset_index(drop=False)

    history: dict[str, list[dict[str, int]]] = defaultdict(list)
    if history_path is not None and history_path.exists():
        history.update(json.loads(history_path.read_text(encoding="utf-8")))

    counts = np.zeros(len(work), dtype=np.int32)
    successes = np.zeros(len(work), dtype=np.int32)
    failures = np.zeros(len(work), dtype=np.int32)
    months_since_last = np.full(len(work), -1, dtype=np.int32)

    has_label = "Label" in work.columns

    for row_idx, row in work.iterrows():
        pk = str(row["PrimaryKey"])
        ym = int(row["_year_month"])
        past = [item for item in history.get(pk, []) if item["year_month"] < ym]
        counts[row_idx] = len(past)
        if past:
            successes[row_idx] = sum(1 for h in past if h.get("label") == 1)
            failures[row_idx] = sum(1 for h in past if h.get("label") == 0)
            months_since_last[row_idx] = ym - max(h["year_month"] for h in past)

        if has_label and update_history:
            label_value = row["Label"]
            if pd.notna(label_value):
                history[pk].append({"year_month": ym, "label": int(label_value)})

    work["pk_past_count"] = counts
    work["pk_past_success_count"] = successes
    work["pk_past_fail_count"] = failures
    work["pk_past_success_rate"] = np.where(counts > 0, successes / np.maximum(counts, 1), 0.0).astype(np.float32)
    work["pk_months_since_last_exam"] = months_since_last

    if update_history and history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

    work = work.sort_values("index").drop(columns=["index", "_year_month"]).reset_index(drop=True)
    return work


__all__ = ["build_history_features"]
