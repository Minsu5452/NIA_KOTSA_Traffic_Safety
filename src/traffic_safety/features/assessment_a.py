"""신규 자격 검사(A 검사) 기반 도메인 피처.

대회 컬럼 규약(요약):

| 컬럼 | 의미 |
| --- | --- |
| `A1-1` | 자극 위치(좌/우) | `A1-2` 자극 속도(저/중/고) | `A1-3` 응답 정오 | `A1-4` 반응시간 |
| `A2-1` | 조건1 | `A2-2` 조건2 | `A2-3` 응답 | `A2-4` 반응시간 |
| `A3-1..3-4` | 자극 메타 | `A3-5` 응답 라벨 | `A3-6` 응답 정오 | `A3-7` 반응시간 |
| `A4-1` | 일치/불일치 | `A4-2` 색 | `A4-3` 정확도 | `A4-4` 응답 | `A4-5` 반응시간 |
| `A5-1` | 변화 유형 | `A5-2` 정확도 | `A5-3` 응답 |
| `A6-1`, `A7-1`, `A8-*`, `A9-*` | 보조 점수 |

각 시퀀스 컬럼은 콤마 구분 csv 문자열이며 길이는 모든 행에서 동일하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .primitives import (
    seq_cond_mean,
    seq_count,
    seq_mean,
    seq_rate,
    seq_std,
    to_sequence_matrix,
)

# 대회 사전에서 명시된 검사 시퀀스 길이
SEQUENCE_LENGTHS: dict[str, int] = {
    "A1": 18,
    "A2": 18,
    "A3": 32,
    "A4": 80,
    "A5": 36,
}

# 응답 누락 표시 값(`*-3` 또는 `*-6` 등에 들어 있는 0)
MISSING_RESPONSE = 0


def _eps_div(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return numerator / (denominator + eps)


def build_assessment_a_features(frame: pd.DataFrame) -> pd.DataFrame:
    """A 검사 원본 데이터프레임 → 모델 입력용 피처 프레임."""

    out: dict[str, np.ndarray] = {}

    # --- A1: 좌/우 자극 + 속도 조건 ---
    a1_loc = to_sequence_matrix(frame["A1-1"], SEQUENCE_LENGTHS["A1"], np.int8)
    a1_speed = to_sequence_matrix(frame["A1-2"], SEQUENCE_LENGTHS["A1"], np.int8)
    a1_resp = to_sequence_matrix(frame["A1-3"], SEQUENCE_LENGTHS["A1"], np.int8)
    a1_rt = to_sequence_matrix(frame["A1-4"], SEQUENCE_LENGTHS["A1"], np.float32)

    out["A1_resp_correct_rate"] = seq_rate(a1_resp, 1)
    out["A1_resp_left_rate"] = seq_cond_mean(a1_loc, a1_resp == 1, 1)
    out["A1_resp_right_rate"] = seq_cond_mean(a1_loc, a1_resp == 1, 2)
    out["A1_resp_lr_diff"] = out["A1_resp_left_rate"] - out["A1_resp_right_rate"]

    out["A1_rt_mean"] = seq_mean(a1_rt, mask=(a1_resp != MISSING_RESPONSE))
    out["A1_rt_std"] = seq_std(a1_rt, mask=(a1_resp != MISSING_RESPONSE))
    out["A1_rt_cv"] = _eps_div(out["A1_rt_std"], out["A1_rt_mean"])
    out["A1_rt_slow"] = seq_cond_mean(a1_speed, a1_rt, 1, miss=a1_resp, miss_values=[1, 2])
    out["A1_rt_fast"] = seq_cond_mean(a1_speed, a1_rt, 3, miss=a1_resp, miss_values=[1, 2])
    out["A1_rt_speed_diff"] = out["A1_rt_slow"] - out["A1_rt_fast"]
    out["A1_miss_rate"] = (a1_resp == MISSING_RESPONSE).mean(axis=1).astype(np.float32)

    # --- A2: 두 조건 동시 변화 ---
    a2_cond1 = to_sequence_matrix(frame["A2-1"], SEQUENCE_LENGTHS["A2"], np.int8)
    a2_cond2 = to_sequence_matrix(frame["A2-2"], SEQUENCE_LENGTHS["A2"], np.int8)
    a2_resp = to_sequence_matrix(frame["A2-3"], SEQUENCE_LENGTHS["A2"], np.int8)
    a2_rt = to_sequence_matrix(frame["A2-4"], SEQUENCE_LENGTHS["A2"], np.float32)

    out["A2_resp_correct_rate"] = seq_rate(a2_resp, 1)
    for cond_name, cond_mat in (("c1", a2_cond1), ("c2", a2_cond2)):
        out[f"A2_resp_{cond_name}_slow"] = seq_cond_mean(cond_mat, a2_resp == 1, 1)
        out[f"A2_resp_{cond_name}_fast"] = seq_cond_mean(cond_mat, a2_resp == 1, 3)
        out[f"A2_rt_{cond_name}_slow"] = seq_cond_mean(cond_mat, a2_rt, 1, miss=a2_resp, miss_values=[1, 2])
        out[f"A2_rt_{cond_name}_fast"] = seq_cond_mean(cond_mat, a2_rt, 3, miss=a2_resp, miss_values=[1, 2])
        out[f"A2_rt_{cond_name}_diff"] = out[f"A2_rt_{cond_name}_slow"] - out[f"A2_rt_{cond_name}_fast"]
    out["A2_rt_mean"] = seq_mean(a2_rt, mask=(a2_resp != MISSING_RESPONSE))
    out["A2_rt_std"] = seq_std(a2_rt, mask=(a2_resp != MISSING_RESPONSE))
    out["A2_rt_cv"] = _eps_div(out["A2_rt_std"], out["A2_rt_mean"])
    out["A2_miss_rate"] = (a2_resp == MISSING_RESPONSE).mean(axis=1).astype(np.float32)

    # --- A3: 자극 크기/위치 매칭 ---
    a3_size = to_sequence_matrix(frame["A3-1"], SEQUENCE_LENGTHS["A3"], np.int8)
    a3_loc = to_sequence_matrix(frame["A3-3"], SEQUENCE_LENGTHS["A3"], np.int8)
    a3_label = to_sequence_matrix(frame["A3-5"], SEQUENCE_LENGTHS["A3"], np.int8)
    a3_resp = to_sequence_matrix(frame["A3-6"], SEQUENCE_LENGTHS["A3"], np.int8)
    a3_rt = to_sequence_matrix(frame["A3-7"], SEQUENCE_LENGTHS["A3"], np.float32)

    a3_correct = ((a3_label == 1) | (a3_label == 3)) & (a3_resp != MISSING_RESPONSE)
    out["A3_correct_ratio"] = a3_correct.mean(axis=1).astype(np.float32)
    out["A3_invalid_ratio"] = ((a3_label == 4) & (a3_resp != MISSING_RESPONSE)).mean(axis=1).astype(np.float32)

    out["A3_rt_mean"] = seq_mean(a3_rt, mask=(a3_resp != MISSING_RESPONSE))
    out["A3_rt_std"] = seq_std(a3_rt, mask=(a3_resp != MISSING_RESPONSE))
    out["A3_rt_cv"] = _eps_div(out["A3_rt_std"], out["A3_rt_mean"])
    out["A3_rt_size_diff"] = (
        seq_cond_mean(a3_size, a3_rt, 1, miss=a3_resp, miss_values=[1, 2])
        - seq_cond_mean(a3_size, a3_rt, 2, miss=a3_resp, miss_values=[1, 2])
    )
    out["A3_rt_lr_diff"] = (
        seq_cond_mean(a3_loc, a3_rt, 1, miss=a3_resp, miss_values=[1, 2])
        - seq_cond_mean(a3_loc, a3_rt, 2, miss=a3_resp, miss_values=[1, 2])
    )
    out["A3_miss_rate"] = (a3_resp == MISSING_RESPONSE).mean(axis=1).astype(np.float32)

    # --- A4: Stroop ---
    a4_congruency = to_sequence_matrix(frame["A4-1"], SEQUENCE_LENGTHS["A4"], np.int8)
    a4_color = to_sequence_matrix(frame["A4-2"], SEQUENCE_LENGTHS["A4"], np.int8)
    a4_correct = to_sequence_matrix(frame["A4-3"], SEQUENCE_LENGTHS["A4"], np.int8)
    a4_resp = to_sequence_matrix(frame["A4-4"], SEQUENCE_LENGTHS["A4"], np.int8)
    a4_rt = to_sequence_matrix(frame["A4-5"], SEQUENCE_LENGTHS["A4"], np.float32)

    out["A4_acc_rate"] = seq_rate(a4_correct, 1)
    out["A4_acc_congruent"] = seq_cond_mean(a4_congruency, a4_correct == 1, 1)
    out["A4_acc_incongruent"] = seq_cond_mean(a4_congruency, a4_correct == 1, 2)
    out["A4_acc_stroop_gap"] = out["A4_acc_congruent"] - out["A4_acc_incongruent"]
    out["A4_rt_mean"] = seq_mean(a4_rt, mask=(a4_resp != MISSING_RESPONSE))
    out["A4_rt_std"] = seq_std(a4_rt, mask=(a4_resp != MISSING_RESPONSE))
    out["A4_rt_cv"] = _eps_div(out["A4_rt_std"], out["A4_rt_mean"])
    out["A4_rt_stroop_gap"] = (
        seq_cond_mean(a4_congruency, a4_rt, 2, miss=a4_resp, miss_values=[1, 2])
        - seq_cond_mean(a4_congruency, a4_rt, 1, miss=a4_resp, miss_values=[1, 2])
    )
    out["A4_rt_color_gap"] = (
        seq_cond_mean(a4_color, a4_rt, 1, miss=a4_resp, miss_values=[1, 2])
        - seq_cond_mean(a4_color, a4_rt, 2, miss=a4_resp, miss_values=[1, 2])
    )
    out["A4_miss_rate"] = (a4_resp == MISSING_RESPONSE).mean(axis=1).astype(np.float32)

    # --- A5: 변화 감지 ---
    a5_change = to_sequence_matrix(frame["A5-1"], SEQUENCE_LENGTHS["A5"], np.int8)
    a5_correct = to_sequence_matrix(frame["A5-2"], SEQUENCE_LENGTHS["A5"], np.int8)
    a5_resp = to_sequence_matrix(frame["A5-3"], SEQUENCE_LENGTHS["A5"], np.int8)

    out["A5_acc_rate"] = seq_rate(a5_correct, 1)
    out["A5_acc_nonchange"] = seq_cond_mean(a5_change, a5_correct == 1, 1)
    out["A5_acc_change"] = seq_cond_mean(a5_change, a5_correct == 1, [2, 3, 4])
    out["A5_acc_change_gap"] = out["A5_acc_change"] - out["A5_acc_nonchange"]
    out["A5_resp_change_rate"] = seq_cond_mean(a5_change, a5_resp == 1, [2, 3, 4])
    out["A5_miss_rate"] = (a5_resp == MISSING_RESPONSE).mean(axis=1).astype(np.float32)

    # --- A6 / A7: 보조 점수 (점수 만점 기준 정규화) ---
    if "A6-1" in frame.columns:
        out["A6_correct_ratio"] = (frame["A6-1"].fillna(0) / 14.0).to_numpy(np.float32)
    if "A7-1" in frame.columns:
        out["A7_correct_ratio"] = (frame["A7-1"].fillna(0) / 18.0).to_numpy(np.float32)

    feature_frame = pd.DataFrame(out)
    feature_frame.insert(0, "Test_id", frame["Test_id"].to_numpy())
    return feature_frame


__all__ = ["build_assessment_a_features", "SEQUENCE_LENGTHS"]
