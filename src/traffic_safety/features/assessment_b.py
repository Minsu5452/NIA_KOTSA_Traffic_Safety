"""자격 유지 검사(B 검사) 기반 도메인 피처.

B 검사 컬럼은 5개 인지 과제(B1~B5) + 짧은 정답 시퀀스(B6~B8) + 청각/시각 점수
(B9, B10) 로 구성된다. 본 모듈은 각 과제별 정확도/반응시간 통계와 청각·시각
점수의 정규화·결합 피처를 만든다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .primitives import seq_cond_mean, seq_count, seq_mean, seq_rate, seq_std, to_sequence_matrix

SEQUENCE_LENGTHS: dict[str, int] = {
    "B1": 16,
    "B2": 16,
    "B3": 15,
    "B4": 60,
    "B5": 20,
    "B6": 15,
    "B7": 15,
    "B8": 12,
}

MISSING_RESPONSE = 0


def _eps_div(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return numerator / (denominator + eps)


def build_assessment_b_features(frame: pd.DataFrame) -> pd.DataFrame:
    """B 검사 원본 데이터프레임 → 피처 프레임."""

    out: dict[str, np.ndarray] = {}

    # --- B1, B2: 위치 + 색 매칭 ---
    for prefix in ("B1", "B2"):
        loc = to_sequence_matrix(frame[f"{prefix}-1"], SEQUENCE_LENGTHS[prefix], np.int8)
        rt = to_sequence_matrix(frame[f"{prefix}-2"], SEQUENCE_LENGTHS[prefix], np.float32)
        color = to_sequence_matrix(frame[f"{prefix}-3"], SEQUENCE_LENGTHS[prefix], np.int8)

        out[f"{prefix}_loc_acc"] = seq_rate(loc, 1)
        out[f"{prefix}_color_acc"] = seq_rate(color, [1, 3])
        out[f"{prefix}_rt_mean"] = seq_mean(rt)
        out[f"{prefix}_rt_std"] = seq_std(rt)
        out[f"{prefix}_rt_cv"] = _eps_div(out[f"{prefix}_rt_std"], out[f"{prefix}_rt_mean"])

    out["B1_to_B2_loc_acc_diff"] = out["B1_loc_acc"] - out["B2_loc_acc"]
    out["B1_to_B2_color_acc_diff"] = out["B1_color_acc"] - out["B2_color_acc"]

    # --- B3 ---
    b3_resp = to_sequence_matrix(frame["B3-1"], SEQUENCE_LENGTHS["B3"], np.int8)
    b3_rt = to_sequence_matrix(frame["B3-2"], SEQUENCE_LENGTHS["B3"], np.float32)
    out["B3_acc"] = seq_rate(b3_resp, 1)
    out["B3_rt_mean"] = seq_mean(b3_rt, mask=(b3_resp != MISSING_RESPONSE))
    out["B3_rt_std"] = seq_std(b3_rt, mask=(b3_resp != MISSING_RESPONSE))
    out["B3_rt_cv"] = _eps_div(out["B3_rt_std"], out["B3_rt_mean"])

    # --- B4: Stroop 유사 (1=일치 정답, 3/5=불일치 정답) ---
    b4_resp = to_sequence_matrix(frame["B4-1"], SEQUENCE_LENGTHS["B4"], np.int8)
    b4_rt = to_sequence_matrix(frame["B4-2"], SEQUENCE_LENGTHS["B4"], np.float32)
    congruent_mask = np.isin(b4_resp, [1, 2])
    incongruent_mask = np.isin(b4_resp, [3, 4, 5, 6])
    congruent_count = congruent_mask.sum(axis=1).astype(np.float32)
    incongruent_count = incongruent_mask.sum(axis=1).astype(np.float32)
    out["B4_acc"] = seq_rate(b4_resp, 1)
    out["B4_acc_congruent"] = _eps_div((b4_resp == 1).sum(axis=1).astype(np.float32), congruent_count)
    out["B4_acc_incongruent"] = _eps_div(np.isin(b4_resp, [3, 5]).sum(axis=1).astype(np.float32), incongruent_count)
    out["B4_acc_stroop_gap"] = out["B4_acc_congruent"] - out["B4_acc_incongruent"]
    out["B4_rt_mean"] = seq_mean(b4_rt, mask=(b4_resp != MISSING_RESPONSE))
    out["B4_rt_std"] = seq_std(b4_rt, mask=(b4_resp != MISSING_RESPONSE))
    out["B4_rt_cv"] = _eps_div(out["B4_rt_std"], out["B4_rt_mean"])
    out["B4_rt_congruent"] = seq_cond_mean(b4_resp, b4_rt, [1, 2])
    out["B4_rt_incongruent"] = seq_cond_mean(b4_resp, b4_rt, [3, 4, 5, 6])
    out["B4_rt_stroop_gap"] = out["B4_rt_incongruent"] - out["B4_rt_congruent"]

    # --- B5 ---
    b5_resp = to_sequence_matrix(frame["B5-1"], SEQUENCE_LENGTHS["B5"], np.int8)
    out["B5_acc"] = seq_rate(b5_resp, 1)

    # --- B6, B7, B8 (정답 시퀀스만) ---
    for prefix, length in (("B6", SEQUENCE_LENGTHS["B6"]), ("B7", SEQUENCE_LENGTHS["B7"]), ("B8", SEQUENCE_LENGTHS["B8"])):
        resp = to_sequence_matrix(frame[prefix], length, np.int8)
        out[f"{prefix}_acc"] = seq_rate(resp, 1)
    out["B6to8_acc_mean"] = (out["B6_acc"] + out["B7_acc"] + out["B8_acc"]) / 3.0

    # --- B9: 청각 검사 ---
    if {"B9-1", "B9-4", "B9-5"}.issubset(frame.columns):
        b91 = pd.to_numeric(frame["B9-1"], errors="coerce").fillna(0).to_numpy(np.float32)
        b94 = pd.to_numeric(frame["B9-4"], errors="coerce").fillna(0).to_numpy(np.float32)
        b95 = pd.to_numeric(frame["B9-5"], errors="coerce").fillna(0).to_numpy(np.float32)
        out["B9_score"] = b91 + b94 - b95
        out["B9_hit_ratio"] = b91 / 50.0

    # --- B10: 시각 검사 ---
    if {"B10-1", "B10-4", "B10-5", "B10-6"}.issubset(frame.columns):
        b101 = pd.to_numeric(frame["B10-1"], errors="coerce").fillna(0).to_numpy(np.float32)
        b104 = pd.to_numeric(frame["B10-4"], errors="coerce").fillna(0).to_numpy(np.float32)
        b105 = pd.to_numeric(frame["B10-5"], errors="coerce").fillna(0).to_numpy(np.float32)
        b106 = pd.to_numeric(frame["B10-6"], errors="coerce").fillna(0).to_numpy(np.float32)
        out["B10_score"] = b101 + b104 - b105 + b106
        out["B10_err_ratio"] = (b105 / 32.0).astype(np.float32)

    if "B9_score" in out and "B10_score" in out:
        out["B9_to_B10_score_gap"] = out["B9_score"] - out["B10_score"]

    feature_frame = pd.DataFrame(out)
    feature_frame.insert(0, "Test_id", frame["Test_id"].to_numpy())
    return feature_frame


__all__ = ["build_assessment_b_features", "SEQUENCE_LENGTHS"]
