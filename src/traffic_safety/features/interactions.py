"""파생/상호작용 피처.

도메인 피처가 만들어진 뒤 한 번 더 결합 통계를 만든다. 부정 정확도-RT 트레이드
오프와 변동계수 휴리스틱 위험 점수가 핵심.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_ACC_RT_PAIRS_A: tuple[tuple[str, str], ...] = (
    ("A1_resp_correct_rate", "A1_rt_mean"),
    ("A2_resp_correct_rate", "A2_rt_mean"),
    ("A3_correct_ratio", "A3_rt_mean"),
    ("A4_acc_rate", "A4_rt_mean"),
)

_ACC_RT_PAIRS_B: tuple[tuple[str, str], ...] = (
    ("B1_loc_acc", "B1_rt_mean"),
    ("B2_loc_acc", "B2_rt_mean"),
    ("B3_acc", "B3_rt_mean"),
    ("B4_acc", "B4_rt_mean"),
)


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eps = 1e-6

    for acc, rt in _ACC_RT_PAIRS_A + _ACC_RT_PAIRS_B:
        if acc in out.columns and rt in out.columns:
            out[f"{acc}_to_{rt}__tradeoff"] = out[rt] / (out[acc] + eps)

    risk_components: list[pd.Series] = []
    for column, weight in (
        ("A4_acc_stroop_gap", 0.25),
        ("A4_rt_stroop_gap", 0.25),
        ("A1_rt_cv", 0.15),
        ("A2_rt_cv", 0.15),
        ("A3_rt_cv", 0.10),
        ("A4_rt_cv", 0.10),
    ):
        if column in out.columns:
            risk_components.append(weight * out[column].fillna(0.0))
    if risk_components:
        out["RiskScore_A"] = sum(risk_components).astype(np.float32)

    risk_components_b: list[pd.Series] = []
    for column, weight in (
        ("B4_acc_stroop_gap", 0.30),
        ("B4_rt_stroop_gap", 0.20),
        ("B1_rt_cv", 0.15),
        ("B2_rt_cv", 0.15),
        ("B3_rt_cv", 0.10),
        ("B4_rt_cv", 0.10),
    ):
        if column in out.columns:
            risk_components_b.append(weight * out[column].fillna(0.0))
    if risk_components_b:
        out["RiskScore_B"] = sum(risk_components_b).astype(np.float32)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


__all__ = ["add_interaction_features"]
