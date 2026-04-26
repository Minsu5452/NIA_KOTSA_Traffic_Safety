"""CSV 문자열 시퀀스에 대한 numpy 기반 통계 헬퍼.

대회 원본 컬럼(예: `A1-3`, `B4-1`) 은 `"1,2,1,3,2"` 처럼 콤마로 이어진 정수
시퀀스 문자열로 들어 있다. 시퀀스 길이는 검사 항목마다 고정이라, 모든 행을 같은
폭의 numpy 행렬로 변환하면 row-wise 마스킹·평균·표준편차를 빠르게 계산할 수
있다.

본 모듈은 그 행렬화와 마스크 연산 헬퍼를 제공한다.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def to_sequence_matrix(series: pd.Series, length: int, dtype: type = np.float32) -> np.ndarray:
    """`"1,2,3"` 형식의 csv 문자열 series 를 `(n, length)` 행렬로 변환.

    누락 셀은 NaN 으로 채운다.
    """

    if length <= 0:
        raise ValueError("length must be positive")

    result = np.full((len(series), length), np.nan, dtype=np.float32)
    values = series.fillna("").to_numpy()
    for row, raw in enumerate(values):
        if not raw:
            continue
        try:
            parsed = np.fromstring(raw, sep=",", dtype=np.float32)
        except ValueError:
            continue
        n = min(parsed.shape[0], length)
        result[row, :n] = parsed[:n]

    return result.astype(dtype, copy=False) if dtype != np.float32 else result


def _build_mask(matrix: np.ndarray, allowed: int | float | Iterable[int | float]) -> np.ndarray:
    if np.isscalar(allowed):
        return matrix == allowed
    targets = np.asarray(list(allowed))
    return np.isin(matrix, targets)


def seq_mean(matrix: np.ndarray, *, mask: np.ndarray | None = None, absolute: bool = False) -> np.ndarray:
    """행 단위 평균. NaN 과 mask 를 함께 고려한다."""

    values = np.abs(matrix) if absolute else matrix
    valid = ~np.isnan(values) if mask is None else (mask & ~np.isnan(values))
    counts = valid.sum(axis=1)
    sums = np.where(valid, values, 0.0).sum(axis=1)
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def seq_std(matrix: np.ndarray, *, mask: np.ndarray | None = None, absolute: bool = False) -> np.ndarray:
    """행 단위 표준편차 (분모는 valid count). 표본이 1개 이하면 NaN."""

    values = np.abs(matrix) if absolute else matrix
    valid = ~np.isnan(values) if mask is None else (mask & ~np.isnan(values))
    counts = valid.sum(axis=1)
    sums = np.where(valid, values, 0.0).sum(axis=1)
    sq_sums = np.where(valid, values * values, 0.0).sum(axis=1)
    means = np.zeros_like(sums)
    np.divide(sums, counts, out=means, where=counts > 0)
    variances = np.zeros_like(sums)
    np.divide(sq_sums, counts, out=variances, where=counts > 0)
    variances = np.maximum(variances - means * means, 0.0)
    out = np.sqrt(variances, dtype=np.float32)
    out[counts <= 1] = np.nan
    return out


def seq_rate(matrix: np.ndarray, target: int | float | Iterable[int | float]) -> np.ndarray:
    """전체 valid 셀 중 `target` 에 해당하는 비율."""

    target_mask = _build_mask(matrix, target)
    valid = ~np.isnan(matrix)
    counts = valid.sum(axis=1)
    hits = (target_mask & valid).sum(axis=1)
    out = np.full(counts.shape, np.nan, dtype=np.float32)
    np.divide(hits, counts, out=out, where=counts > 0)
    return out


def seq_cond_mean(
    cond: np.ndarray,
    value: np.ndarray,
    cond_values: int | float | Iterable[int | float],
    *,
    miss: np.ndarray | None = None,
    miss_values: int | float | Iterable[int | float] | None = None,
    absolute: bool = False,
) -> np.ndarray:
    """`cond ∈ cond_values` 인 셀에서 `value` 의 평균을 계산.

    선택적으로 `miss` 행렬과 `miss_values` 를 넘겨 "응답 누락 셀" 을 함께 제외할 수
    있다 (예: A 검사의 시간 시퀀스에서 미응답 인덱스를 지움).
    """

    mask = _build_mask(cond, cond_values)
    if miss is not None and miss_values is not None:
        mask &= _build_mask(miss, miss_values)
    return seq_mean(value, mask=mask, absolute=absolute)


def seq_count(matrix: np.ndarray, target: int | float | Iterable[int | float]) -> np.ndarray:
    """`target` 에 해당하는 셀 개수."""

    return _build_mask(matrix, target).sum(axis=1).astype(np.int32)


def seq_in_set_ratio(matrix: np.ndarray, target_set: Sequence[int | float]) -> np.ndarray:
    """전체 valid 중 `target_set` 에 속한 비율."""

    return seq_rate(matrix, list(target_set))
