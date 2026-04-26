"""LightGBM + CatBoost 기반 KFold 학습/추론 루틴."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import TrainingConfig


@dataclass(frozen=True)
class FoldArtifact:
    fold_index: int
    model_path: Path
    val_auc: float


@dataclass(frozen=True)
class ModelReport:
    name: str
    fold_aucs: list[float]
    oof: np.ndarray
    test_pred: np.ndarray
    artifacts: list[FoldArtifact]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def std_auc(self) -> float:
        return float(np.std(self.fold_aucs))


def _make_splitter(config: TrainingConfig, groups: pd.Series | None):
    if groups is not None and config.split.group_column:
        return StratifiedGroupKFold(
            n_splits=config.split.n_splits,
            shuffle=config.split.shuffle,
            random_state=config.split.random_state,
        )
    return StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=config.split.random_state,
    )


def _fit_lightgbm(params: dict[str, Any], X_tr, y_tr, X_va, y_va) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(0)],
    )
    return model


def _fit_catboost(params: dict[str, Any], X_tr, y_tr, X_va, y_va) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )
    return model


def _kfold_train(
    name: str,
    fit_fn,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    config: TrainingConfig,
    groups: pd.Series | None,
) -> ModelReport:
    splitter = _make_splitter(config, groups)
    split_args = (X, y, groups) if groups is not None and config.split.group_column else (X, y)

    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: list[float] = []
    artifacts: list[FoldArtifact] = []

    config.output.model_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(*split_args), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        model = fit_fn(params, X_tr, y_tr, X_va, y_va)

        valid_pred = model.predict_proba(X_va)[:, 1]
        oof[valid_idx] = valid_pred
        fold_auc = roc_auc_score(y_va, valid_pred)
        fold_aucs.append(fold_auc)

        test_pred += model.predict_proba(X_test)[:, 1] / config.split.n_splits

        model_path = config.output.model_dir / f"{name}_fold{fold_index}.joblib"
        joblib.dump(model, model_path)
        artifacts.append(FoldArtifact(fold_index=fold_index, model_path=model_path, val_auc=fold_auc))

    return ModelReport(
        name=name,
        fold_aucs=fold_aucs,
        oof=oof,
        test_pred=test_pred,
        artifacts=artifacts,
    )


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    config: TrainingConfig,
    groups: pd.Series | None,
) -> dict[str, ModelReport]:
    reports: dict[str, ModelReport] = {}
    if "lightgbm" in config.models:
        reports["lightgbm"] = _kfold_train(
            "lightgbm",
            _fit_lightgbm,
            {**config.models["lightgbm"], "random_state": config.split.random_state},
            X,
            y,
            X_test,
            config,
            groups,
        )
    if "catboost" in config.models:
        reports["catboost"] = _kfold_train(
            "catboost",
            _fit_catboost,
            {**config.models["catboost"], "random_seed": config.split.random_state},
            X,
            y,
            X_test,
            config,
            groups,
        )
    return reports


def blend_reports(reports: dict[str, ModelReport], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    total_weight = sum(weights[name] for name in reports)
    if total_weight <= 0:
        raise ValueError("ensemble weights must be positive")

    oof_blend = np.zeros_like(next(iter(reports.values())).oof)
    test_blend = np.zeros_like(next(iter(reports.values())).test_pred)
    for name, report in reports.items():
        weight = weights[name] / total_weight
        oof_blend += weight * report.oof
        test_blend += weight * report.test_pred
    return oof_blend, test_blend


def predict_with_artifacts(model_dir: Path, prefix: str, X: pd.DataFrame) -> np.ndarray:
    """저장된 fold 모델을 모두 로드해 단순 평균 예측을 만든다."""

    paths = sorted(model_dir.glob(f"{prefix}_fold*.joblib"))
    if not paths:
        raise FileNotFoundError(f"no saved models for prefix={prefix} under {model_dir}")
    preds = np.zeros(len(X), dtype=np.float64)
    for path in paths:
        model = joblib.load(path)
        preds += model.predict_proba(X)[:, 1] / len(paths)
    return preds


__all__ = [
    "FoldArtifact",
    "ModelReport",
    "blend_reports",
    "predict_with_artifacts",
    "train_models",
]
