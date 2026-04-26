"""학습·추론 entry point.

실행 예시
---------

```
python -m traffic_safety.cli train --config configs/default.yaml
python -m traffic_safety.cli infer --config configs/default.yaml
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import TrainingConfig
from .io import load_competition_data
from .models import blend_reports, predict_with_artifacts, train_models
from .pipeline import build_dataset

LOGGER = logging.getLogger("traffic_safety")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-safety")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="모델 학습 + OOF/제출 예측 생성")
    train_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))

    infer_parser = sub.add_parser("infer", help="저장된 모델로 테스트 예측 생성")
    infer_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser


def _save_submission(sample_submission: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> None:
    submission = sample_submission.merge(predictions, on="Test_id", how="left")
    submission["Label"] = submission["prob"].fillna(0.0).astype(float)
    submission = submission.drop(columns=["prob"])
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def train(config: TrainingConfig) -> None:
    _setup_logging(config.output.log_path)
    LOGGER.info("loading competition data...")
    frames = load_competition_data(config.data)

    LOGGER.info("building feature dataset...")
    dataset = build_dataset(frames, config)
    LOGGER.info("train shape=%s, test shape=%s, features=%d", dataset.train.shape, dataset.test.shape, len(dataset.feature_columns))

    X = dataset.train[dataset.feature_columns].fillna(0.0)
    y = dataset.train["Label"].astype(int)
    X_test = dataset.test[dataset.feature_columns].fillna(0.0)
    groups = dataset.train[config.split.group_column] if config.split.group_column else None

    LOGGER.info("training models...")
    reports = train_models(X, y, X_test, config, groups=groups)
    for name, report in reports.items():
        LOGGER.info(
            "%s: fold AUCs=%s | mean=%.5f ± %.5f",
            name,
            [f"{score:.5f}" for score in report.fold_aucs],
            report.mean_auc,
            report.std_auc,
        )

    oof_blend, test_blend = blend_reports(reports, config.ensemble_weights)
    blended_auc = roc_auc_score(y, oof_blend)
    LOGGER.info("ensemble OOF AUC=%.5f", blended_auc)

    config.output.oof_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Test_id": dataset.train["Test_id"].to_numpy(),
            "Label": y.to_numpy(),
            "oof_prob": oof_blend,
            **{f"oof_{name}": report.oof for name, report in reports.items()},
        }
    ).to_csv(config.output.oof_path, index=False)

    predictions = pd.DataFrame({"Test_id": dataset.test["Test_id"].to_numpy(), "prob": test_blend})
    _save_submission(frames.sample_submission, predictions, config.output.submission_path)
    LOGGER.info("saved submission to %s", config.output.submission_path)


def infer(config: TrainingConfig) -> None:
    _setup_logging(config.output.log_path)
    LOGGER.info("loading competition data for inference...")
    frames = load_competition_data(config.data)
    dataset = build_dataset(frames, config)
    X_test = dataset.test[dataset.feature_columns].fillna(0.0)

    blended = np.zeros(len(X_test), dtype=np.float64)
    total_weight = 0.0
    for name, weight in config.ensemble_weights.items():
        try:
            preds = predict_with_artifacts(config.output.model_dir, name, X_test)
        except FileNotFoundError:
            LOGGER.warning("no saved fold models for %s — skipping", name)
            continue
        blended += weight * preds
        total_weight += weight
    if total_weight <= 0:
        raise RuntimeError("no model artifacts found — train first")
    blended /= total_weight

    predictions = pd.DataFrame({"Test_id": dataset.test["Test_id"].to_numpy(), "prob": blended})
    _save_submission(frames.sample_submission, predictions, config.output.submission_path)
    LOGGER.info("saved submission to %s", config.output.submission_path)


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)
    if args.command == "train":
        train(config)
    elif args.command == "infer":
        infer(config)


def train_cli() -> None:
    config = TrainingConfig.from_yaml(Path("configs/default.yaml"))
    train(config)


def infer_cli() -> None:
    config = TrainingConfig.from_yaml(Path("configs/default.yaml"))
    infer(config)


if __name__ == "__main__":
    main()
