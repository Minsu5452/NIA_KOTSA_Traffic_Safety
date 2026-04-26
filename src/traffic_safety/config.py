"""학습/추론 파이프라인용 설정 dataclass.

YAML(`configs/default.yaml`) 을 그대로 dataclass 로 매핑한다. CLI 에서 `--config`
로 다른 yaml 을 넘기면 동일 인터페이스로 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data")
    train_meta: str = "train.csv"
    test_meta: str = "test.csv"
    sample_submission: str = "sample_submission.csv"
    train_subdir: str = "train"
    test_subdir: str = "test"


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    stratify_column: str = "Label"
    group_column: str = "PrimaryKey"


@dataclass(frozen=True)
class OutputConfig:
    model_dir: Path = Path("models")
    submission_path: Path = Path("submissions/submission.csv")
    oof_path: Path = Path("artifacts/oof.csv")
    log_path: Path = Path("logs/run.log")


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=lambda: {"lightgbm": 0.55, "catboost": 0.45})
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(
                data_dir=Path(raw["data"]["data_dir"]),
                train_meta=raw["data"]["train_meta"],
                test_meta=raw["data"]["test_meta"],
                sample_submission=raw["data"]["sample_submission"],
                train_subdir=raw["data"]["train_subdir"],
                test_subdir=raw["data"]["test_subdir"],
            ),
            split=SplitConfig(
                n_splits=raw["split"]["n_splits"],
                shuffle=raw["split"]["shuffle"],
                random_state=raw["split"]["random_state"],
                stratify_column=raw["split"]["stratify_column"],
                group_column=raw["split"]["group_column"],
            ),
            models=dict(raw["models"]),
            ensemble_weights=dict(raw["ensemble"]["weights"]),
            output=OutputConfig(
                model_dir=Path(raw["output"]["model_dir"]),
                submission_path=Path(raw["output"]["submission_path"]),
                oof_path=Path(raw["output"]["oof_path"]),
                log_path=Path(raw["output"]["log_path"]),
            ),
        )
