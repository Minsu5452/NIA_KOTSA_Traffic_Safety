"""대회 데이터 로드 헬퍼.

`train.csv`, `test.csv` 는 (`Test_id`, `Test`, `PrimaryKey`, `Age`, `TestDate`,
`Label`) 메타 정보만 들고, 실제 검사 결과는 `train/A.csv`, `train/B.csv`,
`test/A.csv`, `test/B.csv` 에 들어 있다. 본 모듈은 이 네 파일을 메타와 join 해
A 검사·B 검사 데이터프레임 두 벌을 만들어 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DataConfig


@dataclass(frozen=True)
class CompetitionFrames:
    """A 검사 / B 검사 데이터를 담는 컨테이너."""

    a_train: pd.DataFrame
    b_train: pd.DataFrame
    a_test: pd.DataFrame
    b_test: pd.DataFrame
    sample_submission: pd.DataFrame


def _attach_meta(meta: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["Test_id", "Test"]
    merged = detail.merge(meta, on=keys, how="left", validate="1:1")
    if "PrimaryKey" in merged.columns:
        merged["PrimaryKey"] = merged["PrimaryKey"].astype(str)
    return merged


def load_competition_data(config: DataConfig) -> CompetitionFrames:
    """대회 폴더 구조를 가정해 메타 + 상세 검사 결과를 한 번에 로드한다."""

    base: Path = config.data_dir
    train_meta = pd.read_csv(base / config.train_meta)
    test_meta = pd.read_csv(base / config.test_meta)
    sample_submission = pd.read_csv(base / config.sample_submission)

    a_train_raw = pd.read_csv(base / config.train_subdir / "A.csv")
    b_train_raw = pd.read_csv(base / config.train_subdir / "B.csv")
    a_test_raw = pd.read_csv(base / config.test_subdir / "A.csv")
    b_test_raw = pd.read_csv(base / config.test_subdir / "B.csv")

    a_train = _attach_meta(train_meta[train_meta["Test"] == "A"], a_train_raw)
    b_train = _attach_meta(train_meta[train_meta["Test"] == "B"], b_train_raw)
    a_test = _attach_meta(test_meta[test_meta["Test"] == "A"], a_test_raw)
    b_test = _attach_meta(test_meta[test_meta["Test"] == "B"], b_test_raw)

    return CompetitionFrames(
        a_train=a_train,
        b_train=b_train,
        a_test=a_test,
        b_test=b_test,
        sample_submission=sample_submission,
    )
