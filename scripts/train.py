"""편의용 wrapper — `python scripts/train.py --config configs/default.yaml`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_safety.cli import train  # noqa: E402
from traffic_safety.config import TrainingConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    train(TrainingConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
