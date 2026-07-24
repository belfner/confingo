from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from confingo import ConfigRoot


@dataclass
class OptimizerConfig:
    name: Literal["adamw", "sgd"]
    lr: float = 3e-4


@dataclass
class TrainingConfig(ConfigRoot):
    optimizer: OptimizerConfig
    seed: int = 0
    output_dir: Path = Path("runs")
