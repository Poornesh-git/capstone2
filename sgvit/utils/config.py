from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sgvit.models.sgvit import SGViTConfig


@dataclass
class TrainConfig:
    data_root: str
    train_split: str
    val_split: str
    test_split: str
    output_dir: str = "outputs"
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.05
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    resume: str = ""
    model: SGViTConfig = field(default_factory=SGViTConfig)


def _merge_model_config(raw: dict[str, Any]) -> SGViTConfig:
    model_raw = raw.get("model", {})
    return SGViTConfig(**model_raw)


def load_config(path: str) -> TrainConfig:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    raw["model"] = _merge_model_config(raw)
    return TrainConfig(**raw)
