from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from confingo import (
    ConfigError,
    config_hash,
    dumps_json,
    to_dict,
)
from tests.schemas import (
    Device,
    RootConfig,
    Trainer,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_from_dict_classmethod():
    cfg = RootConfig.from_dict({"device": "cuda", "trainer": {"lr": 1e-4}})
    assert isinstance(cfg, RootConfig)
    assert cfg.device is Device.CUDA
    assert cfg.trainer.lr == 1e-4


def test_from_dict_collects_issues():
    with pytest.raises(ConfigError):
        RootConfig.from_dict({"device": "tpu"})


def test_to_dict_method_matches_free_function():
    cfg = RootConfig(device=Device.CUDA, seed=3)
    assert cfg.to_dict() == to_dict(cfg)


def test_dumps_json_method_matches_free_function():
    cfg = RootConfig(trainer=Trainer(lr=1e-4))
    assert cfg.dumps_json() == dumps_json(cfg)


def test_config_hash_method_matches_free_function():
    cfg = RootConfig(seed=1)
    assert cfg.config_hash() == config_hash(cfg)
    assert len(cfg.config_hash(length=8)) == 8


def test_save_load_json_methods_round_trip(tmp_path: Path):
    cfg = RootConfig(device=Device.CUDA, seed=7)
    path = cfg.save_json(tmp_path / "config.json")
    assert RootConfig.load_json(path) == cfg
