from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from confingo import (
    ConfigError,
    from_file,
    to_file,
)
from tests.schemas import (
    Device,
    RootConfig,
    Session,
    Training,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_round_trip_by_extension(tmp_path: Path, suffix: str):
    cfg = Training(device=Device.CUDA, seed=1, sessions=[Session("a")])
    path = to_file(cfg, tmp_path / f"config{suffix}")
    assert from_file(Training, path) == cfg


def test_to_file_returns_written_path(tmp_path: Path):
    path = to_file(Training(), tmp_path / "nested" / "config.json")
    assert path.exists()


def test_unsupported_extension_raises(tmp_path: Path):
    with pytest.raises(ConfigError) as info:
        to_file(Training(), tmp_path / "config.toml")
    assert any(".toml" in issue.message for issue in info.value.issues)


def test_missing_extension_raises(tmp_path: Path):
    with pytest.raises(ConfigError) as info:
        from_file(Training, tmp_path / "config")
    assert any("no extension" in issue.message for issue in info.value.issues)


def test_extension_is_case_insensitive(tmp_path: Path):
    cfg = Training(seed=4)
    path = to_file(cfg, tmp_path / "config.JSON")
    assert from_file(Training, path) == cfg


def test_config_node_file_methods_round_trip(tmp_path: Path):
    cfg = RootConfig(device=Device.CUDA, seed=9)
    path = cfg.cfg.to_file(tmp_path / "config.yaml")
    assert RootConfig.cfg.from_file(path) == cfg
