from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from confingo import (
    ConfigError,
    load_json,
    save_json,
)


if TYPE_CHECKING:
    from pathlib import Path

from tests.schemas import (
    Device,
    Session,
    Training,
)


def test_save_load_round_trip(tmp_path: Path):
    cfg = Training(device=Device.CUDA, seed=1, sessions=[Session("a")])
    path = save_json(cfg, tmp_path / "config.json")
    assert load_json(Training, path) == cfg


def test_save_leaves_no_tmp(tmp_path: Path):
    save_json(Training(), tmp_path / "config.json")
    assert not (tmp_path / "config.json.tmp").exists()


def test_save_creates_parents(tmp_path: Path):
    path = save_json(Training(), tmp_path / "nested" / "dir" / "config.json")
    assert path.exists()


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError) as info:
        load_json(Training, tmp_path / "absent.json")
    assert "absent.json" in info.value.context


def test_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_json(Training, path)


def test_non_object_document(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_json(Training, path)
    assert any("mapping document" in issue.message for issue in info.value.issues)


def test_null_document_uses_defaults(tmp_path: Path):
    path = tmp_path / "null.json"
    path.write_text("null", encoding="utf-8")
    assert load_json(Training, path) == Training()
