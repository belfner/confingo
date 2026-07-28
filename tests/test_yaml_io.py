from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from confingo import ConfigError
from confingo.functional import (
    dumps_json,
    dumps_yaml,
    load_json,
    load_yaml,
    save_json,
    save_yaml,
)
from tests.schemas import (
    Device,
    RootConfig,
    Session,
    Training,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_save_load_round_trip(tmp_path: Path):
    cfg = Training(device=Device.CUDA, seed=1, sessions=[Session("a")])
    path = save_yaml(cfg, tmp_path / "config.yaml")
    assert load_yaml(Training, path) == cfg


def test_save_leaves_no_tmp(tmp_path: Path):
    save_yaml(Training(), tmp_path / "config.yaml")
    assert not (tmp_path / "config.yaml.tmp").exists()


def test_save_creates_parents(tmp_path: Path):
    path = save_yaml(Training(), tmp_path / "nested" / "dir" / "config.yaml")
    assert path.exists()


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError) as info:
        load_yaml(Training, tmp_path / "absent.yaml")
    assert "absent.yaml" in info.value.context


def test_malformed_yaml(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(Training, path)


def test_non_mapping_document(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_yaml(Training, path)
    assert any("mapping document" in issue.message for issue in info.value.issues)


def test_null_document_uses_defaults(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("\n", encoding="utf-8")
    assert load_yaml(Training, path) == Training()


def test_dump_holds_only_plain_structures():
    body = dumps_yaml(Training(device=Device.CUDA, sessions=[Session("a")]))
    assert "!!" not in body


def test_dump_preserves_field_order():
    body = dumps_yaml(Training(seed=5))
    top_level = [line.split(":", 1)[0] for line in body.splitlines() if line[:1] not in {" ", "-", ""}]
    assert top_level == ["device", "seed", "buffer_size", "output_dir", "trainer", "sessions"]


def test_yaml_and_json_load_equal(tmp_path: Path):
    cfg = Training(device=Device.CUDA, seed=2, sessions=[Session("x", 0.5)])
    yaml_path = save_yaml(cfg, tmp_path / "c.yaml")
    json_path = save_json(cfg, tmp_path / "c.json")
    assert load_yaml(Training, yaml_path) == load_json(Training, json_path)


def test_yaml_reads_json_file(tmp_path: Path):
    cfg = Training(device=Device.CPU, seed=7)
    json_path = save_json(cfg, tmp_path / "c.json")
    assert load_yaml(Training, json_path) == cfg


def test_yaml_and_json_dumps_hold_same_data():
    cfg = Training(device=Device.CUDA, seed=3, sessions=[Session("a"), Session("b")])
    assert yaml.safe_load(dumps_yaml(cfg)) == json.loads(dumps_json(cfg))


def test_config_node_yaml_methods_round_trip(tmp_path: Path):
    cfg = RootConfig(device=Device.CUDA, seed=9)
    path = cfg.cfg.save_yaml(tmp_path / "config.yaml")
    assert RootConfig.cfg.load_yaml(path) == cfg
    assert cfg.cfg.dumps_yaml() == dumps_yaml(cfg)
