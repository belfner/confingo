from __future__ import annotations

from confingo.functional import config_hash
from tests.schemas import (
    Trainer,
    Training,
)


def test_equal_configs_hash_equal():
    assert config_hash(Training()) == config_hash(Training())


def test_change_changes_hash():
    assert config_hash(Training(seed=1)) != config_hash(Training(seed=2))


def test_nested_change_changes_hash():
    a = Training(trainer=Trainer(lr=1e-4))
    b = Training(trainer=Trainer(lr=2e-4))
    assert config_hash(a) != config_hash(b)


def test_length_honored():
    assert len(config_hash(Training(), length=8)) == 8
