from __future__ import annotations

from pathlib import Path

import pytest

from confingo import (
    ConfigError,
    from_dict,
)
from tests.schemas import (
    Containers,
    Device,
    LiteralInts,
    Trainer,
    Training,
)


def test_integral_float_to_int():
    cfg = from_dict(Training, {"buffer_size": 2e6})
    assert cfg.buffer_size == 2_000_000
    assert isinstance(cfg.buffer_size, int)


def test_enum_by_value():
    assert from_dict(Training, {"device": "cuda"}).device is Device.CUDA


def test_enum_by_name():
    assert from_dict(Training, {"device": "CUDA"}).device is Device.CUDA


def test_str_to_path():
    assert from_dict(Training, {"output_dir": "runs/exp"}).output_dir == Path("runs/exp")


def test_bool_rejected_on_int():
    with pytest.raises(ConfigError) as info:
        from_dict(Training, {"buffer_size": True})
    assert any(issue.path == "buffer_size" for issue in info.value.issues)


def test_literal_membership():
    assert from_dict(Trainer, {"algorithm": "sac"}).algorithm == "sac"
    with pytest.raises(ConfigError):
        from_dict(Trainer, {"algorithm": "ppo"})


def test_literal_bool_int_distinction():
    assert from_dict(LiteralInts, {"level": 2}).level == 2
    # True == 1 but is a bool, so it fails a Literal[1, 2] of ints.
    with pytest.raises(ConfigError):
        from_dict(LiteralInts, {"level": True})
    with pytest.raises(ConfigError):
        from_dict(LiteralInts, {"level": 3})


def test_sequence_container_types():
    cfg = from_dict(Containers, {"names": ["a", "b"], "frozen": [1, 2], "bare_set": [1, 2]})
    assert cfg.names == {"a", "b"}
    assert isinstance(cfg.names, set)
    assert cfg.frozen == frozenset({1, 2})
    assert isinstance(cfg.frozen, frozenset)
    assert isinstance(cfg.bare_set, set)


def test_fixed_tuple_arity_mismatch():
    with pytest.raises(ConfigError):
        from_dict(Containers, {"pair": [1, 2, 3]})


def test_variadic_tuple():
    assert from_dict(Containers, {"variadic": [1, 2, 3, 4]}).variadic == (1, 2, 3, 4)
