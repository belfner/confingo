from __future__ import annotations

from dataclasses import make_dataclass

import pytest

from confingo import (
    ConfigError,
    from_dict,
)
from tests.schemas import (
    IntKeyed,
    PostInit,
    StrKeyed,
    Trainer,
    Training,
)


def test_collect_all_issues():
    with pytest.raises(ConfigError) as info:
        from_dict(
            Training,
            {
                "device": "tpu",  # bad enum value
                "sed": 123,  # unknown key
                "buffer_size": "large",  # type mismatch
                "trainer": {"algorithm": "ppo"},  # bad literal
                "sessions": [{"weight": 2.0}],  # missing required name
            },
        )
    paths = {issue.path for issue in info.value.issues}
    assert {"device", "sed", "buffer_size", "trainer.algorithm", "sessions.0.name"} <= paths


def test_validate_messages_surface():
    with pytest.raises(ConfigError) as info:
        from_dict(Trainer, {"lr": -1.0, "hidden": [128]})
    assert any("lr must be positive" in issue.message for issue in info.value.issues)


def test_post_init_error_surfaces():
    with pytest.raises(ConfigError) as info:
        from_dict(PostInit, {"value": -5})
    assert any("value must be >= 0" in issue.message for issue in info.value.issues)


def test_partial_mapping_uses_defaults():
    cfg = from_dict(Training, {"seed": 7})
    assert cfg.seed == 7
    assert cfg.buffer_size == 1_000_000
    assert cfg.trainer == Trainer()


def test_str_keyed_dict_ok():
    cfg = from_dict(StrKeyed, {"mapping": {"a": 0.5, "b": 1.5}})
    assert cfg.mapping == {"a": 0.5, "b": 1.5}


def test_non_str_keyed_dict_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(IntKeyed, {"mapping": {"1": 0.5}})
    assert any(issue.path == "mapping" and "only str keys" in issue.message for issue in info.value.issues)


def test_unresolvable_annotation_reports_schema_error():
    bad = make_dataclass("Bad", [("x", "Missing")])
    with pytest.raises(ConfigError) as info:
        from_dict(bad, {"x": 1})
    assert info.value.context == "config schema"
