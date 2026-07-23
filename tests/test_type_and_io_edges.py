"""Type-boundary and file-IO edge cases, each pinned against regression.

Unsupported annotations, init=False runtime state, unhashable set elements,
nested optional paths, float overflow, str-keyed mappings, invalid UTF-8,
non-finite floats, atomic writes, and fingerprint stability across hash seeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable  # noqa: TC003  (needed at runtime by get_type_hints)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NewType,
)

import pytest

from confingo import (
    ConfigError,
    dumps_json,
    from_dict,
    from_file,
    load_json,
    load_yaml,
    to_file,
)
from tests.schemas import (
    Containers,
    Trainer,
)


if TYPE_CHECKING:
    from pathlib import Path

UserId = NewType("UserId", int)


# --- unsupported annotations are rejected on both sides ------------------


@dataclass
class HasIterable:
    xs: Iterable[int] = ()


@dataclass
class HasNewType:
    uid: UserId = UserId(0)


def test_iterable_annotation_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(HasIterable, {"xs": [1, 2]})
    assert any("unsupported field type" in i.message for i in info.value.issues)


def test_newtype_annotation_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(HasNewType, {"uid": 3})
    assert any("unsupported field type" in i.message for i in info.value.issues)


# --- init=False schemas are rejected -------------------------------------


@dataclass
class Derived:
    x: int = 0
    computed: int = field(init=False, default=1)


def test_init_false_rejected_on_load():
    with pytest.raises(ConfigError) as info:
        from_dict(Derived, {"x": 2})
    assert any("init=False" in i.message for i in info.value.issues)


def test_init_false_rejected_on_dump():
    with pytest.raises(ConfigError) as info:
        dumps_json(Derived(2))
    assert "init=False" in str(info.value)


# --- unhashable set elements report an issue instead of crashing ---------


@dataclass
class SetHolder:
    s: set[Any] = field(default_factory=set)
    n: int = 0


def test_unhashable_set_element_collected():
    with pytest.raises(ConfigError) as info:
        from_dict(SetHolder, {"s": [{}], "n": "bad"})
    messages = [i.message for i in info.value.issues]
    # No raw TypeError escaped, and the sibling issue is still collected.
    assert any("cannot build" in m for m in messages)
    assert any("expected int" in m for m in messages)


# --- Optional[nested] preserves nested collect-all detail ----------------


@dataclass
class Child:
    a: int = 0
    b: int = 0


@dataclass
class Parent:
    child: Child | None = None


def test_optional_nested_reports_leaf_paths():
    with pytest.raises(ConfigError) as info:
        from_dict(Parent, {"child": {"a": "bad", "b": "also bad"}})
    paths = {i.path for i in info.value.issues}
    assert "child.a" in paths
    assert "child.b" in paths


# --- huge int on a float field is a collected issue, not a crash --------


@dataclass
class Floats:
    x: float = 0.0
    y: int = 0


def test_float_overflow_collected():
    with pytest.raises(ConfigError) as info:
        from_dict(Floats, {"x": 10**1000, "y": "bad"})
    messages = [i.message for i in info.value.issues]
    assert any("too large" in m for m in messages)
    assert any("expected int" in m for m in messages)


# --- bare dict enforces str keys ----------------------------------------


def test_bare_dict_rejects_non_str_key():
    with pytest.raises(ConfigError) as info:
        from_dict(Containers, {"bare_dict": {1: "one"}})
    assert any("expected str" in i.message for i in info.value.issues)


def test_bare_dict_accepts_str_key():
    cfg = from_dict(Containers, {"bare_dict": {"a": 1}})
    assert cfg.bare_dict == {"a": 1}


def test_bare_dict_yaml_int_key_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("bare_dict:\n  1: one\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(Containers, path)


# --- invalid UTF-8 is a ConfigError -------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_invalid_utf8_is_config_error(tmp_path: Path, suffix: str):
    path = tmp_path / f"bad{suffix}"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError):
        from_file(Floats, path)


# --- non-finite floats are rejected on both sides -----------------------


def test_nan_load_rejected(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_json(Floats, path)


def test_infinity_dump_rejected():
    with pytest.raises(ConfigError) as info:
        dumps_json(Floats(x=float("inf")))
    assert "non-finite" in str(info.value)


# --- atomic write uses a unique temp -------------------------------------


def test_atomic_write_leaves_foreign_tmp_untouched(tmp_path: Path):
    foreign = tmp_path / "config.json.tmp"
    foreign.write_text("keep me", encoding="utf-8")
    to_file(Floats(x=1.5), tmp_path / "config.json")
    assert foreign.read_text(encoding="utf-8") == "keep me"
    strays = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp" and p != foreign]
    assert strays == []


# --- config_hash is stable across process hash seeds ---------------------

_HASH_PROG = (
    "from dataclasses import dataclass, field\n"
    "from confingo import from_dict, config_hash\n"
    "@dataclass\n"
    "class M:\n"
    "    s: frozenset[int | str] = field(default_factory=frozenset)\n"
    "print(config_hash(from_dict(M, {'s': [3, 1, 'b', 'a']})))\n"
)


def test_config_hash_stable_across_hash_seeds():
    digests = set()
    for seed in ("0", "1", "2", "3"):
        out = subprocess.run(
            [sys.executable, "-c", _HASH_PROG],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# --- the yaml extra stays isolated when PyYAML is absent ----------------

_ISOLATION_PROG = (
    "import sys\n"
    "sys.modules['yaml'] = None\n"  # make `import yaml` raise ImportError
    "import tempfile, pathlib\n"
    "from dataclasses import dataclass\n"
    "import confingo\n"
    "from confingo import from_file, to_file\n"
    "@dataclass\n"
    "class C:\n"
    "    x: int = 1\n"
    "with tempfile.TemporaryDirectory() as d:\n"
    "    p = pathlib.Path(d) / 'c.json'\n"
    "    to_file(C(x=5), p)\n"
    "    assert from_file(C, p).x == 5\n"
    "try:\n"
    "    confingo.load_yaml\n"
    "    raise SystemExit('yaml did not raise')\n"
    "except ImportError as e:\n"
    "    assert 'confingo[yaml]' in str(e), str(e)\n"
    "print('ISOLATION_OK')\n"
)


def test_yaml_isolated_without_pyyaml():
    out = subprocess.run([sys.executable, "-c", _ISOLATION_PROG], capture_output=True, text=True, check=True)
    assert "ISOLATION_OK" in out.stdout


def test_supported_types_still_work():
    # A guard that the tightening did not break a normal supported schema.
    cfg = from_dict(Trainer, {"lr": 1e-4, "hidden": [128, 64]})
    assert cfg.lr == 1e-4
    assert cfg.hidden == (128, 64)
