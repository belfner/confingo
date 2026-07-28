"""Smoke tests for the public API surface after the module split.

Confirms that ``import confingo`` resolves and every advertised public name is
importable from the package root, independent of which internal module now
defines it.
"""

from __future__ import annotations

import importlib
import pathlib
import tomllib
import typing

import confingo
import confingo.functional


def test_package_imports_cold() -> None:
    """A fresh import of the package resolves without error."""
    module = importlib.import_module("confingo")
    assert module is confingo


def test_config_error_init_type_hints_resolve() -> None:
    """Runtime type-hint introspection of the public ConfigError resolves."""
    hints = typing.get_type_hints(confingo.ConfigError.__init__)
    assert "issues" in hints


def test_the_root_carries_the_names_a_schema_is_written_with() -> None:
    """Every name a schema declaration needs is reachable from the package root."""
    assert set(confingo.__all__) == {
        "ConfigError",
        "ConfigIssue",
        "ConfigNode",
        "ConfigScalar",
        "ConfigValue",
        "__version__",
    }
    for name in confingo.__all__:
        assert hasattr(confingo, name), name


def test_functional_carries_every_operation_run_over_a_schema() -> None:
    """Every free-function operation is reachable from ``confingo.functional``."""
    assert set(confingo.functional.__all__) == {
        "config_equal",
        "config_hash",
        "dumps_json",
        "dumps_yaml",
        "from_dict",
        "from_file",
        "load_json",
        "load_yaml",
        "save_json",
        "save_yaml",
        "to_dict",
        "to_file",
        "validate_schema",
    }
    for name in confingo.functional.__all__:
        assert hasattr(confingo.functional, name), name


def test_the_two_routes_reach_one_implementation() -> None:
    """A node's accessor and the free function are the same operation."""
    for name in ("from_dict", "load_json", "load_yaml", "from_file", "validate_schema"):
        assert hasattr(confingo.ConfigNode.cfg, name), name
    for name in ("to_dict", "dumps_json", "dumps_yaml", "save_json", "save_yaml", "to_file", "hash"):
        assert hasattr(confingo.ConfigNode.cfg, name), name


def test_the_declared_version_matches_the_packaged_one() -> None:
    """``__version__`` and the version in ``pyproject.toml`` name one release."""
    root = pathlib.Path(__file__).resolve().parent.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert confingo.__version__ == declared
