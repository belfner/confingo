"""Smoke tests for the public API surface after the module split.

Confirms that ``import confingo`` resolves and every advertised public name is
importable from the package root, independent of which internal module now
defines it.
"""

from __future__ import annotations

import importlib
import typing

import confingo


def test_package_imports_cold() -> None:
    """A fresh import of the package resolves without error."""
    module = importlib.import_module("confingo")
    assert module is confingo


def test_config_error_init_type_hints_resolve() -> None:
    """Runtime type-hint introspection of the public ConfigError resolves."""
    hints = typing.get_type_hints(confingo.ConfigError.__init__)
    assert "issues" in hints


def test_all_public_names_resolve() -> None:
    """Every name in ``__all__`` is reachable from the package root."""
    expected = {
        "ConfigError",
        "ConfigIssue",
        "ConfigRoot",
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
    }
    assert expected <= set(confingo.__all__)
    for name in confingo.__all__:
        assert hasattr(confingo, name), name
