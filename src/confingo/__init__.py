"""confingo: a dataclass-driven configuration toolkit.

Define settings once as typed dataclasses, then marshal them to plain data and
unmarshal config files back into validated dataclass trees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from confingo._core import (
    ConfigError,
    ConfigIssue,
    config_hash,
    from_dict,
    to_dict,
)
from confingo._file import (
    from_file,
    to_file,
)
from confingo._json import (
    dumps_json,
    load_json,
    save_json,
)
from confingo._root import ConfigRoot


# Kept in sync with the version declared in pyproject.toml.
__version__ = "0.2.0"

if TYPE_CHECKING:
    from confingo._yaml import dumps_yaml as dumps_yaml
    from confingo._yaml import load_yaml as load_yaml
    from confingo._yaml import save_yaml as save_yaml

_LAZY_YAML = frozenset({"dumps_yaml", "load_yaml", "save_yaml"})


def __getattr__(name: str) -> object:
    """Resolve the optional YAML helpers on first access.

    The YAML helpers live behind the ``yaml`` extra so the core imports no
    third-party packages. Accessing one imports it on demand, raising a helpful
    error when PyYAML is absent.

    Args:
        name: The attribute being accessed on the package.

    Returns:
        The requested YAML helper function.

    Raises:
        ImportError: When a YAML helper is accessed and PyYAML is not installed.
        AttributeError: When the name is not a package member.
    """
    if name in _LAZY_YAML:
        from confingo import _yaml  # noqa: PLC0415

        return getattr(_yaml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConfigError",
    "ConfigIssue",
    "ConfigRoot",
    "config_hash",
    "dumps_json",
    "from_dict",
    "from_file",
    "load_json",
    "save_json",
    "to_dict",
    "to_file",
]
