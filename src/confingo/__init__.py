"""confingo: a dataclass-driven configuration toolkit.

Define settings once as typed dataclasses, then marshal them to plain data and
unmarshal config files back into validated dataclass trees.
"""

from __future__ import annotations

from confingo._core import (
    ConfigError,
    ConfigIssue,
    config_hash,
    from_dict,
    to_dict,
)
from confingo._json import (
    dumps_json,
    load_json,
    save_json,
)
from confingo._root import ConfigRoot


# Kept in sync with the version declared in pyproject.toml.
__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "ConfigIssue",
    "ConfigRoot",
    "config_hash",
    "dumps_json",
    "from_dict",
    "load_json",
    "save_json",
    "to_dict",
]
