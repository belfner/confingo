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
from confingo._equality import config_equal
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
from confingo._yaml import (
    dumps_yaml,
    load_yaml,
    save_yaml,
)


# Kept in sync with the version declared in pyproject.toml.
__version__ = "0.3.0"


__all__ = [
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
]
