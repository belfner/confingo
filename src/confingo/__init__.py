"""confingo: a dataclass-driven configuration toolkit.

Define settings once as typed dataclasses, then marshal them to plain data and
unmarshal config files back into validated dataclass trees.

The root carries the names a schema is written with: ``ConfigNode`` for a class
that answers its own operations under ``cfg``, ``ConfigValue`` and
``ConfigScalar`` for a field holding open-ended plain data, and ``ConfigError``
with ``ConfigIssue`` for what a rejection carries. The operations run over a
schema live in ``confingo.functional``::

    from confingo import ConfigNode, ConfigValue
    from confingo.functional import from_dict, to_dict
"""

from __future__ import annotations

from confingo._errors import (
    ConfigError,
    ConfigIssue,
)
from confingo._node import ConfigNode
from confingo.typing import (
    ConfigScalar,
    ConfigValue,
)


# Kept in sync with the version declared in pyproject.toml.
__version__ = "2.1.0"


__all__ = [
    "ConfigError",
    "ConfigIssue",
    "ConfigNode",
    "ConfigScalar",
    "ConfigValue",
    "__version__",
]
