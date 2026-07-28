"""Every config operation as a free function over a plain dataclass.

This is the route that needs nothing of a config class but its annotations, so a
dataclass written without any knowledge of confingo builds, exports,
fingerprints, and validates through it::

    from confingo.functional import from_dict, to_dict

    config = from_dict(Config, {"trainer": {"lr": 0.001}})
    plain = to_dict(config)

A class that subclasses ``ConfigNode`` reaches the same operations through its
``cfg`` accessor, scoped to the node it is called on. The two routes call one
implementation, so a result is the same whichever one produced it.

The package root carries the names a schema is written with -- ``ConfigNode``,
``ConfigValue``, ``ConfigScalar``, ``ConfigError``, ``ConfigIssue`` -- and this
module carries the operations run over one.
"""

from __future__ import annotations

from confingo._core import (
    from_dict,
    validate_schema,
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
from confingo._serialize import (
    config_hash,
    to_dict,
)
from confingo._yaml import (
    dumps_yaml,
    load_yaml,
    save_yaml,
)


__all__ = [
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
]
