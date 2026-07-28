"""Type aliases naming the data domain confingo builds from and writes back.

``ConfigValue`` is the plain-data domain a config file carries: the JSON scalars,
lists of them, and string-keyed mappings of them, nested up to the 64 levels every
walk over a value follows. Annotate a field with it to hold open-ended data whose
shape the schema leaves to the file, and confingo still guarantees the value
round-trips and fingerprints. ``ConfigScalar`` names the leaf half of that domain
for a field that holds one value.

Both are ordinary PEP 695 aliases, so a type checker reads them structurally and
reports a value outside the domain at the assignment. confingo matches them by
identity, the way it matches an array annotation, and applies the plain-data
rules to whatever the file supplied.
"""

from __future__ import annotations


type ConfigScalar = bool | int | float | str | None
"""One plain leaf value: a JSON scalar or ``None``."""

type ConfigValue = bool | int | float | str | None | list[ConfigValue] | dict[str, ConfigValue]
"""Plain data of any shape: scalars, lists, and string-keyed mappings, nested up to 64 levels."""


__all__ = [
    "ConfigScalar",
    "ConfigValue",
]
