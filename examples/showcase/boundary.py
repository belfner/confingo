"""Schemas that sit just outside the boundary, one per rule that draws it.

The other files in this example show the last values confingo carries. These are
the first shapes it declines, each paired with the rule it crosses, so the edge is
visible from both sides. Every class here is rejected at schema preflight, before
any config data is read, and every message names what to write instead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import (
    dataclass,
    field,
)
from decimal import Decimal
from enum import (
    Enum,
    EnumType,
)
from pathlib import Path  # noqa: TC003  (needed at runtime by get_type_hints)
from typing import (
    Any,
    override,
)


@dataclass
class Section:
    """An ordinary section, used below as an element a set cannot hold."""

    lr: float = 1e-3


class SeparatedInt(int):
    """An int subclass whose instances stay distinct from one another."""

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeparatedInt):
            return self is other
        return int.__eq__(self, other)

    __hash__ = int.__hash__


class SubclassValued(Enum):
    """Two members separated only by an int subclass, so both write ``1``."""

    A = SeparatedInt(1)
    B = SeparatedInt(1)  # noqa: PIE796  (the shared plain form is the point)


class RedirectingMeta(EnumType):
    """A metaclass answering every lookup with one member."""

    def __call__(cls, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Answer with the first member whatever was asked for.

        Args:
          value (Any): The value looked up.
          *args (Any): Ignored.
          **kwargs (Any): Ignored.

        Returns:
          Any: The first member.
        """
        return next(iter(cls))


class RedirectedMode(Enum, metaclass=RedirectingMeta):
    A = "a"
    B = "b"


class Timestamp(dt.datetime):
    """A datetime subclass, used below as an annotation a load cannot build."""


@dataclass
class UnionInASet:
    values: set[str | Path] = field(default_factory=set)


@dataclass
class SectionInASet:
    sections: frozenset[Section] = field(default_factory=frozenset)


@dataclass
class UnhashableElement:
    values: set[list[int]] = field(default_factory=set)


@dataclass
class BareContainer:
    # pyrefly: ignore[implicit-any-type-argument]  (the argument-free form is the shape shown)
    values: list = field(default_factory=list)


@dataclass
class OutsideTheBoundary:
    amount: Decimal = Decimal(0)


@dataclass
class NonStringKeys:
    weights: dict[int, float] = field(default_factory=dict)


@dataclass
class SubclassEnumValues:
    choice: SubclassValued = SubclassValued.A


@dataclass
class RedirectedLookup:
    choice: RedirectedMode = RedirectedMode.A


RECORDED_AT = Timestamp(2026, 1, 1)


@dataclass
class ScalarSubclass:
    recorded_at: Timestamp = RECORDED_AT


@dataclass
class TakesTypeParameters[ElementP]:
    count: int = 0


@dataclass
class CoercedDefault:
    # pyrefly: ignore[bad-assignment]  (a default the annotation would have to coerce is the shape shown)
    output_dir: Path = "runs"


REJECTED: tuple[tuple[str, type], ...] = (
    ("a union inside a set", UnionInASet),
    ("a section inside a set", SectionInASet),
    ("an element rebuilding unhashable", UnhashableElement),
    ("an argument-free container", BareContainer),
    ("a type outside the boundary", OutsideTheBoundary),
    ("a mapping keyed by something other than str", NonStringKeys),
    ("enum values separated only by a subclass", SubclassEnumValues),
    ("an enum binding its own member lookup", RedirectedLookup),
    ("a subclass of a supported scalar", ScalarSubclass),
    ("a schema taking type parameters", TakesTypeParameters),
    ("a default needing coercion", CoercedDefault),
)
"""Each rejected shape beside the rule it crosses, in the order ``run.py`` prints."""
