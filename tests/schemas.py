"""Module-level dataclasses used across the test suite.

These live at module level so ``get_type_hints`` resolves their annotations in
this module's namespace.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
)

from confingo import ConfigRoot


class Device(Enum):
    """Value-keyed enum for exercising enum coercion."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class Trainer:
    """Nested section exercising ``__validate__``."""

    lr: float = 3e-4
    algorithm: Literal["sac", "td3"] = "td3"
    hidden: tuple[int, ...] = (256, 256)

    def __validate__(self):
        if self.lr <= 0.0:
            yield f"lr must be positive, got {self.lr}"
        if len(self.hidden) == 0:
            yield "hidden must name at least one layer width"


@dataclass(frozen=True)
class Session:
    """Element of a list-of-dataclasses field."""

    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class Training:
    """Root schema covering enums, optionals, paths, and nested dataclasses."""

    device: Device = Device.CPU
    seed: int | None = None
    buffer_size: int = 1_000_000
    output_dir: Path = Path("runs")
    trainer: Trainer = field(default_factory=Trainer)
    sessions: list[Session] = field(default_factory=list)


@dataclass(frozen=True)
class Containers:
    """Schema covering every supported container shape."""

    ints: list[int] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    frozen: frozenset[int] = field(default_factory=frozenset)
    pair: tuple[int, str] = (0, "")
    variadic: tuple[int, ...] = ()
    tags: dict[str, int] = field(default_factory=dict)
    bare_tuple: tuple = ()
    bare_list: list = field(default_factory=list)
    bare_set: set = field(default_factory=set)
    bare_dict: dict = field(default_factory=dict)
    anything: Any = None


@dataclass(frozen=True)
class LiteralInts:
    """Schema for the Literal bool-vs-int distinction."""

    level: Literal[1, 2] = 1


@dataclass(frozen=True)
class PostInit:
    """Schema whose ``__post_init__`` rejects negative values."""

    value: int = 0

    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"value must be >= 0, got {self.value}")


@dataclass(frozen=True)
class StrKeyed:
    """Schema with a supported str-keyed dict."""

    mapping: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IntKeyed:
    """Schema with an unsupported non-str-keyed dict."""

    mapping: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RootConfig(ConfigRoot):
    """Root schema subclassing ConfigRoot to exercise the method facade."""

    device: Device = Device.CPU
    seed: int | None = None
    trainer: Trainer = field(default_factory=Trainer)
