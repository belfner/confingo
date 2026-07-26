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

from confingo import ConfigNode


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
class RootConfig(ConfigNode):
    """Root schema subclassing ConfigNode to exercise the method facade."""

    device: Device = Device.CPU
    seed: int | None = None
    trainer: Trainer = field(default_factory=Trainer)


HOOK_CALLS: list[str] = []
"""Construction-hook call log, appended to by the parallel-tree fixtures below."""


# Two parallel trees with identical fields: PlainTree nests ordinary dataclasses,
# NodeTree nests ConfigNode subclasses at every level. Both entry classes carry
# the facade so the same operations can be run against each, which isolates the
# effect of node membership on the nested declarations alone.


@dataclass
class PlainLeaf:
    """Leaf section of the plain parallel tree."""

    name: Literal["adamw", "sgd"]
    lr: float = 3e-4

    def __post_init__(self):
        HOOK_CALLS.append("plain_leaf_post_init")

    def __validate__(self):
        HOOK_CALLS.append("plain_leaf_validate")
        if self.lr <= 0.0:
            yield f"lr must be positive, got {self.lr}"


@dataclass
class PlainMid:
    """Middle section of the plain parallel tree."""

    leaf: PlainLeaf
    tag: str = "mid"


@dataclass(frozen=True)
class PlainFrozenLeaf:
    """Frozen nested section of the plain parallel tree."""

    w: int = 1


@dataclass(slots=True)
class PlainSlotsLeaf:
    """Slotted nested section of the plain parallel tree."""

    s: int = 1


@dataclass
class PlainAlt:
    """Second union member of the plain parallel tree."""

    kind: Literal["alt"] = "alt"
    scale: float = 1.0


@dataclass
class PlainTree(ConfigNode):
    """Entry class of the plain parallel tree."""

    mid: PlainMid
    items: list[PlainLeaf] = field(default_factory=list)
    lookup: dict[str, PlainLeaf] = field(default_factory=dict)
    maybe: PlainLeaf | None = None
    choice: PlainLeaf | PlainAlt | None = None
    extra: Any = None
    frozen_leaf: PlainFrozenLeaf = field(default_factory=PlainFrozenLeaf)
    slots_leaf: PlainSlotsLeaf = field(default_factory=PlainSlotsLeaf)
    note: str = field(default="n", compare=False)
    weight: float = field(default=1.0, hash=False)
    seed: int = 0
    derived: int = field(init=False)

    def __post_init__(self):
        self.derived = self.seed * 2


@dataclass
class NodeLeaf(ConfigNode):
    """Leaf section of the node parallel tree."""

    name: Literal["adamw", "sgd"]
    lr: float = 3e-4

    def __post_init__(self):
        HOOK_CALLS.append("node_leaf_post_init")

    def __validate__(self):
        HOOK_CALLS.append("node_leaf_validate")
        if self.lr <= 0.0:
            yield f"lr must be positive, got {self.lr}"


@dataclass
class NodeMid(ConfigNode):
    """Middle section of the node parallel tree."""

    leaf: NodeLeaf
    tag: str = "mid"


@dataclass(frozen=True)
class NodeFrozenLeaf(ConfigNode):
    """Frozen nested section of the node parallel tree."""

    w: int = 1


@dataclass(slots=True)
class NodeSlotsLeaf(ConfigNode):
    """Slotted nested section of the node parallel tree."""

    s: int = 1


@dataclass
class NodeAlt(ConfigNode):
    """Second union member of the node parallel tree."""

    kind: Literal["alt"] = "alt"
    scale: float = 1.0


@dataclass
class NodeTree(ConfigNode):
    """Entry class of the node parallel tree."""

    mid: NodeMid
    items: list[NodeLeaf] = field(default_factory=list)
    lookup: dict[str, NodeLeaf] = field(default_factory=dict)
    maybe: NodeLeaf | None = None
    choice: NodeLeaf | NodeAlt | None = None
    extra: Any = None
    frozen_leaf: NodeFrozenLeaf = field(default_factory=NodeFrozenLeaf)
    slots_leaf: NodeSlotsLeaf = field(default_factory=NodeSlotsLeaf)
    note: str = field(default="n", compare=False)
    weight: float = field(default=1.0, hash=False)
    seed: int = 0
    derived: int = field(init=False)

    def __post_init__(self):
        self.derived = self.seed * 2


@dataclass
class UnhashedSection(ConfigNode):
    """Enclosing node holding a section excluded from its own digest."""

    inner: NodeLeaf = field(default_factory=lambda: NodeLeaf(name="adamw"), compare=False, hash=False)
    seed: int = 0


@dataclass
class NodeBase(ConfigNode):
    """Node base used to exercise subclassing of a node."""

    a: int = 1


@dataclass
class NodeDerived(NodeBase):
    """Decorated subclass of a node, which stays a node."""

    b: int = 2


@dataclass
class NodeLeafTwin(ConfigNode):
    """Distinct node class whose hashing fields encode exactly as ``NodeLeaf``."""

    name: Literal["adamw", "sgd"]
    lr: float = 3e-4


@dataclass
class PlainBadLeaf:
    """Plain section carrying an annotation outside the supported set."""

    handler: complex = 0j


@dataclass
class PlainBadTree(ConfigNode):
    """Entry class holding a plain section with an unsupported annotation."""

    bad: PlainBadLeaf = field(default_factory=PlainBadLeaf)


@dataclass
class NodeBadLeaf(ConfigNode):
    """Node section carrying an annotation outside the supported set."""

    handler: complex = 0j


@dataclass
class NodeBadTree(ConfigNode):
    """Entry class holding a node section with an unsupported annotation."""

    bad: NodeBadLeaf = field(default_factory=NodeBadLeaf)
