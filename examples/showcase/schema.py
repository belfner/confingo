"""A schema exercising every annotation category and field option confingo supports.

The tree is a plausible training configuration, and each section concentrates on
one part of the type boundary so a reader can find a feature by its section:

- ``OptimizerConfig``: scalars, ``Literal``, enums, and the union rules.
- ``ScheduleConfig``: a variant group, one of whose variants is frozen and slotted.
- ``DataConfig``: paths, temporal scalars, and every container shape.
- ``TensorConfig``: the numpy and torch annotation forms.
- ``TelemetryConfig``: open data, and the set element rules.
- ``RuntimeConfig``: ``init=False`` runtime state populated in ``__post_init__``.
- ``TrainingConfig``: the root, carrying field projections and ``__validate__``.

``run.py`` loads this schema and drives the values that sit exactly at the
library's declared limits.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import (  # noqa: TC003  (needed at runtime by get_type_hints)
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Literal,
    override,
)

import numpy as np
import numpy.typing as npt
import torch

from confingo import (
    ConfigChoice,
    ConfigNode,
    ConfigScalar,
    ConfigValue,
)


class Precision(Enum):
    """Member values are exactly ``str``, which is what a member lookup reads.

    The ``_missing_`` hook maps spellings outside the member values. A lookup
    reaches it only after those values miss, so every member still rebuilds from
    the value it writes.
    """

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

    @override
    @classmethod
    def _missing_(cls, value: object) -> Precision | None:
        """Map a legacy spelling onto the member that replaced it.

        Args:
          value (object): The value a file carried.

        Returns:
          Precision | None: The member the spelling names, or None to report the
            value as unmatched.
        """
        legacy: dict[object, Precision] = {"float32": cls.FP32, "bfloat16": cls.BF16, "half": cls.FP16}
        return legacy.get(value)


class Stage(Enum):
    """Member values are exactly ``int``, the other primitive an enum may carry."""

    WARMUP = 0
    MAIN = 1
    COOLDOWN = 2


@dataclass
class OptimizerConfig(ConfigNode):
    """Scalars, a literal, enums, and the two union rules.

    A ``ConfigNode`` subclass reaches every operation through ``cfg``, scoped to
    this node, so ``config.optimizer.cfg.hash()`` fingerprints this subtree alone.
    """

    name: Literal["adamw", "sgd", "lion"]
    amsgrad: bool = False
    lr: float = 3e-4
    weight_decay: float = 0.01
    momentum: float | None = None
    warmup_steps: int = 0
    grad_clip: int | float = 1.0
    label_smoothing: float | int = 0.0
    precision: Precision = Precision.FP32
    stage: Stage = Stage.WARMUP

    def __validate__(self) -> list[str]:
        """Report every invariant this node breaks, each as its own issue.

        Returns:
          list[str]: One message per broken invariant, empty when the node holds.
        """
        problems: list[str] = []
        if self.lr <= 0.0:
            problems.append(f"lr must be positive, got {self.lr}")
        if self.name == "sgd" and self.momentum is None:
            problems.append("sgd requires momentum; set optimizer.momentum")
        return problems


@dataclass(frozen=True)
class ScheduleConfig(ConfigChoice, tag_key="kind"):
    """A variant group: one annotation standing for the schedules a run may pick.

    A field annotated with this group takes any of the variants below, and the
    ``kind`` key in the config section names which one to build. The key is this
    group's own, chosen here, and the fields declared on the group are shared by
    every variant.
    """

    total_steps: int = 10_000


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CosineSchedule(ScheduleConfig, tag="cosine"):
    """The variant a file selects with ``kind: cosine``, frozen and slotted.

    ``frozen``, ``slots``, and ``weakref_slot`` are all supported, and this
    variant declares all three. Freezing a config class makes ``config_hash`` the
    way to carry its value identity, since confingo owns hashing on a schema
    class.
    """

    min_lr_ratio: float = 0.1


@dataclass(frozen=True)
class LinearSchedule(ScheduleConfig, tag="linear"):
    """The variant a file selects with ``kind: linear``, decaying to a floor."""

    end_lr_ratio: float = 0.0


@dataclass(frozen=True)
class ConstantSchedule(ScheduleConfig, tag="constant"):
    """The variant a file selects with ``kind: constant``.

    It declares fields of its own nowhere, so it carries the group's shared
    ``total_steps`` alone, which is a complete variant.
    """


@dataclass
class DataConfig(ConfigNode):
    """Paths, temporal scalars, and every container shape the boundary accepts."""

    root: Path
    shards: list[Path]
    splits: dict[str, float]
    image_size: tuple[int, int]
    hidden_widths: tuple[int, ...] = (512, 256)
    nothing: tuple[()] = ()
    tags: Sequence[str] = field(default_factory=list)
    weights: Mapping[str, float] = field(default_factory=dict)
    collected_on: dt.date = dt.date(2026, 1, 1)
    window_start: dt.time = dt.time(9, 0)
    snapshot_at: dt.datetime = dt.datetime(2026, 1, 1, 9, 0)  # noqa: DTZ001  (a naive datetime round-trips as one)


@dataclass
class TensorConfig:
    """Every numpy and torch annotation form, each carrying its own dtype claim.

    A bare annotation infers its dtype from the values; a concrete dtype and a
    dtype family each rebuild to the type they name; and a fixed-arity shape tuple
    enforces exactly that dimensionality.
    """

    inferred: np.ndarray
    concrete: npt.NDArray[np.float32]
    family: npt.NDArray[np.floating]
    integral: npt.NDArray[np.int32]
    shaped: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    pinned_tensor: torch.Tensor
    typed_tensor: Annotated[torch.Tensor, torch.float32]
    shaped_tensor: Annotated[torch.Tensor, torch.float32, tuple[int, int]]
    shape_only_tensor: Annotated[torch.Tensor, tuple[int, int]]
    empty_leading: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class TelemetryConfig:
    """Open data, and every set element shape the preflight admits.

    ``ConfigValue`` names the whole plain-data domain a file carries and
    ``ConfigScalar`` names its leaf half, so a field annotated with either takes
    whatever shape the file states, checked against that domain rather than
    against a declared structure.
    """

    extra: ConfigValue
    marker: ConfigScalar = 0
    labels: set[str] = field(default_factory=set)
    codes: frozenset[int] = field(default_factory=frozenset)
    levels: set[Precision] = field(default_factory=set)
    optional_names: set[str | None] = field(default_factory=set)
    coordinates: set[tuple[str, int]] = field(default_factory=set)
    groups: set[frozenset[str]] = field(default_factory=set)
    mixed: set[ConfigScalar] = field(default_factory=set)
    nested_pairs: frozenset[tuple[tuple[int, str], frozenset[int]]] = field(default_factory=frozenset)
    payloads: list[ConfigValue] = field(default_factory=list)
    lookup: dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    """Runtime state alongside loaded values.

    An ``init=False`` field is populated by ``__post_init__`` rather than by the
    file, and its annotation is exempt from the schema boundary, so it may hold
    any resolvable runtime object. Loading, export, equality, and the fingerprint
    all draw from the ``init=True`` fields.
    """

    device: Literal["cpu", "cuda"] = "cpu"
    workers: int = 4
    generator: torch.Generator = field(init=False)
    worker_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        """Populate every ``init=False`` field, which is checked after construction."""
        self.generator = torch.Generator()
        self.worker_names = [f"worker-{index}" for index in range(self.workers)]


@dataclass
class CheckpointConfig(ConfigNode):
    """A section whose baseline differs from its own field defaults.

    ``TrainingConfig`` selects this through an explicit ``default_factory``, which
    takes precedence over the implicit build from an empty mapping.
    """

    every_steps: int = 1_000
    keep_last: int = 3
    directory: Path = Path("checkpoints")


@dataclass
class TrainingConfig(ConfigNode):
    """The root, carrying the three field projections and a cross-section invariant.

    The sub-config fields carry bare annotations, so an omitted section builds
    from an empty mapping and any value it still requires is reported at its own
    nested path. ``checkpoint`` is the exception, selecting a baseline whose values
    differ from the section's own defaults.
    """

    optimizer: OptimizerConfig
    data: DataConfig
    tensors: TensorConfig
    telemetry: TelemetryConfig
    schedule: ScheduleConfig
    runtime: RuntimeConfig
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    run_name: str = "run"
    output_dir: Path = Path("runs")
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(every_steps=500, keep_last=1))
    notes: str = field(default="", compare=False)
    resumed_from: Path | None = field(default=None, hash=False)

    def __validate__(self) -> list[str]:
        """Report invariants that span more than one section.

        Returns:
          list[str]: One message per broken invariant, empty when the root holds.
        """
        problems: list[str] = []
        if self.epochs <= 0:
            problems.append(f"epochs must be positive, got {self.epochs}")
        if self.optimizer.warmup_steps > self.schedule.total_steps:
            problems.append(
                f"optimizer.warmup_steps ({self.optimizer.warmup_steps}) exceeds "
                f"schedule.total_steps ({self.schedule.total_steps}); lower it to fit the schedule"
            )
        return problems
