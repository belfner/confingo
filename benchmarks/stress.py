"""Stress benchmark for confingo's marshal / unmarshal core.

Generates a large suite of big, deeply nested, randomized configs and times
each hot operation over many trials, reporting best / median / mean +/- stdev
wall time per trial plus derived throughput (configs/sec, microseconds/config).

Operations covered:

  from_dict   : plain dict -> validated dataclass tree (unmarshal)
  to_dict     : dataclass tree -> plain serializable data (marshal)
  round_trip  : from_dict(to_dict(obj)) -- the invariant confingo guarantees
  config_hash : stable fingerprint over the config tree
  dumps_json  : marshal + JSON text
  dumps_yaml  : marshal + YAML text

The schema exercises three nesting levels, a list of dataclasses (layers), a
dict of dataclasses (datasets), scalar lists/dicts, enums, and optional
(`| None`) fields. A correctness gate first asserts the round-trip invariant
holds for every config in the suite, so the timings measure real, valid work.

Usage:
  uv run python benchmarks/stress.py [suite_size] [repeat]

Defaults: suite_size=1500, repeat=10.
"""

from __future__ import annotations

import gc
import random
import statistics
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
)

import confingo


if TYPE_CHECKING:
    from collections.abc import Callable


class Activation(Enum):
    relu = "relu"
    gelu = "gelu"
    tanh = "tanh"
    silu = "silu"


class Precision(Enum):
    fp16 = "fp16"
    bf16 = "bf16"
    fp32 = "fp32"


@dataclass
class Layer:
    kind: str
    units: int
    activation: Activation
    dropout: float
    bias: bool


@dataclass
class Optimizer:
    name: str
    lr: float
    weight_decay: float
    schedule: dict[str, float]
    momentum: float | None


@dataclass
class Dataset:
    path: str
    shuffle: bool
    num_workers: int
    transforms: list[str]


@dataclass
class Model:
    name: str
    precision: Precision
    layers: list[Layer]
    hidden: dict[str, int]
    notes: str | None


@dataclass
class Trainer:
    max_epochs: int
    grad_clip: float | None
    optimizer: Optimizer
    metrics: list[str]


@dataclass
class ExperimentConfig:
    seed: int
    tags: list[str]
    model: Model
    trainer: Trainer
    datasets: dict[str, Dataset]
    extra: dict[str, int]


_ACTIVATIONS = [a.value for a in Activation]
_PRECISIONS = [p.value for p in Precision]
_KINDS = ["linear", "conv", "attention", "norm", "embedding"]
_TRANSFORMS = ["resize", "crop", "flip", "normalize", "jitter", "blur", "erase"]
_METRICS = ["loss", "acc", "f1", "auroc", "precision", "recall"]


def make_config(rng: random.Random) -> dict[str, Any]:
    """Generate one large, randomized config dict.

    Args:
      rng (random.Random): Seeded RNG driving every size and value so the suite
        is reproducible.

    Returns:
      dict[str, Any]: A plain-data config matching the ExperimentConfig schema,
        with randomized collection sizes to stress varied per-item work.
    """
    n_layers = rng.randint(12, 48)
    layers = [
        {
            "kind": rng.choice(_KINDS),
            "units": rng.randint(16, 4096),
            "activation": rng.choice(_ACTIVATIONS),
            "dropout": round(rng.random() * 0.5, 4),
            "bias": rng.random() > 0.5,
        }
        for _ in range(n_layers)
    ]
    hidden = {f"h{i}": rng.randint(8, 2048) for i in range(rng.randint(4, 16))}
    schedule = {f"step{i}": round(rng.random(), 5) for i in range(rng.randint(3, 10))}
    datasets = {
        f"ds_{i}": {
            "path": f"/data/shard_{rng.randint(0, 9999)}",
            "shuffle": rng.random() > 0.3,
            "num_workers": rng.randint(0, 32),
            "transforms": rng.sample(_TRANSFORMS, rng.randint(2, len(_TRANSFORMS))),
        }
        for i in range(rng.randint(3, 12))
    }
    return {
        "seed": rng.randint(0, 2**31),
        "tags": [f"tag{rng.randint(0, 999)}" for _ in range(rng.randint(3, 10))],
        "model": {
            "name": f"model_{rng.randint(0, 9999)}",
            "precision": rng.choice(_PRECISIONS),
            "layers": layers,
            "hidden": hidden,
            "notes": None if rng.random() > 0.5 else "auto-generated",
        },
        "trainer": {
            "max_epochs": rng.randint(1, 300),
            "grad_clip": None if rng.random() > 0.5 else round(rng.random() * 5, 3),
            "optimizer": {
                "name": rng.choice(["adam", "sgd", "adamw", "lamb"]),
                "lr": round(rng.random() * 0.01, 6),
                "weight_decay": round(rng.random() * 0.1, 6),
                "schedule": schedule,
                "momentum": None if rng.random() > 0.5 else round(rng.random(), 4),
            },
            "metrics": rng.sample(_METRICS, rng.randint(2, len(_METRICS))),
        },
        "datasets": datasets,
        "extra": {f"k{i}": rng.randint(0, 10**6) for i in range(rng.randint(4, 12))},
    }


def generate_suite(count: int, seed: int = 1234) -> list[dict[str, Any]]:
    """Generate a reproducible suite of config dicts.

    Args:
      count (int): Number of distinct configs to generate.
      seed (int = 1234): RNG seed for reproducibility.

    Returns:
      list[dict[str, Any]]: The generated config dicts.
    """
    rng = random.Random(seed)
    return [make_config(rng) for _ in range(count)]


def count_nested(configs: list[dict[str, Any]]) -> int:
    """Count layers plus datasets across the suite as a scale readout.

    Args:
      configs (list[dict[str, Any]]): Generated config dicts.

    Returns:
      int: Total nested sub-objects, a rough proxy for per-pass work.
    """
    return sum(len(c["model"]["layers"]) + len(c["datasets"]) for c in configs)


def correctness_gate(objs: list[ExperimentConfig]) -> None:
    """Assert the round-trip invariant holds for every built config.

    Args:
      objs (list[ExperimentConfig]): Configs already unmarshalled from the suite.

    Raises:
      AssertionError: A config fails from_dict(to_dict(obj)) == obj.
    """
    failures = 0
    for obj in objs:
        if confingo.from_dict(ExperimentConfig, confingo.to_dict(obj)) != obj:
            failures += 1
    if failures > 0:
        raise AssertionError(f"round-trip invariant failed for {failures} configs")
    print(f"correctness gate: round-trip holds for all {len(objs):,} configs")


def build_ops(
    configs: list[dict[str, Any]],
    objs: list[ExperimentConfig],
) -> dict[str, Callable[[], None]]:
    """Build whole-suite operations to time.

    Args:
      configs (list[dict[str, Any]]): The plain-data suite (inputs to from_dict).
      objs (list[ExperimentConfig]): Pre-built configs (inputs to marshal ops).

    Returns:
      dict[str, Callable[[], None]]: Operation name -> a callable that runs that
        operation across the entire suite once.
    """

    def op_from_dict() -> None:
        for c in configs:
            confingo.from_dict(ExperimentConfig, c)

    def op_to_dict() -> None:
        for o in objs:
            confingo.to_dict(o)

    def op_round_trip() -> None:
        for o in objs:
            confingo.from_dict(ExperimentConfig, confingo.to_dict(o))

    def op_config_hash() -> None:
        for o in objs:
            confingo.config_hash(o)

    def op_dumps_json() -> None:
        for o in objs:
            confingo.dumps_json(o)

    def op_dumps_yaml() -> None:
        for o in objs:
            confingo.dumps_yaml(o)

    return {
        "from_dict": op_from_dict,
        "to_dict": op_to_dict,
        "round_trip": op_round_trip,
        "config_hash": op_config_hash,
        "dumps_json": op_dumps_json,
        "dumps_yaml": op_dumps_yaml,
    }


def time_op(op: Callable[[], None], repeat: int) -> list[float]:
    """Time a whole-suite operation across trials with GC paused.

    Args:
      op (Callable[[], None]): Operation running over the entire suite once.
      repeat (int): Number of trials.

    Returns:
      list[float]: Seconds per trial.
    """
    op()  # warm caches
    trials: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            t0 = time.perf_counter()
            op()
            trials.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return trials


@dataclass
class _Stat:
    name: str
    best: float
    median: float
    mean: float
    stdev: float
    per_sec: float
    us_per: float


def report(results: dict[str, list[float]], suite_size: int) -> None:
    """Print a stats table over all operations, ordered by best time.

    Args:
      results (dict[str, list[float]]): Operation -> per-trial seconds.
      suite_size (int): Number of configs per trial.
    """
    rows: list[_Stat] = []
    for name, trials in results.items():
        best = min(trials)
        rows.append(
            _Stat(
                name=name,
                best=best,
                median=statistics.median(trials),
                mean=statistics.fmean(trials),
                stdev=statistics.pstdev(trials),
                per_sec=suite_size / best,
                us_per=best / suite_size * 1e6,
            )
        )
    rows.sort(key=lambda r: r.best)

    print(f"\nresults   (suite of {suite_size:,} configs per trial, best trial reported)")
    print(
        f"  {'operation':12s} {'best(ms)':>9} {'median(ms)':>11} "
        f"{'mean(ms)':>9} {'stdev(ms)':>9} {'cfgs/sec':>10} {'us/cfg':>9}"
    )
    print("  " + "-" * 82)
    for r in rows:
        print(
            f"  {r.name:12s} {r.best * 1e3:9.2f} {r.median * 1e3:11.2f} "
            f"{r.mean * 1e3:9.2f} {r.stdev * 1e3:9.2f} "
            f"{r.per_sec:10,.0f} {r.us_per:9.1f}"
        )


def main() -> None:
    """Generate the suite, run the correctness gate, and benchmark every op."""
    suite_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"confingo {confingo.__version__}  python {sys.version.split()[0]}")
    configs = generate_suite(suite_size)
    nested = count_nested(configs)
    print(
        f"suite_size={suite_size:,} configs  repeat={repeat}  "
        f"nested sub-objects (layers+datasets)={nested:,}  "
        f"(~{nested / suite_size:.0f} per config)"
    )

    objs = [confingo.from_dict(ExperimentConfig, c) for c in configs]
    correctness_gate(objs)

    ops = build_ops(configs, objs)
    results = {name: time_op(op, repeat) for name, op in ops.items()}
    report(results, suite_size)


if __name__ == "__main__":
    main()
