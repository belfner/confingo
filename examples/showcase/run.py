"""Drive the showcase schema, including the values that sit at confingo's limits.

Run from this directory::

    uv run python run.py
"""

from __future__ import annotations

import tempfile
import textwrap
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    override,
)

import numpy as np
from boundary import REJECTED
from schema import (
    Precision,
    Stage,
    TrainingConfig,
)

from confingo import (
    ConfigError,
    ConfigValue,
)
from confingo.functional import (
    config_equal,
    config_hash,
    dumps_json,
    dumps_yaml,
    from_dict,
    from_file,
    load_json,
    load_yaml,
    save_json,
    save_yaml,
    to_dict,
    to_file,
    validate_schema,
)


HERE = Path(__file__).parent
RULE = "-" * 78


def banner(title: str) -> None:
    """Print a section heading.

    Args:
      title (str): The heading text.
    """
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_schema_check() -> None:
    """Validate the schema from the declaration alone, reading no config data."""
    banner("1. Schema preflight, reading the declaration alone")
    # The two surfaces are the same operation: the free function takes the class,
    # and the node accessor reaches it from the class that owns the schema.
    validate_schema(TrainingConfig)
    TrainingConfig.cfg.validate_schema()
    print("TrainingConfig validates through both surfaces: every annotation is")
    print("inside the boundary, and every authored default already carries the")
    print("runtime type its annotation names.")


def show_scalars_and_enums(config: TrainingConfig) -> None:
    """Report the scalar, literal, enum, and union results.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("2. Scalars, literals, enums, unions")
    optimizer = config.optimizer
    # The legacy spelling is loaded rather than looked up by hand, so this shows
    # the hook running inside from_dict where a config file would reach it.
    legacy_precision = from_dict(
        TrainingConfig, {**to_dict(config), "optimizer": {**to_dict(config.optimizer), "precision": "bfloat16"}}
    ).optimizer.precision
    print(f"batch_size          {config.batch_size!r:<28} 32000.0 in the file, int on an int field")
    print(f"output_dir          {config.output_dir!r:<28} a string builds a Path")
    print(f"name                {optimizer.name!r:<28} Literal option")
    print(f"precision           {optimizer.precision!r:<28} enum matched by member value 'bf16'")
    print(f"stage               {optimizer.stage!r:<28} enum matched by member name 'MAIN'")
    print(f"precision legacy    {legacy_precision!r:<28} _missing_ maps a spelling the members lack")
    print(f"amsgrad             {optimizer.amsgrad!r:<28} bool, exact in both directions")
    print(f"momentum            {optimizer.momentum!r:<28} T | None carrying a value")
    print(f"grad_clip           {optimizer.grad_clip!r:<28} int | float keeps the float the file stated")
    print(f"label_smoothing     {optimizer.label_smoothing!r:<28} float | int keeps the int the file stated")
    print(f"collected_on        {config.data.collected_on!r:<28} ISO 8601 date")
    print(f"window_start        {config.data.window_start!r:<28} ISO 8601 time")
    print(f"snapshot_at         {config.data.snapshot_at!r}")


def show_containers(config: TrainingConfig) -> None:
    """Report the container shapes and the set element rules.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("3. Containers, and every set element shape")
    data = config.data
    telemetry = config.telemetry
    print(f"shards           {data.shards!r}")
    print(f"image_size       {data.image_size!r}   fixed-arity tuple")
    print(f"hidden_widths    {data.hidden_widths!r}  variadic tuple")
    print(f"nothing          {data.nothing!r}  tuple[()] holds nothing")
    print(f"tags             {data.tags!r}  Sequence[T] builds a list")
    print(f"weights          {data.weights!r}")
    print(f"splits           {data.splits!r}")
    print()
    print(f"labels           {sorted(telemetry.labels)!r}  set[str], the repeated 'alpha' deduplicated")
    print(f"codes            {sorted(telemetry.codes)!r}  frozenset[int]")
    print(f"levels           {sorted(level.value for level in telemetry.levels)!r}  set of enum members")
    print(f"optional_names   {sorted(str(name) for name in telemetry.optional_names)!r}  set[str | None]")
    print(f"coordinates      {sorted(telemetry.coordinates)!r}  set of tuples")
    print(f"groups           {sorted(sorted(group) for group in telemetry.groups)!r}  set of frozensets")
    print(f"nested_pairs     {sorted(telemetry.nested_pairs)!r}")
    print(f"mixed            {telemetry.mixed!r}")
    print("                 set[ConfigScalar]; the file states 1, 'two', and true, and")
    print("                 True == 1 in Python, so the set itself holds two elements")


def show_open_data(config: TrainingConfig) -> None:
    """Report the open-data fields.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("4. Open data: ConfigValue and ConfigScalar")
    telemetry = config.telemetry
    print(f"extra            {telemetry.extra!r}")
    print(f"marker           {telemetry.marker!r}")
    print(f"payloads         {telemetry.payloads!r}")
    print(f"lookup           {telemetry.lookup!r}")
    print("Each takes the shape the file states, checked against the plain-data")
    print("domain rather than against a declared structure.")


def show_arrays(config: TrainingConfig) -> None:
    """Report each array annotation's rebuilt dtype and shape.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("5. Arrays and tensors, one row per annotation form")
    tensors = config.tensors
    rows = [
        ("np.ndarray", tensors.inferred),
        ("npt.NDArray[np.float32]", tensors.concrete),
        ("npt.NDArray[np.floating]", tensors.family),
        ("npt.NDArray[np.int32]", tensors.integral),
        ("np.ndarray[tuple[int, int], np.dtype[np.float64]]", tensors.shaped),
        ("torch.Tensor", tensors.pinned_tensor),
        ("Annotated[Tensor, float32]", tensors.typed_tensor),
        ("Annotated[Tensor, float32, (i, i)]", tensors.shaped_tensor),
        ("Annotated[Tensor, (i, i)]", tensors.shape_only_tensor),
        ("npt.NDArray[np.float64] (0, 3)", tensors.empty_leading),
    ]
    for annotation, value in rows:
        print(f"{annotation:<44} dtype={value.dtype!s:<10} shape={tuple(value.shape)}")
    print()
    print(f"empty_leading serializes as {to_dict(config)['tensors']['empty_leading']!r}:")
    print("an empty axis ends the encoding, so the shape (0, 3) writes one level.")


def show_field_options(config: TrainingConfig) -> None:
    """Report how the three field projections differ.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("6. Field options: init, compare, hash")
    exported = to_dict(config)
    print(f"runtime.workers          {config.runtime.workers}")
    print(f"runtime.worker_names     {config.runtime.worker_names}   init=False, built in __post_init__")
    print(f"runtime.generator        {type(config.runtime.generator).__name__}    init=False, any runtime type")
    print(f"'generator' exported     {'generator' in exported['runtime']}")
    print()

    renamed = from_dict(TrainingConfig, {**exported, "notes": "a different note"})
    print(f"notes is compare=False   equal after changing it: {config_equal(config, renamed)}")
    print(f"                         fingerprint holds:       {config_hash(config) == config_hash(renamed)}")

    resumed = from_dict(TrainingConfig, {**exported, "resumed_from": "runs/other.pt"})
    print(f"resumed_from is hash=False  equal after changing it: {config_equal(config, resumed)}")
    print(f"                            fingerprint holds:       {config_hash(config) == config_hash(resumed)}")
    print("Export carries every init=True field; equality ranges over the compared")
    print("fields, and the fingerprint over the hashing fields within those.")


def show_weakref(config: TrainingConfig) -> None:
    """Take a weak reference to the section that declares ``weakref_slot=True``.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    reference = weakref.ref(config.schedule)
    print(f"weakref.ref(schedule)  {reference() is config.schedule}   weakref_slot=True on a slotted section")


def show_node_scope(config: TrainingConfig) -> None:
    """Report the subtree scoping a ``ConfigNode`` gives each section.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("7. ConfigNode: every operation scoped to the node it is called on")
    print(f"root fingerprint       {config.cfg.hash()}")
    print(f"optimizer fingerprint  {config.optimizer.cfg.hash()}")
    print(f"data fingerprint       {config.data.cfg.hash()}")
    print(f"optimizer.to_dict()    {config.optimizer.cfg.to_dict()}")
    rebuilt = TrainingConfig.cfg.from_dict(config.cfg.to_dict())
    print(f"root round trip        {config_equal(config, rebuilt)}")
    show_weakref(config)


def show_round_trips(config: TrainingConfig) -> None:
    """Round-trip the config through both file formats.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("8. Round trips across JSON and YAML")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        json_path = config.cfg.to_file(root / "resolved.json")
        yaml_path = config.cfg.to_file(root / "resolved.yaml")
        from_json = TrainingConfig.cfg.from_file(json_path)
        from_yaml = TrainingConfig.cfg.from_file(yaml_path)
        print(f"dispatch by suffix     {json_path.name}, {yaml_path.name}")
        print(f"JSON equals original   {config_equal(config, from_json)}")
        print(f"YAML equals original   {config_equal(config, from_yaml)}")
        print(f"one fingerprint        {config_hash(from_json) == config_hash(from_yaml)}")
        show_surface_parity(config, root)
        print("The fingerprint is a SHA-256 prefix over the canonical JSON of the")
        print("hashing fields, so it is the same string across formats, processes,")
        print("and hash seeds.")


def show_surface_parity(config: TrainingConfig, root: Path) -> None:
    """Call both forms of every operation that has one, and report their agreement.

    The free function takes the class or the value; the ``cfg`` accessor reaches
    the same implementation from a node. Each row runs both and compares, so the
    claim that the two surfaces are one operation is exercised rather than stated.

    Args:
      config (TrainingConfig): The loaded configuration.
      root (Path): A directory the file operations may write into.
    """
    print("\nBoth surfaces, one implementation:")
    rows: list[tuple[str, object, object]] = [
        ("to_dict", to_dict(config), config.cfg.to_dict()),
        ("config_hash", config_hash(config), config.cfg.hash()),
        ("from_dict", from_dict(TrainingConfig, to_dict(config)), TrainingConfig.cfg.from_dict(to_dict(config))),
        ("validate_schema", validate_schema(TrainingConfig), TrainingConfig.cfg.validate_schema()),
        ("dumps_json", dumps_json(config), config.cfg.dumps_json()),
        ("dumps_yaml", dumps_yaml(config), config.cfg.dumps_yaml()),
        (
            "save_json / load_json",
            load_json(TrainingConfig, save_json(config, root / "free.json")),
            TrainingConfig.cfg.load_json(config.cfg.save_json(root / "node.json")),
        ),
        (
            "save_yaml / load_yaml",
            load_yaml(TrainingConfig, save_yaml(config, root / "free.yaml")),
            TrainingConfig.cfg.load_yaml(config.cfg.save_yaml(root / "node.yaml")),
        ),
        (
            "from_file / to_file",
            from_file(TrainingConfig, to_file(config, root / "free.dispatch.json")),
            TrainingConfig.cfg.from_file(config.cfg.to_file(root / "node.dispatch.yaml")),
        ),
    ]
    for name, free_result, node_result in rows:
        print(f"  {name:<22} agree: {free_result == node_result}")
    print(f"  {'config_equal':<22} agree: {config_equal(config, config)}   free function only")


@dataclass
class DeepHolder:
    """A root whose one field takes open data of any shape."""

    value: ConfigValue = None


@dataclass
class ArrayHolder:
    """A root whose one required field takes an array of any dtype."""

    values: np.ndarray


class RenderLink(np.ndarray):
    """An array whose plain form is another array, for a set number of hops.

    ``tolist`` belongs to an array's own class, so a class may answer with another
    array. The hop at zero answers with a number, which follows into nothing.
    """

    remaining = 0

    @override
    def tolist(self) -> Any:
        """Answer with the next array in the chain, or with the terminal number.

        Returns:
          Any: A further array while hops remain, otherwise the number 7.
        """
        if self.remaining == 0:
            return 7
        following = np.zeros(()).view(RenderLink)
        following.remaining = self.remaining - 1
        return following


def _render_chain(hops: int) -> Any:
    """Build an array whose rendering follows into ``hops`` further arrays.

    Args:
      hops (int): Array-to-array transitions before the terminal number.

    Returns:
      Any: The head of the chain.
    """
    head = np.zeros(()).view(RenderLink)
    head.remaining = hops
    return head


def show_limits(config: TrainingConfig) -> None:
    """Drive each declared limit at the last value it carries and the first it reports.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("9. Right up to the edge")

    print("Plain-data nesting, limit 64 levels:")
    for levels in (63, 64):
        node: ConfigValue = "leaf"
        for _ in range(levels):
            node = [node]
        try:
            to_dict(from_dict(DeepHolder, {"value": node}))
            print(f"  {levels} levels   carried")
        except ConfigError as error:
            print(f"  {levels} levels   reported at {error.issues[0].path[:24]}... ({len(error.issues)} issue)")

    print("\nArray elements, limit 1,000,000 per field:")
    for count, shape in ((1_000_000, (1000, 1000)), (1_000_001, (1_000_001,))):
        try:
            built = from_dict(ArrayHolder, {"values": np.zeros(shape)})
            print(f"  {count:>9,}  carried, shape {tuple(built.values.shape)}")
        except ConfigError as error:
            print(f"  {count:>9,}  {error.issues[0].message[:58]}...")

    print("\nFingerprint length, an int from 1 to 64:")
    print(f"  length=1    {config_hash(config, length=1)}")
    print(f"  length=64   {config_hash(config, length=64)}")
    for length in (0, 65):
        try:
            config_hash(config, length=length)
        except ConfigError as error:
            print(f"  length={length:<4} {error.issues[0].message[:56]}...")

    print("\nArray render hops, limit 64 arrays followed into one another:")
    for hops in (64, 65):
        try:
            rendered = to_dict(from_dict(ArrayHolder, {"values": _render_chain(hops)}))
            print(f"  {hops} hops    carried, writes {rendered['values']!r}")
        except ConfigError as error:
            print(f"  {hops} hops    {error.issues[0].message[:52]}...")

    print("\nZero-size arrays, where an empty axis ends the encoding:")
    for shape in ((0, 3), (2, 0), (2, 0, 5)):
        built = from_dict(ArrayHolder, {"values": np.zeros(shape)})
        print(f"  shape {shape!s:<10} writes {to_dict(built)['values']!r}")


def show_schema_boundary() -> None:
    """Report the first shape declined by each rule that draws the boundary.

    The sections above drive the last values confingo carries. These are the
    shapes it declines, read from the declaration alone before any config data
    exists, each with the remedy its message names.
    """
    banner("10. The schema boundary, from the other side")
    for rule, config_cls in REJECTED:
        try:
            validate_schema(config_cls)
        except ConfigError as error:
            wrapped = textwrap.fill(error.issues[0].message, width=74, subsequent_indent="    ")
            print(f"  {rule}:\n    {wrapped}")
        else:
            print(f"  {rule}: admitted")


def show_error_collection() -> None:
    """Report every problem in one pass, each at its own dotted path."""
    banner("11. Collect-all validation, every issue at its own path")
    broken: dict[str, ConfigValue] = {
        "epochs": -1,
        "optimizer": {"name": "adagrad", "lr": -0.5, "momentum": "fast"},
        "data": {"root": "/data", "shards": "one-shard", "splits": {"train": "most"}, "image_size": [1, 2, 3]},
        "telemetry": {"extra": None, "labels": [["unhashable"]]},
        "tensors": {},
        "runtime": {"device": "tpu"},
    }
    try:
        from_dict(TrainingConfig, broken, context="broken.yaml")
    except ConfigError as error:
        print(f"context: {error.context}")
        print(f"issues:  {len(error.issues)}\n")
        for issue in error.issues:
            print(f"  {issue.path or '<root>':<28} {issue.message[:70]}")


def show_invariants(config: TrainingConfig) -> None:
    """Report the ``__validate__`` hooks, which run on a fully populated node.

    Args:
      config (TrainingConfig): The loaded configuration.
    """
    banner("12. Dataclass invariants, reported at the node that owns them")
    exported = to_dict(config)
    exported["epochs"] = -1
    exported["optimizer"] = {**exported["optimizer"], "name": "sgd", "momentum": None, "warmup_steps": 99_999}
    try:
        from_dict(TrainingConfig, exported, context="invariants.yaml")
    except ConfigError as error:
        for issue in error.issues:
            print(f"  {issue.path or '<root>':<12} {issue.message}")
    print("\nEvery field coerced, so each node was constructed and its hook ran.")
    print("One hook reports several independent problems, each as its own issue,")
    print("and the root's hook is what sees across two sections at once.")


def show_implicit_sections() -> None:
    """Report how an omitted section hoists its required values to nested paths."""
    banner("13. Implicit sections, and a selected baseline")
    try:
        from_dict(TrainingConfig, {}, context="empty.yaml")
    except ConfigError as error:
        print("Loading an empty mapping builds every section from an empty mapping,")
        print("so each value that is still required is named at its own nested path:\n")
        for issue in error.issues:
            print(f"  {issue.path:<28} {issue.message}")
    print("\nThe checkpoint section selects an explicit default_factory baseline,")
    print(f"whose values differ from the section's own defaults: {CHECKPOINT_BASELINE}")


CHECKPOINT_BASELINE = "every_steps=500, keep_last=1"


def main() -> None:
    """Load the showcase config and drive every feature over it."""
    show_schema_check()
    config = TrainingConfig.cfg.from_file(HERE / "showcase.yaml")
    show_scalars_and_enums(config)
    show_containers(config)
    show_open_data(config)
    show_arrays(config)
    show_field_options(config)
    show_node_scope(config)
    show_round_trips(config)
    show_limits(config)
    show_schema_boundary()
    show_error_collection()
    show_invariants(config)
    show_implicit_sections()
    banner("Done")
    print(f"Precision members: {[member.value for member in Precision]}")
    print(f"Stage members:     {[member.value for member in Stage]}")
    print(f"Run fingerprint:   {config.cfg.hash()}")


if __name__ == "__main__":
    main()
