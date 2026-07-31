# Showcase

Every annotation category, field option, and operation confingo supports, in one schema, driven right up to the library's declared limits. Each is exercised rather than named: `tests/test_docs_examples.py` runs this example and pins the output line each feature produces.

```bash
cd examples/showcase
uv run python run.py
```

## Files

| File | What it holds |
| --- | --- |
| `schema.py` | The schema. One section per part of the type boundary. |
| `boundary.py` | The first shape each boundary rule declines, one per rule. |
| `showcase.yaml` | A config file supplying every field, with comments on the values that demonstrate a rule. |
| `run.py` | Loads the schema and prints what each feature produced, in fourteen sections. Calls all 13 `confingo.functional` operations, and every one that also has a `cfg` form through both surfaces, comparing the two. |

## What the schema covers

**Scalars and choices** (`OptimizerConfig`, `DataConfig`): `bool`, `int`, `float`, `str`, `Path`, `datetime`, `date`, `time`, `Literal[...]`, and enums with `str` and `int` member values, matched by member value, by member name, and through a `_missing_` hook that maps a spelling the members lack. Unions cover `T | None` and both orders of a numeric union.

**Sections** (`CheckpointConfig`): sub-config fields carrying bare annotations so an omitted section builds from an empty mapping, and one section selected through an explicit `default_factory` baseline whose values differ from its own field defaults.

**Variant groups** (`ScheduleConfig`): one annotation standing for the three schedules a run may pick, selected by the `kind` key the group names. `total_steps` is declared once on the group and shared, each variant adds fields of its own, and `ConstantSchedule` carries the shared field alone. `CosineSchedule` is also the `frozen=True, slots=True, weakref_slot=True` section, so the variant a file selects is the one a weak reference is taken to.

**Containers** (`DataConfig`): `list[T]`, `tuple[X, Y]`, `tuple[T, ...]`, `tuple[()]`, `dict[str, T]`, `Sequence[T]`, `Mapping[str, T]`.

**Sets** (`TelemetryConfig`): every element shape preflight admits, which is a scalar or a `tuple` / `frozenset` whose own arguments recursively satisfy the same rule. `set[str]`, `frozenset[int]`, a set of enum members, `set[str | None]`, `set[tuple[str, int]]`, `set[frozenset[str]]`, `set[ConfigScalar]`, and a nested `frozenset[tuple[tuple[int, str], frozenset[int]]]`.

**Open data** (`TelemetryConfig`): `ConfigValue` and `ConfigScalar`, direct and inside containers.

**Arrays** (`TensorConfig`): a bare `np.ndarray`, a concrete dtype, a dtype family, a shape-typed `np.ndarray[tuple[int, int], np.dtype[...]]`, a bare `torch.Tensor`, and the three `Annotated` tensor forms: a dtype, a dtype with a shape, and a shape alone.

**Field options** (`RuntimeConfig`, `TrainingConfig`): `init=False` runtime state populated in `__post_init__` and holding types outside the schema boundary, plus `compare=False` and `hash=False` showing how export, equality, and the fingerprint nest.

**Invariants**: `__validate__` on a section and on the root, the root's hook reading across two sections at once.

**Operations**: `from_dict`, `to_dict`, `config_equal`, `config_hash`, `validate_schema`, `dumps_json`, `save_json`, `load_json`, `dumps_yaml`, `save_yaml`, `load_yaml`, `from_file`, `to_file`. Each of the twelve that has both a free-function and a `cfg` form is called both ways and the two results compared, so dropping either form fails the test; `config_equal` is the free function that has no `cfg` form. A nested node also fingerprints and exports its own subtree. `weakref.ref` on the section declaring `weakref_slot=True`.

## The limits section

Section 9 of `run.py` drives each declared limit at the last value it carries and the first value it reports:

| Limit | Carried | Reported |
| --- | --- | --- |
| Plain-data nesting | 63 levels | 64 levels, at its own dotted path |
| Array elements per field | 1,000,000 | 1,000,001 |
| `config_hash` length | 1 and 64 | 0 and 65 |
| Array render hops | 64 arrays followed into one another | 65 |

It also shows zero-size arrays, where an empty axis ends the encoding, so `(0, 3)` writes `[]` and `(2, 0)` writes `[[], []]`.

## The boundary from the other side

Section 10 walks `boundary.py`, which holds one rejected schema per rule that draws the boundary: a union inside a set, a section inside a set, an element that rebuilds unhashable, an argument-free container, a type outside the accepted set, a mapping keyed by something other than `str`, enum values separated only by a subclass, an enum binding its own member lookup, a subclass of a supported scalar, a schema taking type parameters, and a default that would need coercion. Each prints the message preflight produces, in full, so the remedy each names is visible. Every one is pinned in `tests/test_docs_examples.py`.

## The reporting sections

Section 10 loads a config with problems spread across five sections and prints every issue from one pass, each at its dotted path. Section 11 shows `__validate__` running once every field has coerced. Section 12 loads an empty mapping, where every section builds from an empty mapping and each value that is still required is named at its own nested path.
