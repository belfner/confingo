[Documentation home](README.md)

# Types and coercion

This is the authoritative reference for the type boundary: which annotations a schema may use, which input shapes each accepts during `from_dict`, and which serialized form each produces under `to_dict`.

Two kinds of problems come out of this page's rules. An input outside an accepted shape produces a collected issue at that value's path. An annotation outside the accepted set produces a schema error before construction begins.


## Choose an annotation

Start from the value a field holds and read down to the rules that govern it:

| To configure | Annotation | Details |
| --- | --- | --- |
| a single value | `bool`, `int`, `float`, `str`, `Path`, `datetime`, `date`, `time` | [Scalars](#scalars) |
| one of a fixed set | `Literal[...]`, or an `Enum` subclass | [Enums and literals](#enums-and-literals) |
| a nullable value | `T \| None` | [Unions and optionals](#unions-and-optionals) |
| a first-match choice of types | `X \| Y` in declaration order | [Unions and optionals](#unions-and-optionals) |
| a nested section | a dataclass type | [Nested dataclasses](#nested-dataclasses); builds implicitly from its own leaves |
| an ordered collection | `list[T]`, `tuple[T, ...]`, `tuple[X, Y]`, `Sequence[T]` | [Sequences and tuples](#sequences-and-tuples) |
| a unique collection | `set[T]`, `frozenset[T]` | [Sequences and tuples](#sequences-and-tuples) |
| a keyed collection | `dict[str, T]`, `Mapping[str, T]` | [Mappings](#mappings); keys are strings |
| free-form data | `Any` | [`Any` and plain data](#any-and-plain-data) |
| a numeric array | `np.ndarray`, `npt.NDArray[...]`, `torch.Tensor` | [Arrays and tensors](arrays-and-tensors.md); the backend is imported by the application |
| a value with a default fallback | `field(default=...)`, `field(default_factory=...)` | [leaf defaults and precedence](schema-design.md#leaf-defaults-and-precedence) |

The sections below are the authoritative rules for each row.


## How coercion works

- `from_dict` moves each supplied value toward its field's annotation, collecting an issue when the value's shape is outside what the annotation accepts.
- Defaults are validated against their annotation and their plain form, then used exactly as authored. Coercion applies to supplied values alone, so a default has to be written in the type its annotation names. See [defaults and precedence](schema-design.md#leaf-defaults-and-precedence).
- `to_dict` converts the built tree back to plain data: dicts, lists, strings, numbers, booleans, and `None`.


## Scalars

| Annotation | Accepted inputs | Runtime value | Serialized form |
| --- | --- | --- | --- |
| `bool` | `bool` | `bool` | `true` / `false` |
| `int` | `int`; `float` with an integral value (`2e6` -> `2_000_000`) | `int` | number |
| `float` | `float`; `int` (converted, result must be finite) | `float` | number |
| `str` | `str` | `str` | string |
| `Path` | `str`; `Path` | `Path` | string (`str(path)`) |
| `datetime` | `datetime`; ISO 8601 `str` | `datetime` | ISO 8601 string |
| `date` | `date`; ISO 8601 `str` | `date` | ISO 8601 string |
| `time` | `time`; ISO 8601 `str` | `time` | ISO 8601 string |
| `None` (as `NoneType` or in unions) | `None` | `None` | `null` |

Booleans are exact. `bool` fields accept `bool` values only, and `int` / `float` fields report `bool` inputs as type mismatches, keeping `flag: bool` and `count: int` fully separate even though Python's `bool` subclasses `int`.

Strings are exact in both directions. `str` fields accept `str` only, and numeric fields report strings such as `"5"` as type mismatches: a config file states numbers as numbers.

```python
from_dict(DataConfig, {"dataset_path": "d", "batch_size": 2e6})
# batch_size == 2_000_000; the integral float landed on an int field
```


## Enums and literals

Enum members must carry primitive values (`str`, `int`, or `bool`). An input matches an enum field by member value first, then by member name:

```python
class Optimizer(Enum):
    ADAMW = "adamw"
    SGD = "sgd"

# both inputs resolve to Optimizer.ADAMW:
#   "adamw"  (by value)
#   "ADAMW"  (by member name)
```

`to_dict` serializes an enum as its `.value`. Error messages for a failed match list the valid values.

`Literal[...]` options must be primitives or `None`, and matching compares value and exact runtime type together: `Literal["cpu", "cuda"]` accepts exactly those strings, and `Literal[1, 2]` reports `True` as a mismatch even though `True == 1` in Python.


## Nested dataclasses

A mapping supplied for a dataclass-annotated field recursively constructs that section, with issues collected under the section's dotted path.

Dataclasses held in containers (`list[StageConfig]`, `dict[str, DatasetConfig]`) construct the same way, with the index or key in the path (`stages.0.name`).

A bare-annotated dataclass field defaults to an implicit build from an empty mapping, so its required leaves are reported at their nested paths. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements).

The annotation drives construction, so a section annotated with a [config node](schema-design.md#config-nodes) class rebuilds as that class, in a direct field, in a container, or as a union member. Construction runs through the same recursion that builds a plain dataclass section, so each section's `__post_init__` and `__validate__` run once for the value that is kept.


## Sequences and tuples

Every sequence-container annotation accepts a sequence, a `set`, or a `frozenset` as input:

| Annotation | Accepted inputs | Runtime value | Serialized form |
| --- | --- | --- | --- |
| `list[T]` | sequence/set of `T`-coercible items | `list` | array |
| `tuple[T, ...]` | sequence/set, each item `T`-coercible | `tuple` | array |
| `tuple[X, Y]` | sequence/set of exactly that arity | `tuple` | array |
| `tuple[()]` / `typing.Tuple[()]` | empty sequence/set | `()` | `[]` |
| `set[T]` / `frozenset[T]` | sequence/set of hashable `T`-coercible items | `set` / `frozenset` | array, deterministically ordered |
| `Sequence[T]` | sequence/set of `T`-coercible items | `list` | array |
| bare `list` / `tuple` / `set` / `frozenset` | sequence/set; elements pass through as `Any` | matching container | array |

A few rules apply across every row of the table:

- Elements coerce individually, and element issues carry their index in the path (`hidden_widths.1`).
- String and bytes inputs follow scalar handling: a `list[str]` field reports a bare `"abc"` as one type mismatch.
- An unhashable element headed into a set becomes a collected issue alongside any sibling issues.
- A `set[T]` / `frozenset[T]` whose element type carries a config section is reported at preflight, since [config objects are unhashable](equality-and-hashing.md#config-objects-are-unhashable). Hold sections in a `list` or `tuple`, and key them by `config_hash(section)` when uniqueness matters.
- A container field uses its default when it has one, and a required container needs a supplied value. An intentionally empty container is authored as `field(default_factory=list)`.

Typical ML shapes: `hidden_widths: tuple[int, ...]` for layer sizes, `tuple[int, int]` for a fixed `(warmup_steps, total_steps)` schedule pair, and `metrics: set[str]` for tracked metrics.


## Mappings

`dict[str, T]`, `Mapping[str, T]`, and bare `dict` accept mappings with `str` keys and construct a concrete `dict`. Values of a bare `dict` pass through as `Any`.

- Every key is checked as a string at load time, which catches YAML documents whose keys parsed as numbers.
- Split definitions like `datasets: dict[str, DatasetConfig]` construct each value against the annotated section type, with the key in the issue path (`datasets.train.path`).
- A mapping field uses its default when it has one, and a required mapping needs a supplied value; `field(default_factory=dict)` authors an intentionally empty mapping.


## Unions and optionals

Union members are tried in declaration order and the first member that coerces cleanly wins, so order unions deliberately: `int | str` sends `5` to `int` and `"5"` to `str`, while `OptimizerConfig | SchedulerConfig` tries `OptimizerConfig` first for every mapping.

When every member fails, the field reports a summary naming the whole union and the member that came closest, followed by that one member's own issues at their own paths:

```text
config has 2 issues:
  - optimizer: expected AdamW | SGD; best match SGD failed with 1 issue
  - optimizer.lr: expected float, got str
```

"Closest" is the member whose attempt collected the fewest issues, and an equal count goes to the first declared member. That tie is what two structurally identical variants produce when only their discriminator `Literal` differs and the file carries a typo: each fails once, so the first declared member supplies the detail while the summary still names the whole union.

`T | None` is special-cased: `None` is accepted directly, and any other input coerces straight through `T`, preserving nested issue paths and running the section's construction hooks exactly once.


## `Any` and plain data

`Any` fields pass plain data through unchanged: mappings, sequences, scalars, and `None` all survive as-is.

Finite-float validation still recurses through the value, including nested mapping keys and list elements, because every value in the tree needs a JSON form.

Under `to_dict`, tuple- and set-shaped values held by an `Any` field serialize as lists. An explicit container annotation is what restores the exact container type on the next load. See [cross-format round trips](files-and-identity.md#cross-format-round-trip).

A mapping supplied to an `Any` field stays a mapping: the annotation names no class, so it reaches the field as plain data. A config object assigned to one programmatically serializes through `to_dict` as its plain form, and a [node](schema-design.md#config-nodes) or dataclass annotation is what reconstructs the object on the next load.

Arrays and tensors follow the same rule under `Any`: a supplied array validates on the way in (supported dtype, finite elements, the size cap) and stays in memory as the object you passed, `to_dict` renders it as plain scalars and lists, and reloading yields that plain data. An [array annotation](#arrays-and-tensors) is what rebuilds a backend object on the next load.


## Arrays and tensors

NumPy arrays and PyTorch tensors are field types whenever the host application has already imported the backend; the integration reads that application-loaded backend, while confingo's base runtime stays the standard-library core plus PyYAML-backed YAML I/O. A bare annotation (`np.ndarray`, `torch.Tensor`) rebuilds the array with a value-stable inferred dtype; a concrete annotation (`npt.NDArray[np.float32]`, `Annotated[torch.Tensor, torch.float32]`) pins the dtype, and a fixed-arity shape tuple pins the dimensionality. Values serialize as plain JSON (a scalar for a 0-d value, nested lists otherwise) and rebuild against the annotation on the next load.

The full contract -- backend activation, the annotation table, accepted inputs, dtype normalization, serialization state, finiteness and size limits, and array participation in round trips, equality, and hashing -- lives in [Arrays and tensors](arrays-and-tensors.md).


## Finite numbers and temporal exactness

JSON represents only finite numbers, so a non-finite float anywhere in supplied data produces an issue, and `to_dict` raises `ConfigError` when asked to serialize one. An integer too large for `float` conversion produces a collected overflow issue.

On temporal fields the subtype ordering is enforced: a `datetime` value supplied for a `date` field is reported as a type issue. The two types stay distinct through every load.


## Accepted schema boundary

The accepted annotation set is explicit and closed:

| Category | Accepted annotations |
| --- | --- |
| Scalars | `bool`, `int`, `float`, `str`, `Path`, `datetime`, `date`, `time`, `None` |
| Enums / literals | `Enum` subclasses with primitive member values; `Literal` with primitive or `None` options |
| Containers | `list`, `tuple`, `set`, `frozenset`, `dict`, `Sequence`, `Mapping` (str keys for mappings), bare or parameterized |
| Structure | dataclasses (each `init=True` field boundary-checked; an `init=False` field holds runtime state and is exempt), unions of accepted members, `Optional[T]`, `Any` |
| Arrays | `np.ndarray` forms and `torch.Tensor` forms from [arrays and tensors](#arrays-and-tensors), when the backend is loaded |
| Wrappers | `Annotated[T, ...]`, treated as `T`; on tensors, a `torch.dtype` entry pins the dtype and a fixed-arity all-`int` shape tuple such as `tuple[int, int]` enforces dimensionality, each usable alone or together; every other metadata entry passes through as ordinary annotation metadata |

An `init=True` annotation outside this set produces a `ConfigError` during schema preflight, even for a field that would have used its default. Rejected shapes include:

- `Decimal`, `TypedDict`, `Iterable[T]`, and `NewType`
- mappings with keys other than `str`, including `dict[Any, T]`
- enums with object values, and enum-backed `Literal` options

An [`init=False`](schema-design.md#field-options) field holds runtime state populated in `__post_init__` and is exempt from this boundary; its annotation need only resolve. The preflight runs before any value is coerced. See [validation phases](validation-and-errors.md#two-validation-phases).


## Working near the boundary

For a value whose natural type sits outside the accepted set, store a supported primitive and validate its syntax in the schema:

```python
from decimal import Decimal, InvalidOperation


@dataclass
class BudgetConfig:
    max_cost: str  # decimal string, e.g. "12.50"

    def __post_init__(self) -> None:
        try:
            Decimal(self.max_cost)
        except InvalidOperation as exc:
            raise ValueError(f"max_cost must be a decimal string: {self.max_cost!r}") from exc

    @property
    def max_cost_decimal(self) -> Decimal:
        return Decimal(self.max_cost)
```

Raising `ValueError` from `__post_init__` routes the failure into the collect-all report. See [dataclass invariants](validation-and-errors.md#dataclass-invariants).

The stored field stays serializable and hashable, round trips through every format, and the application reads the parsed value through the property.


---

[Previous: Schema design](schema-design.md) | [Home](README.md) | [Next: Validation and errors](validation-and-errors.md)
