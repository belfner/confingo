[Documentation home](README.md)

# Types and coercion

This is the authoritative reference for the type boundary: which annotations a schema may use, which input shapes each accepts during `from_dict`, and which serialized form each produces under `to_dict`.

Two kinds of problems come out of this page's rules. An input outside an accepted shape produces a collected issue at that value's path. An annotation outside the accepted set produces a schema error before construction begins.


## How coercion works

- `from_dict` moves each supplied value toward its field's annotation, collecting an issue when the value's shape is outside what the annotation accepts.
- Defaults retain their authored values and skip coercion. See [defaults and precedence](schema-design.md#leaf-defaults-and-precedence).
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

A dataclass field absent from the input builds implicitly from an empty mapping, so its required leaves are reported at their nested paths. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements).


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
- A container field absent from the input is a missing required value when it carries no default. An intentionally empty container is authored as `field(default_factory=list)`.

Typical ML shapes: `hidden_widths: tuple[int, ...]` for layer sizes, `tuple[int, int]` for a fixed `(warmup_steps, total_steps)` schedule pair, and `metrics: set[str]` for tracked metrics.


## Mappings

`dict[str, T]`, `Mapping[str, T]`, and bare `dict` accept mappings with `str` keys and construct a concrete `dict`. Values of a bare `dict` pass through as `Any`.

- Every key is checked as a string at load time, which catches YAML documents whose keys parsed as numbers.
- Split definitions like `datasets: dict[str, DatasetConfig]` construct each value against the annotated section type, with the key in the issue path (`datasets.train.path`).
- A mapping field absent from the input is a missing required value when it carries no default; `field(default_factory=dict)` authors an intentionally empty mapping.


## Unions and optionals

Union members are tried in declaration order and the first member that coerces cleanly wins, so order unions deliberately: `int | str` sends `5` to `int` and `"5"` to `str`, while `OptimizerConfig | SchedulerConfig` tries `OptimizerConfig` first for every mapping. When every member fails, the field reports a single union mismatch issue.

`T | None` is special-cased: `None` is accepted directly, and any other input coerces straight through `T`, preserving nested issue paths and running the section's construction hooks exactly once.


## `Any` and plain data

`Any` fields pass plain data through unchanged: mappings, sequences, scalars, and `None` all survive as-is.

Finite-float validation still recurses through the value, including nested mapping keys and list elements, because every value in the tree needs a JSON form.

Under `to_dict`, tuple- and set-shaped values held by an `Any` field serialize as lists. An explicit container annotation is what restores the exact container type on the next load. See [cross-format round trips](files-and-identity.md#cross-format-round-trip).


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
| Structure | dataclasses (all fields `init=True`), unions of accepted members, `Optional[T]`, `Any` |
| Wrappers | `Annotated[T, ...]`, treated as `T` |

Annotations outside this set produce a `ConfigError` during schema preflight, even when the offending field is omitted from the input and would have used its default. Rejected shapes include:

- `Decimal`, `TypedDict`, `Iterable[T]`, and `NewType`
- mappings with keys other than `str`, including `dict[Any, T]`
- enums with object values, and enum-backed `Literal` options
- `init=False` fields anywhere in the tree

The preflight runs before any value is coerced. See [validation phases](validation-and-errors.md#two-validation-phases).


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
