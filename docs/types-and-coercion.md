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

Arrays and tensors follow the same rule under `Any`: a supplied array validates on the way in (supported dtype, finite elements, the size cap) and stays in memory as the object you passed, `to_dict` renders it as plain scalars and lists, and reloading yields that plain data. An [array annotation](#arrays-and-tensors) is what rebuilds a backend object on the next load.


## Arrays and tensors

NumPy arrays and PyTorch tensors are field types whenever the host application has already imported the backend. Detection reads `sys.modules` on each call and matches annotations and values against the loaded module's own classes, so `import confingo` keeps its stdlib-only core, and a numpy-only program keeps torch unloaded. The backend packages install with the application itself, on the application's own terms.

The wire form is the array's validated `tolist()` result: a JSON scalar for a 0-d value, nested lists otherwise. The file carries values and nesting only; the annotation carries the dtype claim.

| Annotation | Rebuilt dtype | Promise |
| --- | --- | --- |
| `np.ndarray`, `npt.NDArray[Any]` | inferred: `bool`, `int64` / `uint64` by value, `float64` | values |
| `npt.NDArray[np.float32]` (any concrete `bool` / integer / float dtype up to 64 bits) | the annotated dtype | dtype + values |
| `npt.NDArray[np.floating]` (also `integer`, `signedinteger`, `unsignedinteger`, `number`) | the family's target: `float64`, `int64` / `uint64` by value | family + values |
| `np.ndarray[tuple[int, int], np.dtype[np.float64]]` | as the dtype rules | as above, and the fixed-arity shape tuple enforces exactly that dimensionality |
| `torch.Tensor` | pinned: `bool` / `int64` / `float64` | values |
| `Annotated[torch.Tensor, torch.float32]` (any supported `torch.dtype`, `bfloat16` included) | the annotated dtype | dtype + values |
| `Annotated[torch.Tensor, torch.float32, tuple[int, int]]` | as the dtype rules | as above, and the fixed-arity all-`int` shape tuple enforces exactly that dimensionality |
| `Annotated[torch.Tensor, tuple[int, int]]` | pinned: `bool` / `int64` / `float64` | values, and the shape tuple enforces exactly that dimensionality |

Accepted input for an array field: a same-backend array or tensor, nested lists / tuples, or a single `bool` / `int` / `float` for a 0-d value. NumPy scalar leaves normalize to their exact Python equivalents. Strings, mappings, sets, and cross-backend objects are reported as type mismatches; a torch value headed into a numpy field converts explicitly at the call site via `tolist()`.

Plain input is validated leaf by leaf before any backend call, with element issues at exact indexed paths (`weights.2.0: expected a number for array dtype float32, got str`). The checks cover leaf category (`bool` stays fully separate from numbers, exactly as scalar fields keep them), integral values for integer dtypes, dtype range bounds, finiteness, and rectangularity, where a ragged row reports the divergent index (`weights.1: ragged array: expected 3 items, got 2`).

A supplied array or tensor that already satisfies its annotation is stored as-is, preserving device placement and gradient state until serialization. A supplied array needing a concrete dtype conversion produces a new converted object, with the same per-element range and finiteness checks.

Dtype normalization is value-preserving by construction:

- Bare `torch.Tensor` rebuilds with pinned dtypes (`bool`, `int64`, `float64`) independent of `torch.set_default_dtype`, so every serialized value reloads exactly and `config_hash` stays stable across processes. `Annotated[torch.Tensor, torch.float32]` is the spelling that pins a narrower dtype.
- The broad `np.integer` / `np.number` families select their integer target by value: `int64` when every value fits, `uint64` for nonnegative values above the `int64` range, so a retained `uint64` array holding `2**63` survives a save/load cycle.
- Small float dtypes (`float16`, `bfloat16`, `float32`) widen exactly into JSON numbers, since every value they represent is exactly a binary64 float.

Serialization normalizes tensor execution state: values detach from any autograd graph and copy to the CPU, so device, `requires_grad`, and layout/stride details stay out of the file. Dense strided tensors serialize; sparse, quantized, nested, and meta forms are reported as issues, as are complex, object, structured, temporal, and string dtypes on the numpy side.

Every element must be finite, matching the scalar float rule, and an array field holds at most 1,000,000 elements; both directions check the limit before materializing data.

Round trips hold as canonical serialized equality: `to_dict(from_dict(cls, to_dict(config))) == to_dict(config)` for every supported input, and concrete-dtype annotations additionally guarantee dtype and bit-exact values. On [`@configclass`](schema-design.md#configclass-and-equality) schemas the plain `==` operator implements this contract, so `from_dict(cls, to_dict(config)) == config` reads literally; the `__eq__` a plain `@dataclass` generates raises on multi-element array fields, which is what the once-per-class `ConfigWarning` points out.

Two shape details are visible in the encoding. A 0-d array serializes as a JSON scalar and rebuilds 0-d. Dimensions after the first zero-length axis have no list representation (`(0, 3)` serializes as `[]`), so zero-size arrays rebuild with the sizes the encoding retains; under a fixed-dimensionality annotation they pad with trailing zero-length axes to the annotated rank, and the padded form serializes identically.

For consumers outside Python, the file stays ordinary JSON: nested arrays of numbers. A reader that parses every number as a double sees the usual precision limits for integers above 2**53; confingo's own round trip keeps full 64-bit integer exactness.


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
| Arrays | `np.ndarray` forms and `torch.Tensor` forms from [arrays and tensors](#arrays-and-tensors), when the backend is loaded |
| Wrappers | `Annotated[T, ...]`, treated as `T`; on tensors, a `torch.dtype` entry pins the dtype and a fixed-arity all-`int` shape tuple such as `tuple[int, int]` enforces dimensionality, each usable alone or together; every other metadata entry passes through as ordinary annotation metadata |

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
