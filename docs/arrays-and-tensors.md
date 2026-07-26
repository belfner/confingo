[Documentation home](README.md)

# Arrays and tensors

NumPy arrays and PyTorch tensors are confingo field types whenever the host
application has already imported the backend. Annotate the field, and the file
carries plain nested arrays of numbers while the config object holds a real array
or tensor.

Read the first three sections to declare, load, and save an array field. The
[exact contracts](#exact-contracts) below them cover backend activation, dtype
normalization, tensor execution state, round trips and hashing, zero-size shapes,
and `Any`. The core scalar and container rules live in
[Types and coercion](types-and-coercion.md).

For a small fixed group of numbers, a tuple is the simpler choice: write
`image_mean: tuple[float, float, float]` for three channel means, and confingo
enforces the arity on the core install. Reach for an array when the payload is
large, homogeneous, or headed into numeric code.


## Two starting schemas
A bare annotation rebuilds the array with an inferred dtype; a concrete annotation
pins the dtype and, with a fixed-arity shape tuple, the dimensionality.

```python
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from confingo import ConfigNode


@dataclass
class InferredArrays(ConfigNode):
    values: np.ndarray                       # dtype inferred by value on load


@dataclass
class TypedArrays(ConfigNode):
    kernel: npt.NDArray[np.float32]          # rebuilt as float32
    grid: np.ndarray[tuple[int, int], np.dtype[np.float64]]  # 2-D, float64
```

For PyTorch, `Annotated` metadata carries the dtype and shape claims:

```python
from dataclasses import dataclass
from typing import Annotated

import torch

from confingo import ConfigNode


@dataclass
class TensorField(ConfigNode):
    bias: torch.Tensor                                       # pinned bool/int64/float64
    weight: Annotated[torch.Tensor, torch.float32, tuple[int, int]]  # float32, 2-D
```


## Wire form and the dtype claim
The wire form is the array's validated `tolist()` result: a JSON scalar for a 0-d
value, nested lists otherwise. The file carries values and nesting only; the
annotation carries the dtype claim.

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

Accepted input for an array field: a same-backend array or tensor, nested lists /
tuples, or a single `bool` / `int` / `float` for a 0-d value. NumPy scalar leaves
normalize to their exact Python equivalents. Strings, mappings, sets, and
cross-backend objects are reported as type mismatches; a torch value headed into a
numpy field converts explicitly at the call site via `tolist()`.

Plain input is validated leaf by leaf before any backend call, with element issues
at exact indexed paths (`weights.2.0: expected a number for array dtype float32,
got str`). The checks cover leaf category (`bool` stays fully separate from
numbers, exactly as scalar fields keep them), integral values for integer dtypes,
dtype range bounds, finiteness, and rectangularity, where a ragged row reports the
divergent index (`weights.1: ragged array: expected 3 items, got 2`).

A supplied array or tensor that already satisfies its annotation is stored as-is,
preserving device placement and gradient state until serialization. A supplied
array needing a concrete dtype conversion produces a new converted object, with
the same per-element range and finiteness checks.


## Load and save an array field
Loading and saving needs nothing extra. The file carries plain nested numbers, and
the annotation rebuilds the array:

```python
import numpy as np
import numpy.typing as npt


@dataclass
class NormalizeConfig(ConfigNode):
    channel_mean: npt.NDArray[np.float64]
    channel_std: npt.NDArray[np.float64]
```

```json
{
  "channel_mean": [0.485, 0.456, 0.406],
  "channel_std": [0.229, 0.224, 0.225]
}
```

```python
config = NormalizeConfig.cfg.load_json("normalize.json")
config.channel_mean            # array([0.485, 0.456, 0.406]), dtype float64
config.cfg.save_json("resolved.json")
run_id = config.cfg.hash()
```

Two limits apply to every array field, in both directions. Every element must be
finite, matching the scalar float rule, and a field holds at most 1,000,000
elements; both are checked before any data is materialized.


## Exact contracts
The rules below decide edge cases: when the backends activate, how a dtype is
chosen, what happens to a tensor's execution state, how arrays take part in round
trips and hashing, how zero-size shapes encode, and how an array behaves under
`Any`. Reach for them when an array field does something unexpected.


## Backend activation
Detection reads `sys.modules` on each call and matches annotations and values
against the loaded module's own classes: the array/tensor integration activates
from the backend the application has imported, while confingo's base runtime stays
the standard-library core plus PyYAML-backed YAML I/O. The backend packages
install with the application itself, on the application's own terms.


## Dtype normalization
Dtype normalization is value-preserving by construction:

- Bare `torch.Tensor` rebuilds with pinned dtypes (`bool`, `int64`, `float64`)
  independent of `torch.set_default_dtype`, so every serialized value reloads
  exactly and `config_hash` stays stable across processes.
  `Annotated[torch.Tensor, torch.float32]` is the spelling that pins a narrower
  dtype.
- The broad `np.integer` / `np.number` families select their integer target by
  value: `int64` when every value fits, `uint64` for nonnegative values above the
  `int64` range, so a retained `uint64` array holding `2**63` survives a save/load
  cycle.
- Small float dtypes (`float16`, `bfloat16`, `float32`) widen exactly into JSON
  numbers, since every value they represent is exactly a binary64 float.


## Serialization state
Serialization normalizes tensor execution state: values detach from any autograd
graph and copy to the CPU, so the file records the detached CPU values in dense
strided form. Dense strided tensors serialize; sparse, quantized, nested, and meta
forms are reported as issues, as are complex, object, structured, temporal, and
string dtypes on the numpy side.


## Round trips, equality, and hashing
Round trips hold as canonical serialized equality:
`to_dict(from_dict(cls, to_dict(config))) == to_dict(config)` for every supported
input, and concrete-dtype annotations additionally guarantee dtype and bit-exact
values. The `==` operator implements this contract on every schema class. A
`ConfigNode` subclass carries [canonical equality](equality-and-hashing.md#canonical-equality)
from class-creation time, and every other schema dataclass receives it at first
schema processing, so `from_dict(cls, to_dict(config)) == config` reads
literally, with array fields compared through the backends' vectorized operations.
The `config_equal` free function exposes the same relation ahead of any engine
call.

[Hashing](equality-and-hashing.md#stable-run-identity) treats array and tensor
fields by their encoded values and nesting, exactly what the file records: two
bare-annotated arrays holding the same values at different dtype widths hash equal,
integer and float widths alike, as do tensors differing only in device, gradient
state, or stride and storage arrangement. A concrete dtype annotation is schema,
so it shapes the rebuilt value while the hash tracks the encoded values.


## Shape details in the encoding
Two shape details are visible in the encoding. A 0-d array serializes as a JSON
scalar and rebuilds 0-d. The first zero-length axis determines the retained
encoded dimensions (`(0, 3)` serializes as `[]`), so zero-size arrays rebuild with
the sizes the encoding retains; under a fixed-dimensionality annotation they pad
with trailing zero-length axes to the annotated rank, and the padded form
serializes identically.

For consumers in other languages, the file stays ordinary JSON: nested arrays of
numbers. A reader that parses every number as a double sees the usual precision
limits for integers above 2**53; confingo's own round trip keeps full 64-bit
integer exactness.


## Arrays under `Any`
A field annotated `Any` validates a supplied array on the way in (supported dtype,
finite elements, the size cap) and stays in memory as the object you passed;
`to_dict` renders it as plain scalars and lists, and reloading yields that plain
data. An array annotation is what rebuilds a backend object on the next load. See
[`Any` and plain data](types-and-coercion.md#any-and-plain-data).

---

Essentials: [Getting started](getting-started.md) | [Files, formats, and run identity](files-and-identity.md) | [Recipes](recipes.md) | [Documentation home](README.md)
