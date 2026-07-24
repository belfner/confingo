"""Presence-detected NumPy array and PyTorch tensor support.

This adapter recognizes array annotations and values for backends the host
application has already imported. Detection reads ``sys.modules`` on every call
and matches annotations and values by identity and ``isinstance`` against the
loaded module's own classes, so confingo itself imports only the standard
library and works identically whether or not a backend is installed.

The wire form is the validated ``tolist()`` result: a JSON scalar for a 0-d
value, nested lists otherwise. Validation is hybrid: raw list/tuple/scalar
input runs through a small recursive walker that checks node types,
rectangularity, and per-leaf category/bounds with exact indexed issue paths,
while supplied native arrays are checked with vectorized masks. Construction
runs only after clean validation.

``_core.py`` talks to this module through a small adapter protocol:
``inspect_annotation`` classifies a resolved hint into an :class:`ArraySpec`,
and the value functions return ``NOT_ARRAY`` when a value is unrelated to the
loaded backends, ``FAILED`` when it matched but was invalid, or the coerced /
serialized result.
"""

from __future__ import annotations

import math
import sys
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    get_args,
    get_origin,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    IssueSink = Callable[[str, str], None]
    """Callback receiving ``(path, message)`` for each problem found."""

NOT_ARRAY = object()
"""Returned when a value or hint is unrelated to any loaded array backend."""

FAILED = object()
"""Returned when a value matched an array form but failed validation."""

_ELEMENT_CAP = 1_000_000
"""Hard limit on the element count of any single array field."""

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1
_FLOAT64_MAX = sys.float_info.max

_SUPPORTED_KINDS = frozenset("biuf")
"""NumPy dtype kinds inside the boundary: bool, signed and unsigned integer, float."""

_SUPPORTED_MESSAGE = "supported dtypes are bool, integer, and float up to 64 bits"

_TORCH_DTYPE_NAMES = (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
"""Names of the torch dtypes inside the boundary."""

_NUMPY_FAMILY_NAMES = ("floating", "integer", "signedinteger", "unsignedinteger", "number")
"""Names of the abstract numpy scalar families accepted as dtype arguments."""


@dataclass(frozen=True)
class ArraySpec:
    """Classified array annotation, carrying everything coercion needs.

    Attributes:
      backend (Literal["numpy", "torch"]): The owning backend, ``"numpy"`` or
        ``"torch"``.
      dtype (Any | None): Concrete target dtype object, or None for bare and
        family forms.
      family (Any | None): Abstract numpy scalar family the dtype must satisfy,
        or None.
      ndim (int | None): Enforced dimensionality from an authored fixed-arity
        shape tuple, or None when the annotation carries no shape claim.
      display (str): Short annotation name used in issue messages.
    """

    backend: Literal["numpy", "torch"]
    dtype: Any | None
    family: Any | None
    ndim: int | None
    display: str


@dataclass(frozen=True)
class AnnotationMatch:
    """Result of classifying one resolved type hint.

    Attributes:
      spec (ArraySpec | None): The classification when the hint is a valid array
        form, else None.
      error (str | None): A specific schema message when the hint is an array
        form with an unsupported or malformed parameterization, else None.
    """

    spec: ArraySpec | None = None
    error: str | None = None

    @property
    def matched(self) -> bool:
        """Whether the hint names an array type of a loaded backend."""
        return self.spec is not None or self.error is not None


def _numpy() -> Any | None:
    """Return the numpy module when the application has imported it.

    Returns:
      Any | None: The loaded numpy module, or None.
    """
    return sys.modules.get("numpy")


def _torch() -> Any | None:
    """Return the torch module when the application has imported it.

    Returns:
      Any | None: The loaded torch module, or None.
    """
    return sys.modules.get("torch")


# ---------------------------------------------------------------------------
# Annotation classification
# ---------------------------------------------------------------------------


def is_array_value(value: Any) -> bool:
    """Report whether a value is a native array of a loaded backend.

    Args:
      value (Any): Any value.

    Returns:
      bool: True for ``np.ndarray`` and ``torch.Tensor`` instances (subclasses
        included) of backends the application has imported.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        return True
    torch = _torch()
    return torch is not None and isinstance(value, torch.Tensor)


def inspect_annotation(hint: Any) -> AnnotationMatch:
    """Classify a resolved type hint against the loaded array backends.

    Args:
      hint (Any): The resolved hint, with any ``Annotated`` wrapper intact.

    Returns:
      AnnotationMatch: The classification: unmatched for hints unrelated to
        loaded backends, matched with a spec for supported forms, matched with
        an error message for array forms carrying unsupported parameterizations.
    """
    base = hint
    metadata: tuple[Any, ...] = ()
    if get_origin(base) is Annotated:
        annotated_args = get_args(base)
        base, metadata = annotated_args[0], annotated_args[1:]

    torch = _torch()
    if torch is not None and base is torch.Tensor:
        return _inspect_torch(torch, metadata)

    np = _numpy()
    if np is not None:
        if base is np.ndarray:
            return AnnotationMatch(ArraySpec("numpy", None, None, None, "ndarray"))
        if get_origin(base) is np.ndarray:
            return _inspect_numpy_generic(np, base)
    return AnnotationMatch()


def _inspect_torch(torch: Any, metadata: tuple[Any, ...]) -> AnnotationMatch:
    """Classify a ``torch.Tensor`` hint and its ``Annotated`` metadata.

    A ``torch.dtype`` metadata object pins the concrete dtype, and a
    fixed-arity all-``int`` tuple type such as ``tuple[int, int]`` encodes
    exactly its arity as the tensor's dimensionality, matching the shape rule
    of ``np.ndarray[tuple[int, int], ...]``. Unrelated metadata coexists.

    Args:
      torch (Any): The loaded torch module.
      metadata (tuple[Any, ...]): The ``Annotated`` metadata tuple, empty for a
        bare hint.

    Returns:
      AnnotationMatch: The classification for the tensor annotation.
    """
    dtype_meta = [item for item in metadata if isinstance(item, torch.dtype)]
    shape_meta = [item for item in metadata if _fixed_ndim(item) is not None]
    if len(dtype_meta) > 1:
        names = ", ".join(str(item) for item in dtype_meta)
        return AnnotationMatch(error=f"conflicting torch dtype metadata: {names}")
    if len(shape_meta) > 1:
        names = ", ".join(str(item) for item in shape_meta)
        return AnnotationMatch(error=f"conflicting tensor shape metadata: {names}")
    ndim = _fixed_ndim(shape_meta[0]) if len(shape_meta) == 1 else None
    if len(dtype_meta) == 0:
        return AnnotationMatch(ArraySpec("torch", None, None, ndim, "Tensor"))
    dtype = dtype_meta[0]
    if dtype not in _supported_torch_dtypes(torch):
        return AnnotationMatch(error=f"unsupported array dtype {_torch_dtype_name(dtype)}; {_SUPPORTED_MESSAGE}")
    display = f"Tensor[{_torch_dtype_name(dtype)}]"
    return AnnotationMatch(ArraySpec("torch", dtype, None, ndim, display))


def _inspect_numpy_generic(np: Any, base: Any) -> AnnotationMatch:
    """Classify a parameterized ``np.ndarray[Shape, np.dtype[...]]`` hint.

    Args:
      np (Any): The loaded numpy module.
      base (Any): The generic-alias hint whose origin is ``np.ndarray``.

    Returns:
      AnnotationMatch: The classification for the array annotation.
    """
    args = get_args(base)
    if len(args) != 2:
        return AnnotationMatch(error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")
    shape_arg, dtype_arg = args
    ndim = _fixed_ndim(shape_arg)

    if get_origin(dtype_arg) is not np.dtype:
        if dtype_arg is np.dtype:
            return AnnotationMatch(ArraySpec("numpy", None, None, ndim, "ndarray"))
        return AnnotationMatch(error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")

    scalar_arg = get_args(dtype_arg)[0]
    if scalar_arg is Any or type(scalar_arg).__name__ == "TypeVar":
        return AnnotationMatch(ArraySpec("numpy", None, None, ndim, "ndarray"))
    for family_name in _NUMPY_FAMILY_NAMES:
        if scalar_arg is getattr(np, family_name):
            display = f"ndarray[numpy.{family_name}]"
            return AnnotationMatch(ArraySpec("numpy", None, scalar_arg, ndim, display))
    if isinstance(scalar_arg, type) and issubclass(scalar_arg, np.generic):
        if _is_abstract_numpy_scalar(np, scalar_arg):
            return AnnotationMatch(error=f"unsupported numpy dtype family numpy.{scalar_arg.__name__}")
        dtype = np.dtype(scalar_arg)
        if dtype.kind not in _SUPPORTED_KINDS or dtype.itemsize > 8:
            return AnnotationMatch(error=f"unsupported array dtype {dtype.name}; {_SUPPORTED_MESSAGE}")
        return AnnotationMatch(ArraySpec("numpy", dtype, None, ndim, f"ndarray[{dtype.name}]"))
    return AnnotationMatch(error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")


def _is_abstract_numpy_scalar(np: Any, scalar_type: type) -> bool:
    """Report whether a numpy scalar type is abstract rather than instantiable.

    Args:
      np (Any): The loaded numpy module.
      scalar_type (type): The ``np.generic`` subclass named in the annotation.

    Returns:
      bool: True for abstract hierarchy members such as ``np.inexact`` or
        ``np.generic`` itself.
    """
    try:
        np.dtype(scalar_type)
    except TypeError:
        return True
    return False


def _fixed_ndim(shape_arg: Any) -> int | None:
    """Extract an enforced dimensionality from an authored shape argument.

    A fixed-arity all-``int`` tuple such as ``tuple[int, int]`` encodes exactly
    its arity as the array's dimensionality. Library placeholder shapes and
    variadic tuples carry no claim.

    Args:
      shape_arg (Any): The first argument of an ``np.ndarray[...]`` annotation.

    Returns:
      int | None: The enforced number of dimensions, or None when the shape
        carries no claim.
    """
    if get_origin(shape_arg) is not tuple:
        return None
    entries = get_args(shape_arg)
    if len(entries) == 0 or any(entry is not int for entry in entries):
        return None
    return len(entries)


def _supported_torch_dtypes(torch: Any) -> tuple[Any, ...]:
    """List the torch dtype objects inside the supported boundary.

    Args:
      torch (Any): The loaded torch module.

    Returns:
      tuple[Any, ...]: The supported ``torch.dtype`` objects.
    """
    return tuple(getattr(torch, name) for name in _TORCH_DTYPE_NAMES)


def _torch_dtype_name(dtype: Any) -> str:
    """Render a torch dtype as its short name.

    Args:
      dtype (Any): The ``torch.dtype`` object.

    Returns:
      str: The name with the ``torch.`` prefix removed, such as ``float32``.
    """
    return str(dtype).removeprefix("torch.")


# ---------------------------------------------------------------------------
# NumPy scalar leaves
# ---------------------------------------------------------------------------


def normalize_numpy_scalar(value: Any) -> tuple[bool, Any]:
    """Convert a supported numpy scalar into its exact Python equivalent.

    Args:
      value (Any): Any value; only ``np.generic`` instances of supported kinds
        normalize.

    Returns:
      tuple[bool, Any]: ``(True, scalar)`` with a Python bool/int/float when the
        value is a supported numpy scalar, else ``(False, None)``.
    """
    np = _numpy()
    if np is None or not isinstance(value, np.generic):
        return (False, None)
    dtype = value.dtype
    if dtype.kind not in _SUPPORTED_KINDS or dtype.itemsize > 8:
        return (False, None)
    item = value.item()
    if type(item) in (bool, int, float):
        return (True, item)
    return (False, None)


# ---------------------------------------------------------------------------
# Plain-input walker
# ---------------------------------------------------------------------------


@dataclass
class _WalkState:
    """Mutable facts collected while walking raw plain input.

    Attributes:
        ok: Whether every node seen so far validated.
        count: Number of leaf elements seen.
        truncated: Whether the walk stopped at the element cap.
        category: Leaf category, fixed by the target dtype or by the first leaf
          under an inferred form: ``"bool"``, ``"int"``, ``"float"``, or None
          until the first leaf decides.
        has_float: Whether any leaf arrived as a Python float.
        int_lo: Inclusive lower bound leaves must satisfy, or None.
        int_hi: Inclusive upper bound leaves must satisfy, or None.
        float_bound: Largest magnitude the target float dtype represents, or
          None when any finite float is fine.
        collect_range: Whether integer leaves defer range selection, recording
          out-of-int64 values in ``overs`` instead of failing immediately.
        allow_float_leaves: Whether float leaves are acceptable at all.
        negatives: Whether any integer leaf is negative.
        overs: ``(path, value)`` pairs for integer leaves above the int64 range,
          empty until one is seen.
        dims: Expected length at each nesting depth, grown on first visit, empty
          until the first sequence node.
        leaf_depth: Nesting depth of the first leaf, fixing the array's shape.
    """

    ok: bool = True
    count: int = 0
    truncated: bool = False
    category: str | None = None
    label: str | None = None
    has_float: bool = False
    int_lo: int | None = None
    int_hi: int | None = None
    float_bound: float | None = None
    collect_range: bool = False
    allow_float_leaves: bool = True
    negatives: bool = False
    overs: list[tuple[str, int]] = field(default_factory=list)
    dims: list[int] = field(default_factory=list)
    leaf_depth: int | None = None

    def reject(self, issue: IssueSink, path: str, message: str) -> None:
        """Record a validation issue and mark the walk failed.

        Args:
          issue (IssueSink): Destination for the problem.
          path (str): Dotted, index-suffixed path of the offending node.
          message (str): Human-readable description of the problem.
        """
        self.ok = False
        issue(path, message)


def _walk_plain(node: Any, depth: int, path: str, field_path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one node of raw plain input, recursing into sequences.

    Args:
      node (Any): The list/tuple or scalar leaf at this position.
      depth (int): Nesting depth of this node, 0 at the field's own value.
      path (str): Dotted, index-suffixed path of this node.
      field_path (str): Dotted path of the whole array field, for field-level
        issues.
      issue (IssueSink): Destination for problems found.
      state (_WalkState): Shared walk facts, updated in place.
    """
    if state.truncated:
        return
    if isinstance(node, (list, tuple)):
        if state.leaf_depth is not None and depth >= state.leaf_depth:
            state.reject(issue, path, "ragged array: expected a scalar element, got a sequence")
            return
        if depth == len(state.dims):
            state.dims.append(len(node))
        elif len(node) != state.dims[depth]:
            state.reject(issue, path, f"ragged array: expected {state.dims[depth]} items, got {len(node)}")
            return
        for index, child in enumerate(node):
            _walk_plain(child, depth + 1, f"{path}.{index}", field_path, issue, state)
            if state.truncated:
                return
        return
    if state.leaf_depth is None:
        if len(state.dims) > depth:
            # A sibling at this depth was a sequence, so a scalar here is ragged.
            state.reject(issue, path, "ragged array: expected a nested sequence, got a scalar")
            return
        state.leaf_depth = depth
    elif depth != state.leaf_depth:
        state.reject(issue, path, "ragged array: expected a nested sequence, got a scalar")
        return
    state.count += 1
    if state.count > _ELEMENT_CAP:
        state.truncated = True
        state.reject(issue, field_path, f"array has more than {_ELEMENT_CAP} elements; maximum is {_ELEMENT_CAP}")
        return
    _check_leaf(node, path, issue, state)


def _check_leaf(leaf: Any, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one leaf value against the walk's target category.

    Args:
      leaf (Any): The scalar at this position, numpy scalars already normalized.
      path (str): Dotted, index-suffixed path of this leaf.
      issue (IssueSink): Destination for problems found.
      state (_WalkState): Shared walk facts, updated in place.
    """
    is_scalar, normalized = normalize_numpy_scalar(leaf)
    if is_scalar:
        leaf = normalized
    if state.category is None:
        state.category = "bool" if type(leaf) is bool else "number"
    if state.category == "bool":
        if type(leaf) is not bool:
            state.reject(issue, path, f"expected bool for array dtype bool, got {type(leaf).__name__}")
        return
    if type(leaf) is bool:
        state.reject(issue, path, f"expected a number{_dtype_clause(state.label)}, got bool")
        return
    if type(leaf) is int:
        _check_int_leaf(leaf, path, issue, state)
        return
    if type(leaf) is float:
        _check_float_leaf(leaf, path, issue, state)
        return
    state.reject(issue, path, f"expected a number{_dtype_clause(state.label)}, got {type(leaf).__name__}")


def _check_int_leaf(leaf: int, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one integer leaf against the target's bounds.

    Args:
      leaf (int): The integer value.
      path (str): Dotted, index-suffixed path of this leaf.
      issue (IssueSink): Destination for problems found.
      state (_WalkState): Shared walk facts, updated in place.
    """
    if leaf < 0:
        state.negatives = True
    if state.collect_range:
        if leaf > _INT64_MAX or leaf < _INT64_MIN:
            # Range selection is deferred until the walk finishes, so values
            # outside int64 are recorded and judged against the selected target.
            state.overs.append((path, leaf))
        return
    if state.int_lo is not None and state.int_hi is not None:
        if not state.int_lo <= leaf <= state.int_hi:
            state.reject(issue, path, f"value {leaf} is out of range for array dtype {state.label}")
        return
    # Float targets: integer leaves honor the target dtype's magnitude bound.
    if state.float_bound is not None and abs(leaf) > state.float_bound:
        state.reject(issue, path, f"value {leaf} is out of range for array dtype {state.label}")


def _check_float_leaf(leaf: float, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one float leaf: finiteness, integrality for integer targets, bounds.

    Args:
      leaf (float): The float value.
      path (str): Dotted, index-suffixed path of this leaf.
      issue (IssueSink): Destination for problems found.
      state (_WalkState): Shared walk facts, updated in place.
    """
    if not math.isfinite(leaf):
        state.reject(issue, path, f"expected a finite float, got {leaf!r}")
        return
    if state.allow_float_leaves:
        state.has_float = True
        if state.float_bound is not None and abs(leaf) > state.float_bound:
            state.reject(issue, path, f"value {leaf} is out of range for array dtype {state.label}")
        return
    if leaf != int(leaf):
        state.reject(issue, path, f"expected an integral value for array dtype {state.label}, got {leaf}")
        return
    _check_int_leaf(int(leaf), path, issue, state)


def _dtype_clause(label: str | None) -> str:
    """Render the target-dtype clause of a leaf issue message.

    Args:
      label (str | None): The target dtype label, or None under an inferred
        form.

    Returns:
      str: A ``" for array dtype <label>"`` clause, or an empty string.
    """
    if label is None:
        return ""
    return f" for array dtype {label}"


def _index_path(path: str, indices: tuple[int, ...] | list[int]) -> str:
    """Append element indices to a field path.

    Args:
      path (str): Dotted path of the array field.
      indices (tuple[int, ...] | list[int]): One index per dimension, empty for
        a 0-d value.

    Returns:
      str: The indexed path, or the field path itself for a 0-d value.
    """
    if len(indices) == 0:
        return path
    return path + "." + ".".join(str(index) for index in indices)


def _sanitize(exc: BaseException) -> str:
    """Flatten a backend exception message into one bounded line.

    Args:
      exc (BaseException): The exception raised by a backend call.

    Returns:
      str: The message with whitespace collapsed, truncated to 200 characters.
    """
    text = " ".join(str(exc).split())
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _configure_walk(spec: ArraySpec, np: Any, torch: Any) -> _WalkState:
    """Build the walk state that encodes a spec's leaf rules.

    Args:
      spec (ArraySpec): The classified annotation being coerced.
      np (Any): The loaded numpy module, or None.
      torch (Any): The loaded torch module, or None.

    Returns:
      _WalkState: A fresh walk state with category, bounds, and labels preset.
    """
    state = _WalkState()
    if spec.backend == "numpy" and spec.dtype is not None:
        kind = spec.dtype.kind
        state.label = spec.dtype.name
        if kind == "b":
            state.category = "bool"
        elif kind in "iu":
            state.category = "number"
            state.allow_float_leaves = False
            info = np.iinfo(spec.dtype)
            state.int_lo, state.int_hi = int(info.min), int(info.max)
        else:
            state.category = "number"
            state.float_bound = float(np.finfo(spec.dtype).max)
    elif spec.backend == "numpy" and spec.family is not None:
        state.category = "number"
        name = spec.family.__name__
        if name == "floating":
            state.label = "float64"
            state.float_bound = float(np.finfo(np.float64).max)
        elif name == "signedinteger":
            state.label = "int64"
            state.allow_float_leaves = False
            state.int_lo, state.int_hi = _INT64_MIN, _INT64_MAX
        elif name == "unsignedinteger":
            state.label = "uint64"
            state.allow_float_leaves = False
            state.int_lo, state.int_hi = 0, _UINT64_MAX
        else:
            state.label = f"numpy.{name}"
            state.collect_range = True
            state.allow_float_leaves = name == "number"
    elif spec.backend == "torch" and spec.dtype is not None:
        state.label = _torch_dtype_name(spec.dtype)
        if spec.dtype is torch.bool:
            state.category = "bool"
        elif spec.dtype.is_floating_point:
            state.category = "number"
            state.float_bound = float(torch.finfo(spec.dtype).max)
        else:
            state.category = "number"
            state.allow_float_leaves = False
            info = torch.iinfo(spec.dtype)
            state.int_lo, state.int_hi = int(info.min), int(info.max)
    else:
        # Bare forms infer: the first leaf fixes the category, integer range
        # selection is deferred to the target-selection step.
        state.collect_range = True
    return state


def _select_integer_dtype(state: _WalkState, allow_uint64: bool, issue: IssueSink) -> str | None:
    """Pick the value-aware integer target for a collected walk.

    Args:
      state (_WalkState): The finished walk facts, carrying negatives and
        over-int64 values.
      allow_uint64 (bool): Whether uint64 is available as a widening target.
      issue (IssueSink): Destination for out-of-range issues.

    Returns:
      str | None: ``"int64"`` or ``"uint64"``, or None when some value fits
        neither and indexed range issues were emitted.
    """
    if len(state.overs) == 0:
        return "int64"
    if allow_uint64 and not state.negatives:
        selected, low, high = "uint64", 0, _UINT64_MAX
    else:
        selected, low, high = "int64", _INT64_MIN, _INT64_MAX
    failed = False
    for over_path, value in state.overs:
        if not low <= value <= high:
            failed = True
            issue(over_path, f"value {value} is out of range for array dtype {selected}")
    if failed:
        return None
    return selected


def _overs_fit_float64(state: _WalkState, issue: IssueSink) -> bool:
    """Check recorded huge integer leaves against the float64 magnitude bound.

    Args:
      state (_WalkState): The finished walk facts carrying values outside the
        int64 range.
      issue (IssueSink): Destination for out-of-range issues.

    Returns:
      bool: True when every recorded value is representable as a finite float64.
    """
    fits = True
    for over_path, value in state.overs:
        if abs(value) > _FLOAT64_MAX:
            fits = False
            issue(over_path, f"value {value} is out of range for array dtype float64")
    return fits


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def coerce_array(value: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Coerce a field value toward its array annotation.

    Native same-backend objects validate with vectorized masks and follow the
    three construction paths: already satisfying the annotation they return
    unchanged, needing a dtype conversion they produce a new object. Plain
    scalar / list / tuple input runs the recursive walker and constructs a new
    backend object.

    Args:
      value (Any): The incoming field value.
      spec (ArraySpec): The classified annotation.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The coerced backend object, or ``FAILED`` after reporting issues.
    """
    np = _numpy()
    torch = _torch()
    if spec.backend == "numpy" and np is not None and isinstance(value, np.ndarray):
        return _coerce_native_numpy(np, value, spec, path, issue)
    if spec.backend == "torch" and torch is not None and isinstance(value, torch.Tensor):
        return _coerce_native_torch(torch, value, spec, path, issue)

    is_scalar, normalized = normalize_numpy_scalar(value)
    if is_scalar:
        value = normalized
    if not isinstance(value, (list, tuple)) and type(value) not in (bool, int, float):
        typename = type(value).__name__
        issue(path, f"expected an array-compatible scalar or sequence for {spec.display}, got {typename}")
        return FAILED

    state = _configure_walk(spec, np, torch)
    _walk_plain(value, 0, path, path, issue, state)
    if not state.ok:
        return FAILED
    return _build_from_plain(value, spec, state, np, torch, path, issue)


def _build_from_plain(
    value: Any, spec: ArraySpec, state: _WalkState, np: Any, torch: Any, path: str, issue: IssueSink
) -> Any:
    """Construct the backend object for validated plain input.

    Args:
      value (Any): The validated plain scalar / list / tuple input.
      spec (ArraySpec): The classified annotation.
      state (_WalkState): The finished walk facts driving value-aware target
        selection.
      np (Any): The loaded numpy module, or None.
      torch (Any): The loaded torch module, or None.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The constructed backend object, or ``FAILED``.
    """
    if spec.backend == "numpy":
        target = _numpy_plain_target(spec, state, np, issue)
    else:
        target = _torch_plain_target(spec, state, torch, issue)
    if target is None:
        return FAILED
    try:
        built = np.array(value, dtype=target) if spec.backend == "numpy" else torch.tensor(value, dtype=target)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        issue(path, f"cannot build {spec.display}: {_sanitize(exc)}")
        return FAILED
    return _apply_ndim(built, spec, path, issue)


def _numpy_plain_target(spec: ArraySpec, state: _WalkState, np: Any, issue: IssueSink) -> Any | None:
    """Select the numpy dtype a validated plain input builds with.

    Args:
      spec (ArraySpec): The classified annotation.
      state (_WalkState): The finished walk facts.
      np (Any): The loaded numpy module.
      issue (IssueSink): Destination for out-of-range issues under value-aware
        families.

    Returns:
      Any | None: The target ``np.dtype``, or None when range selection failed.
    """
    if spec.dtype is not None:
        return spec.dtype
    family_name = spec.family.__name__ if spec.family is not None else None
    if family_name == "floating":
        return np.dtype(np.float64)
    if family_name == "signedinteger":
        return np.dtype(np.int64)
    if family_name == "unsignedinteger":
        return np.dtype(np.uint64)
    if state.category == "bool":
        return np.dtype(np.bool_)
    if state.category is None:
        # An empty value carries no leaves: families fall back to their integer
        # default, the bare form keeps numpy's own empty-input inference.
        return np.dtype(np.int64) if family_name is not None else np.dtype(np.float64)
    if state.has_float:
        if not _overs_fit_float64(state, issue):
            return None
        return np.dtype(np.float64)
    selected = _select_integer_dtype(state, allow_uint64=True, issue=issue)
    if selected is None:
        return None
    return np.dtype(np.int64) if selected == "int64" else np.dtype(np.uint64)


def _torch_plain_target(spec: ArraySpec, state: _WalkState, torch: Any, issue: IssueSink) -> Any | None:
    """Select the torch dtype a validated plain input builds with.

    Bare tensors pin bool / int64 / float64 by value shape, independent of
    ``torch.get_default_dtype``, so rebuilt values and hashes are stable.

    Args:
      spec (ArraySpec): The classified annotation.
      state (_WalkState): The finished walk facts.
      torch (Any): The loaded torch module.
      issue (IssueSink): Destination for out-of-range issues.

    Returns:
      Any | None: The target ``torch.dtype``, or None when range selection
        failed.
    """
    if spec.dtype is not None:
        return spec.dtype
    if state.category == "bool":
        return torch.bool
    if state.category is None:
        return torch.float64
    if state.has_float:
        if not _overs_fit_float64(state, issue):
            return None
        return torch.float64
    selected = _select_integer_dtype(state, allow_uint64=False, issue=issue)
    if selected is None:
        return None
    return torch.int64


def _apply_ndim(built: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Enforce an annotated dimensionality, padding zero-size values.

    A zero-element value with fewer dimensions than annotated gains zero-length
    trailing axes until the dimensionality matches, so the empty-shape encoding
    and fixed-ndim enforcement compose.

    Args:
      built (Any): The constructed or supplied backend object.
      spec (ArraySpec): The classified annotation carrying the ndim claim.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for dimensionality issues.

    Returns:
      Any: The (possibly reshaped) object, or ``FAILED``.
    """
    if spec.ndim is None or built.ndim == spec.ndim:
        return built
    count = int(built.size) if spec.backend == "numpy" else int(built.numel())
    if count == 0 and built.ndim < spec.ndim:
        return built.reshape(tuple(built.shape) + (0,) * (spec.ndim - built.ndim))
    issue(path, f"expected a {spec.ndim}-dimensional array, got {built.ndim} dimensions")
    return FAILED


# ---------------------------------------------------------------------------
# Native-array coercion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Backend:
    """Backend-specific primitives the shared native-array kernels call.

    One record per loaded backend lets ``_native_prefix_ok`` and
    ``_report_nonfinite`` run one gate chain over numpy arrays and torch tensors
    alike, differing only in these primitives.

    Attributes:
      count (Callable[[Any], int]): Element count of a value.
      dtype_supported (Callable[[Any], bool]): Whether the value's dtype is
        inside the supported boundary.
      dtype_name (Callable[[Any], str]): Short dtype name for issue messages.
      is_float (Callable[[Any], bool]): Whether the value carries a float dtype.
      nonfinite_indices (Callable[[Any], list[tuple[Any, ...]]]): Row-major index
        tuples of the value's non-finite elements in the backend's native
        integer type, empty when all are finite.
      element_at (Callable[[Any, tuple[Any, ...]], float]): The float element at
        a native-coordinate index tuple.
      form_issue (Callable[[Any, str], str | None]): Structural-form issue for a
        value under the naming verb, or None; numpy always returns None.
    """

    count: Callable[[Any], int]
    dtype_supported: Callable[[Any], bool]
    dtype_name: Callable[[Any], str]
    is_float: Callable[[Any], bool]
    nonfinite_indices: Callable[[Any], list[tuple[Any, ...]]]
    element_at: Callable[[Any, tuple[Any, ...]], float]
    form_issue: Callable[[Any, str], str | None]


def _numpy_dtype_ok(dtype: Any) -> bool:
    """Report whether a numpy dtype sits inside the supported boundary.

    Args:
      dtype (Any): The ``np.dtype`` of a supplied array.

    Returns:
      bool: True for dense bool, integer, and float dtypes up to 64 bits.
    """
    return dtype.kind in _SUPPORTED_KINDS and dtype.itemsize <= 8


def _numpy_backend(np: Any) -> _Backend:
    """Build the backend primitives for numpy arrays.

    Args:
      np (Any): The loaded numpy module.

    Returns:
      _Backend: The numpy primitives.
    """

    def nonfinite(value: Any) -> list[tuple[Any, ...]]:
        # Keep each coordinate in its backend-native integer type so element
        # access matches an ndarray subclass's own __getitem__; the reporter
        # normalizes a separate copy for the dotted path.
        mask = ~np.isfinite(value)
        if not bool(mask.any()):
            return []
        return [tuple(indices) for indices in np.argwhere(mask)]

    return _Backend(
        count=lambda value: int(value.size),
        dtype_supported=lambda value: _numpy_dtype_ok(value.dtype),
        dtype_name=lambda value: value.dtype.name,
        is_float=lambda value: value.dtype.kind == "f",
        nonfinite_indices=nonfinite,
        element_at=lambda value, idx: float(value[idx]),
        form_issue=lambda value, verb: None,
    )


def _torch_backend(torch: Any) -> _Backend:
    """Build the backend primitives for torch tensors.

    Args:
      torch (Any): The loaded torch module.

    Returns:
      _Backend: The torch primitives.
    """

    def nonfinite(value: Any) -> list[tuple[Any, ...]]:
        mask = ~torch.isfinite(value)
        if not bool(mask.any()):
            return []
        return [tuple(indices) for indices in torch.nonzero(mask).cpu().tolist()]

    return _Backend(
        count=lambda value: int(value.numel()),
        dtype_supported=lambda value: value.dtype in _supported_torch_dtypes(torch),
        dtype_name=lambda value: _torch_dtype_name(value.dtype),
        is_float=lambda value: bool(value.dtype.is_floating_point),
        nonfinite_indices=nonfinite,
        element_at=lambda value, idx: float(value[idx]),
        form_issue=lambda value, verb: _torch_form_issue(torch, value, verb),
    )


def _native_backend_for(value: Any) -> _Backend | None:
    """Return the backend primitives for a native array value, or None.

    Args:
      value (Any): Any value.

    Returns:
      _Backend | None: The matching backend, or None when the value is not a
        native array of a loaded backend.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        return _numpy_backend(np)
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return _torch_backend(torch)
    return None


def _report_nonfinite(
    backend: _Backend, value: Any, path: str, issue: IssueSink, message: Callable[[float], str]
) -> bool:
    """Report every non-finite element of a float array in backend order.

    Args:
      backend (_Backend): The backend primitives for the value.
      value (Any): The float-kind array or tensor to scan.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for indexed finiteness issues.
      message (Callable[[float], str]): Builds the issue text from a bad element.

    Returns:
      bool: True when every element is finite.
    """
    indices = backend.nonfinite_indices(value)
    for idx in indices:
        # element_at receives the backend-native coordinate tuple; the path takes
        # a Python-int copy so dotted paths render identically across backends.
        issue(_index_path(path, [int(i) for i in idx]), message(backend.element_at(value, idx)))
    return len(indices) == 0


def _load_dtype_message(name: str) -> str:
    """Build the load-time unsupported-dtype message.

    Args:
      name (str): The offending dtype's name.

    Returns:
      str: The issue message.
    """
    return f"unsupported array dtype {name}; {_SUPPORTED_MESSAGE}"


def _finite_load_message(element: float) -> str:
    """Build the load-time non-finite-element message.

    Args:
      element (float): The offending element.

    Returns:
      str: The issue message.
    """
    return f"expected a finite float, got {element!r}"


def _serialize_finite_message(element: float) -> str:
    """Build the marshal-time non-finite-element message.

    Args:
      element (float): The offending element.

    Returns:
      str: The issue message.
    """
    return f"cannot serialize non-finite float {element!r}"


def _native_prefix_ok(
    backend: _Backend,
    value: Any,
    path: str,
    issue: IssueSink,
    *,
    dtype_message: Callable[[str], str],
    nonfinite_message: Callable[[float], str],
    verb: str,
) -> bool:
    """Run the shared native-array gate chain in order.

    The gates fire in the order structural form (torch only) -> dtype support ->
    element cap -> finiteness, and an earlier gate's failure emits no later-gate
    issues, matching the per-backend chains callers rely on.

    Args:
      backend (_Backend): The backend primitives for the value.
      value (Any): The supplied array or tensor.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.
      dtype_message (Callable[[str], str]): Builds the unsupported-dtype message.
      nonfinite_message (Callable[[float], str]): Builds the non-finite message.
      verb (str): The clause naming the operation for a form issue.

    Returns:
      bool: True when the value clears every gate.
    """
    form = backend.form_issue(value, verb)
    if form is not None:
        issue(path, form)
        return False
    if not backend.dtype_supported(value):
        issue(path, dtype_message(backend.dtype_name(value)))
        return False
    count = backend.count(value)
    if count > _ELEMENT_CAP:
        issue(path, f"array has {count} elements; maximum is {_ELEMENT_CAP}")
        return False
    if not backend.is_float(value):
        return True
    return _report_nonfinite(backend, value, path, issue, nonfinite_message)


def _coerce_native_numpy(np: Any, value: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Coerce a supplied ``np.ndarray`` toward its annotation.

    Args:
      np (Any): The loaded numpy module.
      value (Any): The supplied array.
      spec (ArraySpec): The classified annotation.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The array itself when it already satisfies the annotation, a new
        converted array when the concrete dtype differs, or ``FAILED``.
    """
    if not _native_prefix_ok(
        _numpy_backend(np),
        value,
        path,
        issue,
        dtype_message=_load_dtype_message,
        nonfinite_message=_finite_load_message,
        verb="are supported",
    ):
        return FAILED
    dtype = value.dtype
    if spec.family is not None:
        if not issubclass(dtype.type, spec.family):
            issue(path, f"supplied dtype {dtype.name} does not satisfy numpy.{spec.family.__name__}")
            return FAILED
        return _apply_ndim(value, spec, path, issue)
    if spec.dtype is None or dtype == spec.dtype:
        return _apply_ndim(value, spec, path, issue)
    converted = _convert_numpy(np, value, spec.dtype, path, issue)
    if converted is FAILED:
        return FAILED
    return _apply_ndim(converted, spec, path, issue)


def _convert_numpy(np: Any, value: Any, target: Any, path: str, issue: IssueSink) -> Any:
    """Convert a supplied numpy array to a different concrete dtype, safely.

    Integer targets verify each element survives a cast round trip, which
    catches non-integral floats, out-of-range values, and silent wrapping in
    one comparison. Float targets check for overflow to infinity after the
    cast.

    Args:
      np (Any): The loaded numpy module.
      value (Any): The supplied array, already finite where float.
      target (Any): The concrete target ``np.dtype``.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The converted array, or ``FAILED``.
    """
    src = value.dtype
    if (src.kind == "b") != (target.kind == "b"):
        issue(path, f"supplied dtype {src.name} does not satisfy array dtype {target.name}")
        return FAILED
    with np.errstate(all="ignore"):
        if target.kind in "iu" and src.kind in "iu":
            # Integer-to-integer conversions check the target's range directly:
            # a cast round trip wraps bijectively across signedness at equal
            # widths, so it would pass values the target cannot hold.
            low, high = int(np.iinfo(target).min), int(np.iinfo(target).max)
            if int(value.size) > 0 and not (low <= int(value.min()) and int(value.max()) <= high):
                exact = value.astype(object)
                out_of_range = (exact < low) | (exact > high)
                for indices in np.argwhere(out_of_range):
                    element = value[tuple(indices)].item()
                    element_path = _index_path(path, [int(i) for i in indices])
                    issue(element_path, f"value {element} is out of range for array dtype {target.name}")
                return FAILED
            return value.astype(target)
        converted = value.astype(target)
        if target.kind in "iu":
            back = converted.astype(src)
            mismatched = back != value
            if bool(mismatched.any()):
                for indices in np.argwhere(mismatched):
                    element = value[tuple(indices)].item()
                    element_path = _index_path(path, [int(i) for i in indices])
                    if element != int(element):
                        issue(element_path, f"expected an integral value for array dtype {target.name}, got {element}")
                    else:
                        issue(element_path, f"value {element} is out of range for array dtype {target.name}")
                return FAILED
        elif target.kind == "f":
            overflowed = ~np.isfinite(converted)
            if bool(overflowed.any()):
                for indices in np.argwhere(overflowed):
                    element = value[tuple(indices)].item()
                    element_path = _index_path(path, [int(i) for i in indices])
                    issue(element_path, f"value {element} is out of range for array dtype {target.name}")
                return FAILED
    return converted


def _torch_form_issue(torch: Any, value: Any, verb: str) -> str | None:
    """Check a tensor's structural form: dense strided, materialized storage.

    Args:
      torch (Any): The loaded torch module.
      value (Any): The tensor to check.
      verb (str): The clause naming the operation, ``"are supported"`` at load
        and ``"can be serialized"`` at marshal.

    Returns:
      str | None: The issue message for nested, sparse/quantized-layout, or meta
        tensors, or None for a dense strided tensor with real storage.
    """
    if bool(value.is_nested):
        return f"only dense strided torch tensors {verb}; got a nested tensor"
    if value.layout is not torch.strided:
        return f"only dense strided torch tensors {verb}; got {value.layout}"
    if value.device.type == "meta":
        return "meta torch tensors carry no element values"
    return None


def _coerce_native_torch(torch: Any, value: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Coerce a supplied ``torch.Tensor`` toward its annotation.

    A satisfying tensor returns unchanged, preserving its device and grad
    state; serialization is the normalization boundary.

    Args:
      torch (Any): The loaded torch module.
      value (Any): The supplied tensor.
      spec (ArraySpec): The classified annotation.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The tensor itself, a new dtype-converted tensor, or ``FAILED``.
    """
    if not _native_prefix_ok(
        _torch_backend(torch),
        value,
        path,
        issue,
        dtype_message=_load_dtype_message,
        nonfinite_message=_finite_load_message,
        verb="are supported",
    ):
        return FAILED
    if spec.dtype is None or value.dtype is spec.dtype:
        return _apply_ndim(value, spec, path, issue)
    converted = _convert_torch(torch, value, spec.dtype, path, issue)
    if converted is FAILED:
        return FAILED
    return _apply_ndim(converted, spec, path, issue)


def _convert_torch(torch: Any, value: Any, target: Any, path: str, issue: IssueSink) -> Any:
    """Convert a supplied tensor to a different concrete dtype, safely.

    Args:
      torch (Any): The loaded torch module.
      value (Any): The supplied tensor, already finite where float.
      target (Any): The concrete target ``torch.dtype``.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: The converted tensor, or ``FAILED``.
    """
    src = value.dtype
    if (src is torch.bool) != (target is torch.bool):
        src_name, target_name = _torch_dtype_name(src), _torch_dtype_name(target)
        issue(path, f"supplied dtype {src_name} does not satisfy array dtype {target_name}")
        return FAILED
    target_name = _torch_dtype_name(target)
    if not target.is_floating_point and target is not torch.bool and not src.is_floating_point:
        # Integer-to-integer conversions check the target's range directly,
        # since torch casts wrap silently. Widening within the target's range
        # is always safe; otherwise the comparison runs on an int64 copy,
        # where every supported integer value and bound is exact. Comparing
        # the source tensor itself would cast each Python-int bound into the
        # source dtype and fail on bounds outside its range.
        src_info = torch.iinfo(src)
        info = torch.iinfo(target)
        if src_info.min >= info.min and src_info.max <= info.max:
            return value.to(dtype=target)
        wide = value.to(dtype=torch.int64)
        out_of_range = (wide < info.min) | (wide > info.max)
        if bool(out_of_range.any()):
            for indices in torch.nonzero(out_of_range).cpu().tolist():
                element = value[tuple(indices)].item()
                issue(_index_path(path, indices), f"value {element} is out of range for array dtype {target_name}")
            return FAILED
        return value.to(dtype=target)
    converted = value.to(dtype=target)
    if not target.is_floating_point and target is not torch.bool:
        back = converted.to(dtype=src)
        mismatched = back != value
        if bool(mismatched.any()):
            for indices in torch.nonzero(mismatched).cpu().tolist():
                element = value[tuple(indices)].item()
                element_path = _index_path(path, indices)
                if element != int(element):
                    issue(element_path, f"expected an integral value for array dtype {target_name}, got {element}")
                else:
                    issue(element_path, f"value {element} is out of range for array dtype {target_name}")
            return FAILED
    elif target.is_floating_point:
        overflowed = ~torch.isfinite(converted)
        if bool(overflowed.any()):
            for indices in torch.nonzero(overflowed).cpu().tolist():
                element = value[tuple(indices)].item()
                element_path = _index_path(path, indices)
                issue(element_path, f"value {element} is out of range for array dtype {target_name}")
            return FAILED
    return converted


# ---------------------------------------------------------------------------
# Any-field validation and marshalling
# ---------------------------------------------------------------------------


def validate_array_value(value: Any, path: str, issue: IssueSink) -> Any:
    """Validate a native array under an ``Any`` field, retaining it as-is.

    Args:
      value (Any): The incoming value.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: ``NOT_ARRAY`` when the value is unrelated to loaded backends, the
        value itself when it validates, or ``FAILED``.
    """
    backend = _native_backend_for(value)
    if backend is None:
        return NOT_ARRAY
    if not _native_prefix_ok(
        backend,
        value,
        path,
        issue,
        dtype_message=_load_dtype_message,
        nonfinite_message=_finite_load_message,
        verb="are supported",
    ):
        return FAILED
    return value


def array_to_plain(value: Any, path: str, issue: IssueSink) -> Any:
    """Serialize a native array to its plain ``tolist()`` form.

    Torch tensors normalize on the way out: values detach from any autograd
    graph and copy to the CPU, so device, grad state, and layout are erased
    from the wire form.

    Args:
      value (Any): The value being marshalled.
      path (str): Dotted path of the field.
      issue (IssueSink): Destination for problems found.

    Returns:
      Any: ``NOT_ARRAY`` when the value is unrelated to loaded backends, the
        plain scalar / nested-list form when serialization succeeds, or
        ``FAILED``.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        if not _native_prefix_ok(
            _numpy_backend(np),
            value,
            path,
            issue,
            dtype_message=lambda name: f"unsupported numpy dtype {name}",
            nonfinite_message=_serialize_finite_message,
            verb="are supported",
        ):
            return FAILED
        return value.tolist()
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        if not _native_prefix_ok(
            _torch_backend(torch),
            value,
            path,
            issue,
            dtype_message=lambda name: f"unsupported torch dtype {name}",
            nonfinite_message=_serialize_finite_message,
            verb="can be serialized",
        ):
            return FAILED
        try:
            return value.detach().to(device="cpu").tolist()
        except (RuntimeError, TypeError, ValueError) as exc:
            issue(path, f"cannot copy torch tensor to CPU for serialization: {_sanitize(exc)}")
            return FAILED
    return NOT_ARRAY


# ---------------------------------------------------------------------------
# Native equality
# ---------------------------------------------------------------------------

NOT_COMPARABLE = object()
"""Returned by ``native_equal`` for pairs outside the exact vectorized path."""


def native_equal(a: Any, b: Any) -> Any:
    """Compare two backend array values with vectorized operations.

    The fast path covers pairs whose vectorized comparison provably matches
    comparison of their serialized plain forms: both values are backend
    arrays holding at least one element, in supported dense form, with dtype
    kinds that promote exactly under elementwise comparison. A tensor
    compared against a numpy array converts through ``detach().cpu().numpy()``
    with bfloat16 widening exactly to float32 first, and tensor pairs on
    different devices compare on the CPU. Every other pair -- inexactly
    promoting kind mixes, zero-size values whose trailing dimensions collapse
    in the encoding, subclasses of either backend's array type (whose
    overridable operators could skew the verdict), and unsupported dtypes or
    tensor forms -- reports ``NOT_COMPARABLE`` so the caller compares plain
    forms.

    Args:
      a (Any): The left-hand value.
      b (Any): The right-hand value.

    Returns:
      Any: ``NOT_COMPARABLE`` when the pair falls outside the fast path, else
        whether the two values match in shape and every element.
    """
    np = _numpy()
    torch = _torch()
    a_tensor = torch is not None and type(a) is torch.Tensor
    b_tensor = torch is not None and type(b) is torch.Tensor
    a_array = np is not None and type(a) is np.ndarray
    b_array = np is not None and type(b) is np.ndarray
    if a_tensor and b_tensor:
        return _tensor_pair_equal(torch, a, b)
    if a_array and b_array:
        return _numpy_pair_equal(np, a, b)
    if a_tensor and b_array:
        converted = _tensor_as_numpy(torch, a)
        return NOT_COMPARABLE if converted is None else _numpy_pair_equal(np, converted, b)
    if a_array and b_tensor:
        converted = _tensor_as_numpy(torch, b)
        return NOT_COMPARABLE if converted is None else _numpy_pair_equal(np, a, converted)
    return NOT_COMPARABLE


def _kinds_promote_exactly(kind_a: str, size_a: int, kind_b: str, size_b: int) -> bool:
    """Report whether elementwise comparison across two dtypes is value-exact.

    Args:
      kind_a (str): NumPy-style dtype kind of the left value.
      size_a (int): Item size in bytes of the left dtype.
      kind_b (str): NumPy-style dtype kind of the right value.
      size_b (int): Item size in bytes of the right dtype.

    Returns:
      bool: True when comparison promotes without changing the plain form each
        value serializes to: matching kinds widen within the kind, and
        signed/unsigned integers mix exactly while the unsigned side stays below
        64 bits (a 64-bit unsigned operand against a signed one promotes to
        float64 and loses integer precision). A bool operand pairs natively only
        with another bool: bool serializes to ``true`` / ``false`` while a
        numeric ``1`` / ``0`` serializes differently, so a bool-versus-numeric
        pair falls through to the plain-form comparison, keeping equality aligned
        with the fingerprint.
    """
    if kind_a == kind_b:
        return True
    if {kind_a, kind_b} == {"i", "u"}:
        unsigned_size = size_a if kind_a == "u" else size_b
        return unsigned_size < 8
    return False


def _numpy_pair_equal(np: Any, a: Any, b: Any) -> Any:
    """Compare two numpy arrays where vectorized equality is provably exact.

    Args:
      np (Any): The loaded numpy module.
      a (Any): The left-hand array.
      b (Any): The right-hand array.

    Returns:
      Any: ``NOT_COMPARABLE`` for unsupported kinds, zero-size operands, or an
        inexactly promoting kind mix, else whether the arrays serialize to the
        same plain form: ``np.array_equal`` value equality, and for float kinds
        an additional sign-bit match so ``0.0`` and ``-0.0`` (equal by value but
        distinct in canonical JSON) compare unequal, keeping native equality
        aligned with the fingerprint.
    """
    kind_a, kind_b = a.dtype.kind, b.dtype.kind
    if kind_a not in _SUPPORTED_KINDS or kind_b not in _SUPPORTED_KINDS:
        return NOT_COMPARABLE
    if int(a.size) == 0 or int(b.size) == 0:
        return NOT_COMPARABLE
    if not _kinds_promote_exactly(kind_a, a.dtype.itemsize, kind_b, b.dtype.itemsize):
        return NOT_COMPARABLE
    if not bool(np.array_equal(a, b)):
        return False
    if kind_a == "f" or kind_b == "f":
        # Equal-but-oppositely-signed zeros serialize to distinct JSON tokens;
        # value equality alone would conflate them.
        return bool(np.array_equal(np.signbit(a), np.signbit(b)))
    return True


def _torch_kind(torch: Any, dtype: Any) -> str:
    """Map a supported torch dtype to its numpy-style kind letter.

    Args:
      torch (Any): The loaded torch module.
      dtype (Any): A dtype from the supported set.

    Returns:
      str: ``"b"`` for bool, ``"f"`` for floating dtypes, ``"u"`` for uint8, and
        ``"i"`` for the signed integer dtypes.
    """
    if dtype is torch.bool:
        return "b"
    if bool(dtype.is_floating_point):
        return "f"
    return "u" if dtype is torch.uint8 else "i"


def _tensor_pair_equal(torch: Any, a: Any, b: Any) -> Any:
    """Compare two tensors where vectorized equality is provably exact.

    Args:
      torch (Any): The loaded torch module.
      a (Any): The left-hand tensor.
      b (Any): The right-hand tensor.

    Returns:
      Any: ``NOT_COMPARABLE`` for unsupported dtypes or forms, zero-size
        operands, or an inexactly promoting kind mix, else whether shapes and
        every element match; a cross-device pair compares on the CPU. For float
        kinds a sign-bit match is required so ``0.0`` and ``-0.0`` (equal by
        value but distinct in canonical JSON) compare unequal.
    """
    supported = _supported_torch_dtypes(torch)
    if a.dtype not in supported or b.dtype not in supported:
        return NOT_COMPARABLE
    if _torch_form_issue(torch, a, "are supported") is not None:
        return NOT_COMPARABLE
    if _torch_form_issue(torch, b, "are supported") is not None:
        return NOT_COMPARABLE
    if int(a.numel()) == 0 or int(b.numel()) == 0:
        return NOT_COMPARABLE
    if not _kinds_promote_exactly(
        _torch_kind(torch, a.dtype), a.element_size(), _torch_kind(torch, b.dtype), b.element_size()
    ):
        return NOT_COMPARABLE
    if tuple(a.shape) != tuple(b.shape):
        return False
    left, right = a, b
    if left.device != right.device:
        left, right = left.detach().cpu(), right.detach().cpu()
    if not bool((left == right).all().item()):
        return False
    if _torch_kind(torch, a.dtype) == "f" or _torch_kind(torch, b.dtype) == "f":
        # 0.0 and -0.0 are equal by value but serialize to distinct JSON tokens.
        return bool((torch.signbit(left) == torch.signbit(right)).all().item())
    return True


def _tensor_as_numpy(torch: Any, value: Any) -> Any | None:
    """Convert a tensor to a numpy array for a cross-backend comparison.

    Args:
      torch (Any): The loaded torch module.
      value (Any): The tensor to convert.

    Returns:
      Any | None: A CPU numpy view of the detached tensor, with bfloat16 widened
        exactly to float32 and a lazy negative bit materialized, or None for
        tensors outside the supported dense forms.
    """
    if value.dtype not in _supported_torch_dtypes(torch):
        return None
    if _torch_form_issue(torch, value, "are supported") is not None:
        return None
    out = value.detach()
    if bool(out.is_neg()):
        out = out.resolve_neg()
    if out.dtype is torch.bfloat16:
        out = out.to(torch.float32)
    if out.device.type != "cpu":
        out = out.cpu()
    return out.numpy()
