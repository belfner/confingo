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
from dataclasses import dataclass
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
        backend: The owning backend, ``"numpy"`` or ``"torch"``.
        dtype: Concrete target dtype object, or None for bare and family forms.
        family: Abstract numpy scalar family the dtype must satisfy, or None.
        ndim: Enforced dimensionality from an authored fixed-arity shape tuple,
          or None when the annotation carries no shape claim.
        display: Short annotation name used in issue messages.
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
        matched: Whether the hint names an array type of a loaded backend.
        spec: The classification when the hint is valid, else None.
        error: A specific schema message when the hint is an array form with an
          unsupported or malformed parameterization, else None.
    """

    matched: bool
    spec: ArraySpec | None = None
    error: str | None = None


def _numpy() -> Any | None:
    """Return the numpy module when the application has imported it.

    Returns:
        The loaded numpy module, or None.
    """
    return sys.modules.get("numpy")


def _torch() -> Any | None:
    """Return the torch module when the application has imported it.

    Returns:
        The loaded torch module, or None.
    """
    return sys.modules.get("torch")


# ---------------------------------------------------------------------------
# Annotation classification
# ---------------------------------------------------------------------------


def inspect_annotation(hint: Any) -> AnnotationMatch:
    """Classify a resolved type hint against the loaded array backends.

    Args:
        hint: The resolved hint, with any ``Annotated`` wrapper intact.

    Returns:
        The classification: unmatched for hints unrelated to loaded backends,
        matched with a spec for supported forms, matched with an error message
        for array forms carrying unsupported parameterizations.
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
            return AnnotationMatch(True, ArraySpec("numpy", None, None, None, "ndarray"))
        if get_origin(base) is np.ndarray:
            return _inspect_numpy_generic(np, base)
    return AnnotationMatch(False)


def _inspect_torch(torch: Any, metadata: tuple[Any, ...]) -> AnnotationMatch:
    """Classify a ``torch.Tensor`` hint and its ``Annotated`` dtype metadata.

    Args:
        torch: The loaded torch module.
        metadata: The ``Annotated`` metadata tuple, empty for a bare hint.

    Returns:
        The classification for the tensor annotation.
    """
    dtype_meta = [item for item in metadata if isinstance(item, torch.dtype)]
    if len(dtype_meta) == 0:
        return AnnotationMatch(True, ArraySpec("torch", None, None, None, "Tensor"))
    if len(dtype_meta) > 1:
        names = ", ".join(str(item) for item in dtype_meta)
        return AnnotationMatch(True, error=f"conflicting torch dtype metadata: {names}")
    dtype = dtype_meta[0]
    if dtype not in _supported_torch_dtypes(torch):
        return AnnotationMatch(True, error=f"unsupported array dtype {_torch_dtype_name(dtype)}; {_SUPPORTED_MESSAGE}")
    display = f"Tensor[{_torch_dtype_name(dtype)}]"
    return AnnotationMatch(True, ArraySpec("torch", dtype, None, None, display))


def _inspect_numpy_generic(np: Any, base: Any) -> AnnotationMatch:
    """Classify a parameterized ``np.ndarray[Shape, np.dtype[...]]`` hint.

    Args:
        np: The loaded numpy module.
        base: The generic-alias hint whose origin is ``np.ndarray``.

    Returns:
        The classification for the array annotation.
    """
    args = get_args(base)
    if len(args) != 2:
        return AnnotationMatch(True, error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")
    shape_arg, dtype_arg = args
    ndim = _fixed_ndim(shape_arg)

    if get_origin(dtype_arg) is not np.dtype:
        if dtype_arg is np.dtype:
            return AnnotationMatch(True, ArraySpec("numpy", None, None, ndim, "ndarray"))
        return AnnotationMatch(True, error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")

    scalar_arg = get_args(dtype_arg)[0]
    if scalar_arg is Any or type(scalar_arg).__name__ == "TypeVar":
        return AnnotationMatch(True, ArraySpec("numpy", None, None, ndim, "ndarray"))
    for family_name in _NUMPY_FAMILY_NAMES:
        if scalar_arg is getattr(np, family_name):
            display = f"ndarray[numpy.{family_name}]"
            return AnnotationMatch(True, ArraySpec("numpy", None, scalar_arg, ndim, display))
    if isinstance(scalar_arg, type) and issubclass(scalar_arg, np.generic):
        if _is_abstract_numpy_scalar(np, scalar_arg):
            return AnnotationMatch(True, error=f"unsupported numpy dtype family numpy.{scalar_arg.__name__}")
        dtype = np.dtype(scalar_arg)
        if dtype.kind not in _SUPPORTED_KINDS or dtype.itemsize > 8:
            return AnnotationMatch(True, error=f"unsupported array dtype {dtype.name}; {_SUPPORTED_MESSAGE}")
        return AnnotationMatch(True, ArraySpec("numpy", dtype, None, ndim, f"ndarray[{dtype.name}]"))
    return AnnotationMatch(True, error="malformed ndarray annotation; expected ndarray[Shape, np.dtype[...]]")


def _is_abstract_numpy_scalar(np: Any, scalar_type: type) -> bool:
    """Report whether a numpy scalar type is abstract rather than instantiable.

    Args:
        np: The loaded numpy module.
        scalar_type: The ``np.generic`` subclass named in the annotation.

    Returns:
        True for abstract hierarchy members such as ``np.inexact`` or
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
        shape_arg: The first argument of an ``np.ndarray[...]`` annotation.

    Returns:
        The enforced number of dimensions, or None when the shape carries no
        claim.
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
        torch: The loaded torch module.

    Returns:
        The supported ``torch.dtype`` objects.
    """
    return tuple(getattr(torch, name) for name in _TORCH_DTYPE_NAMES)


def _torch_dtype_name(dtype: Any) -> str:
    """Render a torch dtype as its short name.

    Args:
        dtype: The ``torch.dtype`` object.

    Returns:
        The name with the ``torch.`` prefix removed, such as ``float32``.
    """
    return str(dtype).removeprefix("torch.")


# ---------------------------------------------------------------------------
# NumPy scalar leaves
# ---------------------------------------------------------------------------


def normalize_numpy_scalar(value: Any) -> tuple[bool, Any]:
    """Convert a supported numpy scalar into its exact Python equivalent.

    Args:
        value: Any value; only ``np.generic`` instances of supported kinds
          normalize.

    Returns:
        ``(True, scalar)`` with a Python bool/int/float when the value is a
        supported numpy scalar, else ``(False, None)``.
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
        overs: ``(path, value)`` pairs for integer leaves above the int64 range.
        dims: Expected length at each nesting depth, grown on first visit.
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
    overs: list[tuple[str, int]] | None = None
    dims: list[int] | None = None
    leaf_depth: int | None = None


def _walk_plain(node: Any, depth: int, path: str, field_path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one node of raw plain input, recursing into sequences.

    Args:
        node: The list/tuple or scalar leaf at this position.
        depth: Nesting depth of this node, 0 at the field's own value.
        path: Dotted, index-suffixed path of this node.
        field_path: Dotted path of the whole array field, for field-level issues.
        issue: Destination for problems found.
        state: Shared walk facts, updated in place.
    """
    if state.truncated:
        return
    if isinstance(node, (list, tuple)):
        if state.leaf_depth is not None and depth >= state.leaf_depth:
            state.ok = False
            issue(path, "ragged array: expected a scalar element, got a sequence")
            return
        if state.dims is None:
            state.dims = []
        if depth == len(state.dims):
            state.dims.append(len(node))
        elif len(node) != state.dims[depth]:
            state.ok = False
            issue(path, f"ragged array: expected {state.dims[depth]} items, got {len(node)}")
            return
        for index, child in enumerate(node):
            _walk_plain(child, depth + 1, f"{path}.{index}", field_path, issue, state)
            if state.truncated:
                return
        return
    if state.leaf_depth is None:
        state.leaf_depth = depth
    elif depth != state.leaf_depth:
        state.ok = False
        issue(path, "ragged array: expected a nested sequence, got a scalar")
        return
    state.count += 1
    if state.count > _ELEMENT_CAP:
        state.ok = False
        state.truncated = True
        issue(field_path, f"array has more than {_ELEMENT_CAP} elements; maximum is {_ELEMENT_CAP}")
        return
    _check_leaf(node, path, issue, state)


def _check_leaf(leaf: Any, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one leaf value against the walk's target category.

    Args:
        leaf: The scalar at this position, numpy scalars already normalized.
        path: Dotted, index-suffixed path of this leaf.
        issue: Destination for problems found.
        state: Shared walk facts, updated in place.
    """
    is_scalar, normalized = normalize_numpy_scalar(leaf)
    if is_scalar:
        leaf = normalized
    if state.category is None:
        state.category = "bool" if type(leaf) is bool else "number"
    if state.category == "bool":
        if type(leaf) is not bool:
            state.ok = False
            issue(path, f"expected bool for array dtype bool, got {type(leaf).__name__}")
        return
    if type(leaf) is bool:
        state.ok = False
        issue(path, f"expected a number{_dtype_clause(state.label)}, got bool")
        return
    if type(leaf) is int:
        _check_int_leaf(leaf, path, issue, state)
        return
    if type(leaf) is float:
        _check_float_leaf(leaf, path, issue, state)
        return
    state.ok = False
    issue(path, f"expected a number{_dtype_clause(state.label)}, got {type(leaf).__name__}")


def _check_int_leaf(leaf: int, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one integer leaf against the target's bounds.

    Args:
        leaf: The integer value.
        path: Dotted, index-suffixed path of this leaf.
        issue: Destination for problems found.
        state: Shared walk facts, updated in place.
    """
    if leaf < 0:
        state.negatives = True
    if state.collect_range:
        if leaf > _INT64_MAX:
            if state.overs is None:
                state.overs = []
            state.overs.append((path, leaf))
        elif leaf < _INT64_MIN:
            state.ok = False
            issue(path, f"value {leaf} is out of range for array dtype int64")
        return
    if state.int_lo is not None and state.int_hi is not None and not state.int_lo <= leaf <= state.int_hi:
        state.ok = False
        issue(path, f"value {leaf} is out of range for array dtype {state.label}")


def _check_float_leaf(leaf: float, path: str, issue: IssueSink, state: _WalkState) -> None:
    """Validate one float leaf: finiteness, integrality for integer targets, bounds.

    Args:
        leaf: The float value.
        path: Dotted, index-suffixed path of this leaf.
        issue: Destination for problems found.
        state: Shared walk facts, updated in place.
    """
    if not math.isfinite(leaf):
        state.ok = False
        issue(path, f"expected a finite float, got {leaf!r}")
        return
    if state.allow_float_leaves:
        state.has_float = True
        if state.float_bound is not None and abs(leaf) > state.float_bound:
            state.ok = False
            issue(path, f"value {leaf} is out of range for array dtype {state.label}")
        return
    if leaf != int(leaf):
        state.ok = False
        issue(path, f"expected an integral value for array dtype {state.label}, got {leaf}")
        return
    _check_int_leaf(int(leaf), path, issue, state)


def _dtype_clause(label: str | None) -> str:
    """Render the target-dtype clause of a leaf issue message.

    Args:
        label: The target dtype label, or None under an inferred form.

    Returns:
        A ``" for array dtype <label>"`` clause, or an empty string.
    """
    if label is None:
        return ""
    return f" for array dtype {label}"


def _index_path(path: str, indices: tuple[int, ...] | list[int]) -> str:
    """Append element indices to a field path.

    Args:
        path: Dotted path of the array field.
        indices: One index per dimension, empty for a 0-d value.

    Returns:
        The indexed path, or the field path itself for a 0-d value.
    """
    if len(indices) == 0:
        return path
    return path + "." + ".".join(str(index) for index in indices)


def _sanitize(exc: BaseException) -> str:
    """Flatten a backend exception message into one bounded line.

    Args:
        exc: The exception raised by a backend call.

    Returns:
        The message with whitespace collapsed, truncated to 200 characters.
    """
    text = " ".join(str(exc).split())
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _configure_walk(spec: ArraySpec, np: Any, torch: Any) -> _WalkState:
    """Build the walk state that encodes a spec's leaf rules.

    Args:
        spec: The classified annotation being coerced.
        np: The loaded numpy module, or None.
        torch: The loaded torch module, or None.

    Returns:
        A fresh walk state with category, bounds, and labels preset.
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
            if spec.dtype.itemsize < 8:
                state.float_bound = float(np.finfo(spec.dtype).max)
    elif spec.backend == "numpy" and spec.family is not None:
        state.category = "number"
        name = spec.family.__name__
        if name == "floating":
            state.label = "float64"
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
            if spec.dtype is not torch.float64:
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
        state: The finished walk facts, carrying negatives and over-int64 values.
        allow_uint64: Whether uint64 is available as a widening target.
        issue: Destination for out-of-range issues.

    Returns:
        ``"int64"`` or ``"uint64"``, or None when some value fits neither and
        indexed range issues were emitted.
    """
    overs: list[tuple[str, int]] = state.overs if state.overs is not None else []
    if len(overs) == 0:
        return "int64"
    if allow_uint64 and not state.negatives and all(value <= _UINT64_MAX for _, value in overs):
        return "uint64"
    if allow_uint64 and not state.negatives:
        selected = "uint64"
        bound = _UINT64_MAX
    else:
        selected = "int64"
        bound = _INT64_MAX
    failed = False
    for over_path, value in overs:
        if value > bound:
            failed = True
            issue(over_path, f"value {value} is out of range for array dtype {selected}")
    if failed:
        return None
    return selected


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
        value: The incoming field value.
        spec: The classified annotation.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The coerced backend object, or ``FAILED`` after reporting issues.
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
        value: The validated plain scalar / list / tuple input.
        spec: The classified annotation.
        state: The finished walk facts driving value-aware target selection.
        np: The loaded numpy module, or None.
        torch: The loaded torch module, or None.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The constructed backend object, or ``FAILED``.
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
        spec: The classified annotation.
        state: The finished walk facts.
        np: The loaded numpy module.
        issue: Destination for out-of-range issues under value-aware families.

    Returns:
        The target ``np.dtype``, or None when range selection failed.
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
        spec: The classified annotation.
        state: The finished walk facts.
        torch: The loaded torch module.
        issue: Destination for out-of-range issues.

    Returns:
        The target ``torch.dtype``, or None when range selection failed.
    """
    if spec.dtype is not None:
        return spec.dtype
    if state.category == "bool":
        return torch.bool
    if state.category is None:
        return torch.float64
    if state.has_float:
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
        built: The constructed or supplied backend object.
        spec: The classified annotation carrying the ndim claim.
        path: Dotted path of the field.
        issue: Destination for dimensionality issues.

    Returns:
        The (possibly reshaped) object, or ``FAILED``.
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


def _numpy_dtype_ok(dtype: Any) -> bool:
    """Report whether a numpy dtype sits inside the supported boundary.

    Args:
        dtype: The ``np.dtype`` of a supplied array.

    Returns:
        True for dense bool, integer, and float dtypes up to 64 bits.
    """
    return dtype.kind in _SUPPORTED_KINDS and dtype.itemsize <= 8


def _report_nonfinite(np: Any, value: Any, path: str, issue: IssueSink) -> bool:
    """Report every non-finite element of a float numpy array.

    Args:
        np: The loaded numpy module.
        value: The float-kind array to scan.
        path: Dotted path of the field.
        issue: Destination for indexed finiteness issues.

    Returns:
        True when every element is finite.
    """
    bad = ~np.isfinite(value)
    if not bool(bad.any()):
        return True
    for indices in np.argwhere(bad):
        element = float(value[tuple(indices)])
        issue(_index_path(path, [int(i) for i in indices]), f"expected a finite float, got {element!r}")
    return False


def _coerce_native_numpy(np: Any, value: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Coerce a supplied ``np.ndarray`` toward its annotation.

    Args:
        np: The loaded numpy module.
        value: The supplied array.
        spec: The classified annotation.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The array itself when it already satisfies the annotation, a new
        converted array when the concrete dtype differs, or ``FAILED``.
    """
    dtype = value.dtype
    if not _numpy_dtype_ok(dtype):
        issue(path, f"unsupported array dtype {dtype.name}; {_SUPPORTED_MESSAGE}")
        return FAILED
    if int(value.size) > _ELEMENT_CAP:
        issue(path, f"array has {int(value.size)} elements; maximum is {_ELEMENT_CAP}")
        return FAILED
    if dtype.kind == "f" and not _report_nonfinite(np, value, path, issue):
        return FAILED

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
        np: The loaded numpy module.
        value: The supplied array, already finite where float.
        target: The concrete target ``np.dtype``.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The converted array, or ``FAILED``.
    """
    src = value.dtype
    if (src.kind == "b") != (target.kind == "b"):
        issue(path, f"supplied dtype {src.name} does not satisfy array dtype {target.name}")
        return FAILED
    with np.errstate(all="ignore"):
        converted = value.astype(target)
        if target.kind in "iu":
            back = converted.astype(src)
            mismatched = back != value
            if bool(mismatched.any()):
                for indices in np.argwhere(mismatched):
                    element = value[tuple(indices)].item()
                    element_path = _index_path(path, [int(i) for i in indices])
                    if src.kind == "f" and element != int(element):
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


def _coerce_native_torch(torch: Any, value: Any, spec: ArraySpec, path: str, issue: IssueSink) -> Any:
    """Coerce a supplied ``torch.Tensor`` toward its annotation.

    A satisfying tensor returns unchanged, preserving its device and grad
    state; serialization is the normalization boundary.

    Args:
        torch: The loaded torch module.
        value: The supplied tensor.
        spec: The classified annotation.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The tensor itself, a new dtype-converted tensor, or ``FAILED``.
    """
    if value.layout is not torch.strided:
        issue(path, f"only dense strided torch tensors are supported; got {value.layout}")
        return FAILED
    if value.dtype not in _supported_torch_dtypes(torch):
        issue(path, f"unsupported array dtype {_torch_dtype_name(value.dtype)}; {_SUPPORTED_MESSAGE}")
        return FAILED
    if int(value.numel()) > _ELEMENT_CAP:
        issue(path, f"array has {int(value.numel())} elements; maximum is {_ELEMENT_CAP}")
        return FAILED
    if value.dtype.is_floating_point and not _report_nonfinite_torch(torch, value, path, issue):
        return FAILED
    if spec.dtype is None or value.dtype is spec.dtype:
        return value
    return _convert_torch(torch, value, spec.dtype, path, issue)


def _report_nonfinite_torch(torch: Any, value: Any, path: str, issue: IssueSink) -> bool:
    """Report every non-finite element of a float torch tensor.

    Args:
        torch: The loaded torch module.
        value: The float tensor to scan.
        path: Dotted path of the field.
        issue: Destination for indexed finiteness issues.

    Returns:
        True when every element is finite.
    """
    bad = ~torch.isfinite(value)
    if not bool(bad.any()):
        return True
    for indices in torch.nonzero(bad).cpu().tolist():
        element = float(value[tuple(indices)])
        issue(_index_path(path, indices), f"expected a finite float, got {element!r}")
    return False


def _convert_torch(torch: Any, value: Any, target: Any, path: str, issue: IssueSink) -> Any:
    """Convert a supplied tensor to a different concrete dtype, safely.

    Args:
        torch: The loaded torch module.
        value: The supplied tensor, already finite where float.
        target: The concrete target ``torch.dtype``.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        The converted tensor, or ``FAILED``.
    """
    src = value.dtype
    if (src is torch.bool) != (target is torch.bool):
        src_name, target_name = _torch_dtype_name(src), _torch_dtype_name(target)
        issue(path, f"supplied dtype {src_name} does not satisfy array dtype {target_name}")
        return FAILED
    converted = value.to(dtype=target)
    target_name = _torch_dtype_name(target)
    if not target.is_floating_point and target is not torch.bool:
        back = converted.to(dtype=src)
        mismatched = back != value
        if bool(mismatched.any()):
            for indices in torch.nonzero(mismatched).cpu().tolist():
                element = value[tuple(indices)].item()
                element_path = _index_path(path, indices)
                if src.is_floating_point and element != int(element):
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
        value: The incoming value.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        ``NOT_ARRAY`` when the value is unrelated to loaded backends, the value
        itself when it validates, or ``FAILED``.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        if not _numpy_dtype_ok(value.dtype):
            issue(path, f"unsupported array dtype {value.dtype.name}; {_SUPPORTED_MESSAGE}")
            return FAILED
        if int(value.size) > _ELEMENT_CAP:
            issue(path, f"array has {int(value.size)} elements; maximum is {_ELEMENT_CAP}")
            return FAILED
        if value.dtype.kind == "f" and not _report_nonfinite(np, value, path, issue):
            return FAILED
        return value
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            issue(path, f"only dense strided torch tensors are supported; got {value.layout}")
            return FAILED
        if value.dtype not in _supported_torch_dtypes(torch):
            issue(path, f"unsupported array dtype {_torch_dtype_name(value.dtype)}; {_SUPPORTED_MESSAGE}")
            return FAILED
        if int(value.numel()) > _ELEMENT_CAP:
            issue(path, f"array has {int(value.numel())} elements; maximum is {_ELEMENT_CAP}")
            return FAILED
        if value.dtype.is_floating_point and not _report_nonfinite_torch(torch, value, path, issue):
            return FAILED
        return value
    return NOT_ARRAY


def array_to_plain(value: Any, path: str, issue: IssueSink) -> Any:
    """Serialize a native array to its plain ``tolist()`` form.

    Torch tensors normalize on the way out: values detach from any autograd
    graph and copy to the CPU, so device, grad state, and layout are erased
    from the wire form.

    Args:
        value: The value being marshalled.
        path: Dotted path of the field.
        issue: Destination for problems found.

    Returns:
        ``NOT_ARRAY`` when the value is unrelated to loaded backends, the plain
        scalar / nested-list form when serialization succeeds, or ``FAILED``.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        if not _numpy_dtype_ok(value.dtype):
            issue(path, f"unsupported numpy dtype {value.dtype.name}")
            return FAILED
        if int(value.size) > _ELEMENT_CAP:
            issue(path, f"array has {int(value.size)} elements; maximum is {_ELEMENT_CAP}")
            return FAILED
        if value.dtype.kind == "f" and bool((~np.isfinite(value)).any()):
            for indices in np.argwhere(~np.isfinite(value)):
                element = float(value[tuple(indices)])
                issue(_index_path(path, [int(i) for i in indices]), f"cannot serialize non-finite float {element!r}")
            return FAILED
        return value.tolist()
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            issue(path, f"only dense strided torch tensors can be serialized; got {value.layout}")
            return FAILED
        if value.dtype not in _supported_torch_dtypes(torch):
            issue(path, f"unsupported torch dtype {_torch_dtype_name(value.dtype)}")
            return FAILED
        if int(value.numel()) > _ELEMENT_CAP:
            issue(path, f"array has {int(value.numel())} elements; maximum is {_ELEMENT_CAP}")
            return FAILED
        if value.dtype.is_floating_point and bool((~torch.isfinite(value)).any()):
            for indices in torch.nonzero(~torch.isfinite(value)).cpu().tolist():
                element = float(value[tuple(indices)])
                issue(_index_path(path, indices), f"cannot serialize non-finite float {element!r}")
            return FAILED
        try:
            return value.detach().to(device="cpu").tolist()
        except (RuntimeError, TypeError, ValueError) as exc:
            issue(path, f"cannot copy torch tensor to CPU for serialization: {_sanitize(exc)}")
            return FAILED
    return NOT_ARRAY
