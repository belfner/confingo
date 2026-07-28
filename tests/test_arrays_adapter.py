"""Adapter-level tests for ``confingo._arrays``.

These exercise the adapter protocol directly: annotation classification, the
plain-input walker, native-array validation and conversion, the ``Any``-field
path, and marshalling. Integration through ``from_dict`` / ``to_dict`` is
covered by the core-hook tests.
"""

from __future__ import annotations

import sys
from typing import (
    Annotated,
    Any,
)

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
npt = pytest.importorskip("numpy.typing")

from confingo import _arrays  # noqa: E402


Sink = list[tuple[str, str]]


@pytest.fixture
def issues() -> Sink:
    return []


def sink(issues: Sink):
    return lambda path, message: issues.append((path, message))


def spec_of(hint: Any) -> _arrays.ArraySpec:
    match = _arrays.inspect_annotation(hint)
    assert match.matched, match
    assert match.spec is not None, match
    return match.spec


def error_of(hint: Any) -> str:
    match = _arrays.inspect_annotation(hint)
    assert match.matched, match
    assert match.error is not None, match
    return match.error


# --- annotation classification ------------------------------------------------


def test_bare_ndarray_and_nd_any_classify_as_bare():
    assert spec_of(np.ndarray).dtype is None
    assert spec_of(npt.NDArray[Any]).dtype is None


def test_concrete_dtypes_classify():
    for scalar, name in [(np.float16, "float16"), (np.uint64, "uint64"), (np.int8, "int8"), (np.bool_, "bool")]:
        spec = spec_of(npt.NDArray[scalar])
        assert spec.dtype == np.dtype(scalar)
        assert spec.display == f"ndarray[{name}]"


def test_families_classify():
    for family in (np.floating, np.integer, np.signedinteger, np.unsignedinteger, np.number):
        spec = spec_of(npt.NDArray[family])
        assert spec.family is family
        assert spec.dtype is None


def test_rejected_numpy_dtypes_and_families():
    assert "unsupported array dtype complex64" in error_of(npt.NDArray[np.complex64])
    assert "unsupported numpy dtype family numpy.inexact" in error_of(npt.NDArray[np.inexact])
    assert "unsupported numpy dtype family numpy.generic" in error_of(npt.NDArray[np.generic])
    if np.dtype(np.longdouble).itemsize > 8:
        assert "unsupported array dtype" in error_of(npt.NDArray[np.longdouble])


def test_fixed_arity_shape_encodes_ndim():
    assert spec_of(np.ndarray[tuple[int, int], np.dtype[np.float64]]).ndim == 2
    assert spec_of(np.ndarray[tuple[int, int, int], np.dtype[np.float64]]).ndim == 3
    assert spec_of(np.ndarray[Any, np.dtype[np.float64]]).ndim is None
    assert spec_of(np.ndarray[tuple[int, ...], np.dtype[np.float64]]).ndim is None
    assert spec_of(npt.NDArray[np.float64]).ndim is None


def test_torch_forms_classify():
    assert spec_of(torch.Tensor).dtype is None
    concrete = spec_of(Annotated[torch.Tensor, torch.bfloat16])
    assert concrete.dtype is torch.bfloat16
    assert concrete.display == "Tensor[bfloat16]"


def test_torch_metadata_errors():
    conflict = error_of(Annotated[torch.Tensor, torch.float32, torch.float64])
    assert conflict == "conflicting torch dtype metadata: torch.float32, torch.float64"
    assert "conflicting torch dtype metadata" in error_of(Annotated[torch.Tensor, torch.float32, torch.float32])
    assert "unsupported array dtype complex64" in error_of(Annotated[torch.Tensor, torch.complex64])


def test_unrelated_metadata_coexists_with_torch_dtype():
    spec = spec_of(Annotated[torch.Tensor, "unit: kg", torch.float32])
    assert spec.dtype is torch.float32


def test_tensor_shape_metadata_encodes_ndim():
    both = spec_of(Annotated[torch.Tensor, torch.float32, tuple[int, int]])
    assert both.dtype is torch.float32
    assert both.ndim == 2
    bare = spec_of(Annotated[torch.Tensor, tuple[int, int, int]])
    assert bare.dtype is None
    assert bare.ndim == 3
    with_extras = spec_of(Annotated[torch.Tensor, "unit: kg", tuple[int, int], torch.int32])
    assert with_extras.ndim == 2
    assert with_extras.dtype is torch.int32


def test_tensor_shape_metadata_without_a_claim_is_ignored():
    assert spec_of(Annotated[torch.Tensor, torch.float32, tuple[int, ...]]).ndim is None
    assert spec_of(Annotated[torch.Tensor, torch.float32, tuple[int, str]]).ndim is None


def test_conflicting_tensor_shape_metadata_errors():
    message = error_of(Annotated[torch.Tensor, tuple[int, int], tuple[int, int, int]])
    assert "conflicting tensor shape metadata" in message


def test_tensor_ndim_is_enforced(issues: Sink):
    spec = spec_of(Annotated[torch.Tensor, torch.float32, tuple[int, int]])
    built = _arrays.coerce_array([[1.0, 2.0]], spec, "w", sink(issues))
    assert issues == []
    assert built.shape == (1, 2)
    assert _arrays.coerce_array([1.0, 2.0], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "expected a 2-dimensional array, got 1 dimensions")]
    issues.clear()
    native = torch.zeros(2, 3, 4)
    assert _arrays.coerce_array(native, spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "expected a 2-dimensional array, got 3 dimensions")]


def test_tensor_zero_size_pads_to_the_annotated_ndim(issues: Sink):
    spec = spec_of(Annotated[torch.Tensor, torch.float32, tuple[int, int]])
    padded = _arrays.coerce_array([], spec, "w", sink(issues))
    assert padded.shape == (0, 0)
    native = _arrays.coerce_array(torch.zeros(0, dtype=torch.float32), spec, "w", sink(issues))
    assert native.shape == (0, 0)
    assert issues == []


def test_annotated_numpy_metadata_is_transparent():
    spec = spec_of(Annotated[npt.NDArray[np.float32], "unit: kg"])
    assert spec.dtype == np.dtype(np.float32)


def test_unrelated_hints_stay_unmatched():
    for hint in (int, list[int], dict, Any):
        assert _arrays.inspect_annotation(hint).matched is False


def test_hidden_backends_make_everything_unmatched(monkeypatch: pytest.MonkeyPatch):
    hint = npt.NDArray[np.float32]
    value = np.array([1.0])
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _arrays.inspect_annotation(hint).matched is False
    record: Sink = []
    assert _arrays.array_to_plain(value, "w", sink(record)) is _arrays.NOT_ARRAY
    assert _arrays.normalize_numpy_scalar(np.float32(1.0)) == (False, None)
    assert record == []


# --- plain-input walker -------------------------------------------------------


def test_walker_flags_categories_with_indexed_paths(issues: Sink):
    spec = spec_of(npt.NDArray[np.float32])
    result = _arrays.coerce_array([[1.0, "2"], [True, float("inf")]], spec, "w", sink(issues))
    assert result is _arrays.FAILED
    assert ("w.0.1", "expected a number for array dtype float32, got str") in issues
    assert ("w.1.0", "expected a number for array dtype float32, got bool") in issues
    assert ("w.1.1", "expected a finite float, got inf") in issues


def test_walker_flags_fractional_and_overflow_for_integer_targets(issues: Sink):
    spec = spec_of(npt.NDArray[np.uint8])
    result = _arrays.coerce_array([1.5, 256, -1, 3], spec, "w", sink(issues))
    assert result is _arrays.FAILED
    assert ("w.0", "expected an integral value for array dtype uint8, got 1.5") in issues
    assert ("w.1", "value 256 is out of range for array dtype uint8") in issues
    assert ("w.2", "value -1 is out of range for array dtype uint8") in issues


def test_walker_accepts_integral_floats_for_integer_targets(issues: Sink):
    spec = spec_of(npt.NDArray[np.int64])
    result = _arrays.coerce_array([1.0, 2.0], spec, "w", sink(issues))
    assert issues == []
    assert result.dtype == np.dtype(np.int64)
    assert result.tolist() == [1, 2]


def test_ragged_rows_report_the_divergent_row(issues: Sink):
    spec = spec_of(npt.NDArray[np.float64])
    assert _arrays.coerce_array([[1.0, 2.0, 3.0], [1.0, 2.0]], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "ragged array: expected 3 items, got 2")]


def test_inconsistent_nesting_reports(issues: Sink):
    spec = spec_of(npt.NDArray[np.float64])
    assert _arrays.coerce_array([[1.0], 2.0], spec, "w", sink(issues)) is _arrays.FAILED
    assert "ragged array" in issues[0][1]


def test_rejected_input_types(issues: Sink):
    spec = spec_of(npt.NDArray[np.float32])
    for value, typename in [("abc", "str"), (b"x", "bytes"), ({"a": 1}, "dict"), ({1, 2}, "set")]:
        issues.clear()
        assert _arrays.coerce_array(value, spec, "w", sink(issues)) is _arrays.FAILED
        expected = f"expected an array-compatible scalar or sequence for ndarray[float32], got {typename}"
        assert issues == [("w", expected)]


def test_cross_backend_objects_are_rejected(issues: Sink):
    numpy_spec = spec_of(npt.NDArray[np.float32])
    assert _arrays.coerce_array(torch.tensor([1.0]), numpy_spec, "w", sink(issues)) is _arrays.FAILED
    assert "got Tensor" in issues[0][1]
    issues.clear()
    torch_spec = spec_of(torch.Tensor)
    assert _arrays.coerce_array(np.array([1.0]), torch_spec, "w", sink(issues)) is _arrays.FAILED
    assert "got ndarray" in issues[0][1]


def test_numpy_scalar_leaves_normalize_inside_plain_input(issues: Sink):
    spec = spec_of(npt.NDArray[np.float32])
    result = _arrays.coerce_array([np.float32(1.5), np.int16(2)], spec, "w", sink(issues))
    assert issues == []
    assert result.tolist() == [1.5, 2.0]


def test_scalar_input_builds_zero_dimensional(issues: Sink):
    spec = spec_of(npt.NDArray[np.float64])
    result = _arrays.coerce_array(2.5, spec, "w", sink(issues))
    assert result.shape == ()
    assert result.tolist() == 2.5


# --- value-aware family targets -----------------------------------------------


def test_broad_integer_family_selects_by_value(issues: Sink):
    spec = spec_of(npt.NDArray[np.integer])
    assert _arrays.coerce_array([1, -2], spec, "w", sink(issues)).dtype == np.dtype(np.int64)
    assert _arrays.coerce_array([1, 2**63], spec, "w", sink(issues)).dtype == np.dtype(np.uint64)
    assert issues == []
    assert _arrays.coerce_array([-1, 2**63], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "value 9223372036854775808 is out of range for array dtype int64")]
    issues.clear()
    assert _arrays.coerce_array([1, 2**64], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", f"value {2**64} is out of range for array dtype uint64")]


def test_number_family_prefers_float64_when_floats_appear(issues: Sink):
    spec = spec_of(npt.NDArray[np.number])
    assert _arrays.coerce_array([1, 2.5], spec, "w", sink(issues)).dtype == np.dtype(np.float64)
    assert _arrays.coerce_array([1, 2], spec, "w", sink(issues)).dtype == np.dtype(np.int64)
    assert issues == []


def test_narrow_families_pin_their_width(issues: Sink):
    assert _arrays.coerce_array([1.5], spec_of(npt.NDArray[np.floating]), "w", sink(issues)).dtype == np.float64
    assert _arrays.coerce_array([1], spec_of(npt.NDArray[np.signedinteger]), "w", sink(issues)).dtype == np.int64
    assert _arrays.coerce_array([1], spec_of(npt.NDArray[np.unsignedinteger]), "w", sink(issues)).dtype == np.uint64
    assert issues == []
    assert _arrays.coerce_array([-1], spec_of(npt.NDArray[np.unsignedinteger]), "w", sink(issues)) is _arrays.FAILED


def test_bare_forms_infer_by_first_leaf(issues: Sink):
    spec = spec_of(np.ndarray)
    assert _arrays.coerce_array([True, False], spec, "w", sink(issues)).dtype == np.dtype(np.bool_)
    assert _arrays.coerce_array([1, 2], spec, "w", sink(issues)).dtype == np.dtype(np.int64)
    assert _arrays.coerce_array([1.0, 2], spec, "w", sink(issues)).dtype == np.dtype(np.float64)
    assert issues == []
    assert _arrays.coerce_array([1, True], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "expected a number, got bool")]


# --- bare torch pinning -------------------------------------------------------


def test_bare_torch_pins_dtypes_against_the_process_default(issues: Sink):
    spec = spec_of(torch.Tensor)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        floats = _arrays.coerce_array([1.5, 2.5], spec, "w", sink(issues))
        ints = _arrays.coerce_array([1, 2], spec, "w", sink(issues))
        bools = _arrays.coerce_array([True], spec, "w", sink(issues))
    finally:
        torch.set_default_dtype(previous)
    assert floats.dtype is torch.float64
    assert ints.dtype is torch.int64
    assert bools.dtype is torch.bool
    assert issues == []


def test_bare_torch_has_no_uint64_widening(issues: Sink):
    spec = spec_of(torch.Tensor)
    assert _arrays.coerce_array([2**63], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "value 9223372036854775808 is out of range for array dtype int64")]


def test_annotated_torch_builds_concrete(issues: Sink):
    spec = spec_of(Annotated[torch.Tensor, torch.float16])
    result = _arrays.coerce_array([1.5, 65504.0], spec, "w", sink(issues))
    assert result.dtype is torch.float16
    assert issues == []
    assert _arrays.coerce_array([1e10], spec, "w", sink(issues)) is _arrays.FAILED
    assert "out of range for array dtype float16" in issues[0][1]


# --- ndim enforcement and padding ---------------------------------------------


def test_zero_size_values_pad_to_the_annotated_ndim(issues: Sink):
    two_d = spec_of(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    assert _arrays.coerce_array([], two_d, "w", sink(issues)).shape == (0, 0)
    three_d = spec_of(np.ndarray[tuple[int, int, int], np.dtype[np.float64]])
    assert _arrays.coerce_array([[], []], three_d, "w", sink(issues)).shape == (2, 0, 0)
    assert issues == []


def test_nonzero_values_fail_ndim_mismatch(issues: Sink):
    two_d = spec_of(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    assert _arrays.coerce_array([[[1.0]]], two_d, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "expected a 2-dimensional array, got 3 dimensions")]
    issues.clear()
    assert _arrays.coerce_array(1.0, two_d, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "expected a 2-dimensional array, got 0 dimensions")]


def test_excess_nesting_of_empty_values_still_fails(issues: Sink):
    two_d = spec_of(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    assert _arrays.coerce_array([[[], []]], two_d, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "expected a 2-dimensional array, got 3 dimensions")]


# --- native arrays: three construction paths ----------------------------------


def test_satisfying_native_arrays_return_unchanged(issues: Sink):
    value = np.array([1.0], dtype=np.float32)
    assert _arrays.coerce_array(value, spec_of(npt.NDArray[np.float32]), "w", sink(issues)) is value
    assert _arrays.coerce_array(value, spec_of(np.ndarray), "w", sink(issues)) is value
    kept = np.array([1], dtype=np.uint64)
    assert _arrays.coerce_array(kept, spec_of(npt.NDArray[np.integer]), "w", sink(issues)) is kept
    tensor = torch.tensor([1.0], requires_grad=True)
    assert _arrays.coerce_array(tensor, spec_of(torch.Tensor), "w", sink(issues)) is tensor
    assert issues == []


def test_native_dtype_conversion_builds_a_new_object(issues: Sink):
    source = np.array([1, 2], dtype=np.int16)
    converted = _arrays.coerce_array(source, spec_of(npt.NDArray[np.float32]), "w", sink(issues))
    assert converted.dtype == np.dtype(np.float32)
    assert converted is not source
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = _arrays.coerce_array(tensor, spec_of(Annotated[torch.Tensor, torch.float32]), "w", sink(issues))
    assert out.dtype is torch.float32
    assert out is not tensor
    assert issues == []


def test_native_conversion_failures_are_indexed(issues: Sink):
    int_spec = spec_of(npt.NDArray[np.int64])
    assert _arrays.coerce_array(np.array([1.5, 2.0]), int_spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "expected an integral value for array dtype int64, got 1.5")]
    issues.clear()
    tiny = spec_of(npt.NDArray[np.uint8])
    assert _arrays.coerce_array(np.array([1, 300], dtype=np.int64), tiny, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "value 300 is out of range for array dtype uint8")]
    issues.clear()
    narrow = spec_of(npt.NDArray[np.float32])
    assert _arrays.coerce_array(np.array([1.0, 1e300]), narrow, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "value 1e+300 is out of range for array dtype float32")]


def test_native_bool_never_crosses_the_numeric_boundary(issues: Sink):
    int_spec = spec_of(npt.NDArray[np.int64])
    assert _arrays.coerce_array(np.array([True]), int_spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "supplied dtype bool does not satisfy array dtype int64")]
    issues.clear()
    bool_spec = spec_of(npt.NDArray[np.bool_])
    assert _arrays.coerce_array(np.array([1]), bool_spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "supplied dtype int64 does not satisfy array dtype bool")]


def test_native_cross_signedness_wrap_is_rejected(issues: Sink):
    unsigned = spec_of(npt.NDArray[np.uint8])
    assert _arrays.coerce_array(np.array([-1], dtype=np.int8), unsigned, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "value -1 is out of range for array dtype uint8")]
    issues.clear()
    signed = spec_of(npt.NDArray[np.int8])
    assert _arrays.coerce_array(np.array([255], dtype=np.uint8), signed, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "value 255 is out of range for array dtype int8")]
    issues.clear()
    torch_unsigned = spec_of(Annotated[torch.Tensor, torch.uint8])
    tensor = torch.tensor([-1], dtype=torch.int8)
    assert _arrays.coerce_array(tensor, torch_unsigned, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "value -1 is out of range for array dtype uint8")]


def test_native_int64_boundary_converts_to_uint64_exactly(issues: Sink):
    spec = spec_of(npt.NDArray[np.uint64])
    converted = _arrays.coerce_array(np.array([1, 2**62], dtype=np.int64), spec, "w", sink(issues))
    assert issues == []
    assert converted.dtype == np.dtype(np.uint64)
    assert converted.tolist() == [1, 2**62]


def test_integer_leaves_honor_float_target_bounds(issues: Sink):
    half = spec_of(npt.NDArray[np.float16])
    assert _arrays.coerce_array([10**100], half, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", f"value {10**100} is out of range for array dtype float16")]
    issues.clear()
    double = spec_of(npt.NDArray[np.float64])
    assert _arrays.coerce_array([10**400], double, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", f"value {10**400} is out of range for array dtype float64")]
    issues.clear()
    number = spec_of(npt.NDArray[np.number])
    assert _arrays.coerce_array([1.5, 10**400], number, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", f"value {10**400} is out of range for array dtype float64")]
    issues.clear()
    torch_half = spec_of(Annotated[torch.Tensor, torch.float16])
    assert _arrays.coerce_array([10**100], torch_half, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", f"value {10**100} is out of range for array dtype float16")]


def test_meta_and_nested_tensors_are_rejected(issues: Sink):
    spec = spec_of(torch.Tensor)
    meta = torch.empty(2, device="meta")
    assert _arrays.coerce_array(meta, spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "meta torch tensors carry no element values")]
    issues.clear()
    assert _arrays.array_to_plain(meta, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "meta torch tensors carry no element values")]
    issues.clear()
    nested = torch.nested.nested_tensor([[1.0], [2.0, 3.0]], layout=torch.jagged)
    assert _arrays.coerce_array(nested, spec, "w", sink(issues)) is _arrays.FAILED
    assert "nested tensor" in issues[0][1]
    issues.clear()
    assert _arrays.array_to_plain(nested, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "only dense strided torch tensors can be serialized; got a nested tensor")]


def test_empty_row_before_a_scalar_is_ragged(issues: Sink):
    spec = spec_of(npt.NDArray[np.float64])
    assert _arrays.coerce_array([[], 1.0], spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "ragged array: expected a nested sequence, got a scalar")]


def test_torch_integer_widening_converts_cleanly(issues: Sink):
    wide = spec_of(Annotated[torch.Tensor, torch.int16])
    converted = _arrays.coerce_array(torch.tensor([-128, 127], dtype=torch.int8), wide, "w", sink(issues))
    assert issues == []
    assert converted.dtype is torch.int16
    assert converted.tolist() == [-128, 127]
    huge = spec_of(Annotated[torch.Tensor, torch.int64])
    converted = _arrays.coerce_array(torch.tensor([-32768, 32767], dtype=torch.int16), huge, "w", sink(issues))
    assert issues == []
    assert converted.tolist() == [-32768, 32767]
    signed = spec_of(Annotated[torch.Tensor, torch.int16])
    converted = _arrays.coerce_array(torch.tensor([255], dtype=torch.uint8), signed, "w", sink(issues))
    assert issues == []
    assert converted.tolist() == [255]
    narrow = spec_of(Annotated[torch.Tensor, torch.uint8])
    assert _arrays.coerce_array(torch.tensor([300], dtype=torch.int16), narrow, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.0", "value 300 is out of range for array dtype uint8")]


def test_native_family_mismatch_reports(issues: Sink):
    spec = spec_of(npt.NDArray[np.unsignedinteger])
    assert _arrays.coerce_array(np.array([1], dtype=np.int64), spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "supplied dtype int64 does not satisfy numpy.unsignedinteger")]


def test_native_unsupported_dtypes_report(issues: Sink):
    spec = spec_of(np.ndarray)
    assert _arrays.coerce_array(np.array([1 + 2j]), spec, "w", sink(issues)) is _arrays.FAILED
    assert "unsupported array dtype complex128" in issues[0][1]
    issues.clear()
    assert _arrays.coerce_array(np.array([object()]), spec, "w", sink(issues)) is _arrays.FAILED
    assert "unsupported array dtype object" in issues[0][1]


def test_native_nonfinite_reports_indexed_paths(issues: Sink):
    spec = spec_of(npt.NDArray[np.float64])
    value = np.array([[1.0, float("nan")], [float("-inf"), 2.0]])
    assert _arrays.coerce_array(value, spec, "w", sink(issues)) is _arrays.FAILED
    assert ("w.0.1", "expected a finite float, got nan") in issues
    assert ("w.1.0", "expected a finite float, got -inf") in issues


def test_sparse_tensors_are_rejected(issues: Sink):
    sparse = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64), torch.zeros(0), (2, 2), check_invariants=False
    )
    assert _arrays.coerce_array(sparse, spec_of(torch.Tensor), "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "only dense strided torch tensors are supported; got torch.sparse_coo")]


def test_native_zero_size_pads_under_fixed_ndim(issues: Sink):
    spec = spec_of(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    padded = _arrays.coerce_array(np.zeros((0,)), spec, "w", sink(issues))
    assert padded.shape == (0, 0)
    assert issues == []


# --- element cap ----------------------------------------------------------------


def test_cap_allows_exactly_one_million(issues: Sink):
    spec = spec_of(npt.NDArray[np.uint8])
    value = np.zeros(1_000_000, dtype=np.uint8)
    assert _arrays.coerce_array(value, spec, "w", sink(issues)) is value


def test_cap_rejects_native_arrays_above_the_limit(issues: Sink):
    spec = spec_of(npt.NDArray[np.uint8])
    value = np.zeros(1_000_001, dtype=np.uint8)
    assert _arrays.coerce_array(value, spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", _arrays.element_cap_message("1000001"))]


def test_cap_stops_the_walker_on_plain_input(issues: Sink):
    spec = spec_of(npt.NDArray[np.uint8])
    assert _arrays.coerce_array([0] * 1_000_001, spec, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", _arrays.element_cap_message("more than 1000000"))]


def test_cap_rejects_on_marshal_before_materializing(issues: Sink):
    value = np.zeros(1_000_001, dtype=np.uint8)
    assert _arrays.array_to_plain(value, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", _arrays.element_cap_message("1000001"))]


# --- marshalling ----------------------------------------------------------------


def test_marshal_produces_nested_lists_and_scalars(issues: Sink):
    assert _arrays.array_to_plain(np.array([[1, 2]], dtype=np.int32), "w", sink(issues)) == [[1, 2]]
    assert _arrays.array_to_plain(np.array(2.5), "w", sink(issues)) == 2.5
    assert _arrays.array_to_plain(torch.tensor(3, dtype=torch.int64), "w", sink(issues)) == 3
    assert issues == []


def test_marshal_normalizes_grad_and_noncontiguous_tensors(issues: Sink):
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    assert _arrays.array_to_plain(tensor, "w", sink(issues)) == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    grad = torch.tensor([1.0], requires_grad=True)
    assert _arrays.array_to_plain(grad, "w", sink(issues)) == [1.0]
    assert issues == []


def test_marshal_widens_small_float_dtypes_exactly(issues: Sink):
    tensor = torch.tensor([1.5, 2.25], dtype=torch.bfloat16)
    assert _arrays.array_to_plain(tensor, "w", sink(issues)) == [1.5, 2.25]
    half = np.array([65504.0], dtype=np.float16)
    assert _arrays.array_to_plain(half, "w", sink(issues)) == [65504.0]
    assert issues == []


def test_marshal_rejects_nonfinite_with_indexed_paths(issues: Sink):
    value = np.array([1.0, float("nan")])
    assert _arrays.array_to_plain(value, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w.1", "cannot serialize non-finite float nan")]


def test_marshal_rejects_unsupported_forms(issues: Sink):
    assert _arrays.array_to_plain(np.array([object()]), "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "unsupported numpy dtype object")]
    issues.clear()
    sparse = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64), torch.zeros(0), (2, 2), check_invariants=False
    )
    assert _arrays.array_to_plain(sparse, "w", sink(issues)) is _arrays.FAILED
    assert issues == [("w", "only dense strided torch tensors can be serialized; got torch.sparse_coo")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_marshal_copies_cuda_tensors_to_cpu(issues: Sink):
    tensor = torch.tensor([1.0, 2.0], device="cuda")
    assert _arrays.array_to_plain(tensor, "w", sink(issues)) == [1.0, 2.0]
    assert issues == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_tensors_are_kept_on_device_at_load(issues: Sink):
    tensor = torch.tensor([1.0, 2.0], device="cuda")
    kept = _arrays.coerce_array(tensor, spec_of(torch.Tensor), "w", sink(issues))
    assert kept is tensor
    assert kept.device.type == "cuda"


# --- numpy scalars --------------------------------------------------------------


def test_numpy_scalars_normalize_exactly():
    assert _arrays.normalize_numpy_scalar(np.float32(1.5)) == (True, 1.5)
    assert _arrays.normalize_numpy_scalar(np.float64(2.5)) == (True, 2.5)
    assert _arrays.normalize_numpy_scalar(np.int64(-(2**62))) == (True, -(2**62))
    assert _arrays.normalize_numpy_scalar(np.uint64(2**63)) == (True, 2**63)
    ok, value = _arrays.normalize_numpy_scalar(np.bool_(True))
    assert ok
    assert value is True


def test_python_and_extended_scalars_stay_unnormalized():
    assert _arrays.normalize_numpy_scalar(1.5) == (False, None)
    assert _arrays.normalize_numpy_scalar(np.complex128(1 + 2j)) == (False, None)
    if np.dtype(np.longdouble).itemsize > 8:
        assert _arrays.normalize_numpy_scalar(np.longdouble(1.5)) == (False, None)


# --- native equality -------------------------------------------------------------


def test_native_equal_same_dtype_pairs():
    assert _arrays.native_equal(np.array([1, 2]), np.array([1, 2])) is True
    assert _arrays.native_equal(np.array([1, 2]), np.array([1, 3])) is False
    assert _arrays.native_equal(torch.tensor([1.5]), torch.tensor([1.5])) is True
    assert _arrays.native_equal(torch.tensor([1.5]), torch.tensor([2.5])) is False


def test_native_equal_shape_mismatch_is_a_verdict():
    assert _arrays.native_equal(np.array([1, 2]), np.array([[1, 2]])) is False
    assert _arrays.native_equal(torch.tensor([1]), torch.tensor([[1]])) is False


def test_native_equal_same_kind_widths_compare_exactly():
    assert _arrays.native_equal(np.array([5], dtype=np.int8), np.array([5], dtype=np.int64)) is True
    assert _arrays.native_equal(np.array([0.5], dtype=np.float16), np.array([0.5])) is True
    assert _arrays.native_equal(torch.tensor([7], dtype=torch.int16), torch.tensor([7], dtype=torch.int64)) is True
    assert (
        _arrays.native_equal(torch.tensor([0.5], dtype=torch.bfloat16), torch.tensor([0.5], dtype=torch.float64))
        is True
    )


def test_native_equal_cross_backend_converts_through_numpy():
    assert _arrays.native_equal(torch.tensor([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32)) is True
    assert _arrays.native_equal(np.array([[1, 2]]), torch.tensor([[1, 2]])) is True
    assert _arrays.native_equal(torch.tensor([1.0, 2.0]), np.array([1.0, 2.5])) is False
    grad = torch.tensor([1.0], requires_grad=True)
    assert _arrays.native_equal(grad, np.array([1.0])) is True


def test_native_equal_declines_inexact_and_degenerate_pairs():
    declined = [
        (np.array([1], dtype=np.int64), np.array([1.0])),
        (np.array([1], dtype=np.uint64), np.array([1], dtype=np.int64)),
        (torch.tensor([1]), torch.tensor([1.0])),
        (np.zeros((0, 3)), np.zeros((0, 4))),
        (torch.zeros(0), torch.zeros(0)),
        (np.array([1, 2]), [1, 2]),
        ([1, 2], [1, 2]),
    ]
    for left, right in declined:
        assert _arrays.native_equal(left, right) is _arrays.NOT_COMPARABLE


def test_native_equal_small_unsigned_mixes_and_bool_defers_to_plain_form():
    # bool serializes to true/false and numeric to 1/0, so a bool-versus-numeric
    # pair is not a native match; it defers to the token-aware plain-form path,
    # keeping equality aligned with the fingerprint. bool/bool stays native.
    assert _arrays.native_equal(np.array([True]), np.array([1])) is _arrays.NOT_COMPARABLE
    assert _arrays.native_equal(np.array([True]), np.array([True])) is True
    assert _arrays.native_equal(np.array([1], dtype=np.uint8), np.array([1], dtype=np.int64)) is True
    assert _arrays.native_equal(torch.tensor([1], dtype=torch.uint8), torch.tensor([1], dtype=torch.int8)) is True


def test_native_equal_rejects_unsupported_forms():
    sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,), check_invariants=False)
    assert _arrays.native_equal(sparse, torch.tensor([1.0, 0.0])) is _arrays.NOT_COMPARABLE
    assert _arrays.native_equal(sparse, np.array([1.0, 0.0])) is _arrays.NOT_COMPARABLE
    strings = np.array(["a"])
    assert _arrays.native_equal(strings, strings) is _arrays.NOT_COMPARABLE


def test_native_equal_declines_tensor_subclasses():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    assert _arrays.native_equal(param, torch.tensor([1.0])) is _arrays.NOT_COMPARABLE
    assert _arrays.native_equal(param, np.array([1.0])) is _arrays.NOT_COMPARABLE


def test_native_equal_resolves_negative_bit_views():
    view = torch._neg_view(torch.tensor([1.0, -2.0]))
    assert _arrays.native_equal(view, np.array([-1.0, 2.0])) is True
    assert _arrays.native_equal(view, torch.tensor([-1.0, 2.0])) is True
