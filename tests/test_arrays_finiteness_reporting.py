"""Finiteness reporting over arrays with backend-native index coordinates.

Covers the array non-finite reporter preserving backend-native index
coordinates for element access, so an ``np.ndarray`` subclass with a custom
``__getitem__`` still yields the collected finiteness issue.
"""

from __future__ import annotations

import pytest

from confingo import _arrays


def test_numpy_subclass_nonfinite_reports_issue() -> None:
    """A supported ndarray subclass reports its non-finite element, not a raw error."""
    np = pytest.importorskip("numpy")

    class StrictIndex(np.ndarray):
        """ndarray subclass whose __getitem__ rejects plain Python-int coordinates."""

        def __getitem__(self, key: object) -> object:
            if isinstance(key, tuple) and any(type(part) is int for part in key):
                raise TypeError("python-int index rejected")
            return super().__getitem__(key)

    array = np.array([1.0, float("nan"), 3.0]).view(StrictIndex)
    issues: list[tuple[str, str]] = []
    result = _arrays.validate_array_value(array, "w", lambda path, message: issues.append((path, message)))

    assert result is _arrays.FAILED
    assert issues == [("w.1", "expected a finite float, got nan")]


def test_numpy_subclass_marshal_reports_issue() -> None:
    """The marshal path reports a subclass's non-finite element with its own message."""
    np = pytest.importorskip("numpy")

    class StrictIndex(np.ndarray):
        """ndarray subclass whose __getitem__ rejects plain Python-int coordinates."""

        def __getitem__(self, key: object) -> object:
            if isinstance(key, tuple) and any(type(part) is int for part in key):
                raise TypeError("python-int index rejected")
            return super().__getitem__(key)

    array = np.array([[1.0, 2.0], [float("inf"), 4.0]]).view(StrictIndex)
    issues: list[tuple[str, str]] = []
    result = _arrays.array_to_plain(array, "w", lambda path, message: issues.append((path, message)))

    assert result is _arrays.FAILED
    assert issues == [("w.1.0", "cannot serialize non-finite float inf")]
