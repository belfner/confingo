"""Config-build issue reporting shared across the marshal / unmarshal engine.

``ConfigIssue`` is one problem tagged with a dotted path; ``ConfigError`` carries
every issue found in one build; ``_IssueCollector`` accumulates them so a single
walk reports all problems at once. ``_reject`` bundles the record-and-fail step
that ends a failed coercion branch, returning the ``_UNSET`` sentinel the
construction and serialization walks propagate.
"""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003 (runtime: keeps get_type_hints resolvable)
from dataclasses import dataclass
from typing import Any

from confingo._backend import (
    BackendSnapshot,
    capture_backend_snapshot,
)


_UNSET = object()
"""Sentinel returned by coercion helpers when a value failed to convert."""


@dataclass(frozen=True)
class ConfigIssue:
    """A single problem found while building a config.

    Args:
      path (str): Dotted path to the offending value, such as ``training.trainer.lr``.
        The entry object of the current operation carries an empty path, rendered as ``<root>``.
      message (str): Human-readable description of the problem.
    """

    path: str
    message: str

    def __str__(self) -> str:
        label = "<root>" if self.path == "" else self.path
        return f"{label}: {self.message}"


class ConfigError(ValueError):
    """Raised when a config fails to build, carrying every issue found.

    Args:
      issues (Iterable[ConfigIssue]): The problems collected during the build.
      context (str): Description of the config source, used in the summary line.

    Attributes:
      issues (tuple[ConfigIssue, ...]): The problems collected during the build, in discovery order.
      context (str): Description of the config source.
    """

    def __init__(self, issues: Iterable[ConfigIssue], *, context: str) -> None:
        self.issues = tuple(issues)
        self.context = context
        count = len(self.issues)
        noun = "issue" if count == 1 else "issues"
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"{context} has {count} {noun}:\n{detail}")

    @classmethod
    def single(cls, message: str, *, context: str, path: str = "") -> ConfigError:
        """Build an error carrying exactly one issue.

        Args:
          message (str): Human-readable description of the problem.
          context (str): Description of the config source, used in the summary line.
          path (str = ""): Dotted path to the offending value, by default the root.

        Returns:
          ConfigError: An error whose ``issues`` holds the single problem.
        """
        return cls([ConfigIssue(path=path, message=message)], context=context)


class _IssueCollector:
    """Accumulates config issues so one build reports all of them at once.

    Also carries the array-backend snapshot for the operation, captured once and
    shared across the whole walk so array handling is gated on one consistent
    view of which backends are loaded.
    """

    def __init__(self, backend: BackendSnapshot | None = None) -> None:
        self.issues: list[ConfigIssue] = []
        self.backend: BackendSnapshot = backend if backend is not None else capture_backend_snapshot()

    def add(self, path: str, message: str) -> None:
        """Record one issue.

        Args:
          path (str): Dotted path to the offending value.
          message (str): Human-readable description of the problem.
        """
        self.issues.append(ConfigIssue(path=path, message=message))

    def extend(self, issues: Iterable[ConfigIssue]) -> None:
        """Record issues a trial collector gathered, keeping their order and paths.

        Args:
          issues (Iterable[ConfigIssue]): The issues to adopt, already carrying the
            paths they were reported under.
        """
        self.issues.extend(issues)

    def clean(self) -> bool:
        """Report whether the build has stayed issue-free so far.

        Returns:
          bool: True while no issue has been recorded.
        """
        return len(self.issues) == 0


def _reject(collector: _IssueCollector, path: str, message: str) -> Any:
    """Record one issue and return the coercion-failure sentinel.

    Bundles the two steps that end every failed coercion branch so a caller
    writes ``return _reject(collector, path, message)`` on one line.

    Args:
      collector (_IssueCollector): Destination for the issue.
      path (str): Dotted path to the offending value.
      message (str): Human-readable description of the problem.

    Returns:
      Any: The ``_UNSET`` sentinel.
    """
    collector.add(path, message)
    return _UNSET
