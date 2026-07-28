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


RESOURCE_ERRORS = (MemoryError, SystemError)
"""Exceptions describing the interpreter's own state rather than the config.

These reach the caller unchanged wherever confingo runs code it does not own,
since an allocation failure or an interpreter defect would be named wrongly as
invalid configuration. A ``RecursionError`` sits outside this set: walking a
supplied value reaches it through the depth of the value itself, which is a
property of the config being built.
"""


_UNSET = object()
"""Sentinel returned by coercion helpers when a value failed to convert."""


def class_label(config_cls: Any) -> str:
    """Name a class for a message without letting the reading fail again.

    Reading ``__name__`` goes through the class's metaclass, which a class may
    supply code for. A metaclass that raises there would replace the problem being
    reported with a problem of its own, so a fixed phrase answers instead and the
    original report still arrives. Every message that names a class the config
    author owns reads it through here.

    Args:
      config_cls (Any): The class to name.

    Returns:
      str: The class's name, or a fixed phrase when reading it fails.
    """
    try:
        return str.__str__(config_cls.__name__)
    except RESOURCE_ERRORS:
        raise
    except Exception:
        return "a class that could not be named"


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


PENDING_RENDER_LIMIT = 5
"""Paths named in the pending-lifecycle line before the remainder becomes a count.

The attribute carries every path it collected; this bounds only the rendered
sentence, which sits under an issue list that is already long.
"""


def _render_pending_lifecycle(paths: tuple[str, ...]) -> str:
    """Word the pending-lifecycle line for the paths a build deferred.

    Args:
      paths (tuple[str, ...]): Dotted paths carrying deferred lifecycle work, in
        discovery order. The root is the empty string.

    Returns:
      str: The rendered line, indented to sit beneath the issue list.
    """
    labels = ["<root>" if path == "" else path for path in paths]
    shown = ", ".join(labels[:PENDING_RENDER_LIMIT])
    remainder = len(labels) - PENDING_RENDER_LIMIT
    more = f" and {remainder} more" if remainder > 0 else ""
    return (
        f"  Pending lifecycle work at {shown}{more}: fix the issues above, then load the config "
        f"again to run the applicable callbacks and checks."
    )


class ConfigError(ValueError):
    """Raised when a config fails to build, carrying every issue found.

    Args:
      issues (Iterable[ConfigIssue]): The problems collected during the build.
      context (str): Description of the config source, used in the summary line.
      pending_lifecycle_paths (Iterable[str] = ()): Paths whose lifecycle work this
        attempt deferred, in discovery order.

    Attributes:
      issues (tuple[ConfigIssue, ...]): The problems collected during the build, in discovery order.
      context (str): Description of the config source.
      pending_lifecycle_paths (tuple[str, ...]): Paths where a ``__post_init__``, ``init=False``
        completeness check, or ``__validate__`` can run on a later load, in discovery order. Each
        entry names a node with lifecycle stages still ahead of it, or an authored-default subtree
        this attempt set aside, covering that path and anything beneath it. The reading is
        deliberately generous, so an entry marks work a repair can reach. The root is the empty
        string.
    """

    def __init__(
        self,
        issues: Iterable[ConfigIssue],
        *,
        context: str,
        pending_lifecycle_paths: Iterable[str] = (),
    ) -> None:
        self.issues = tuple(issues)
        self.context = context
        self.pending_lifecycle_paths = tuple(pending_lifecycle_paths)
        count = len(self.issues)
        noun = "issue" if count == 1 else "issues"
        detail = "\n".join(f"  - {issue}" for issue in self.issues)
        summary = f"{context} has {count} {noun}:\n{detail}"
        if len(self.pending_lifecycle_paths) > 0:
            summary = f"{summary}\n{_render_pending_lifecycle(self.pending_lifecycle_paths)}"
        super().__init__(summary)

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
        self.pending_lifecycle_paths: list[str] = []
        # Membership set beside the ordered list: discovery order is the reported
        # order, and a wide tree would make a scan of the list quadratic.
        self._pending_seen: set[str] = set()

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

    def add_pending_lifecycle(self, path: str) -> None:
        """Record one path whose lifecycle work this build deferred.

        Args:
          path (str): Dotted path to the node or authored subtree, the empty string
            for the root. A path already recorded keeps its first position.
        """
        if path not in self._pending_seen:
            self._pending_seen.add(path)
            self.pending_lifecycle_paths.append(path)

    def extend_pending_lifecycle(self, paths: Iterable[str]) -> None:
        """Record several deferred-lifecycle paths, keeping discovery order.

        Args:
          paths (Iterable[str]): The paths to record.
        """
        for path in paths:
            self.add_pending_lifecycle(path)

    def adopt(self, other: _IssueCollector) -> None:
        """Take over everything a trial collector gathered, issues and pending paths alike.

        A union member is probed with its own collector, so the diagnostics of the
        member the report goes on to name are adopted together through here.
        ``extend`` stays the issue-only operation, which is what the authored-default
        array path calls.

        Args:
          other (_IssueCollector): The trial collector to adopt from.
        """
        self.extend(other.issues)
        self.extend_pending_lifecycle(other.pending_lifecycle_paths)

    def clean(self) -> bool:
        """Report whether the build has stayed issue-free so far.

        Ordinary issues determine this reading. Pending lifecycle paths describe
        work a later load runs, where an issue describes the value itself, and a
        union member is selected on whether its trial stayed clean, so keeping this
        reading to issues keeps member selection with the config.

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
