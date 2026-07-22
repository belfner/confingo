"""Shared file IO helpers for the config format loaders.

The JSON and YAML loaders differ only in how they parse and serialize a document.
Everything around that step -- reading the file, treating a ``null`` document as
defaults, rejecting a non-mapping document, and writing atomically -- lives here
so both formats behave identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

from confingo._core import (
    ConfigError,
    _typename,
    from_dict,
)


T = TypeVar("T")


def read_source_text(path: str | Path) -> tuple[Path, str]:
    """Read a config file's text, reporting a read failure as a config error.

    Args:
        path: Path to the config file.

    Returns:
        The resolved path and the file's decoded text.

    Raises:
        ConfigError: When the file cannot be read.
    """
    source = Path(path)
    try:
        return source, source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc


def build_from_document(config_cls: type[T], data: Any, source: Path) -> T:
    """Build a config object from an already-parsed document.

    A ``null`` document builds the config from its defaults; a document that is
    anything other than a mapping is rejected.

    Args:
        config_cls: The root dataclass to build.
        data: The parsed document.
        source: The file the document came from, used for the error context.

    Returns:
        The constructed config object.

    Raises:
        ConfigError: When the document is not a mapping, or validation fails. A
          validation failure lists every issue found.
    """
    context = f"config file {source}"
    if data is None:
        return from_dict(config_cls, {}, context=context)
    if not isinstance(data, Mapping):
        raise ConfigError.single(f"expected a mapping document, got {_typename(data)}", context=context)
    return from_dict(config_cls, data, context=context)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text to a file, replacing the target atomically.

    The text goes to a ``.tmp`` sibling and is then renamed onto the target, so a
    reader observes either the previous file or the complete new one. Parent
    directories are created as needed.

    Args:
        path: Destination file path.
        text: The full file contents to write.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination
