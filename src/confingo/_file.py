"""Format-dispatching file IO that picks JSON or YAML by the path's extension.

``from_file`` and ``to_file`` map a file extension to the matching loader:
``.json`` uses the JSON functions, and ``.yaml`` / ``.yml`` use the YAML functions
from the ``yaml`` extra. An extension that names no supported format raises a
``ConfigError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

from confingo._core import ConfigError
from confingo._json import (
    load_json,
    save_json,
)


T = TypeVar("T")

_JSON_SUFFIXES = frozenset({".json"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_SUPPORTED = ", ".join(sorted(_JSON_SUFFIXES | _YAML_SUFFIXES))


def _format_for(path: str | Path) -> str:
    """Resolve a file's config format from its extension.

    Args:
        path: The config file path.

    Returns:
        The format name, one of ``"json"`` or ``"yaml"``.

    Raises:
        ConfigError: When the extension names no supported format.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _JSON_SUFFIXES:
        return "json"
    if suffix in _YAML_SUFFIXES:
        return "yaml"
    if suffix == "":
        message = f"config file has no extension to select a format; expected one of {_SUPPORTED}"
    else:
        message = f"unsupported config file extension {suffix!r}; expected one of {_SUPPORTED}"
    raise ConfigError.single(message, context=f"config file {path}")


def from_file(config_cls: type[T], path: str | Path) -> T:
    """Load a config file, choosing the loader from the path's extension.

    A ``.json`` path loads as JSON; a ``.yaml`` or ``.yml`` path loads as YAML,
    which requires the ``yaml`` extra.

    Args:
        config_cls: The root dataclass to build.
        path: Path to the config file.

    Returns:
        The constructed config object.

    Raises:
        ConfigError: When the extension names no supported format, or the file is
          unreadable, malformed, non-mapping, or fails validation.
    """
    if _format_for(path) == "json":
        return load_json(config_cls, path)
    from confingo._yaml import load_yaml  # noqa: PLC0415

    return load_yaml(config_cls, path)


def to_file(config: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write a config file, choosing the writer from the path's extension.

    A ``.json`` path writes JSON; a ``.yaml`` or ``.yml`` path writes YAML, which
    requires the ``yaml`` extra.

    Args:
        config: The config object to write.
        path: Destination file path. Parent directories are created as needed.
        indent: Number of spaces per indentation level, by default 2.

    Returns:
        The path written.

    Raises:
        ConfigError: When the extension names no supported format.
    """
    if _format_for(path) == "json":
        return save_json(config, path, indent=indent)
    from confingo._yaml import save_yaml  # noqa: PLC0415

    return save_yaml(config, path, indent=indent)
