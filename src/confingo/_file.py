"""Format-dispatching file IO that picks JSON or YAML by the path's extension.

``from_file`` and ``to_file`` map a file extension to the matching loader:
``.json`` uses the JSON functions, and ``.yaml`` / ``.yml`` use the YAML
functions. An extension that names no supported format raises a ``ConfigError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from confingo._errors import ConfigError
from confingo._json import (
    load_json,
    save_json,
)
from confingo._yaml import (
    load_yaml,
    save_yaml,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    Reader = Callable[[type[Any], "str | Path"], Any]
    """Loader that builds a config object from a file path."""

    Writer = Callable[..., Path]
    """Writer that serializes a config object to a file path."""

T = TypeVar("T")

_FORMATS: dict[str, tuple[Reader, Writer]] = {
    ".json": (load_json, save_json),
    ".yaml": (load_yaml, save_yaml),
    ".yml": (load_yaml, save_yaml),
}
"""File extension mapped to its ``(loader, writer)`` pair; the supported set is these keys."""

_SUPPORTED = ", ".join(sorted(_FORMATS))


def _handlers_for(path: str | Path) -> tuple[Reader, Writer]:
    """Resolve the loader/writer pair for a config file from its extension.

    Args:
      path (str | Path): The config file path.

    Returns:
      tuple[Reader, Writer]: The loader and writer for the file's format.

    Raises:
      ConfigError: When the extension names no supported format.
    """
    suffix = Path(path).suffix.lower()
    handlers = _FORMATS.get(suffix)
    if handlers is None:
        if suffix == "":
            message = f"config file has no extension to select a format; expected one of {_SUPPORTED}"
        else:
            message = f"unsupported config file extension {suffix!r}; expected one of {_SUPPORTED}"
        raise ConfigError.single(message, context=f"config file {path}")
    return handlers


def from_file(config_cls: type[T], path: str | Path) -> T:
    """Load a config file, choosing the loader from the path's extension.

    A ``.json`` path loads as JSON; a ``.yaml`` or ``.yml`` path loads as YAML.

    Args:
      config_cls (type[T]): The root dataclass to build.
      path (str | Path): Path to the config file.

    Returns:
      T: The constructed config object.

    Raises:
      ConfigError: When the extension names no supported format, or the file is
        unreadable, malformed, non-mapping, or fails validation.
    """
    reader, _writer = _handlers_for(path)
    return reader(config_cls, path)


def to_file(config: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write a config file, choosing the writer from the path's extension.

    A ``.json`` path writes JSON; a ``.yaml`` or ``.yml`` path writes YAML.

    Args:
      config (Any): The config object to write.
      path (str | Path): Destination file path. Parent directories are created
        as needed.
      indent (int = 2): Number of spaces per indentation level.

    Returns:
      Path: The path written.

    Raises:
      ConfigError: When the extension names no supported format.
    """
    _reader, writer = _handlers_for(path)
    return writer(config, path, indent=indent)
