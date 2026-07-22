"""JSON file IO for config objects.

Thin wrappers over the marshal / unmarshal core: ``load_json`` reads a JSON
document into a dataclass tree, and ``dumps_json`` / ``save_json`` render a config
object back to JSON text.
"""

from __future__ import annotations

import json
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
    to_dict,
)


T = TypeVar("T")


def dumps_json(config: Any, *, indent: int = 2) -> str:
    """Render a config object as JSON text.

    Args:
        config: The config object to render.
        indent: Number of spaces per indentation level, by default 2.

    Returns:
        The JSON document, in field-declaration order, ending with a newline.
    """
    body = json.dumps(to_dict(config), indent=indent, ensure_ascii=True)
    return f"{body}\n"


def save_json(config: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write the resolved config to a JSON file, replacing the target atomically.

    The serialized content comes from the in-memory object, so the file records
    the value of every field as the caller holds it, including any change made
    programmatically after loading.

    The write goes to a ``.tmp`` sibling and is then renamed onto the target, so
    a reader observes either the previous file or the complete new one.

    Args:
        config: The config object to write.
        path: Destination file path. Parent directories are created as needed.
        indent: Number of spaces per indentation level, by default 2.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(dumps_json(config, indent=indent), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_json(config_cls: type[T], path: str | Path) -> T:
    """Load a JSON file into a config object.

    Args:
        config_cls: The root dataclass to build.
        path: Path to the JSON file.

    Returns:
        The constructed config object.

    Raises:
        ConfigError: When the file is unreadable, holds invalid JSON, holds a
          non-object document, or fails validation. Validation failures list every
          issue found.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc
    if data is None:
        return from_dict(config_cls, {}, context=f"config file {source}")
    if not isinstance(data, Mapping):
        raise ConfigError.single(f"expected a mapping document, got {_typename(data)}", context=f"config file {source}")
    return from_dict(config_cls, data, context=f"config file {source}")
