"""JSON file IO for config objects.

Thin wrappers over the marshal / unmarshal core: ``load_json`` reads a JSON
document into a dataclass tree, and ``dumps_json`` / ``save_json`` render a config
object back to JSON text.
"""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from confingo._errors import ConfigError
from confingo._fileio import (
    atomic_write_text,
    build_from_document,
    read_source_text,
)
from confingo._serialize import to_dict


if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


def dumps_json(config: Any, *, indent: int = 2) -> str:
    """Render a config object as JSON text.

    Args:
      config (Any): The config object to render.
      indent (int = 2): Number of spaces per indentation level.

    Returns:
      str: The JSON document, in field-declaration order, ending with a newline.
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
      config (Any): The config object to write.
      path (str | Path): Destination file path. Parent directories are created as
        needed.
      indent (int = 2): Number of spaces per indentation level.

    Returns:
      Path: The path written.
    """
    return atomic_write_text(path, dumps_json(config, indent=indent))


def load_json(config_cls: type[T], path: str | Path) -> T:
    """Load a JSON file into a config object.

    Args:
      config_cls (type[T]): The root dataclass to build.
      path (str | Path): Path to the JSON file.

    Returns:
      T: The constructed config object.

    Raises:
      ConfigError: When the file is unreadable, holds invalid JSON, holds a
        non-object document, or fails validation. Validation failures list every
        issue found.
    """
    source, text = read_source_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc
    return build_from_document(config_cls, data, source)
