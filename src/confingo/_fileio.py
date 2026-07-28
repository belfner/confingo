"""Shared file IO helpers for the config format loaders.

The JSON and YAML loaders differ only in how they parse and serialize a document.
Everything around that step -- reading the file, treating a ``null`` document as
defaults, rejecting a non-mapping document, and writing atomically -- lives here
so both formats behave identically.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from confingo._core import from_dict
from confingo._errors import ConfigError
from confingo._schema import _typename


def read_source_text(path: str | Path) -> tuple[Path, str]:
    """Read a config file's text, reporting a read failure as a config error.

    Args:
      path (str | Path): Path to the config file.

    Returns:
      tuple[Path, str]: The resolved path and the file's decoded text.

    Raises:
      ConfigError: When the file cannot be read.
    """
    source = Path(path)
    try:
        return source, source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc


def build_from_document[T](config_cls: type[T], data: Any, source: Path) -> T:
    """Build a config object from an already-parsed document.

    A ``null`` document builds the config from its defaults; a document that is
    anything other than a mapping is rejected.

    Args:
      config_cls (type[T]): The entry dataclass to build.
      data (Any): The parsed document.
      source (Path): The file the document came from, used for the error context.

    Returns:
      T: The constructed config object.

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

    The text goes to a uniquely named temporary file in the destination directory
    and is then renamed onto the target, so a reader observes either the previous
    file or the complete new one, and concurrent writers never share a temporary.
    Parent directories are created as needed.

    The contents reach the disk before the rename, so a file a reader opens is the
    whole file it was written as. The directory entry is flushed after the rename
    where the platform offers it, which is what carries a completed save across a
    power loss; a platform that declines the flush has still written the file, so
    the save succeeds and its durability is the filesystem's own.

    Args:
      path (str | Path): Destination file path.
      text (str): The full file contents to write.

    Returns:
      Path: The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Create the temporary with O_EXCL so the kernel applies the current umask
    # atomically for a new file (matching an ordinary create), with no process-wide
    # umask toggle. A random name keeps concurrent writers from sharing an inode.
    while True:
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
        try:
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Preserve an existing target's mode; a new file keeps the umask-applied
        # mode. One stat answers both whether the target exists and what mode it
        # carries, so a concurrent unlink between two reads cannot turn a
        # successful save into a failure.
        with contextlib.suppress(OSError):
            temporary.chmod(destination.stat().st_mode & 0o777)
        temporary.replace(destination)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    _sync_directory(destination.parent)
    return destination


def _sync_directory(directory: Path) -> None:
    """Flush a directory entry so a completed rename survives a power loss.

    Every step is best-effort: a filesystem that declines to open a directory, or
    to sync one, has written the file either way, and a save that succeeded is not
    turned into a failure by a durability step the platform does not offer.

    Args:
      directory (Path): The directory holding the renamed file.
    """
    with contextlib.suppress(OSError, AttributeError):
        descriptor = os.open(directory, getattr(os, "O_DIRECTORY", os.O_RDONLY))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
