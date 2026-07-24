"""Live snapshot of the optional array backends confingo can serialize.

The marshal / unmarshal core stays stdlib-only; numpy and torch support engages
only when the application has already imported those modules. A snapshot records
which backends are present at the start of one public operation so the whole
walk sees a single, consistent answer, and a backend imported later is honored
by the next operation rather than mid-walk.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendSnapshot:
    """The array backends loaded at one moment.

    Attributes:
      numpy (Any | None): The loaded numpy module, or None when not imported.
      torch (Any | None): The loaded torch module, or None when not imported.
    """

    numpy: Any | None
    torch: Any | None

    @property
    def active(self) -> bool:
        """Whether any array backend is loaded and array handling can apply."""
        return self.numpy is not None or self.torch is not None


def capture_backend_snapshot() -> BackendSnapshot:
    """Capture the array backends currently present in ``sys.modules``.

    Reads live module state so a backend imported after an earlier operation is
    seen by the next call. It never imports a backend itself, keeping the base
    install free of numpy and torch.

    Returns:
      BackendSnapshot: The numpy and torch modules loaded at call time.
    """
    return BackendSnapshot(numpy=sys.modules.get("numpy"), torch=sys.modules.get("torch"))
