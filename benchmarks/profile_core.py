"""cProfile harness for confingo's marshal / unmarshal hot paths.

Profiles from_dict and to_dict over the stress suite and prints the top
functions by internal (tottime) and cumulative time, to locate where confingo
spends the most time. Reuses the schema and generator from stress.py.

Usage:
  uv run python benchmarks/profile_core.py [suite_size]

Default: suite_size=500.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from typing import TYPE_CHECKING

from stress import (  # pyrefly: ignore[missing-import]  # sibling benchmark module, resolved at runtime
    ExperimentConfig,
    generate_suite,
)

import confingo
import confingo.functional


if TYPE_CHECKING:
    from collections.abc import Callable


def profile(label: str, fn: Callable[[], object], top: int = 20) -> None:
    """Profile a callable and print top functions by tottime and cumtime.

    Args:
      label (str): Section heading.
      fn (Callable[[], None]): Zero-arg operation to profile.
      top (int = 20): Rows to print per ranking.
    """
    fn()  # warm caches so profile reflects steady state
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()

    print("\n" + "=" * 78)
    print(f"{label}  -- top {top} by tottime (internal time)")
    print("=" * 78)
    stats = pstats.Stats(pr)
    stats.sort_stats("tottime").print_stats(top)

    print("=" * 78)
    print(f"{label}  -- top {top} by cumtime (cumulative time)")
    print("=" * 78)
    stats.sort_stats("cumulative").print_stats(top)


def main() -> None:
    """Generate the suite and profile from_dict and to_dict."""
    suite_size = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    configs = generate_suite(suite_size)
    objs = [confingo.functional.from_dict(ExperimentConfig, c) for c in configs]
    print(f"confingo {confingo.__version__}  suite_size={suite_size:,}")

    def run_from_dict() -> None:
        for c in configs:
            confingo.functional.from_dict(ExperimentConfig, c)

    def run_to_dict() -> None:
        for o in objs:
            confingo.functional.to_dict(o)

    profile("from_dict", run_from_dict)
    profile("to_dict", run_to_dict)


if __name__ == "__main__":
    main()
