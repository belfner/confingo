from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
QUICKSTART_DIR = REPO_ROOT / "examples" / "quickstart"
QUICKSTART_FILES = ("config.py", "train.json", "run.py")

SHOWCASE_DIR = REPO_ROOT / "examples" / "showcase"
SHOWCASE_FILES = ("schema.py", "boundary.py", "showcase.yaml", "run.py")

# The showcase prints one line per feature, so the lines pinned here are the ones
# that carry a rule rather than a value: each declared limit at the last value it
# carries and the first it reports, the issue count of one collect-all pass, and
# the fingerprint, which holds the whole tree's round trip in one string.
SHOWCASE_EXPECTED_LINES = (
    "  63 levels   carried",
    "  1,000,000  carried, shape (1000, 1000)",
    "  length=1    5",
    "  length=64   5f05d51256bf26f8db33b1f9230db44f3dfc7f4adb1650df14dfb5dee5877c89",
    "  64 hops    carried, writes 7",
    "  shape (0, 3)     writes []",
    "  shape (2, 0)     writes [[], []]",
    "JSON equals original   True",
    "YAML equals original   True",
    "one fingerprint        True",
    "root round trip        True",
    "issues:  16",
    "Run fingerprint:   5f05d51256bf",
)

SHOWCASE_EXPECTED_FRAGMENTS = (
    "64 levels   reported at value.0.0.0",
    "1,000,001  array has 1000001 elements",
    "length=0    config_hash length must be an int from 1 to 64, got 0",
    "length=65   config_hash length must be an int from 1 to 64, got 65",
    "65 hops    rendering the plain form followed 64 arrays into one",
    # One line per rule that draws the schema boundary, so dropping a rule or
    # weakening its message fails here.
    "set[str | Path] cannot be built: a set element names one type",
    "config sections are unhashable, so frozenset[Section] cannot be built",
    "set[list[int]] cannot be built: a set element must rebuild hashable",
    "list carries no element type",
    "unsupported field type Decimal",
    "unsupported dict key type int; only str keys are supported",
    "enum SubclassValued must carry primitive values",
    "enum RedirectedMode looks a member up through RedirectingMeta",
    "Timestamp is a datetime subclass",
    "TakesTypeParameters takes the type parameter ElementP",
    "invalid authored default: expected a value already matching Path, got str",
    "sgd requires momentum",
    "epochs must be positive, got -1",
    # The three statements a review found stating something the library does not
    # do. Each is pinned so the example cannot drift back to describing it wrongly.
    "optimizer.warmup_steps (99999) exceeds schedule.total_steps (10000)",
    "The fingerprint is a SHA-256 prefix over the canonical JSON of the",
    "np.ndarray[tuple[int, int], np.dtype[np.float64]] dtype=float64",
    # One row per supported array annotation form, the shape-only tensor included.
    "Annotated[Tensor, (i, i)]",
    # The features the README claims the schema demonstrates. Each is pinned at
    # the value the library produced, so removing the feature fails the test.
    "amsgrad             True",
    "_missing_ maps a spelling the members lack",
    "weakref.ref(schedule)  True   weakref_slot=True on a slotted section",
    # Every operation confingo.functional carries, called through both surfaces
    # and compared, so dropping either form fails this test.
    "  to_dict                agree: True",
    "  config_hash            agree: True",
    "  from_dict              agree: True",
    "  validate_schema        agree: True",
    "  dumps_json             agree: True",
    "  dumps_yaml             agree: True",
    "  save_json / load_json  agree: True",
    "  save_yaml / load_yaml  agree: True",
    "  from_file / to_file    agree: True",
    "  config_equal           agree: True   free function only",
)

EXPECTED_RUN_ID = "344e28a35dd4"
EXPECTED_OPTIMIZER_ID = "be59896dec38"
EXPECTED_STDOUT = (
    "optimizer.name: adamw\n"
    "optimizer.lr: 0.001\n"
    "seed: 0\n"
    f"optimizer id: {EXPECTED_OPTIMIZER_ID}\n"
    f"run id: {EXPECTED_RUN_ID}\n"
    f"saved: runs/{EXPECTED_RUN_ID}/resolved.json\n"
)
EXPECTED_RESOLVED = {
    "optimizer": {"name": "adamw", "lr": 0.001},
    "seed": 0,
    "output_dir": "runs",
}

# Documentation files that embed canonical quickstart blocks, and the block names
# each must carry. A block is marked with `<!-- canonical:<name> -->` directly
# above its fenced code block; the parity test asserts the fenced body matches the
# matching file under examples/quickstart/ byte for byte, so the byte comparison
# keeps each documentation copy synchronized with the tested source.
DOC_CANONICAL_BLOCKS = {
    "README.md": {"config.py", "train.json"},
    "docs/getting-started.md": {"config.py", "train.json", "run.py"},
}

_CANONICAL_BLOCK = re.compile(
    r"<!--\s*canonical:(?P<name>\S+)\s*-->\s*```[^\n]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _find_canonical_blocks(doc_text: str) -> list[tuple[str, str]]:
    return [(m.group("name"), m.group("body")) for m in _CANONICAL_BLOCK.finditer(doc_text)]


def test_quickstart_example_runs(tmp_path: Path):
    for name in QUICKSTART_FILES:
        shutil.copy(QUICKSTART_DIR / name, tmp_path / name)

    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == EXPECTED_STDOUT

    resolved_path = tmp_path / "runs" / EXPECTED_RUN_ID / "resolved.json"
    assert resolved_path.is_file()
    assert json.loads(resolved_path.read_text()) == EXPECTED_RESOLVED


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("numpy", "torch")),
    reason="the showcase drives the numpy and torch adapters",
)
def test_showcase_example_runs(tmp_path: Path):
    for name in SHOWCASE_FILES:
        shutil.copy(SHOWCASE_DIR / name, tmp_path / name)

    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = result.stdout.splitlines()
    for expected in SHOWCASE_EXPECTED_LINES:
        assert expected in lines, f"{expected!r} is absent from the showcase output"
    for fragment in SHOWCASE_EXPECTED_FRAGMENTS:
        assert fragment in result.stdout, f"{fragment!r} is absent from the showcase output"


def test_docs_canonical_blocks_match_source():
    for doc_rel, expected_names in DOC_CANONICAL_BLOCKS.items():
        doc_text = (REPO_ROOT / doc_rel).read_text()
        blocks = _find_canonical_blocks(doc_text)
        names = [name for name, _ in blocks]

        for name in expected_names:
            assert names.count(name) == 1, (
                f"{doc_rel} must carry exactly one canonical block for '{name}', found {names.count(name)}"
            )

        for name, body in blocks:
            source_bytes = (QUICKSTART_DIR / name).read_bytes()
            assert (body + "\n").encode() == source_bytes, (
                f"{doc_rel} canonical block '{name}' must match examples/quickstart/{name} byte for byte"
            )


DOC_PAGES = [*sorted((REPO_ROOT / "docs").glob("*.md")), REPO_ROOT / "README.md"]

PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _python_blocks(page: Path) -> list[tuple[int, str]]:
    """Return every fenced Python block on a page with its starting line number."""
    text = page.read_text(encoding="utf-8")
    return [(text[: match.start()].count("\n") + 2, match.group(1)) for match in PYTHON_BLOCK.finditer(text)]


def test_every_documented_python_block_parses() -> None:
    """Each fenced Python block in the documentation is syntactically valid Python."""
    failures = []
    for page in DOC_PAGES:
        for line, source in _python_blocks(page):
            try:
                compile(source, f"{page.name}:{line}", "exec")
            except SyntaxError as exc:
                failures.append(f"{page.name}:{line}: {exc}")
    assert failures == [], failures


def test_every_documented_confingo_import_resolves() -> None:
    """Each name a documented block imports from confingo is reachable from the module it names."""
    import importlib  # noqa: PLC0415

    failures = []
    for page in DOC_PAGES:
        for line, source in _python_blocks(page):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if node.module != "confingo" and not node.module.startswith("confingo."):
                    continue
                module = importlib.import_module(node.module)
                failures.extend(
                    f"{page.name}:{line}: {node.module} has no {alias.name}"
                    for alias in node.names
                    if not hasattr(module, alias.name)
                )
    assert failures == [], failures
