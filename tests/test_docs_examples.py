from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
QUICKSTART_DIR = REPO_ROOT / "examples" / "quickstart"
QUICKSTART_FILES = ("config.py", "train.json", "run.py")

EXPECTED_RUN_ID = "344e28a35dd4"
EXPECTED_STDOUT = (
    "optimizer.name: adamw\n"
    "optimizer.lr: 0.001\n"
    "seed: 0\n"
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
