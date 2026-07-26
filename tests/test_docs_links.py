"""Tests that every local Markdown link in the repository resolves.

Documentation is split across two routes through one set of pages, so a heading
rename in a reference page can silently orphan a link from an onboarding page.
This walks every Markdown file, resolves each local target as a path, and checks
each fragment against the headings the target file actually declares.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_ROOTS = (REPO_ROOT, REPO_ROOT / "docs")
"""Directories whose Markdown files carry documentation links."""

_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
"""An inline Markdown link, capturing its target and skipping image embeds.

Inline links are the form this repository uses. A reference-style or raw HTML
link would sit outside this audit, so a new one needs a matching pattern here.
"""

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
"""An ATX heading, capturing its level and text."""

_FENCE = re.compile(r"^(?P<mark>```|~~~).*?^(?P=mark)", re.DOTALL | re.MULTILINE)
"""A fenced code block, whose closing marker matches the character it opened with."""

_EMPHASIS_UNDERSCORE = re.compile(r"(?<![\w`])_+(\S(?:.*?\S)?)_+(?![\w`])")
"""Underscores wrapping a run of text, which are emphasis markup around their content.

An underscore inside a word (``snake_case``) or standing alone is literal text a
viewer keeps in the anchor, so only the wrapping pair is removed.
"""

_EXTERNAL = ("http://", "https://", "mailto:")


def _markdown_files() -> list[Path]:
    """Collect every Markdown file the documentation routes cover.

    Returns:
      list[Path]: Absolute paths, in sorted order for a stable report.
    """
    found: set[Path] = set()
    for root in MARKDOWN_ROOTS:
        found.update(path for path in root.glob("*.md") if path.is_file())
    return sorted(found)


def _slug(heading: str) -> str:
    """Render a heading the way a Markdown viewer builds its fragment.

    Follows GitHub's rule: markup is stripped while its text is kept, the result
    is lowercased, anything outside letters, digits, spaces, hyphens, and
    underscores is dropped, and spaces become hyphens. Emphasis markers are
    markup, so ``_Helpful_`` anchors as ``helpful``; an underscore inside a word
    or inside code is literal text, so ``snake_case`` keeps it.

    Args:
      heading (str): The heading text, without its leading hashes.

    Returns:
      str: The base fragment for that heading, before any duplicate suffix.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|\*|~~)", "", text)
    text = _EMPHASIS_UNDERSCORE.sub(r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def _anchors(path: Path) -> set[str]:
    """Collect the fragments a Markdown file's headings define.

    Args:
      path (Path): The Markdown file to read.

    Returns:
      set[str]: Every fragment the file's headings define, including the ``-1``,
        ``-2``, ... suffixes a viewer appends when one slug repeats.
    """
    text = _FENCE.sub("", path.read_text())
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for match in _HEADING.finditer(text):
        base = _slug(match.group(2))
        count = seen.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        seen[base] = count + 1
    return anchors


def _links(path: Path) -> list[str]:
    """Collect the link targets a Markdown file declares outside code fences.

    Args:
      path (Path): The Markdown file to read.

    Returns:
      list[str]: One target per inline link, in document order.
    """
    return _LINK.findall(_FENCE.sub("", path.read_text()))


def test_every_local_markdown_target_exists():
    broken: list[str] = []
    for path in _markdown_files():
        for target in _links(path):
            if target.startswith(_EXTERNAL) or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if (path.parent / relative).exists():
                continue
            broken.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert broken == []


def test_every_markdown_fragment_names_a_heading():
    broken: list[str] = []
    for path in _markdown_files():
        for target in _links(path):
            if target.startswith(_EXTERNAL) or "#" not in target:
                continue
            relative, fragment = target.split("#", 1)
            destination = path if relative == "" else path.parent / relative
            if not destination.is_file() or destination.suffix != ".md":
                continue
            if fragment not in _anchors(destination):
                broken.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert broken == []
