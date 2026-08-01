"""Paragraph reflow in the marimo-export cleaner."""

import importlib.util
import re
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "clean_marimo_md", Path(__file__).resolve().parent.parent / "scripts" / "clean_marimo_md.py"
)
assert _SPEC and _SPEC.loader
clean_marimo_md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clean_marimo_md)

reflow = clean_marimo_md.reflow


def test_joins_soft_wrapped_paragraphs():
    assert reflow("one two\nthree four\n\nnext para\nhere") == "one two three four\n\nnext para here"


def test_leaves_structure_alone():
    # Each of these carries block meaning that a join would change or destroy.
    for block in (
        "- item one\n- item two",
        "1. first\n2. second",
        "| a | b |\n| --- | --- |",
        "# Heading\n## Sub",
        "> quoted\n> more",
        "    indented code\n    more code",
        "<figure>\n<figcaption>",
        "[^note]: a definition",
        "$$\nx = 1\n$$",
    ):
        assert reflow(block) == block, block


def test_leaves_fenced_code_alone():
    src = "prose here\nwrapped\n\n```python\nx = 1\n\ny = 2\n```\n\nmore\nprose"
    assert reflow(src) == "prose here wrapped\n\n```python\nx = 1\n\ny = 2\n```\n\nmore prose"


def test_structural_line_breaks_the_join():
    # A list may follow a paragraph with no blank line between them.
    assert reflow("lead in\ncontinues\n- item\n- item") == "lead in continues\n- item\n- item"


def test_keeps_hard_line_breaks():
    assert reflow("line one\\\nline two") == "line one\\\nline two"


def test_reflow_preserves_content():
    src = "some prose\nwrapped oddly\n\n- a list\n\n| t | b |\n\ntrailing\ntext\n"
    squash = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    assert squash(reflow(src)) == squash(src)
