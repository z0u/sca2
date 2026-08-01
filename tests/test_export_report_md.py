"""Fence handling and the dropped-cell guard in the report exporter."""

import importlib.util
from pathlib import Path

import pytest
from marimo_md_export.models import Cell

_SPEC = importlib.util.spec_from_file_location(
    "export_report_md", Path(__file__).resolve().parent.parent / "scripts" / "export_report_md.py"
)
assert _SPEC and _SPEC.loader
export_report_md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_report_md)

convert_admonitions = export_report_md.convert_admonitions
fenced_spans = export_report_md.fenced_spans
check_sources_agree = export_report_md.check_sources_agree

# One code cell, holding a `///` admonition inside the Python source. This is
# what an interpolated `mo.md(f"...")` looks like in the markdown export: marimo
# can only unwrap literal `mo.md("...")` cells to plain markdown, so this one
# stays a fence and its source has to survive verbatim to match its output.
CELL_MD = '```python {.marimo hide_code="true"}\nmo.md(rf"""\n/// details | Aside\nBody.\n///\n""")\n```\n'


def test_admonition_outside_code_is_converted():
    assert convert_admonitions("/// tip | Title\nBody.\n///\n") == '!!! tip "Title"\n    Body.\n'


def test_admonition_inside_code_is_left_alone():
    # The bug this guards: rewriting the fence changes the cell's source hash,
    # so its rendered output no longer matches and the cell is dropped.
    assert convert_admonitions(CELL_MD) == CELL_MD


def test_admonitions_around_a_fence_are_still_converted():
    src = f"/// tip | Before\nB.\n///\n\n{CELL_MD}\n/// tip | After\nA.\n///\n"
    out = convert_admonitions(src)
    assert '!!! tip "Before"' in out
    assert '!!! tip "After"' in out
    assert CELL_MD in out


def test_fenced_spans_covers_the_whole_block():
    src = f"lead\n\n{CELL_MD}trail\n"
    ((start, end),) = fenced_spans(src)
    assert src[start:end] == CELL_MD


def test_backticks_inside_a_widened_fence_do_not_close_it():
    # marimo widens a cell's fence when the source contains backticks.
    src = '````python {.marimo}\nmo.md("```\\nx\\n```")\n````\n'
    ((start, end),) = fenced_spans(src)
    assert src[start:end] == src


def test_closing_fence_may_not_carry_an_info_string():
    src = "```\na\n```python\nb\n```\n"
    assert [src[s:e] for s, e in fenced_spans(src)] == ["```\na\n```python\nb\n```\n"]


def _cell(source: str) -> Cell:
    from marimo_md_export.parse_md import _md5

    return Cell(source=source, source_hash=_md5(source.strip()), block_text=source)


def test_check_passes_when_sources_match():
    cell = _cell("x = 1\n")
    check_sources_agree([cell], {cell.source_hash})


def test_check_raises_on_a_rewritten_fence():
    cell = _cell("x = 1\n")
    with pytest.raises(RuntimeError, match="no notebook cell has"):
        check_sources_agree([cell], {"some-other-hash"})
