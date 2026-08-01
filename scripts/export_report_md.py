#!/usr/bin/env python
"""Export a Marimo notebook to Markdown with its outputs, then clean it up.

This replaces the ``marimo-md-export`` console script. It reuses that
package's library — which does the hard part: running both the Markdown and
HTML exports, scraping rendered outputs out of the HTML session JSON, and
injecting them back into the Markdown — but fixes two things in how its CLI
drives that library:

**Admonitions were rewritten inside code fences.** ``marimo-md-export``
applies its ``/// type | Title`` → ``!!! type "Title"`` conversion to the
whole Markdown document before collecting cells. Cells are matched to their
rendered output by MD5 of the cell source, so rewriting a ``///`` block that
lives *inside* a cell's Python source changes that cell's hash, and the
output no longer matches. A cell whose code is hidden and whose output does
not match is deleted outright, so the cell silently disappears from the
export. This hits any ``mo.md(f"...")`` cell containing an admonition:
interpolated cells stay code fences in the Markdown export, whereas literal
``mo.md("...")`` cells are unwrapped to plain Markdown and are unaffected.
:func:`convert_admonitions` here applies the same transform outside fenced
code blocks only.

**Dropped cells were unreported.** ``inject_outputs`` cannot tell a cell that
genuinely produces no output from one whose hash failed to match, so it says
nothing either way. Checking the source text instead is decisive, and catches
the whole class rather than this one instance: the HTML export lists a hash
for every cell in the notebook, so a code fence in the Markdown whose hash is
absent from that list is a fence some transform has rewritten.
:func:`check_sources_agree` raises on that rather than letting the gap reach
the published document.

Usage: ``uv run scripts/export_report_md.py <notebook.py> <out.md>``. Images
are externalized beside the output; see ``clean_marimo_md.py`` for what the
cleanup pass does and for the options this script forwards to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from marimo_md_export.export import export_html, export_md, strip_header_from_frontmatter
from marimo_md_export.inject import inject_outputs
from marimo_md_export.models import Cell
from marimo_md_export.parse_html import _extract_session_cells_raw, extract_outputs
from marimo_md_export.parse_md import collect_cells
from marimo_md_export.transform import convert_admonitions as _convert_admonitions

_SPEC = importlib.util.spec_from_file_location("clean_marimo_md", Path(__file__).with_name("clean_marimo_md.py"))
assert _SPEC and _SPEC.loader
clean_marimo_md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clean_marimo_md)

FENCE_RE = re.compile(r"(`{3,}|~{3,})")


def fenced_spans(md: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` character offsets of each fenced code block.

    Follows CommonMark's rules for the cases that come up in a Marimo export:
    a closing fence uses the same character as its opener and is at least as
    long, and carries no info string. Both matter here, because Marimo widens
    a cell's fence when the cell's own source contains backticks — the inner
    ones must not read as the close.
    """
    spans: list[tuple[int, int]] = []
    open_fence: str | None = None
    start = pos = 0

    for line in md.splitlines(keepends=True):
        stripped = line.lstrip()
        m = FENCE_RE.match(stripped)
        if m:
            token = m.group(1)
            if open_fence is None:
                open_fence, start = token, pos
            elif token[0] == open_fence[0] and len(token) >= len(open_fence) and not stripped[len(token) :].strip():
                spans.append((start, pos + len(line)))
                open_fence = None
        pos += len(line)

    if open_fence is not None:  # unterminated fence: treat the rest as code
        spans.append((start, len(md)))
    return spans


def convert_admonitions(md: str) -> str:
    """Convert ``///`` admonitions to ``!!!`` form, leaving fenced code alone."""
    out: list[str] = []
    prev = 0
    for start, end in fenced_spans(md):
        out.append(_convert_admonitions(md[prev:start]))
        out.append(md[start:end])
        prev = end
    out.append(_convert_admonitions(md[prev:]))
    return "".join(out)


def notebook_source_hashes(html: bytes) -> set[str]:
    """Return the MD5 of every cell source in the notebook, per the HTML export.

    The session JSON carries a ``code_hash`` for every cell, including the ones
    that render nothing; ``extract_outputs`` keeps only the cells that produced
    output, so it can't serve as the reference set here.
    """
    return {cell["code_hash"] for cell in json.loads(_extract_session_cells_raw(html))}


def check_sources_agree(cells: list[Cell], source_hashes: set[str]) -> None:
    """Raise if a code fence in the Markdown carries a source no notebook cell has.

    Cells are matched to their rendered output by MD5 of the source, so any
    rewriting of a fence's contents between the two exports costs that cell its
    output — and, if its code is hidden, the cell itself. Cells that Marimo
    unwrapped into plain Markdown have no fence and so aren't checked; their
    text is in the document either way.
    """
    unknown = [cell for cell in cells if cell.source_hash not in source_hashes]
    if not unknown:
        return
    heads = "\n".join(f"    {cell.source.strip().splitlines()[0][:60]}" for cell in unknown)
    raise RuntimeError(
        f"{len(unknown)} code fence(s) in the Markdown export carry a source that no "
        f"notebook cell has, so their rendered output cannot be matched and the cells "
        f"would be dropped silently. Something rewrote the fence contents. Cells:\n{heads}"
    )


def export(notebook: Path, *, sandbox: bool = False, timeout: int | None = None) -> str:
    """Export ``notebook`` to Markdown with its rendered outputs injected."""
    md = convert_admonitions(export_md(notebook, sandbox=sandbox, timeout=timeout))
    cells = collect_cells(md)
    html = export_html(notebook, sandbox=sandbox, timeout=timeout)
    check_sources_agree(cells, notebook_source_hashes(html))
    result, warnings = inject_outputs(md, cells, extract_outputs(html))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return strip_header_from_frontmatter(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-output", type=Path, help="also write the export before the cleanup pass")
    parser.add_argument("--assets-dir", type=Path, help="defaults to <output-stem>.assets/ beside the output")
    parser.add_argument("--no-reflow", action="store_true", help="keep the source's line wrapping")
    parser.add_argument("--sandbox", action="store_true", help="run marimo export in an isolated uv environment")
    parser.add_argument("--timeout", type=int, help="seconds to wait for each marimo export subprocess")
    args = parser.parse_args()

    raw = export(args.notebook, sandbox=args.sandbox, timeout=args.timeout)
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(raw)

    dst = args.output
    assets_dir = args.assets_dir or dst.with_name(f"{dst.stem}.assets")
    rel_dir = assets_dir.name if assets_dir.parent == dst.parent else str(assets_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(clean_marimo_md.clean(raw, assets_dir, rel_dir, do_reflow=not args.no_reflow))


if __name__ == "__main__":
    main()
