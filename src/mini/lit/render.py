"""
Render a document to its outputs: the woven Markdown, the HTML page, and (on request, :func:`write_outputs`) a PDF.

Outputs land in one directory per document — ``.mini/lit/<key>/`` by default, with ``index.html``, ``index.md``, and the ``_assets/`` the figures were written to — so the same relative URLs work opened from disk, served locally, or published as a bundle through ``mini.reports``. The CLI (``python -m mini.lit render``) writes to the files it is given instead, with ``_assets/`` beside them.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
import time
from dataclasses import dataclass
from pathlib import Path

from mini.lit.document import Document, Runner, Woven, parse
from mini.lit.notes import load_glossary
from mini.lit.page import page, to_html
from mini.reports import (
    Publisher,
    export_key,
    link_externalized,
    mark_verdicts,
    report_glossary,
    report_styles,
    set_report_styles,
)
from mini.runs import data_root

__all__ = ["render", "compose", "Rendered", "write_outputs", "output_dir"]


def output_dir(doc: Path, *, live: bool = False) -> Path:
    """``.mini/lit/<key>/``, keyed the way the report's export bundle is (:func:`mini.reports.export_key`).

    The live server writes to ``.mini/lit-live/<key>/`` instead: its page carries a reload script and versioned asset URLs, so it is a different artifact from a render, and keeping the trees apart means a ``render`` while the server is up never overwrites the page a browser is watching.
    """
    return data_root() / ("lit-live" if live else "lit") / export_key(doc)


@dataclass
class Rendered:
    doc: Document
    woven: Woven
    html: str
    out_dir: Path
    runner: Runner
    seconds: float  # markdown → html

    def write(self, *, markdown: bool = True) -> None:
        """Write ``index.html`` (and ``index.md``) under :attr:`out_dir`.

        The Markdown is for reading as text, so a figure inlined as SVG (:func:`mini.vis.svg_figure`) is a link to its sidecar there (:func:`mini.reports.link_externalized`); the page keeps the inline copy, which its CSS themes.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write(self.out_dir / "index.html", self.html)
        if markdown:
            _write(self.out_dir / "index.md", link_externalized(self.woven.markdown))


def compose(woven: Woven, *, extra_body: str = "") -> tuple[str, float]:
    """The woven Markdown as a complete page, with how long that took.

    Terms the project's shared glossary defines (``docs/glossary.md``, :func:`mini.reports.report_glossary`) are annotated as the document's own are. The project's shared report stylesheet (``docs/report.css``, :func:`mini.reports.report_styles`) goes in last, so a render, the live server and the export all show it — the site build re-inlines the current source on top. Each hypothesis heading is badged with its section's verdict (:func:`mini.reports.mark_verdicts`), which the build also does, for pages exported before the badge.
    """
    t0 = time.perf_counter()
    body = to_html(woven.markdown, glossary=load_glossary(report_glossary(woven.doc.path)))
    html = mark_verdicts(page(body, title=woven.doc.title, extra_body=extra_body))
    html = set_report_styles(html, report_styles(woven.doc.path))
    return html, time.perf_counter() - t0


def render(
    doc: Path | str,
    *,
    out_dir: Path | None = None,
    runner: Runner | None = None,
    live: bool = False,
    extra_body: str = "",
    write: bool = True,
    partial: Callable[[Woven], None] | None = None,
) -> Rendered:
    """Weave *doc* and (with *write*) put ``index.html`` and ``index.md`` under *out_dir*.

    *live* is the interactive setting: asset URLs carry a content stamp so a browser shows a re-drawn figure, and a re-drawn figure may replace one of the same name. Pass the previous call's *runner* to re-run only the cells that changed, and *partial* to be handed the document as it stands before each cell that runs (see :meth:`Runner.weave`).
    """
    path = Path(doc).resolve()
    out = (out_dir or output_dir(path, live=live)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    publish = Publisher(asset_dir=out / "_assets", link="_assets", strict=not live, versioned=live)
    if runner is None:
        runner = Runner(path, publish=publish)
    else:
        runner.publish = publish
    parsed = parse(path)
    woven = runner.weave(parsed, partial=partial)
    html, seconds = compose(woven, extra_body=extra_body)
    rendered = Rendered(parsed, woven, html, out, runner, seconds)
    if write:
        rendered.write()
    return rendered


def write_outputs(rendered: Rendered, outs: list[Path]) -> list[Path]:
    """Write *rendered* to each of *outs*, in the format its suffix names: ``.md``, ``.html``, or ``.pdf``.

    The figures are wherever the weave put them (``_assets/`` under :attr:`Rendered.out_dir`), and the Markdown and HTML link them relatively, so one of those in another directory gets a copy of ``_assets/`` beside it. The PDF is printed by :func:`mini.report_print.print_bundle` from the weave's own ``_assets/``, the print the site build makes, so it matches the one a reviewer reads and needs no copy.
    """
    from mini.report_print import print_bundle

    assets = rendered.out_dir / "_assets"
    written = []
    for out in outs:
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix in (".md", ".html") and out.parent != rendered.out_dir and assets.is_dir():
            shutil.copytree(assets, out.parent / "_assets", dirs_exist_ok=True)
        match out.suffix:
            case ".md":
                _write(out, link_externalized(rendered.woven.markdown))
            case ".html":
                _write(out, rendered.html)
            case ".pdf":
                # Served from a throwaway root beside the weave, holding the page and its _assets/.
                # The "bundle" is any path in the weave dir: with html= given, only its parent
                # (where _assets/ is) matters, and the serve root lands beside it.
                if print_bundle(rendered.out_dir / out.name, out, html=rendered.html) is None:
                    raise RuntimeError(f"{out}: no PDF printed (see the log above)")
            case _:
                raise ValueError(f"{out}: unknown format {out.suffix!r}; use .md, .html, or .pdf")
        written.append(out)
    return written


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")  # per process, so two writers never share a temp file
    tmp.write_text(text)
    tmp.replace(path)
