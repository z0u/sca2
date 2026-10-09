#!/usr/bin/env python
"""Render one report to the files you name: ``.md``, ``.html``, or ``.pdf``, by extension (``./go render``).

Without a PDF this is the weave alone (:mod:`mini.lit`), with the figures under ``_assets/`` beside each output. A PDF is the print a reviewer reads on paper or e-ink, so it is made the way the site makes one, from the report's export bundle (``scripts/export_reports.py``, which adds the provenance footer), with author links resolved to the published site (:func:`build_site.printable`). It also names the commit it was printed from on the edge of its first page, and with ``--since REF`` it bars its margin beside every line changed since REF (:mod:`mini.review_marks`), against a baseline exported from a checkout of REF (``scripts/review_base.py``). The baseline goes through the same exporter, so the two pages differ only where the report does.

Only the named report is exported and printed, and nothing is memoized: a print takes a few seconds, and its stamp changes with every commit anyway.

With ``--cached``, the Markdown comes from a cache shared by every container of the project (:mod:`render_cache`), so a report is woven once per version however many sessions read it.

A Markdown page under ``docs/`` (a design doc, a proposal) renders the same way, through the same page shell and stylesheets. It has no export bundle, so its PDF is printed from the weave (:func:`mini.lit.render`), and its ``--since`` baseline is its old text woven by today's code (:func:`review_base.render_md_at`).
"""

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from build_site import MD_PROSE_CSS, REPORT_CSS, Bundle, LinkResolver, printable
from export_reports import export_one
from render_cache import copy_out, lookup, store, version
from review_base import export_at, render_md_at, resolve

from mini.lit import render
from mini.lit.render import write_outputs
from mini.report_print import print_bundle
from mini.reports import MD_LEAF, export_key, mark_verdicts
from mini.review_marks import baseline_dir, mark_changes, stamp
from mini.store import active_profile

ROOT = Path(__file__).parent.parent.resolve()
DOCS = ROOT / "docs"
FORMATS = (".md", ".html", ".pdf")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


@dataclass(frozen=True)
class Review:
    """What a print for review on paper is marked with, rounds apart (:mod:`mini.review_marks`).

    Every such print names the commit it is of on the edge of its first page, so the next round can be marked against it: with ``since`` (a ref the reader last reviewed), a report that has a baseline there prints with its changes barred in the margin.
    """

    since: str | None  # full commit id
    printed: str  # what the print is of, for the margin note: a short commit id, and whether the tree has edits past it

    @classmethod
    def at(cls, since: str | None) -> "Review":
        head = _git("rev-parse", "--short", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "docs", "src")
        return cls(since, f"{head} + uncommitted edits" if dirty else head)

    def mark(self, page: str, key: str, *, root: Path) -> str:
        """*page* with its change bars, or with the note alone when there is no ``since`` or report *key* has no baseline at it (``review_base.py`` says when it has none).

        *root* is where the page's figures resolve; a text-only report has none, and its page is marked all the same.
        """
        since = self.since
        unmarked = stamp(page, note=f"Printed from {self.printed}")
        if since is None:
            return unmarked
        base = baseline_dir(since, key)
        if not (base / "index.html").is_file():
            return unmarked
        base_html = mark_verdicts((base / "index.html").read_text("utf-8"))
        note = f"Bars mark changes since {since[:7]} · printed from {self.printed}"
        return mark_changes(page, base_html, note=note, root=root, base_root=base)


def review_page(report: Path, bundle: Path, review: Review, links: LinkResolver) -> str:
    """The page a review print is made from: the exported page as the site prints it, then stamped (and marked) by *review*."""
    from_dir = report.parent.relative_to(DOCS).as_posix()
    from_dir = "" if from_dir == "." else from_dir
    key = export_key(report)
    report_css = REPORT_CSS.read_text("utf-8") if REPORT_CSS.exists() else ""
    if report.suffix == ".md":  # its prose prints the way the site shows it, on top of the report frame
        report_css += "\n" + MD_PROSE_CSS.read_text("utf-8")
    html = mark_verdicts((bundle / "index.html").read_text("utf-8"))
    page = printable(Bundle(html), links, from_dir=from_dir, key=key, report_css=report_css)
    return review.mark(page, key, root=bundle)


def write_from_bundle(bundle: Path, page: str, outs: list[Path]) -> None:
    """Write each of *outs* from the export *bundle*: the Markdown and the page as exported, the PDF printed from *page*; ``_assets/`` is copied beside a Markdown or HTML output outside the bundle (a PDF is printed from the bundle's own copy, so it needs none)."""
    assets = bundle / "_assets"
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix in (".md", ".html") and out.parent != bundle and assets.is_dir():
            shutil.copytree(assets, out.parent / "_assets", dirs_exist_ok=True)
        match out.suffix:
            case ".md" if out != bundle / MD_LEAF:
                shutil.copyfile(bundle / MD_LEAF, out)
            case ".html" if out != bundle / "index.html":
                shutil.copyfile(bundle / "index.html", out)
            case ".pdf":
                # Served from a throwaway root beside the bundle, holding the page and its _assets/.
                if print_bundle(bundle, out, html=page) is None:
                    sys.exit(f"{out}: no PDF printed (see the log above)")
        print(out)


def weave(report: Path, outs: list[Path]) -> None:
    """Weave *report* to each of *outs* (Markdown or HTML), exiting on a cell that raised."""
    rendered = render(report, out_dir=outs[0].parent, write=False)
    if errors := rendered.woven.errors:
        for o in errors:
            print(f"error in cell at line {o.cell.line}:\n{o.error}", file=sys.stderr)
        sys.exit(1)
    write_outputs(rendered, outs)


def cached(report: Path, outs: list[Path]) -> None:
    """Print the path of *report*'s Markdown in the shared cache (:mod:`render_cache`), weaving it on a miss, and copy it to each of *outs*.

    With edits in the working tree there is no version to look up, so the render goes to *outs*, or to ``.mini/lit/<key>/index.md`` when there are none, and that path is printed.
    """
    key = export_key(report)
    local = outs or [ROOT / ".mini" / "lit" / key / MD_LEAF]
    ver = version(report, active_profile())
    if ver is None:
        print("render cache: skipped, the working tree has edits under docs/ or src/", file=sys.stderr)
        weave(report, local)
        print(*local, sep="\n")
        return
    if (hit := lookup(key, ver)) is None:
        weave(report, local)
        hit = store(local[0], key, ver, report)
        print(f"render cache: stored {key} at {ver}", file=sys.stderr)
    for out in outs:
        copy_out(hit, out)
    print(hit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "report",
        type=Path,
        help="a report script or a Markdown page, e.g. docs/m2/ex-2.2.15/report.py or docs/m2/d2.2/design.md",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="an output file: .md, .html, or .pdf; repeat for several, all from one weave",
    )
    ap.add_argument(
        "--cached",
        action="store_true",
        help="read the Markdown from the shared render cache, weaving it into the cache on a miss, and print its path"
        " (with -o, copy it there too); a working tree with edits under docs/ or src/ renders without the cache",
    )
    ap.add_argument(
        "--since",
        metavar="REF",
        help="bar the PDF's margin beside every line changed since REF (the commit the reader last reviewed)",
    )
    args = ap.parse_args()
    report = Path(args.report).resolve()
    outs = [Path(o).resolve() for o in args.out]
    if not outs and not args.cached:
        ap.error("name one or more -o files, or pass --cached")
    if args.cached and (bad := [o for o in outs if o.suffix != ".md"]):
        ap.error(f"-o {bad[0]}: --cached holds Markdown only")
    if bad := [o for o in outs if o.suffix not in FORMATS]:
        ap.error(f"-o {bad[0]}: unknown format {bad[0].suffix or '(no extension)'!r}; use one of {', '.join(FORMATS)}")
    if args.since and not any(o.suffix == ".pdf" for o in outs):
        ap.error("--since marks a PDF; add an -o FILE.pdf")

    if args.cached:
        cached(report, outs)
        return

    if not any(o.suffix == ".pdf" for o in outs):
        weave(report, outs)
        print(*outs, sep="\n")
        return

    since = resolve(args.since) if args.since else None  # before the export, so a bad ref fails fast
    if report.suffix == ".md":
        bundle = render(report).out_dir  # index.html and index.md, as an export bundle has them
        if since:
            render_md_at(since, report)
    else:
        # Exits on a cell that raised. No thumbnails: only the site index reads them.
        bundle = export_one(report, thumbs=False)
        if since:
            export_at(since, [report])
    page = review_page(report, bundle, Review.at(since), LinkResolver.discover())
    write_from_bundle(bundle, page, outs)


if __name__ == "__main__":
    main()
