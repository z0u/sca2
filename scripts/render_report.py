#!/usr/bin/env python
"""Render one report to the files you name: ``.md``, ``.html``, or ``.pdf``, by extension (``./go render``).

Without a PDF this is the weave alone (:mod:`mini.lit`), with the figures under ``_assets/`` beside each output. A PDF is the print a reviewer reads on paper or e-ink, so it is made the way the site makes one, from the report's export bundle (``scripts/export_reports.py``, which adds the provenance footer), with author links resolved to the published site (:func:`build_site.printable`). It also names the commit it was printed from on the edge of its first page, and with ``--since REF`` it bars its margin beside every line changed since REF (:mod:`mini.review_marks`), against a baseline exported from a checkout of REF (``scripts/review_base.py``). The baseline goes through the same exporter, so the two pages differ only where the report does. For a Markdown page, ``--since FILE`` names an earlier copy instead, for a draft with no commit to compare against (:func:`review_base.export_file`).

Only the named report is exported and printed, and nothing is memoized: a print takes a few seconds, and its stamp changes with every commit anyway.

A Markdown page under ``docs/`` (a design doc, a proposal) renders the same way, through the same page shell and stylesheets. It has no export bundle, so its PDF is printed from the weave (:func:`mini.lit.render`), and its ``--since`` baseline is its old text woven by today's code (:func:`review_base.render_md_at`). A Markdown draft outside ``docs/`` (a write-up in a shared folder) prints the same way, with the images it links from disk copied in beside it (:func:`local_images`).
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from build_site import Bundle, LinkResolver, open_details, print_css, printable
from export_reports import export_one
from review_base import export_at, export_file, file_slot, render_md_at, resolve

from mini.lit import render
from mini.lit.render import write_outputs
from mini.report_print import print_bundle
from mini.reports import MD_LEAF, export_key, mark_verdicts
from mini.review_marks import baseline_dir, mark_changes, stamp
from mini.store import active_profile

ROOT = Path(__file__).parent.parent.resolve()
DOCS = ROOT / "docs"
FORMATS = (".md", ".html", ".pdf")
ASSETS = "_assets"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


@dataclass(frozen=True)
class Review:
    """What a print for review on paper is marked with, rounds apart (:mod:`mini.review_marks`).

    Every such print names the commit it is of on the edge of its first page, so the next round can be marked against it: with ``since`` (a ref the reader last reviewed), a report that has a baseline there prints with its changes barred in the margin.
    """

    since: str | None  # full commit id, or the slot of a baseline made from a file (review_base.file_slot)
    printed: str  # what the print is of, for the margin note: a short commit id, and whether the tree has edits past it
    since_name: str | None = None  # what the margin note calls the baseline, when it is not a commit

    @classmethod
    def at(cls, since: str | None, *, report: Path | None = None, earlier: Path | None = None) -> "Review":
        """The review of a print of *report*, marked against commit *since*, or against file *earlier* when there is one.

        A draft outside the repo is named by its file, since the commit it was printed from says nothing about it.
        """
        if report is not None and ROOT not in report.parents:
            printed = report.name
        else:
            head = _git("rev-parse", "--short", "HEAD")
            dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "docs", "src")
            printed = f"{head} + uncommitted edits" if dirty else head
        if earlier is not None:
            return cls(file_slot(earlier), printed, since_name=earlier.name)
        return cls(since, printed)

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
        note = f"Bars mark changes since {self.since_name or since[:7]} · printed from {self.printed}"
        return mark_changes(page, base_html, note=note, root=root, base_root=base)


def review_page(report: Path, bundle: Path, review: Review, links: LinkResolver) -> str:
    """The page a review print is made from: the exported page as the site prints it, then stamped (and marked) by *review*."""
    # Where author links resolve. A page outside docs/ (a plan drafted elsewhere) climbs out
    # of it, and its links to repo files resolve to their GitHub source as a report's do.
    from_dir = PurePosixPath(os.path.relpath(report.parent, DOCS)).as_posix()
    from_dir = "" if from_dir == "." else from_dir
    key = export_key(report)
    html = mark_verdicts((bundle / "index.html").read_text("utf-8"))
    if report.suffix == ".md":
        html = open_details(html)
    if report.suffix == ".md" and DOCS not in report.resolve().parents:
        html = local_images(html, page_dir=report.parent, bundle=bundle)
    page = printable(Bundle(html), links, from_dir=from_dir, key=key, report_css=print_css(report))
    return review.mark(page, key, root=bundle)


_IMG_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def local_images(html: str, *, page_dir: Path, bundle: Path) -> str:
    """*html* with each image that names a file on disk (an absolute path, or one relative to *page_dir*) copied into the *bundle*'s ``_assets/`` and linked there, so the print shows it.

    A Markdown page is printed from its weave's directory, where neither kind of path resolves; this is how a draft outside the repo (a write-up in a shared folder, with its figure beside it) prints with its figures. A page under ``docs/`` doesn't need it: its images resolve to the site as its other links do. A URL, an ``_assets/`` figure or a path to nothing is left as it is.
    """

    def bundled(m: re.Match[str]) -> str:
        src = m[2]
        if _URL_SCHEME.match(src) or src.startswith((f"{ASSETS}/", "#")):
            return m[0]
        path = page_dir / unquote(src.split("?", 1)[0])  # an absolute src replaces page_dir
        if not path.is_file():
            return m[0]
        blob = path.read_bytes()
        leaf = f"local-{hashlib.sha256(blob).hexdigest()[:12]}{path.suffix}"
        (bundle / ASSETS).mkdir(parents=True, exist_ok=True)
        (bundle / ASSETS / leaf).write_bytes(blob)
        return f"{m[1]}{ASSETS}/{leaf}{m[3]}"

    return _IMG_SRC.sub(bundled, html)


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


def export_for_review(report: Path, since: str | None, earlier: Path | None) -> Path:
    """Export *report* for a print, and its baseline at commit *since* or from file *earlier*, if either is given. Returns the report's bundle."""
    if report.suffix == ".md":
        bundle = render(report).out_dir  # index.html and index.md, as an export bundle has them
        if since:
            render_md_at(since, report)
    else:
        # Exits on a cell that raised. No thumbnails: only the site index reads them.
        bundle = export_one(report, thumbs=False)
        if since:
            export_at(since, [report])
    if earlier is not None:
        base = export_file(earlier, report)
        if DOCS not in report.parents:
            index = base / "index.html"
            index.write_text(local_images(index.read_text("utf-8"), page_dir=earlier.parent, bundle=base), "utf-8")
    return bundle


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
        required=True,
        metavar="FILE",
        help="an output file: .md, .html, or .pdf; repeat for several, all from one weave",
    )
    ap.add_argument(
        "--since",
        metavar="REF|FILE",
        help="bar the PDF's margin beside every line changed since REF (the commit the reader last reviewed), "
        "or, for a Markdown page, since FILE, an earlier copy of it (a draft with no commit to name)",
    )
    args = ap.parse_args()
    report = Path(args.report).resolve()
    outs = [Path(o).resolve() for o in args.out]
    if bad := [o for o in outs if o.suffix not in FORMATS]:
        ap.error(f"-o {bad[0]}: unknown format {bad[0].suffix or '(no extension)'!r}; use one of {', '.join(FORMATS)}")
    if args.since and not any(o.suffix == ".pdf" for o in outs):
        ap.error("--since marks a PDF; add an -o FILE.pdf")

    if not any(o.suffix == ".pdf" for o in outs):
        rendered = render(report, out_dir=outs[0].parent, write=False)
        if errors := rendered.woven.errors:
            for o in errors:
                print(f"error in cell at line {o.cell.line}:\n{o.error}", file=sys.stderr)
            sys.exit(1)
        for out in write_outputs(rendered, outs):
            print(out)
        return

    # A file of that name is an earlier copy; anything else is a ref, resolved before the export so a bad one fails fast.
    earlier = Path(args.since).resolve() if args.since and Path(args.since).is_file() else None
    since = resolve(args.since) if args.since and earlier is None else None
    if earlier is not None and (report.suffix != ".md" or earlier.suffix != ".md"):
        ap.error(
            f"--since {args.since}: only a Markdown page is marked against an earlier copy; "
            "a report is edited in place, so name the commit last reviewed (HEAD for uncommitted edits)"
        )
    bundle = export_for_review(report, since, earlier)
    page = review_page(report, bundle, Review.at(since, report=report, earlier=earlier), LinkResolver.discover())
    write_from_bundle(bundle, page, outs)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # A failed render under a storage profile has usually missed results that are only on production.
        if e.code not in (0, None, 2) and (profile := active_profile()):
            print(
                f"note: rendered under MINI_PROFILE={profile}; a published report reads production, so render it without MINI_PROFILE",
                file=sys.stderr,
            )
        raise
