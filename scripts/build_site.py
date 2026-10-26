#!/usr/bin/env python
"""Build the static site from the project's reports.

The HTML lives nowhere in Git: each report is exported (``./go publish``) to a self-contained bundle — ``index.html`` + named-keyed ``_assets/`` — and mirrored to the bucket under ``exports/<key>/``. The assembly mode is an explicit choice, never inferred from credentials:

``--externalize`` (CI, ``./go site``) The deterministic, read-only half of publishing: read each *synced* bundle's HTML, resolve author links against the repo, insert one ``<base>`` pointing at the bucket, and write only ``_site/<key>/index.html`` (asset bytes stay on the CDN). Requires a configured store; fails loudly without one.

``--localize`` (local preview, ``./go preview``) Read the bundles from ``.mini/exports/`` and copy their ``_assets/`` beside the HTML, so the site works offline. Never touches the network.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from mini.lit.page import is_lit_page
from mini.report_print import print_bundle, print_stamp
from mini.reports import (
    PDF_LEAF,
    PDF_TYPE,
    PUBLISH_LOCK,
    ReportFigure,
    alternates,
    export_dir,
    export_key,
    insert_base,
    load_pins,
    mark_figures,
    publish_lock,
    report_figures,
    reports,
    rewrite_links,
    set_alternate,
    set_banner,
    set_lightbox,
    mark_verdicts,
    report_styles,
    set_report_styles,
    stray_links,
)

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
SITE_DIR = WORKSPACE_ROOT / "_site"
DOCS_DIR = WORKSPACE_ROOT / "docs"

# The shared report stylesheet, re-inlined into every report at build time (see
# mini.reports.set_report_styles). Read from source each build, so editing it restyles
# every published report with no re-export.
REPORT_CSS = DOCS_DIR / "report.css"
MD_SITE_CSS = WORKSPACE_ROOT / "scripts" / "md.css"  # a Markdown page's site-only rules (the figure strips)

# The relative dir, beside each report's index.html, holding its externalized assets
# (figures, data blobs) written by mini.reports.Publisher.
ASSET_LINK = "_assets"

# Source suffixes that the build renders into a report page (so an author link to one
# resolves to the rendered result, not the dead source file).
_RENDERED_SUFFIXES = (".py", ".md")


def prepare_dirs():
    print("Preparing site directory...")
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()


def _resolve_publish_store():
    """The HF publish tier for ``--externalize``, or a loud exit if unreachable.

    Mode is the caller's explicit choice; this only checks the chosen mode is *possible* — it never silently downgrades to localize.
    """
    from mini.hf_store import HFStore
    from mini.store import store_for

    store = store_for(WORKSPACE_ROOT / ".mini" / "store")
    if not isinstance(store, HFStore):
        sys.exit(
            "--externalize needs the HF publish tier (a read token suffices): "
            "set [tool.mini] store-bucket/publish-repo and run `./go auth`.\n"
            "For an offline build from local bundles, use `./go preview` (--localize)."
        )
    return store


# ---------------------------------------------------------------------------
# Author-link resolution
#
# A report's only *relative* URLs should be its store assets; an author-written link
# (``[src](./experiment.py)``) is repointed by the asset ``<base>`` and would 404. The
# resolver turns each such link into an absolute target — the rendered page for things
# the build renders, the GitHub source otherwise — so it survives the base. In localize
# mode (no base) rendered links stay relative so offline navigation still works.
#
# A root-absolute target (``/eng/gc.md``) is the house style for a cross-tree link
# (see ``todo/eng/markdown-link-check.md``): GitHub and VS Code both read it against
# the repo root, so the resolver does too, rebasing it onto ``docs/`` to take the same
# paths below as a relative one. ``_ANCHORED`` therefore matches only what is already
# absolute or in-page, the same set ``mini.reports`` leaves alone.
# ---------------------------------------------------------------------------

_ANCHORED = re.compile(r"(?:[a-z][a-z0-9+.\-]*:|//|#)", re.IGNORECASE)


def _strip_index(url: str) -> str:
    """Drop a trailing ``index.html`` so a report reads ``<key>/`` not ``<key>/index.html``.

    GitHub Pages serves the directory form, and it's the nicer canonical/shareable URL. Operates before any ``#fragment`` and leaves non-index pages (``foo.html``) untouched. Used only when publishing — offline (``file://``) navigation keeps the explicit file.
    """
    return re.sub(r"(^|/)index\.html(?=$|#)", r"\1", url)


def _repo_slug() -> str | None:
    """``owner/repo`` from ``$MINI_REPO`` or the git ``origin`` remote, or ``None``."""
    url = os.environ.get("MINI_REPO")
    if not url:
        try:
            url = subprocess.run(
                ["git", "-C", str(WORKSPACE_ROOT), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except OSError, subprocess.CalledProcessError:
            return None
    m = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


@dataclass(frozen=True)
class LinkResolver:
    """Maps an author-written relative link to its published target.

    ``render_map`` is docs-relative *source* path → site-relative *output* path for every page the build emits (reports render to ``<key>/index.html``, markdown to ``<name>.html``); ``site_assets`` is what :func:`site_asset_files` copies verbatim into ``_site/`` at the same relative path; ``source_files`` is every file under ``docs/`` (the GitHub-source fallback). ``site_base``/``source_base`` are the absolute roots used when a link must be made absolute (externalize mode).
    """

    render_map: dict[str, str]
    source_files: frozenset[str]
    site_base: str | None
    source_base: str | None
    site_assets: frozenset[str] = frozenset()
    repo_root: Path | None = None  # used to confirm a link escaping docs/ exists in the repo
    # Production's URL whatever ``site_base`` is: a PDF links there from every branch (:func:`printable`).
    production_base: str | None = None

    @classmethod
    def discover(cls) -> "LinkResolver":
        render_map: dict[str, str] = {}
        for md in DOCS_DIR.rglob("*.md"):
            if md.name == "README.md":
                continue
            rel = md.relative_to(DOCS_DIR).as_posix()
            render_map[rel] = PurePosixPath(rel).with_suffix(".html").as_posix()
        for report in reports(DOCS_DIR):
            out = f"{export_key(report)}/index.html"
            stem_rel = report.relative_to(DOCS_DIR)
            # The report came from this script; register every suffix an author might
            # have linked (``report.py`` → its rendered ``<key>/index.html``), plus the
            # bare directory (``../ex-2.1.1/``) — the canonical published URL one report
            # naturally uses to link another.
            for suffix in _RENDERED_SUFFIXES:
                render_map[stem_rel.with_suffix(suffix).as_posix()] = out
            render_map[export_key(report)] = out

        source_files = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in DOCS_DIR.rglob("*") if p.is_file())
        site_assets = frozenset(p.relative_to(DOCS_DIR).as_posix() for p in site_asset_files())

        slug = _repo_slug()
        site_base = os.environ.get("MINI_SITE_URL")
        source_base = os.environ.get("MINI_SOURCE_URL")
        production_base = None
        if slug:
            owner, repo = slug.split("/", 1)
            production_base = f"https://{owner}.github.io/{repo}/"
            site_base = site_base or production_base
            source_base = source_base or f"https://github.com/{slug}/blob/main/"
        return cls(
            render_map,
            source_files,
            site_base,
            source_base,
            site_assets,
            repo_root=WORKSPACE_ROOT,
            production_base=production_base,
        )

    def _in_site(self, out: str, *, out_dir: str, externalizing: bool, frag: str) -> str | None:
        """How a page rendering into ``out_dir`` should link *out*, a site-relative path.

        Externalizing, the page carries an asset ``<base>`` that would repoint a relative URL at the bucket, so it has to be spelled out from ``site_base`` (and a report reads ``<key>/``, not ``<key>/index.html``). Localizing, it stays relative — resolved from where the page *renders*, which for a report differs from its source dir — so offline navigation works.
        """
        if externalizing:
            return None if self.site_base is None else f"{self.site_base}{_strip_index(out)}{frag}"
        rel = os.path.relpath(out, out_dir or ".")
        return f"{PurePosixPath(rel).as_posix()}{frag}"

    def resolve(self, token: str, *, from_dir: str, out_dir: str, externalizing: bool) -> str | None:
        """The rewritten target for author-written link *token* under ``docs/<from_dir>``.

        A relative token is interpreted against ``from_dir`` (where it was written) and a root-absolute one against the repo root; a localized link is made relative to ``out_dir`` (where the emitting page *renders*, which for a report differs from its source dir). ``None`` means "leave it alone" — an external or in-page link, or one whose target the build doesn't know how to reach.
        """
        if not token or _ANCHORED.match(token):
            return None
        path_part, _, frag = token.partition("#")
        frag = f"#{frag}" if frag else ""
        if path_part.startswith("/"):
            # Root-absolute: repo-root-relative, so rebase onto docs/ and let the
            # branches below decide. One under docs/ renders like any other page;
            # one outside it takes the escape branch and points at the source.
            norm = os.path.relpath(os.path.normpath(path_part.lstrip("/")), "docs")
        else:
            norm = os.path.normpath(PurePosixPath(from_dir, path_part).as_posix())
        if norm.startswith(".."):
            # Escaped docs/, but often still inside the repo — a report linking to its
            # source modules (``../src/experiment``, ``../../src/.../README.md``). Point
            # such a link at the GitHub source so it survives the asset <base> (which
            # would otherwise 404 it against the bucket). Bail if there's no source base,
            # it escapes the repo root too, or the target doesn't exist in the repo.
            if self.source_base is None:
                return None
            repo_rel = os.path.normpath(PurePosixPath("docs", norm).as_posix())
            if repo_rel.startswith(".."):
                return None
            if self.repo_root is not None and not (self.repo_root / repo_rel).exists():
                return None
            return f"{self.source_base}{repo_rel}{frag}"

        if norm in self.render_map:
            return self._in_site(self.render_map[norm], out_dir=out_dir, externalizing=externalizing, frag=frag)
        if norm in self.site_assets:
            # A file the build copies verbatim into _site/ — an image, a data blob. The
            # site serves it at the same relative path, so point there. The GitHub
            # fallback below would hand back a ``blob/`` URL, which is an HTML page
            # rather than the bytes: fine to click, but an ``<img>`` renders nothing.
            return self._in_site(norm, out_dir=out_dir, externalizing=externalizing, frag=frag)
        if norm in self.source_files:
            return None if self.source_base is None else f"{self.source_base}docs/{norm}{frag}"
        return None


def prepare_dirs_and_resolver() -> LinkResolver:
    prepare_dirs()
    return LinkResolver.discover()


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    """One report's exported HTML as read from its source, before any assembly.

    ``html`` is ``None`` when there's nothing to assemble (never published, or never exported locally). ``notes`` are log lines the read wants printed — held here rather than printed on the spot, so a concurrent read still logs in report order.
    """

    html: str | None
    base_href: str | None = None  # externalize: the CDN dir the report's _assets/ resolve against
    assets: Path | None = None  # localize: the local _assets/ dir to copy beside the HTML
    # localize: every rendition the page declares (``<link rel="alternate">``) that the export
    # wrote beside it — a literate script's ``index.md`` — copied beside the HTML too. Never
    # the PDF: the build prints that itself (:class:`PdfMemo`), whatever an older export left.
    renditions: tuple[Path, ...] = ()
    notes: tuple[str, ...] = ()


def _read_bundle(report: Path, *, store, pins: dict[str, str], externalizing: bool) -> Bundle:
    """Read one report's exported ``index.html`` — off the bucket, or from ``.mini/exports/``.

    Externalize reads *only* the HTML. The page's ``_assets/`` links stay relative and the ``<base>`` sends them to the bundle on the CDN, so pulling the whole bundle here would fetch megabytes of figures the build has no use for. It's also the one step that waits on the network, which is why it's separable: the reports are independent, so the caller runs these together instead of serially.

    A report pinned in ``docs/publish.lock`` is read *and* based at that revision, so the page serves exactly what its publish uploaded — a later re-publish (e.g. from a branch whose PR hasn't merged) can't swap the assets under this build. An unpinned report falls back to the mutable branch head, with a warning.
    """
    key = export_key(report)
    if not externalizing:
        bundle = export_dir(report)
        if not (bundle / "index.html").exists():
            nb_rel = report.relative_to(WORKSPACE_ROOT).as_posix()
            return Bundle(None, notes=(f"  ! {key}: not exported locally — run `./go preview {nb_rel}` (skipping)",))
        assets = bundle / ASSET_LINK
        html = (bundle / "index.html").read_text("utf-8")
        if not is_lit_page(html):
            nb_rel = report.relative_to(WORKSPACE_ROOT).as_posix()
            return Bundle(
                None, notes=(f"  ! {key}: local export predates mini.lit — run `./go preview {nb_rel}` (skipping)",)
            )
        renditions = tuple(
            p for kind, href in alternates(html).items() if kind != PDF_TYPE and (p := bundle / href).is_file()
        )
        return Bundle(html, assets=assets if assets.is_dir() else None, renditions=renditions)

    notes: list[str] = []
    revision = pins.get(key)
    # Only a git-backed publish tier can pin; on the single-bucket default the mutable
    # head is all there is, so the nudge would be misleading.
    if revision is None and store.publish_repo is not None:
        notes.append(f"  ! {key}: not pinned in {publish_lock()} — serving the mutable head; `./go publish` to pin")
    html = store.read_export_html(key, revision=revision)
    if html is None:
        notes.append(f"  ! {key}: no synced export on the bucket — run `./go publish` (skipping)")
        return Bundle(None, notes=tuple(notes))
    return Bundle(html, base_href=store.export_base(key, revision=revision), notes=tuple(notes))


@dataclass(frozen=True)
class FigureStrip:
    """One built report's figures, for a Markdown page to render as thumbnails.

    ``base_href`` is the CDN dir the bundle's relative asset URLs resolve against — the same (revision-pinned) base the report page itself gets — or ``None`` when localizing, where the assets sit at ``_site/<key>/_assets/`` and a page links them relatively.
    """

    key: str
    base_href: str | None
    figures: tuple[ReportFigure, ...]
    # The PDF's href when the build printed one: absolute into the site when externalizing
    # (the figures' base is the CDN, and the PDF is not there), else the leaf beside the page.
    pdf: str | None = None


Printer = Callable[[Path, Path, str], Path | None]
"""``(serve_from, out, html) -> out`` or ``None``: a test's stand-in for :func:`_print`."""


def _print(serve_from: Path, out: Path, html: str) -> Path | None:
    return print_bundle(serve_from, out, html=html)


@dataclass
class PdfMemo:
    """The PDFs a previous build printed, so this one prints only what changed.

    A report's PDF is a function of the page the build prints from, which already carries the pinned bundle's HTML (naming its figures by immutable, revision-pinned URLs), the current ``report.css``, and every resolved link, plus the print tooling (:func:`mini.report_print.print_stamp`). So the memo keys each report on a hash of those two, kept in ``pdfs.json`` beside the PDFs under ``root``: the same key means the same file, and the report is not printed again. A prose edit reprints one report, a stylesheet edit reprints every report on that branch, and a tooling bump reprints everything once.

    ``root`` is ``$MINI_PDF_MEMO`` when set, which is how the deploy hands a build the previous ``gh-pages`` commit's copy of the site (production's for ``main``, its own preview's for a PR), or ``.mini/pdfs/`` for a local preview. Fresh prints land there too, and the manifest is written as each one lands and again by :meth:`save`, so the site's own copy is the next build's memo, and a local build that stops partway (a tooling change reprints every report, and the sweep is minutes long) keeps what it printed.

    ``fallbacks`` are further memos, read and never written: the rest of ``$MINI_PDF_MEMO``, split on the path separator. A PDF found there under the same key is copied into ``root`` rather than printed. The deploy gives a preview production's memo this way, so a PR prints only the reports it changed: a PDF's links lead to production from every branch (:func:`printable`), which is what makes an unchanged report's key the same on both. A new PR starts with no memo of its own, and would otherwise print every report.
    """

    root: Path
    stamp: str = field(default_factory=print_stamp)
    printer: Printer = _print
    fallbacks: tuple[Path, ...] = ()
    previous: dict[str, str] = field(init=False)
    current: dict[str, str] = field(init=False, default_factory=dict)

    MANIFEST = "pdfs.json"

    def __post_init__(self) -> None:
        self.previous = self._read(self.root)

    @classmethod
    def open(cls) -> "PdfMemo":
        roots = [Path(root) for root in os.environ.get("MINI_PDF_MEMO", "").split(os.pathsep) if root]
        return cls(roots[0], fallbacks=tuple(roots[1:])) if roots else cls(WORKSPACE_ROOT / ".mini" / "pdfs")

    @classmethod
    def _read(cls, root: Path) -> dict[str, str]:
        manifest = root / cls.MANIFEST
        try:
            return json.loads(manifest.read_text("utf-8")) if manifest.is_file() else {}
        except OSError, ValueError:
            return {}

    def _borrow(self, key: str, want: str) -> Path | None:
        """The PDF of *key* in the first fallback memo that printed it from the same page and tooling, if any."""
        for root in self.fallbacks:
            if self._read(root).get(key) == want and (found := root / key / PDF_LEAF).is_file():
                return found
        return None

    def key(self, printable: str) -> str:
        return hashlib.sha256(f"{self.stamp}\n{printable}".encode()).hexdigest()[:32]

    def pdf(self, key: str, printable: str, *, serve_from: Path) -> Path | None:
        """The PDF of *printable* for report *key*: reused from the memo, or printed now. ``None`` when there is no browser to print with."""
        out = self.root / key / PDF_LEAF
        want = self.key(printable)
        fresh = self.previous.get(key) != want or not out.is_file()
        if not fresh:
            print(f"  {key}: PDF unchanged")
        elif (found := self._borrow(key, want)) is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, out)
            print(f"  {key}: PDF unchanged (main's copy)")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"  {key}: printing PDF (headless Chromium; a few seconds)")
            if self.printer(serve_from, out, printable) is None:
                return None
        self.current[key] = want
        if fresh:
            self._write(self.previous | self.current)  # the previous entries stay until save() prunes them
        return out

    def save(self) -> None:
        """Write the manifest of what this build printed or reused; entries for reports the build no longer has are dropped."""
        self._write(self.current)

    def _write(self, manifest: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")


def print_css(source: Path | None) -> str:
    """The shared sheets a page prints with, from source, on top of its own: the current ``report.css``, and for a Markdown page (*source* a ``.md``) its prose sheet after it (:func:`mini.reports.report_styles`)."""
    return report_styles(source or REPORT_CSS)


_CLOSED_DETAILS = re.compile(r"<details(?![^>]*\bopen\b)")


def open_details(html: str) -> str:
    """*html* with every ``<details>`` open, for a Markdown page's print: on paper a fold cannot be opened, and the index keeps its report lists in one."""
    return _CLOSED_DETAILS.sub("<details open", html)


_ASSET_REF = re.compile(r"""(?<=["'(])_assets/""")


def printable(bundle: Bundle, links: LinkResolver, *, from_dir: str, key: str, report_css: str) -> str:
    """The page as the PDF prints it: author links resolved to the site and GitHub, the current ``report.css`` on top, and nothing the build adds for a browser (banner, lightbox, deferred figures).

    Links are resolved the externalizing way in either mode, and against production rather than a preview's URL, so the PDF's links lead to the published site and its bytes are a function of the report alone (relative, they would carry the loopback port the print served from; a preview's, the PR number, and a PR could then never reuse production's PDFs through :class:`PdfMemo`). The page's own ``#fragment`` links stay bare, which is what makes them in-document jumps in the PDF, so the bundle's figures cannot go through a ``<base>``; externalizing, each ``_assets/`` reference is spelled out against the pinned CDN base instead, and the print fetches them through its cache.
    """
    if links.production_base:
        links = replace(links, site_base=links.production_base)
    html = resolve_html_links(bundle.html or "", links, from_dir=from_dir, out_dir=key, externalizing=True)
    html = set_report_styles(html, report_css)
    if bundle.base_href:
        html = _ASSET_REF.sub(f"{bundle.base_href}{ASSET_LINK}/", html)
    return html


def build_reports(
    links: LinkResolver, store, externalizing: bool, memo: PdfMemo | None = None
) -> dict[str, FigureStrip]:
    """Assemble each report bundle into ``_site/<key>/index.html``, with its PDF beside it.

    Externalize: read the synced HTML from the bucket, insert one ``<base>`` at ``exports/<key>/`` so its relative ``_assets/`` resolve there, and write only the HTML into ``_site`` (the bytes stay on the bucket CDN). Localize: read the bundle from ``.mini/exports`` and copy its ``_assets/`` beside the HTML so it works offline. Author links are resolved to absolute/relative targets either way.

    The PDF (``report.pdf``, for reading on paper or e-ink) is printed here from the assembled page, through *memo* (:class:`PdfMemo`, opened from the environment when not given, and closed by :func:`save_pdfs` once the Markdown pages have printed too) so an unchanged report is not printed again. The page links it from the nav chip and declares it as an alternate rendition; when nothing can print (no browser), the page carries neither. A print for review, stamped with its commit and marked against an earlier round, is ``scripts/render_report.py``'s, and never the site's: its stamp would change the memo key on every commit.

    Returns each built report's :class:`FigureStrip` by key, so :func:`convert_markdown` can expand ``mini:figures`` markers from the HTML this pass already fetched.
    """
    print("Building reports...")
    pins = load_pins(WORKSPACE_ROOT) if externalizing else {}
    memo = memo or PdfMemo.open()
    report_css = print_css(None)
    nbs = reports(DOCS_DIR)
    # Externalized, each read is a round trip to the bucket and the reports don't depend
    # on each other — so read them in one wave and the build waits for the slowest report
    # rather than the sum of all of them. Assembly below is CPU-cheap and stays sequential
    # in report order, so the log reads the same however the threads interleaved.
    with ThreadPoolExecutor(max_workers=min(8, max(len(nbs), 1))) as pool:
        bundles = pool.map(
            lambda report: _read_bundle(report, store=store, pins=pins, externalizing=externalizing), nbs
        )

    strips: dict[str, FigureStrip] = {}
    for report, bundle in zip(nbs, list(bundles), strict=True):
        key = export_key(report)
        for note in bundle.notes:
            print(note)
        if bundle.html is None:
            continue
        # The verdict badges, for a page exported before a render wrote them (a no-op otherwise).
        bundle = replace(bundle, html=(page_html := mark_verdicts(bundle.html)))
        figures = tuple(report_figures(page_html, link=ASSET_LINK))
        from_dir = report.parent.relative_to(DOCS_DIR).as_posix()  # where author links resolve
        from_dir = "" if from_dir == "." else from_dir
        nb_rel = report.relative_to(WORKSPACE_ROOT).as_posix()
        dest = SITE_DIR / key / "index.html"
        dest.parent.mkdir(parents=True, exist_ok=True)

        # The page's own published URL: what its `#fragment` links and its PDF link are
        # spelled out against under the <base>, which would otherwise send both to the bucket.
        page_url = links.resolve(key, from_dir="", out_dir=key, externalizing=True) if bundle.base_href else None
        to_print = printable(bundle, links, from_dir=from_dir, key=key, report_css=report_css)
        # Localizing, the print serves the bundle's own _assets/; externalizing, the printable
        # names them on the CDN, so the serve root holds the page alone.
        pdf = memo.pdf(key, to_print, serve_from=bundle.assets.parent if bundle.assets else dest.parent)
        pdf_url = None
        if pdf is not None:
            shutil.copy2(pdf, dest.parent / PDF_LEAF)
            pdf_url = f"{page_url}{PDF_LEAF}" if page_url else PDF_LEAF
        strips[key] = FigureStrip(key, bundle.base_href, figures, pdf=pdf_url)

        html = resolve_html_links(page_html, links, from_dir=from_dir, out_dir=key, externalizing=externalizing)
        html = set_alternate(html, type=PDF_TYPE, href=pdf_url)  # the build's, in place of any an older export declared
        html = mark_figures(html, link=ASSET_LINK)  # defer offscreen figures; mark them zoomable
        html = set_lightbox(html)  # click a figure for the full-size image, over a dimmed page
        index_url, source_url = _nav_urls(links, key=key, nb_rel=nb_rel, externalizing=externalizing)
        html = set_banner(html, index_url=index_url, source_url=source_url, pdf_url=pdf_url)
        html = set_report_styles(html, report_css)  # last, so shared report rules win ties
        if bundle.base_href:
            if page_url is None:
                print(f"  ! {key}: no site URL, so its in-page links (#footnotes, headings) will follow the <base>")
            html = insert_base(html, bundle.base_href, page_url=page_url)
        dest.write_text(html, "utf-8")

        if bundle.assets is not None:
            shutil.copytree(bundle.assets, dest.parent / ASSET_LINK, dirs_exist_ok=True)
        for rendition in bundle.renditions:  # a script's Markdown: served beside the page, as on the bucket
            shutil.copy2(rendition, dest.parent / rendition.name)
        print(f"  {key} -> _site/{key}/index.html{' [+base]' if bundle.base_href else ''}")
    return strips


def save_pdfs(memo: PdfMemo) -> None:
    """Close *memo* once every page is printed, the reports' and the Markdown pages', and put its manifest in the site, so the deployed site is the next build's memo."""
    memo.save()
    shutil.copy2(memo.root / PdfMemo.MANIFEST, SITE_DIR / PdfMemo.MANIFEST)


def _nav_urls(links: LinkResolver, *, key: str, nb_rel: str, externalizing: bool) -> tuple[str | None, str | None]:
    """The report banner's (index, source) links — same absolute/relative policy as author links.

    The source is the script on GitHub (``source_base`` + its repo path). The index is the site root: absolute (``site_base``) when externalizing — the asset ``<base>`` would otherwise repoint a relative link at the bucket — and relative back up from ``_site/<key>/index.html`` when localizing, so offline navigation works. Either is ``None`` if its base is unavailable.
    """
    source_url = f"{links.source_base}{nb_rel}" if links.source_base else None
    if externalizing:
        index_url = links.site_base  # the site root serves index.html
    else:
        index_url = "../" * (key.count("/") + 1) + "index.html"
    return index_url, source_url


def resolve_html_links(html: str, links: LinkResolver, *, from_dir: str, out_dir: str, externalizing: bool) -> str:
    """Rewrite resolvable author links in *html*; warn on the ones left dangling."""
    mapping: dict[str, str] = {}
    for token in stray_links(html, link=ASSET_LINK):
        target = links.resolve(token, from_dir=from_dir, out_dir=out_dir, externalizing=externalizing)
        if target is not None:
            mapping[token] = target
        else:
            print(f"  ! {from_dir or '.'}: unresolved relative link {token!r} — a <base> would break it")
    return rewrite_links(html, mapping) if mapping else html


_ASSET_SKIP_DIRS = {"__pycache__"}
_ASSET_SKIP_SUFFIXES = {".py", ".md", ".pyc", ".pyo"}


def site_asset_files() -> list[Path]:
    """Files under ``docs/`` the build copies verbatim into ``_site/``, at the same relative path.

    One definition, read twice: :func:`copy_assets` copies them, and :class:`LinkResolver` needs the same set to know that a link to one is served by the site. Were the two to drift, an author link would resolve somewhere the copy never put the file.
    """
    out = []
    for item in sorted(DOCS_DIR.rglob("*")):
        if not item.is_file() or item == WORKSPACE_ROOT / PUBLISH_LOCK:  # the pin manifest is build input, not content
            continue
        parts = item.relative_to(DOCS_DIR).parts
        if any(p in _ASSET_SKIP_DIRS or p.startswith(".") for p in parts):
            continue
        if item.suffix in _ASSET_SKIP_SUFFIXES:
            continue
        out.append(item)
    return out


def copy_assets():
    """Copy non-Python, non-Markdown files from docs/ to _site/."""
    print("Copying assets...")
    for item in site_asset_files():
        rel = item.relative_to(DOCS_DIR)
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {item.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}")
        shutil.copy2(item, dest)


def site_root(dest: Path) -> str:
    """Return the relative path prefix from dest back to the site root."""
    depth = len(dest.relative_to(SITE_DIR).parts) - 1
    return "../" * depth


def markdown_pages() -> list[Path]:
    """Every Markdown page the site serves: each ``.md`` under ``docs/`` but its ``README.md``."""
    return sorted(p for p in DOCS_DIR.rglob("*.md") if p.name != "README.md")


def markdown_page(md_file: Path) -> str:
    """*md_file* as ``mini.lit`` renders it: the page shell, stylesheets and glossary notes a report gets, with its author links as written.

    The same page a review print is made from (``scripts/render_report.py``), so the site, the site's PDF and the review print of a Markdown page agree.
    """
    from mini.lit.document import Runner, parse
    from mini.lit.render import compose

    html, _ = compose(Runner(md_file).weave(parse(md_file)))
    return html


def _resolve_md_links(html: str, links: LinkResolver, *, from_dir: str, pretty: bool) -> str:
    """Rewrite the author links in a Markdown page's HTML; warn on the ones left dangling.

    A Markdown page never carries an asset ``<base>``, so its links are resolved in *localize* mode: a rendered target stays a relative link (clickable offline), a source file becomes an absolute GitHub link. When publishing (``pretty``), a report link drops its ``index.html`` so it reads ``<key>/``; offline builds keep the explicit file so ``file://`` navigation still works.
    """
    mapping: dict[str, str] = {}
    for token in stray_links(html, link=ASSET_LINK):
        target = links.resolve(token, from_dir=from_dir, out_dir=from_dir, externalizing=False)
        if target is not None:
            mapping[token] = _strip_index(target) if pretty else target
        else:
            print(f"  ! {from_dir or '.'}: unresolved relative link {token!r}")
    return rewrite_links(html, mapping) if mapping else html


def _md_index_url(rel: str, *, pretty: bool) -> str | None:
    """The nav chip's link from Markdown page *rel* back to the index, relative; ``None`` on the index itself."""
    if rel == "index.md":
        return None
    url = PurePosixPath(os.path.relpath("index.html", PurePosixPath(rel).parent.as_posix() or ".")).as_posix()
    return (_strip_index(url) or "./") if pretty else url


# A figure-strip marker in a Markdown page: `<!-- mini:figures ./m2/ex-2.1.6/report.py -->`
# on its own line, naming a report the way an ordinary link would. Comments pass through
# both GitHub's renderer (invisible there — the source view stays clean) and
# Python-Markdown (so the marker survives into the rendered body, where the build swaps
# it for the strip), and `check_md_links` strips comments before matching, so the path
# inside costs nothing there either.
_FIGURES_MARKER = re.compile(r"<!--\s*mini:figures\s+(\S+)\s*-->")


def _marker_key(token: str, links: LinkResolver, *, from_dir: str) -> str | None:
    """The export key a ``mini:figures`` marker names, or ``None`` for an unknown target.

    Accepts the same spellings a link to the report would use — ``./m2/ex-2.1.6/report.py`` or the bare directory — resolved against the page's own dir via the resolver's ``render_map``, so the marker can't drift from how the rest of the page addresses reports.
    """
    norm = os.path.normpath(PurePosixPath(from_dir, token).as_posix())
    out = links.render_map.get(norm)
    suffix = "/index.html"
    return out[: -len(suffix)] if out is not None and out.endswith(suffix) else None


def _figure_strip_html(strip: FigureStrip, *, from_dir: str, externalizing: bool) -> str:
    """A report's thumbnail strip: each figure a lazy image, themed via ``<picture>``, opening with its PDF.

    Empty for a report with no asset-served figures and no PDF (nothing to show, so no box). The PDF, when the build printed one, is a page-shaped chip at the head of the strip, where it is never scrolled out of view: the strip and the report's nav chip link the same file. Externalizing, URLs use the strip's revision-pinned CDN base — the same assets the report page serves, so the index can never show figures its report doesn't. Localizing they're relative into the copied ``_site/<key>/_assets/``. Each thumbnail is the small copy the export wrote (:func:`mini.reports.write_thumbnails`, a few KB against ~100 KB for the full figure), falling back to the full-size image for a bundle published before thumbnails existed. It reuses the figure's own alt text and the ``width``/``height`` the export stamped: the CSS (``scripts/md.css``) fixes the height, so those only set the aspect ratio, and the row lays out before the images arrive, one theme's file per figure and only as it scrolls into view. Clicking one opens the full-size figure in the lightbox (:func:`~mini.reports.lightbox_chrome`) rather than linking to the PNG: a link navigates away, and the browser paints the figure's transparent background on its white canvas — wrong in dark mode — where the overlay can carry a themed one.
    """
    import html

    if not strip.figures and not strip.pdf:
        return ""
    base = (
        strip.base_href
        if externalizing
        else f"{PurePosixPath(os.path.relpath(strip.key, from_dir or '.')).as_posix()}/"
    )
    parts = []
    if strip.pdf:
        pdf = strip.pdf if externalizing else f"{base}{strip.pdf}"
        parts.append(f'<a class="fig-strip-pdf" href="{pdf}" title="Read the report as a PDF">PDF</a>')
    for fig in strip.figures:
        alt, title = html.escape(fig.alt), html.escape(fig.stem)
        size = f' width="{fig.width}" height="{fig.height}"' if fig.width and fig.height else ""
        light, dark = fig.light_thumb or fig.light, fig.dark_thumb or fig.dark
        # The thumbnail is a few KB and unreadable at strip size, so each one names the
        # full-size figure it was made from; clicking opens that in the lightbox, which
        # is what lets the strip offer the image without a link that navigates to it.
        full = f' data-mini-full="{base}{fig.light}"' + (f' data-mini-full-dark="{base}{fig.dark}"' if fig.dark else "")
        img = (
            f'<img src="{base}{light}" alt="{alt}" title="{title}"{size}'
            f' loading="lazy" tabindex="0" data-mini-zoom{full}>'
        )
        if dark:
            img = f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base}{dark}">{img}</picture>'
        parts.append(img)
    return f'<div class="fig-strip">{"".join(parts)}</div>'


def expand_figure_strips(
    body: str, strips: dict[str, FigureStrip], links: LinkResolver, *, from_dir: str, externalizing: bool
) -> tuple[str, bool]:
    """Swap each ``mini:figures`` marker in a rendered page for its report's thumbnail strip.

    Operates on the rendered HTML rather than the Markdown, so the strip's markup never has to survive Python-Markdown's block parsing (it would read a raw ``<div>`` indented inside a list item as code). A marker whose report wasn't built (unpublished, or skipped) renders as nothing, with a build note.

    Also returns whether the page ended up with a strip, so the caller ships the lightbox only to the pages that have something to enlarge.
    """

    shown = False

    def repl(m: re.Match) -> str:
        nonlocal shown
        token = m.group(1)
        strip = strips.get(_marker_key(token, links, from_dir=from_dir) or "")
        if strip is None:
            print(f"  ! {from_dir or '.'}: mini:figures {token!r} names no built report — dropping the strip")
            return ""
        html = _figure_strip_html(strip, from_dir=from_dir, externalizing=externalizing)
        shown = shown or bool(html)  # a dropped marker, or a figureless report, is not a strip
        return html

    return _FIGURES_MARKER.sub(repl, body), shown


def convert_markdown(
    links: LinkResolver,
    externalizing: bool,
    strips: dict[str, FigureStrip] | None = None,
    memo: PdfMemo | None = None,
):
    """Render each Markdown page under ``docs/`` (but ``README.md``) to ``_site/<name>.html``, with its PDF beside it as ``<name>.pdf``.

    A Markdown page goes through ``mini.lit`` like a report (:func:`markdown_page`), then gets what the site adds to a report: resolved links, the nav chip (index, source, PDF), and the PDF printed through *memo* (:class:`PdfMemo`, keyed by the page's docs-relative path, which no report key ends with). The site-only pieces go on top (``docs/md-prose.css`` is the page's already): ``mini:figures`` strips (with the lightbox, on a page that has one) and ``scripts/md.css``. The print leaves the strips out: a thumbnail row is for scanning on screen, and every entry it sits under links its report already.
    """
    print("Converting Markdown...")
    memo = memo or PdfMemo.open()
    site_css = MD_SITE_CSS.read_text("utf-8")
    # The print serves a page alone (its figures are site files, linked absolute), from a
    # throwaway root beside this empty dir, inside the site, and removed below.
    serve_from = SITE_DIR / ".md-print"
    serve_from.mkdir(parents=True, exist_ok=True)
    for md_file in markdown_pages():
        rel = md_file.relative_to(DOCS_DIR).as_posix()
        dest = SITE_DIR / PurePosixPath(rel).with_suffix(".html")
        dest.parent.mkdir(parents=True, exist_ok=True)
        from_dir = PurePosixPath(rel).parent.as_posix()
        from_dir = "" if from_dir == "." else from_dir
        page_html = markdown_page(md_file)

        to_print = printable(
            Bundle(open_details(_FIGURES_MARKER.sub("", page_html))),
            links,
            from_dir=from_dir,
            key=from_dir,
            report_css=print_css(md_file),
        )
        pdf = memo.pdf(rel, to_print, serve_from=serve_from)
        pdf_url = None
        if pdf is not None:
            shutil.copy2(pdf, dest.with_suffix(".pdf"))
            pdf_url = dest.with_suffix(".pdf").name

        html = _resolve_md_links(page_html, links, from_dir=from_dir, pretty=externalizing)
        html, has_strip = expand_figure_strips(
            html, strips or {}, links, from_dir=from_dir, externalizing=externalizing
        )
        if has_strip:
            html = set_lightbox(html)
        html = set_alternate(html, type=PDF_TYPE, href=pdf_url)
        source_url = f"{links.source_base}docs/{rel}" if links.source_base else None
        html = set_banner(
            html, index_url=_md_index_url(rel, pretty=externalizing), source_url=source_url, pdf_url=pdf_url
        )
        html = set_report_styles(html, site_css)
        dest.write_text(html, "utf-8")
        print(f"  {md_file.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(WORKSPACE_ROOT)}")
    serve_from.rmdir()


def add_nojekyll():
    (SITE_DIR / ".nojekyll").touch()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--externalize",
        action="store_true",
        help="assemble from published bundles; assets stay on the CDN behind a <base> (CI)",
    )
    mode.add_argument(
        "--localize", action="store_true", help="assemble from .mini/exports/ with assets copied in; works offline"
    )
    args = ap.parse_args()

    # Resolve the store *before* wiping _site, so a missing token can't destroy a build.
    if args.externalize:
        store = _resolve_publish_store()
        print(f"  asset mode: externalize ← {store.publish_repo or store.bucket}")
    else:
        store = None
        print("  asset mode: localize (.mini/exports/)")
    links = prepare_dirs_and_resolver()
    memo = PdfMemo.open()
    strips = build_reports(links, store, args.externalize, memo=memo)
    copy_assets()
    convert_markdown(links, args.externalize, strips, memo=memo)
    save_pdfs(memo)
    add_nojekyll()
    print(f"\nSite written to {SITE_DIR.relative_to(WORKSPACE_ROOT)}/")


if __name__ == "__main__":
    main()
