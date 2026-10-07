"""
Watch a script, re-weave it on save, and serve it with a live reload.

The editor stays whatever you write in; the browser is the viewer. The page goes to ``.mini/lit-live/<key>/``, apart from where ``render`` writes, so the two can run at once (they share only the memo cache, which is content-keyed). One process holds the :class:`~mini.lit.document.Runner` (so unchanged cells are not re-run and ``@memo`` hits are in memory), polls the document and the ``.py`` files beside it for changes, rewrites ``index.html`` when something moved, and answers a long-poll from the page so the browser reloads the moment a build lands. A sibling ``.py`` edit (a helper module beside the document) drops that module from ``sys.modules`` and resets the runner, since any cell may have imported it.

A reload is the whole page again, so what the browser may keep matters: figures are served as immutable (their URL carries a stamp of their content, so a re-drawn one arrives as a new URL) and the page is revalidated against an ``ETag``, which keeps a reload from re-fetching every figure — and keeps the page in the browser's cache, where DevTools reads the source behind a CSS rule.

The server listens before the first build, and a build shows its work: when a cell is still running after a moment (a download, a fit), the page is replaced with the document as it stands — everything above it, a running note, and the prose below with pending marks — so a slow cell never hides the rest of the report. The real page replaces it when the build lands.
"""

from __future__ import annotations

import hashlib
import html
import sys
import threading
import time
from collections.abc import Callable
from functools import partial as bind
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mini.lit.document import Woven
from mini.lit.render import Rendered, compose, output_dir, render
from mini.lit.render import _write as write_file

__all__ = ["serve"]

# How long a cell may run before the page shows it as running. Short enough that a
# download is visible at once, long enough that the ordinary run of quick cells does not
# reload the page once per cell.
PARTIAL_AFTER = 0.3

# What the browser may keep. A figure's URL carries a ``?v=`` stamp of its content
# (``mini.reports.Publisher``), so those bytes are immutable under that URL and a re-drawn
# figure arrives as a new URL — hence ``FOREVER``, and a reload repaints from the cache
# instead of pulling every figure down the wire again. The page itself has one URL for
# every version of itself, so it is ``REVALIDATE``: kept, but checked on each load, which
# costs one conditional request and answers ``304`` until a build lands. ``NEVER`` is for
# the version poll, whose whole point is to be current. The distinction matters beyond the
# flicker: ``no-store`` would have the browser discard the page as soon as it is parsed,
# and DevTools with it — no original source behind a CSS rule, no re-reading what was
# served.
FOREVER = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"
NEVER = "no-store"

_RELOAD = """
<script>
(async () => {
  let v = %d;
  for (;;) {
    try {
      const r = await fetch(`/__version?after=${v}`, {cache: "no-store"});
      const n = Number(await r.text());
      // Stop polling once a reload is asked for. Otherwise the next poll answers at once
      // (this page's v is stale now) and reloads again, cancelling the navigation in
      // flight; the tiny poll outruns the page, so the page lands only by luck.
      if (n !== v) { location.reload(); return; }
    } catch (e) { await new Promise(r => setTimeout(r, 1000)); }
  }
})();
</script>
"""


class _Site:
    """The served directory and its version, which the page long-polls; every write goes through :meth:`publish`."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.version = 0
        self.changed = threading.Condition()
        # Reentrant, so a caller can hold it across a check and the publish that depends on it.
        self.lock = threading.RLock()

    def publish(self, html_for: Callable[[str], str], markdown: str | None = None) -> None:
        """Write the page *html_for* builds around the reload script for the next version, then bump."""
        with self.lock:
            with self.changed:
                page = html_for(_RELOAD % (self.version + 1))
                write_file(self.out / "index.html", page)
                if markdown is not None:
                    write_file(self.out / "index.md", markdown)
                self.version += 1
                self.changed.notify_all()

    def wait_past(self, v: int, timeout: float) -> int:
        with self.changed:
            self.changed.wait_for(lambda: self.version != v, timeout=timeout)
            return self.version


class _Handler(SimpleHTTPRequestHandler):
    site: _Site
    _cache: str | None = None
    _head = False

    # Speak HTTP/1.1, so a connection is reused across requests. The default is HTTP/1.0,
    # one connection per request and closed after each; a report page asks for forty-odd
    # figures at once, so a reload opens forty-odd connections in a burst. Whatever sits
    # between the browser and here — VS Code's port forwarding in a dev container, an SSH
    # tunnel — then has to re-dial for every one, and a connection lost in that burst
    # reaches the browser as a body shorter than the ``Content-Length`` said
    # (``ERR_CONTENT_LENGTH_MISMATCH``). With keep-alive the whole page comes down a
    # handful of sockets.
    protocol_version = "HTTP/1.1"
    # Drop an idle keep-alive connection rather than hold its thread forever. Comfortably
    # past the long poll below, so a waiting page is never cut off mid-wait.
    timeout = 90

    def do_GET(self) -> None:  # noqa: N802
        self._cache = None
        if self.path.startswith("/__version"):
            after = int(self.path.rpartition("=")[2] or 0)
            self._send(str(self.site.wait_past(after, timeout=25)).encode(), "text/plain", NEVER)
            return
        if self.path in ("/", ""):
            self.path = "/index.html"
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            super().do_GET()  # a directory listing, or the 404
            return
        # Read the file, then send it, rather than stat-then-stream as the base class
        # does: the bytes we measure are the bytes we write, so the length can't disagree
        # with the body even if a build replaces the file underneath us mid-request.
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        self._send(body, self.guess_type(str(path)), FOREVER if "?v=" in self.path else REVALIDATE)

    def do_HEAD(self) -> None:  # noqa: N802
        """The same answer as a GET, minus the body — so the headers a probe sees are the ones a page gets."""
        self._head = True
        try:
            self.do_GET()
        finally:
            self._head = False

    def _send(self, body: bytes, ctype: str, cache: str) -> None:
        """Answer with *body*, under the *cache* policy, as a 304 if the page already holds these bytes."""
        self._cache = cache
        etag = None if cache == NEVER else f'"{hashlib.sha256(body).hexdigest()[:16]}"'
        if etag is not None and etag in [t.strip() for t in self.headers.get("If-None-Match", "").split(",")]:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if etag is not None:
            self.send_header("ETag", etag)
        self.end_headers()
        if self._head:
            return
        try:
            self.wfile.write(body)
        except OSError:  # the page went away mid-send; this connection is no longer framed
            self.close_connection = True

    def end_headers(self) -> None:
        self.send_header("Cache-Control", self._cache or NEVER)  # a 404 or a listing: never keep it
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # quiet
        pass


class _Server(ThreadingHTTPServer):
    # A page-load burst arrives faster than connections are accepted, and the default
    # backlog of 5 drops the overflow.
    request_queue_size = 128


class _Builder:
    """One build at a time: weave, publishing a partial page when a cell is slow, then the final page."""

    def __init__(self, doc: Path, site: _Site) -> None:
        self.doc = doc
        self.site = site
        self.runner = None
        self._timer: threading.Timer | None = None
        self._live = False  # whether a pending partial may still publish

    def _partial(self, woven: Woven) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(PARTIAL_AFTER, self._publish_partial, [woven])
        self._timer.daemon = True
        self._timer.start()

    def _publish_partial(self, woven: Woven) -> None:
        # Check and publish under one hold of the lock: released in between, the build
        # could finish and publish the final page in the gap, and this partial would then
        # replace it.
        with self.site.lock:
            if not self._live:  # the build finished first
                return
            self.site.publish(lambda reload: compose(woven, extra_body=reload)[0])
        cell = woven.running
        print(
            f"{self.doc.name}: running the cell at line {cell.line if cell else '?'}, page shows the rest", flush=True
        )

    def build(self) -> None:
        self._live = True
        try:
            r = render(
                self.doc, out_dir=self.site.out, live=True, runner=self.runner, write=False, partial=self._partial
            )
            self.runner = r.runner
        finally:
            with self.site.lock:
                self._live = False
            if self._timer is not None:
                self._timer.cancel()
        self.site.publish(lambda reload: compose(r.woven, extra_body=reload)[0], markdown=r.woven.markdown)
        _report(r)


def _mtimes(doc: Path) -> dict[Path, float]:
    files = [doc, *doc.parent.glob("*.py")]
    return {f: f.stat().st_mtime for f in files if f.exists()}


def _watch(doc: Path, builder: _Builder, poll: float) -> None:
    """Build once, then poll the document and its sibling modules and rebuild on a change (resetting the runner if a module moved)."""
    seen = _mtimes(doc)
    builder.build()
    while True:
        time.sleep(poll)
        now = _mtimes(doc)
        if now == seen:
            continue
        for f in {k for k in now.keys() | seen.keys() if now.get(k) != seen.get(k)}:
            if f.suffix == ".py" and f != doc:  # a sibling module (the document itself is the runner's business)
                for name, mod in list(sys.modules.items()):
                    if getattr(mod, "__file__", None) == str(f):
                        del sys.modules[name]
                if builder.runner is not None:
                    builder.runner.reset()
        seen = now
        try:
            builder.build()
        except Exception as e:  # keep serving the last good page
            print(f"build failed: {e}", file=sys.stderr, flush=True)


def serve(
    doc: Path | str, *, out_dir: Path | None = None, port: int = 8765, host: str = "127.0.0.1", poll: float = 0.25
) -> None:
    doc = Path(doc).resolve()
    out = (out_dir or output_dir(doc, live=True)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    site = _Site(out)
    placeholder = f"<!doctype html><meta charset=utf-8><title>{html.escape(doc.name)}</title><p>Rendering {html.escape(doc.name)}…</p>"
    site.publish(lambda reload: placeholder + reload)
    threading.Thread(target=_watch, args=(doc, _Builder(doc, site), poll), daemon=True).start()
    handler = type("Handler", (_Handler,), {"site": site})
    server = _Server((host, port), bind(handler, directory=str(out)))
    print(f"Serving {doc.name} at http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


def _report(r: Rendered) -> None:
    w = r.woven
    note = f", {len(w.errors)} error(s)" if w.errors else (", stopped early" if w.stopped else "")
    print(
        f"{r.doc.path.name}: {w.cells_run} cell(s) run, woven in {w.seconds * 1e3:.0f} ms, page in {r.seconds * 1e3:.0f} ms{note}",
        flush=True,
    )
