"""``python -m mini.lit``: render a literate script, or serve it with live reload."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

FORMATS = (".md", ".html", ".pdf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mini.lit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "render",
        help="weave and write each -o file, in the format its extension names (.md, .html, .pdf); figures go beside each under _assets/",
    )
    r.add_argument("doc", type=Path)
    r.add_argument(
        "-o",
        "--out",
        type=Path,
        action="append",
        required=True,
        metavar="FILE",
        help="an output file: .md, .html, or .pdf (printed with headless Chromium, a few seconds more); repeat for several",
    )
    s = sub.add_parser("serve", help="watch the script, re-render on save, and serve it with live reload")
    s.add_argument("doc", type=Path)
    s.add_argument("-o", "--out", type=Path, default=None, help="output directory (default .mini/lit-live/<key>/)")
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from mini.lit.serve import serve

        serve(args.doc, out_dir=args.out, port=args.port)
        return 0

    from mini.lit.render import render, write_outputs

    if bad := [o for o in args.out if o.suffix not in FORMATS]:
        ap.error(f"-o {bad[0]}: unknown format {bad[0].suffix or '(no extension)'!r}; use one of {', '.join(FORMATS)}")
    t0 = time.perf_counter()
    res = render(args.doc, out_dir=args.out[0].resolve().parent, write=False)
    w = res.woven
    for o in w.errors:
        print(f"error in cell at line {o.cell.line}:\n{o.error}", file=sys.stderr)
    try:
        written = write_outputs(res, args.out)
    except RuntimeError as e:  # no PDF printed; the reason is already logged
        print(e, file=sys.stderr)
        return 1
    for path in written:
        print(path)
    print(
        f"{w.cells_run} cell(s) run, woven in {w.seconds * 1e3:.0f} ms, page in {res.seconds * 1e3:.0f} ms, total {(time.perf_counter() - t0) * 1e3:.0f} ms",
        file=sys.stderr,
    )
    return 1 if w.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
