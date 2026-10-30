#!/usr/bin/env python
"""Report fields in literate prose that span more than one line.

Every top-level string in a literate script (`mini.lit`) is prose, and an f-string paragraph quotes values through `{…}` fields. `ruff format` formats the expression inside a field like any other code: a short one is joined onto one line, and a long one is expanded onto lines of its own, with the `{` and `}` on lines by themselves. The rendered paragraph doesn't change, since the value is the same, but the source paragraph is cut in two, which breaks the one-line-per-paragraph convention and makes the prose hard to read and review.

The fix is to compute the value in the cell above, as a named variable, and quote the name.

Exit status is the finding count clamped to 1, so `./go lint` can gate on it; stdout lists the findings, one per line, and the remedy goes to stderr.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mini.lit import is_literate_script

ROOT = Path(__file__).parent.parent.resolve()


@dataclass(frozen=True, order=True)
class Finding:
    """One prose field that spans lines."""

    path: Path
    line: int

    def __str__(self) -> str:
        where = self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path
        return f"{where.as_posix()}:{self.line}: a field in prose spans lines, so ruff format splits the paragraph"


def multiline_fields(tree: ast.Module) -> Iterator[int]:
    """The first line of each field, in a top-level f-string, that spans more than one line."""
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.JoinedStr)):
            continue
        for piece in node.value.values:
            if isinstance(piece, ast.FormattedValue) and piece.lineno != (piece.end_lineno or piece.lineno):
                yield piece.lineno


def findings_in(path: Path) -> list[Finding]:
    """Every multi-line prose field in *path*, when it is a literate script; a plain module has no prose."""
    if not is_literate_script(path):
        return []
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return []  # not our check's failure to report; ruff and the formatter own that
    return sorted(Finding(path, line) for line in multiline_fields(tree))


def python_files(root: Path) -> list[Path]:
    """Every `.py` file under *root*, sorted; :func:`findings_in` drops the ones that aren't literate."""
    return sorted(p for p in Path(root).rglob("*.py") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="scripts or directories to check (default: docs/)")
    args = ap.parse_args()

    paths = args.paths or [ROOT / "docs"]
    if missing := [p for p in paths if not p.exists()]:
        sys.exit(f"no such path: {', '.join(p.as_posix() for p in missing)}")

    targets = [p for arg in paths for p in (python_files(arg) if arg.is_dir() else [arg])]
    found = sorted(f for path in targets for f in findings_in(path))

    if not found:
        print("✅ No prose field spans lines")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    print(  # the remedy is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} prose field(s) span lines. Compute each value into a named variable in the cell above.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
