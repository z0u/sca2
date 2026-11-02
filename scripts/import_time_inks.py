#!/usr/bin/env python
"""Report `light_dark` calls that run at import, where a themed figure can't see them.

`mini.vis.light_dark` picks its value from the theme active when it is called. A `@themed` figure draws once per theme, so an ink resolved inside the plot follows the theme; one resolved at import (a module-level `INK = {"x": light_dark(...)}`) is resolved once, under the light theme, and the dark render reuses the light ink: a ghost meant to be dark gray comes out near-white on the dark page. The fix is to keep the pair (`INK = {"x": ("#light", "#dark")}`) and resolve it where the figure draws (`light_dark(*INK["x"])`).

A call counts as import-time when no function or lambda encloses it: module level and class bodies both run on import.

Exit status is the finding count clamped to 1, so `./go lint` can gate on it; stdout lists the findings, one per line, and the remedy goes to stderr.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from typing import TypeIs
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


@dataclass(frozen=True, order=True)
class Finding:
    """One `light_dark` call that runs at import."""

    path: Path
    line: int

    def __str__(self) -> str:
        where = self.path.relative_to(ROOT) if self.path.is_relative_to(ROOT) else self.path
        return f"{where.as_posix()}:{self.line}: light_dark runs at import, so a themed figure's dark render gets the light ink"


def _is_light_dark(node: ast.AST) -> TypeIs[ast.Call]:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "light_dark") or (
        isinstance(f, ast.Attribute) and f.attr == "light_dark"
    )


def import_time_calls(node: ast.AST) -> Iterator[int]:
    """The line of each `light_dark` call under *node* that no function or lambda encloses."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            # Decorators and defaults run at definition time, which is import time too.
            for part in [*getattr(child, "decorator_list", []), *child.args.defaults, *child.args.kw_defaults]:
                if part is not None:
                    yield from (n.lineno for n in ast.walk(part) if _is_light_dark(n))
            continue
        if _is_light_dark(child):
            yield child.lineno
        yield from import_time_calls(child)


def findings_in(path: Path) -> list[Finding]:
    """Every import-time `light_dark` call in *path*."""
    try:
        tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except SyntaxError:
        return []  # not our check's failure to report; ruff and the formatter own that
    return sorted({Finding(path, line) for line in import_time_calls(tree)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="files or directories to check (default: docs/ and src/)")
    args = ap.parse_args()

    paths = args.paths or [ROOT / "docs", ROOT / "src"]
    if missing := [p for p in paths if not p.exists()]:
        sys.exit(f"no such path: {', '.join(p.as_posix() for p in missing)}")

    targets = [p for arg in paths for p in (sorted(arg.rglob("*.py")) if arg.is_dir() else [arg])]
    found = sorted(f for path in targets for f in findings_in(path))

    if not found:
        print("✅ No light_dark call runs at import")
        return

    for finding in found:  # stdout is the worklist, so it stays pipeable
        print(finding)

    print(  # the remedy is commentary, so it goes to stderr and out of the pipe
        f"\n{len(found)} light_dark call(s) run at import. Keep each (light, dark) pair as data and call"
        " `light_dark(*pair)` inside the function that draws.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
