#!/usr/bin/env python
"""The report notebooks a branch changed — the work list for CI's auto-publish.

``./go publish`` is the authenticated half of publishing, and it's the half that gets
forgotten: a report merged without it serves the *previous* export's figures until
someone notices. CI runs it instead, and this script decides what "it" covers — every
report notebook the branch touched, minus the ones flagged for hand publishing
(:data:`~mini.reports.NO_AUTO_PUBLISH_MARKER`).

Publishing only the changed notebooks is also what keeps ``docs/publish.lock`` legible:
each run repins the reports that have new content and leaves every other pin as it was,
so the lock diff reads as the list of reports the branch actually changed.

Stdlib-only on purpose (``mini.reports`` imports nothing heavier), so it runs straight
from a checkout with no install: most pushes change no report, and that answer should be
cheap enough to ask on every one.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))  # importable without installing the project

from mini.reports import auto_publishes, export_key  # noqa: E402


def changed_reports(base: str, root: Path = ROOT) -> list[Path]:
    """The auto-publishable report notebooks this branch changed since *base*.

    The diff is three-dot (``base...HEAD``), i.e. against the merge base, so commits that
    landed on the base branch meanwhile aren't mistaken for ours. Deletions are filtered
    out — there's no bundle to export for a report that's gone, and the next publish
    prunes its pin anyway (``export_reports.update_pins``).
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", f"{base}...HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        sys.exit(f"git diff against '{base}' failed — is that ref fetched?\n{diff.stderr.strip()}")
    touched = (root / line for line in diff.stdout.splitlines() if line)
    return sorted(p for p in touched if auto_publishes(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="the ref to diff against, e.g. origin/main")
    ap.add_argument("--keys", action="store_true", help="print export keys instead of notebook paths")
    args = ap.parse_args()

    for nb in changed_reports(args.base):
        print(export_key(nb) if args.keys else nb.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
