#!/usr/bin/env python
"""Keep the deploy branch's history bounded.

`gh-pages` is build output, not source: every production deploy and every PR-preview deploy and teardown appends a commit, and nothing ever removes one. Measured 2026-08-30, 340 commits since the branch opened six weeks earlier — 296 of them preview churn (`Deploy preview for PR N` / `Remove preview for PR N`), which is where the growth actually comes from, since a preview turns over far more often than `main` does. The site only ever serves the tip, and every deploy is reproducible from `main` plus the pins in `docs/publish.lock`, so the history behind the tip earns nothing and costs a bigger clone for everyone who checks the branch out.

This re-roots the branch: the newest `--keep` commits are replayed onto a fresh root with their trees, messages, and author/committer identities intact, and everything older is dropped. Trees are copied verbatim, so **the deployed site is byte-identical before and after** — this changes ancestry, never content. The commits are replayed in order rather than squashed into one, so `git log gh-pages` still shows the recent deploys and you can still diff two of them.

Two decisions worth knowing about:

- **Hysteresis, not a fixed ceiling.** Re-rooting rewrites every kept commit's sha, so pruning on every build would force-push a fresh set of commit objects each time for no gain. Instead the branch is left alone until it passes `--prune-above`, then cut back to `--keep`. At the observed ~6 commits/day that's a rewrite every few weeks, and the branch stays somewhere between one and four weeks deep.

- **The push holds a lease, and yields.** Production deploys run with `force: false` precisely so they rebase onto a concurrent preview deploy instead of dropping it, and previews share this branch under `pr-preview/`. A prune has to force-push, so it can't inherit that protection — instead it pushes with `--force-with-lease` against the tip it fetched. If a preview deploy lands in between, the push is rejected, this exits quietly, and the next `main` build prunes instead. The window is seconds and the outcome of losing the race is "not yet", so the race needs no cleverer handling than that.

`single-commit: true` on `JamesIves/github-pages-deploy-action` was the off-the-shelf alternative. It force-pushes too, so it needed the lease anyway, and it wipes the history entirely rather than keeping a window — no recent deploy left to diff against, and every preview standing on the branch rewritten on every build rather than every few weeks.

Two consequences to be aware of. A rewrite orphans any local `gh-pages` checkout (`git fetch` then `git reset --hard origin/gh-pages` re-syncs one; nobody is expected to have one). And a re-rooted history means old deploy commits stop resolving, so nothing may cite one as a permalink — see `todo/eng/site-permalinks-for-tags.md`, which wants tag-pinned site snapshots and would need a durable home of its own regardless.

Runs from `.github/workflows/publish-docs.yml` after the production deploy. To see what it would do without touching anything:

    uv run scripts/prune_gh_pages.py --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

KEEP = 40
"""Commits left on the branch after a prune — about a week of deploys."""

PRUNE_ABOVE = 120
"""Prune only once the branch is deeper than this, so a rewrite costs a force-push every few weeks rather than every build."""


def git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    """Run git and return its stdout, raising on a non-zero exit."""
    full = {**os.environ, **env} if env else None
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=full).stdout.strip()


def fetch_tip(repo: Path, remote: str, branch: str) -> str:
    """Fetch the deploy branch and return the sha we found it at — the value the push later holds a lease against."""
    git("fetch", "--quiet", remote, branch, cwd=repo)
    return git("rev-parse", "FETCH_HEAD", cwd=repo)


def replay(repo: Path, tip: str, keep: int) -> str:
    """Rebuild the newest `keep` commits as a history of their own, and return the new tip.

    Each replayed commit keeps its predecessor's tree, message, and identities, so only the ancestry changes. Identities are carried through rather than restamped, which also makes this deterministic: the same window of commits always replays to the same shas, so a prune that finds nothing new to cut pushes nothing new.
    """
    parent: str | None = None
    for commit in git("rev-list", "--reverse", f"-n{keep}", tip, cwd=repo).split():
        tree = git("rev-parse", f"{commit}^{{tree}}", cwd=repo)
        message = git("log", "-1", "--format=%B", commit, cwd=repo)
        an, ae, ad, cn, ce, cd = git("log", "-1", "--format=%an%n%ae%n%aI%n%cn%n%ce%n%cI", commit, cwd=repo).split("\n")
        parent = git(
            "commit-tree",
            tree,
            *(["-p", parent] if parent else []),
            "-m",
            message,
            cwd=repo,
            env={
                "GIT_AUTHOR_NAME": an,
                "GIT_AUTHOR_EMAIL": ae,
                "GIT_AUTHOR_DATE": ad,
                "GIT_COMMITTER_NAME": cn,
                "GIT_COMMITTER_EMAIL": ce,
                "GIT_COMMITTER_DATE": cd,
            },
        )
    assert parent, "rev-list returned nothing for a branch we counted"
    return parent


def push(repo: Path, remote: str, branch: str, new_tip: str, leased_on: str) -> bool:
    """Force-push the replayed history, yielding to whoever deployed while we worked.

    Returns False if the lease was stale — a preview deploy landed between our fetch and this push, and the next build will prune instead. Any other failure raises, since it is not something waiting will fix.
    """
    result = subprocess.run(
        [
            "git",
            "push",
            f"--force-with-lease={branch}:{leased_on}",
            remote,
            f"{new_tip}:refs/heads/{branch}",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if "stale info" in result.stderr:
        return False
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def prune(
    repo: Path = ROOT,
    remote: str = "origin",
    branch: str = "gh-pages",
    keep: int = KEEP,
    prune_above: int = PRUNE_ABOVE,
    dry_run: bool = False,
) -> int:
    """Trim the deploy branch if it has grown past the threshold. Returns the process exit code."""
    tip = fetch_tip(repo, remote, branch)
    depth = int(git("rev-list", "--count", tip, cwd=repo))
    if depth <= prune_above:
        print(f"{branch}: {depth} commits, under the {prune_above} threshold — leaving it alone.")
        return 0

    new_tip = replay(repo, tip, keep)
    if dry_run:
        print(f"{branch}: {depth} commits — would re-root at {new_tip[:8]}, keeping {keep} (dry run, nothing pushed).")
        return 0

    if not push(repo, remote, branch, new_tip, leased_on=tip):
        print(
            f"{branch}: moved while we were pruning (a preview deploy, most likely) — skipping, the next build will retry."
        )
        return 0

    print(f"{branch}: {depth} commits re-rooted to the newest {keep}, now at {new_tip[:8]}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository to prune in (default: this checkout)")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--keep", type=int, default=KEEP, help=f"commits to keep after a prune (default: {KEEP})")
    parser.add_argument(
        "--prune-above", type=int, default=PRUNE_ABOVE, help=f"prune only past this depth (default: {PRUNE_ABOVE})"
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would happen; push nothing")
    args = parser.parse_args()
    if args.keep > args.prune_above:
        parser.error(f"--keep {args.keep} exceeds --prune-above {args.prune_above}: a prune would deepen the branch")
    return prune(args.repo, args.remote, args.branch, args.keep, args.prune_above, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
