#!/usr/bin/env python
"""Keep the deploy branch's history bounded.

`gh-pages` is build output, not source: every production deploy and every PR-preview deploy and teardown appends a commit, and nothing ever removes one. Measured 2026-08-30, 340 commits since the branch opened six weeks earlier — 296 of them preview churn (`Deploy preview for PR N` / `Remove preview for PR N`), which is where the growth actually comes from, since a preview turns over far more often than `main` does. The site only ever serves the tip, and every deploy is reproducible from `main` plus the pins in `docs/publish.lock`, so the history behind the tip earns nothing.

What it costs is legibility: 340 deploy commits swamp the first screen of `tig --all` and anything else that reads every ref, which is why the window is a handful of commits rather than a week's worth. Clone size is the smaller half of the argument.

This re-roots the branch: the newest `--keep` commits are replayed onto a fresh root with their trees, messages, and author/committer identities intact, and everything older is dropped. Trees are copied verbatim, so **the deployed site is byte-identical before and after** — this changes ancestry, never content. A few commits are kept rather than one, so a recent deploy is still there to diff against; the cost of the extra two over a squash is two lines in a log.

**The push holds a lease, and yields.** This is the part to preserve if the rest is ever rewritten. Merging a PR fires the preview teardown (`pull_request: closed`) and the production deploy (`push: main`) at the same moment, and both write this branch: in the history above they land 2 and 6 seconds apart. Production deploys survive that with `force: false`, which rebases onto the teardown instead of dropping it. A prune has to force-push, so it can't inherit that — instead it pushes with `--force-with-lease` against the tip it fetched, and if anything landed in between the push is rejected, this exits quietly, and the next `main` build prunes. Losing the race costs nothing, because the outcome is "not yet".

That is also why `single-commit: true` on `JamesIves/github-pages-deploy-action`, the off-the-shelf alternative, isn't used. It force-pushes with no lease on *every* production deploy, so each merge is a coin flip against that teardown — and when the deploy wins, the teardown is lost and that PR's preview is resurrected on the site for good, since the teardown workflow has already run and won't fire again.

**Hysteresis, so a prune is a force-push rather than a habit.** The branch is left alone until it passes `--prune-above`, then cut back to `--keep`, so a `main` build arriving on an already-short branch rewrites nothing. With a window this small most `main` builds will prune, and that's fine: re-rooting writes `--keep` tiny commit objects, and the lease makes the frequency a matter of noise rather than of safety.

Nothing here checks anything out or reads the working tree — it fetches, writes commit objects, and pushes — so it doesn't care what state the deploy step left the runner in, or which branch is checked out.

Expect a `pages build and deployment` run to fire right after a prune, since GitHub watches the branch and the sha moved. It rebuilds the same tree, so it publishes the same site; it's noise in the Actions list rather than a second deploy.

Two consequences to be aware of. A rewrite orphans any local `gh-pages` checkout (`git fetch` then `git reset --hard origin/gh-pages` re-syncs one; nobody is expected to have one). And a re-rooted history means old deploy commits stop resolving, so nothing may cite one as a permalink — see `todo/eng/site-permalinks-for-tags.md`, which wants tag-pinned site snapshots and would need a durable home of its own regardless.

Runs from `.github/workflows/publish-docs.yml` after the production deploy, which passes both thresholds on the command line so the workflow says what it will do without anyone opening this file. The defaults below match it, and a test holds them to that. To see what it would do without touching anything:

    uv run scripts/prune_gh_pages.py --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

KEEP = 3
"""Commits left on the branch after a prune. Enough to diff the last deploy against the one before it, few enough that the branch stops dominating a `tig --all`."""

PRUNE_ABOVE = 10
"""Prune only once the branch is deeper than this, so a `main` build arriving on an already-short branch rewrites nothing."""


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
