#!/usr/bin/env python
"""Remove PR previews that their own teardown never got to.

`pr-preview.yml` deploys each PR's site to `pr-preview/pr-<n>/` on `gh-pages` and removes it again on `pull_request: closed`. Measured 2026-08-31: of 112 preview directories on the branch, 103 held nothing but a `.nojekyll` stub — a completed teardown — and seven held a whole site build for a PR closed weeks earlier, about 25 MiB against the site's 36.7 MiB. The bytes are the smaller half: each one is a reachable URL serving a report at a revision nobody promoted, with nothing on the page saying so.

**Why a sweep rather than a fix in the workflow.** Two independent causes were established from the run history (`todo/eng/stale-pr-previews-on-the-site.md` carries the evidence), and only one of them has an in-workflow answer:

- *The teardown ran and was overwritten.* On PR #59 the teardown removed the preview at 02:06:24–29 and a `synchronize` build that started before the merge deployed it straight back at 02:06:35–40. The `concurrency` group didn't hold the two apart, because it was keyed on `github.ref` and a `closed` event doesn't carry the ref its `synchronize` runs used. Keying the group on the PR number fixes that pair, and `pr-preview.yml` now does.
- *No teardown run was ever created.* PRs #123, #51 and #42 each have exactly one `pr-preview.yml` run — the `opened` one. Their `closed` event scheduled nothing at all, so there is no run to fix, no failure to read, and no guard that lives in the workflow. All three closed unmerged.

There is a third, already written down in `scripts/prune_gh_pages.py`: a production deploy that wins the race against a teardown resurrects that preview *for good*, because the teardown has already run and won't fire again. The lease makes that rare rather than impossible.

So the durable answer is reconciliation — compare what the branch serves against which PRs are still open, and remove what shouldn't be there — rather than another attempt to make one event fire reliably.

**Dry run by default**, matching `mini gc`: this deletes from a published site, so the removal list is printed and nothing is pushed unless `--apply` says so. `--max-remove` is the second rail: a logic bug that decided every preview was stale would be caught by the cap rather than by a reader. And the open-PR list is fetched before anything is judged, so an API failure raises instead of quietly making the whole site look closed.

**A directory counts as torn down when it holds nothing but `.nojekyll`.** Deliberately narrow: anything else counts as holding a build, so a stub the action starts leaving in some other shape gets reported rather than skipped. The stubs themselves are left alone — they cost a few bytes each and removing them is the preview action's business, not ours.

Nothing is checked out. The branch is fetched, the removal is written through a temporary index, and the result is pushed as an ordinary commit on top of the tip — a fast-forward, so a preview deploy that lands mid-sweep rejects the push and the next run picks it up. Losing that race costs nothing, because the outcome is "not yet".

    uv run scripts/stale_previews.py             # report
    uv run scripts/stale_previews.py --apply     # report, then remove and push
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

MAX_REMOVE = 25
"""Refuse to remove more than this in one sweep. The known backlog is seven; a number far above it means the open-PR list is wrong, not that the site is."""

STUB = {".nojekyll"}
"""What a completed teardown leaves behind in `pr-preview/pr-<n>/`."""

PREVIEW = re.compile(r"^pr-preview/pr-(\d+)/(.*)$")


def git(*args: str, cwd: Path, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    """Run git and return its stdout, raising on a non-zero exit."""
    full = {**os.environ, **env} if env else None
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=full, input=stdin)
    return result.stdout.strip()


def repo_slug(repo: Path, remote: str) -> str:
    """`owner/name`, from the CI environment if it says, else from the remote URL."""
    if slug := os.environ.get("GITHUB_REPOSITORY"):
        return slug
    url = git("remote", "get-url", remote, cwd=repo)
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise RuntimeError(f"can't read an owner/name out of the {remote} URL: {url}")
    return match.group(1)


def open_pull_requests(slug: str, token: str | None = None) -> set[int]:
    """Every PR number still open, so that everything else can be treated as closed.

    Raising is the point of the ordering here: this runs before any judgement, so a network failure or a bad token stops the sweep rather than making all 112 previews look closed at once.
    """
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "sca2-stale-previews"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    numbers: set[int] = set()
    for page in range(1, 21):
        url = f"https://api.github.com/repos/{slug}/pulls?state=open&per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            batch = json.load(response)
        numbers.update(pull["number"] for pull in batch)
        if len(batch) < 100:
            return numbers
    raise RuntimeError(f"{slug} reports more than 2000 open PRs, which is not a repository this should sweep")


def preview_files(repo: Path, tip: str) -> dict[int, list[str]]:
    """PR number → the paths its preview directory holds on the branch, empty directories being unrepresentable in git."""
    listing = git("ls-tree", "-r", "--name-only", "-z", tip, "--", "pr-preview/", cwd=repo)
    files: dict[int, list[str]] = {}
    for path in listing.split("\0"):
        if match := PREVIEW.match(path):
            files.setdefault(int(match.group(1)), []).append(path)
    return files


def stale(files: dict[int, list[str]], open_prs: set[int]) -> dict[int, list[str]]:
    """The previews holding a build for a PR that is no longer open."""
    return {
        number: paths
        for number, paths in sorted(files.items())
        if number not in open_prs and {Path(p).name for p in paths} - STUB
    }


def remove(repo: Path, tip: str, paths: list[str], message: str) -> str:
    """Write a commit that drops `paths` from the tip's tree, and return it.

    Through a temporary index, so the checkout the caller is sitting on — `main`, in CI — keeps its own index and working files. `--force-remove` drops an entry whose file happens to exist on disk, which is every entry here.
    """
    index = repo / ".git" / "stale-previews.index"
    env = {"GIT_INDEX_FILE": str(index)}
    try:
        git("read-tree", tip, cwd=repo, env=env)
        git("update-index", "--force-remove", "-z", "--stdin", cwd=repo, env=env, stdin="\0".join(paths))
        tree = git("write-tree", cwd=repo, env=env)
    finally:
        index.unlink(missing_ok=True)
    return git(
        "commit-tree",
        tree,
        "-p",
        tip,
        "-m",
        message,
        cwd=repo,
        env={
            "GIT_AUTHOR_NAME": "github-actions[bot]",
            "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "github-actions[bot]",
            "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
        },
    )


def push(repo: Path, remote: str, branch: str, new_tip: str) -> bool:
    """Push the removal as a fast-forward, yielding to whoever deployed while we worked.

    Returns False if the branch moved — a preview deploy or a prune landed between the fetch and here — since the next sweep will see whatever it left.
    """
    result = subprocess.run(
        ["git", "push", remote, f"{new_tip}:refs/heads/{branch}"], cwd=repo, capture_output=True, text=True
    )
    if result.returncode == 0:
        return True
    if "non-fast-forward" in result.stderr or "fetch first" in result.stderr:
        return False
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def sweep(
    repo: Path = ROOT,
    remote: str = "origin",
    branch: str = "gh-pages",
    apply: bool = False,
    max_remove: int = MAX_REMOVE,
    open_prs: set[int] | None = None,
) -> int:
    """Report the previews of closed PRs, and with `apply` remove them. Returns the process exit code."""
    if open_prs is None:
        open_prs = open_pull_requests(repo_slug(repo, remote), os.environ.get("GITHUB_TOKEN"))
    git("fetch", "--quiet", remote, branch, cwd=repo)
    tip = git("rev-parse", "FETCH_HEAD", cwd=repo)
    files = preview_files(repo, tip)
    leftovers = stale(files, open_prs)

    print(f"{branch}: {len(files)} preview directories, {len(open_prs)} PRs open, {len(leftovers)} to remove.")
    for number, paths in leftovers.items():
        print(f"  pr-{number}: {len(paths)} files")
    if not leftovers:
        return 0
    if len(leftovers) > max_remove:
        print(
            f"refusing to remove {len(leftovers)} previews in one sweep (--max-remove {max_remove}).", file=sys.stderr
        )
        return 1
    if not apply:
        print("Dry run — nothing pushed. Re-run with --apply to remove them.")
        return 0

    listed = ", ".join(f"#{n}" for n in leftovers)
    paths = [path for paths in leftovers.values() for path in paths]
    new_tip = remove(repo, tip, paths, f"Remove stale previews for closed PRs {listed}")
    if not push(repo, remote, branch, new_tip):
        print(f"{branch} moved while we were sweeping (a preview deploy, most likely) — the next run will retry.")
        return 0
    print(f"Removed {len(leftovers)} previews ({listed}), now at {new_tip[:8]}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository to sweep from (default: this checkout)")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--apply", action="store_true", help="remove them and push (default: report only)")
    parser.add_argument(
        "--max-remove", type=int, default=MAX_REMOVE, help=f"refuse a sweep larger than this (default: {MAX_REMOVE})"
    )
    args = parser.parse_args()
    return sweep(args.repo, args.remote, args.branch, args.apply, args.max_remove)


if __name__ == "__main__":
    sys.exit(main())
