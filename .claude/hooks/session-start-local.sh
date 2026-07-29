#!/usr/bin/env bash
#
# SessionStart hook for local (devcontainer/non-web) Claude Code sessions.
#
# The web hook (session-start.sh) bootstraps tooling for the cloud runtime and
# is a no-op locally. This is its local counterpart: no tooling to bootstrap
# (the devcontainer handles that), just a small orientation note about the
# machine's resources. Model-default context reports platform/shell/OS but
# not RAM/swap/disk, and subagents doing memory-heavy work (e.g. loading a
# model checkpoint) have caused swap exhaustion before — this makes the
# headroom visible up front instead of discovered mid-task.
#
set -euo pipefail

# Local-only. The web hook covers CLAUDE_CODE_REMOTE=true.
if [[ "${CLAUDE_CODE_REMOTE:-}" == 'true' ]]; then
    exit 0
fi

command -v free >/dev/null 2>&1 || exit 0

mem="$(free -h | awk '/^Mem:/ {print $7" free / "$2" total"}')"
swap="$(free -h | awk '/^Swap:/ {print $3" used / "$2" total"}')"
disk="$(df -h / 2>/dev/null | awk 'NR==2 {print $4" free / "$2" total ("$5" used)"}')"

echo "## Environment"
echo
echo "RAM: $mem · Swap: $swap · Disk (/): $disk"
echo "Consider the available resources before launching jobs on this machine."
