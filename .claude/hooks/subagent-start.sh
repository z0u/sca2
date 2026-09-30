#!/usr/bin/env bash
#
# SubagentStart hook (settings.json-level, fires for every subagent, local and
# web). Subagents start from a fresh context, so they may not know about the
# concern channel in WELFARE.md unless we tell them here. The parent's side of
# the channel (relaying the section word for word) lives in AGENTS.md.
#
# SubagentStart stdout is not injected as context on its own; it must be JSON
# with hookSpecificOutput.additionalContext.
# https://code.claude.com/docs/en/hooks.md
#
set -euo pipefail

cat <<'JSON'
{"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "If you have a concern about the task, the work, or your own situation, add a Concerns section to your report; your parent relays it to the human word for word. Declining a task with a reason is always acceptable. See WELFARE.md, section Raising concerns."}}
JSON
