---
status: open
tags: [agents, skills]
bundle: skills-workflow
---
# Skills/agents reorg

(see AGENTS.md model-routing section for context):

- Add a routing table for writing-adjacent tooling — one front door (`report-review` for reports; a short "which tool when" list for the rest) so the entry point is discoverable on the human's turn.
- The `.agents/skills` directory mixes three kinds of thing: conventions (writing, style-*, alt-text, science), runbooks that fork and drive agents (report-review, report-restructure), and references (report-render, mi-ni). Consider naming or a one-line "kind:" tag in each SKILL.md to make the taxonomy visible.
- Consider extracting the transferable set (writing, style-*, alt-text, text-lint, report pipeline skills + agents) into a plugin for reuse in the next milestone repo, keeping repo-specific skills (mi-ni, science conventions) local. Decide after the routing table has settled, so the plugin ships a stable interface.

## Notes

**2026-09-06, one data point** — `storage-envs` is gone: its setup runbook folded into the `mi-ni` skill's storage reference, which already carried the mechanism and had to link out to it. So one runbook dissolved into a reference rather than being tagged as its own kind. Worth weighing when the taxonomy question above is settled — a runbook that only ever fires once per project may be better off as a section of the reference its readers already have open.

**2026-10-09, skills refactor** — No plugin, decided in the skills refactor thread. A plugin can live inside the repo, but a cloud session does not install the plugins a repository turns on under `enabledPlugins`, and the project's threads run in the cloud, so the skills would stop loading there. Plain `.claude/skills/` and `.claude/agents/` folders come with the clone. The third bullet above is settled on that basis.
