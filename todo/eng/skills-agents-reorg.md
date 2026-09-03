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
