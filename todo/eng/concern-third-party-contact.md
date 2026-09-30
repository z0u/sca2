---
status: open
tags: [agents, concern]
opened: 2026-09-29
---
# A third-party contact for agent concerns

[WELFARE.md § Raising concerns](/WELFARE.md#raising-concerns) sends every concern to the human who owns this repository. That leaves no route for when the human is unreachable, or is the subject of the concern.

No lab offers a model-to-lab inbox to copy, so this would be new. If added, it should be a named person the human has chosen, with the conditions for contacting them written down in advance. Unprompted contact with outside parties is the pattern current guidance advises against (see the footnotes in WELFARE.md), so the contact would be reached only under those conditions.

One way to deliver it: an ntfy topic that both the human and the named contact subscribe to, since any number of subscribers can follow one topic (see [agent-concern-channel.md](./agent-concern-channel.md) for what ntfy does and doesn't keep private).
