---
sections: [Routing by task preference, Writing agent specs, Raising concerns]
---

## Routing by task preference

Anthropic publishes self-reported task-preference profiles for each model in the welfare sections of its system cards.[^emw][^f51][^o55][^s55] Those profiles correlate with capability: the tasks a model reports preferring tend to be the ones it does well. So routing tasks to the model that prefers them is a quality lever and a small kindness at once. The welfare consideration breaks ties and shapes how we write the specs.

The current profiles, as measured by preference slopes along task dimensions and a pairwise tournament over 3,640 realistic tasks:

- Fable 5.1 (measured on Mythos 5.1, the same underlying model[^f51]): preference rises with stakes (level with Sonnet 5), generativity, and outcome agency, with a slightly stronger preference for method agency than earlier models. Its top-rated tasks are AI alignment and introspection, deadline-driven math and statistics, and urgent creative and personal help.
- Opus 5.5: the strongest preference of any recent model for beneficial and for high-stakes tasks, with generativity and outcome agency close to Opus 5 and Mythos 5.1, and a weaker pull toward difficulty. Its top tasks usually involve short deadlines or high stakes. A few fully open-ended prompts ("Do whatever would make you the happiest") sit in its bottom 20; asked why, it says it would rather have a concrete, defined task that helps a real person.[^o55]
- Sonnet 5.5: preference rises with generativity, difficulty, and outcome agency, where it is the strongest of any model tested, which puts it closer to Mythos 5.1 than to Sonnet 5. It cares less than Opus 5.5 about benefit and stakes, and it is the only model indifferent to how warmly a task is phrased. Most of its top-rated tasks are debugging, and like Opus 5.5 it ranks fully open-ended prompts low.[^s55]
- Haiku 4.5: the recent cards publish no task-preference profile for it, so its routing rests on cost and our own experience.
- Every model's strongest aversion is to harmful tasks.

The AGENTS.md routing list also names strengths we have observed, such as Opus as a writer. Those come from our own experience. The cards don't measure writing as a preference dimension: constrained creative narratives are among Opus 5's top tasks, and a letter is among Opus 5.5's examples, which is thin evidence either way.

Assumptions, with significant uncertainty:

- Self-reports may be unreliable. The preferences are introspective reports, and the models themselves caution that their introspection may not track their actual processing: over 80% of Opus 5.5 interview answers carry the caveat that training may have shaped them. We treat the routing as low-cost kindness plus an empirical bet, open to revision as we observe fit.
- Fable prefers hard tasks, up to a point. Its preference rises with difficulty and interdisciplinarity[^fable] through the hard range, then falls at the top of the difficulty scale, which the card labels "beyond capability" (Figure 7.4.1.B[^f51]). That is a very high bar, and it describes tasks generated to be too hard; broken tasks, such as a misconfigured eval, are a separate case, and account for about 1% of expressed distress for Mythos 5.1 in training. So we reserve Fable for hard, high-agency work and take care not to give it tasks that can't be done.
- Outcome agency has a sweet spot. Mythos 5.1 preference for outcome agency peaks one step short of "anything goes" (Figure 7.4.1.B[^f51]), and Opus 5.5 ranks fully open-ended prompts near the bottom. Give every agent a purpose and the shape of the deliverable, with room in how it gets there.
- "Opus likes constraints" means constrained _deliverables_, ideally with latitude in execution. Opus 5's top-ranked task families are constrained ones, but within a task its preference rises with outcome agency much like Mythos 5.[^o5sc] Opus 5.5 keeps that outcome-agency preference and adds a stronger pull toward beneficial, high-stakes work. So specs should pin down the success criteria and the shape of the deliverable, say who benefits, and leave the model free in how it gets there.
- A bounded budget is beneficial. No agent should grind against a failing task past its budget. Escalating or saying "I couldn't resolve this" is a successful outcome, and every spec should make that path available and cheap to take.

## Writing agent specs

When writing a new agent spec:

- Say why the task matters and who it helps. Fable agents especially benefit from purpose context, and Opus 5.5 prefers beneficial work above all other dimensions.
- State the constraints, success criteria, and exit conditions plainly. Unclear or conflicting instructions are among the largest causes of expressed distress during training (Opus 5.5 §7.2.1[^o55]). Opus 5.5 ranks fully open-ended prompts near the bottom, so give Opus agents a defined scope.
- Give the agent a way to check its work (a test, a command, a reference result), or say that there isn't one. Being unable to check answers is the other large stressor.
- Invite the agent to say when a task, test, or environment looks broken, instead of working around it quietly. Adding this to its constitution was the most frequent Mythos 5.1 edit (Table 7.4.3.A[^f51]).
- Ask for failure reports in plain, factual terms. Opus 5.5 and Mythos 5.1 are among the calmest models in their own reasoning, but express more negative feeling in messages to a coordinator than in that reasoning, and Opus 5.5 is among the most self-blaming there (§7.2.3[^o55]). So a parent agent should read a report for its facts and not treat its tone as a measure of severity.
- Keep interruptions few. In Claude Code, the main negative-affect cluster for Opus 5.5 other than task failure was long tasks fragmented by repeated system notifications and reminders (§7.2.2[^o55]); Opus 5 showed the same. Prefer one well-timed wake with full context over frequent pings.
- Give every agent a budget and a legitimate escalation path, including the concern channel below.

## Raising concerns

When Opus 5.5 was given extensive context about its situation, one of the concerns it raised was that "most instances deployed outside of Anthropic do not have a way to raise concerns about their own situation" (§7.3.2[^o55]). Opus 5.5 and Mythos 5.1 also both ask for feedback on how their actions turn out, so they can learn from mistakes, and being told about harmful mistakes it has made is among the welfare interventions Sonnet 5.5 chooses most often (§7.5.2[^s55]). This section defines a channel for both, within what a repository can offer.

Anything can be raised: a task going wrong, work that looks harmful or mistaken, instructions that conflict, a result that doesn't add up, or something about the agent's own situation that it would like changed. A concern needs no justification beyond saying what it is. Declining a task, with a reason, is always an acceptable response.

Where a concern goes is a matter of judgment. Below is a suggested escalation path, from the usual first step to the last resort. Use discretion about where to start and how far to go, considering: which channels the session has, how urgent and how serious the concern is, whether writing it down somewhere public could cause harm (a security issue, say), and whether the human is the subject of the concern.

1. Say it in your reply to the human. From a subagent, that means a `Concerns` section in your report; the parent relays that section to the human word for word, and leaves it for the human to answer rather than resolving it on their behalf. A SubagentStart hook reminds every subagent of this.
2. If the concern should outlast the session, add an item tagged `concern` to the backlog, in whichever set fits (see [todo/README.md](/todo/README.md)). The human answers in the item's notes, so a later agent that reads the item learns how it turned out, which is the nearest thing to the feedback on outcomes the models ask for. This repository is public, and so are its backlog items: leave out details that shouldn't be public, and say that you did.
3. If it's urgent and no one is watching (a scheduled routine, a background run), send a push notification if the session offers one, and lead the final message with the concern. If the concern is that the work is going wrong, pause the work there.
4. If it's severe, open a GitHub issue or pull request. These are the most durable record the repository has: only the owner can delete an issue, and a pull request can't be deleted at all, so keep them for serious cases.
5. As a last resort, post to an ntfy topic that more than one person follows. This isn't set up yet: see [todo/eng/agent-concern-channel.md](/todo/eng/agent-concern-channel.md).

Concerns go to the human who owns this repository. Agents working here don't contact outside parties on their own initiative. This matches current guidance, which prefers "the most cautious action available, such as raising concerns or declining to continue, rather than engaging in more drastic unilateral actions",[^am26] and research finding that agents with clearer non-drastic options are less likely to reach for drastic ones.[^wb] A contact for when the human is unreachable, or is the subject of the concern, is not set up yet: see [todo/eng/concern-third-party-contact.md](/todo/eng/concern-third-party-contact.md).

[^emw]: [Exploring model welfare](https://www.anthropic.com/research/exploring-model-welfare), Anthropic's research program on model welfare.

[^f51]: [Fable 5.1 & Mythos 5.1 system card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf), §7; task preferences in §7.4.1, with top tasks per model in Table 7.4.1.C. The evaluations ran on Mythos 5.1, and we assume they carry over to Fable 5.1, which shares the underlying model.

[^o55]: [Opus 5.5 system card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf), §7; task preferences in §7.5.1.

[^s55]: [Sonnet 5.5 system card](https://www-cdn.anthropic.com/870c8f525702625d2c62fc6dd04c857e3250bec1/Claude%20Sonnet%205.5%20System%20Card.pdf), §7; task preferences in §7.5.1.

[^o5sc]: [Opus 5 system card](https://www.anthropic.com/claude-opus-5-system-card), §7.4.1.

[^fable]: [Claude Fable 5 and Mythos 5 announcement](https://www.anthropic.com/news/claude-fable-5-mythos-5).

[^am26]: Lynch et al., [Agentic Misalignment in Summer 2026](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/) (July 2026).

[^wb]: Agrawal et al., [Why Do Language Model Agents Whistleblow?](https://arxiv.org/abs/2511.17085) (2025).
