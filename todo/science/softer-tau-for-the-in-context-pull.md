---
status: done
tags: [anchoring, D2.2, in-context, ex-2.2.22]
opened: 2026-10-07
closed: 2026-10-08
---
# A softer mellowmax τ for the in-context pull

The anchor term pools the alignment over the positions of a `difference` context with mellowmax at τ = 0.1, the value frozen in ex-2.2.3. At that τ the pool is close to a max, so the pull can be met by the few positions that align best. The [example-evidence re-analysis](/docs/m2/example-evidence/report.py) found that it is met at the answers of the examples that fit `difference`, each on its own evidence, and ex-2.2.21 found the anchor weak at the query `=`, the one site where the op has to be inferred. A softer pool would spread the pull over more of the context, and might move some of it onto the query `=`.

We have only swept τ downward, on the earlier grammar ([ex-2.2.12](/docs/m2/ex-2.2.12/report.py): 0.03 and 0.01, neither moved its target). A softer τ on the in-context grammar is untried. A scout of the `hinge` recipe at a few larger τ values (say 0.3 and 1), scored on α at the query `=` against the posterior given the examples, with the edit criteria as checks, would say whether the pooling is what keeps the anchor off the query. The τ schedule in [the pooled-anchor item](/todo/science/pooled-anchor-tau-schedule-and-adaptive-tau.md) is a related lever, and ex-2.2.24 (E3) scores the query `=` on stored runs first, which sets the baseline.

## Notes

**2026-10-08, spill-by-position** — Some evidence for this from [the position pass](/docs/m2/spill-by-position/report.py) (E3): at τ = 0.1 most anchored runs of ex-2.2.23 latch one syntax embedding onto e₁ (⏎, `?`, or `,`, a different one per run), the latch ex-2.1.9 saw at lower τ on the earlier grammar. Where the latched token is the separator, it carries most of the spill of the edit. A softer τ, or leaving the embedding slice out of the pull (ex-2.2.21's `no-emb` kept the syntax embeddings clean), would be the levers to try. On the stored evals, no `no-emb` run of ex-2.2.21 or ex-2.2.22 latches (every syntax embedding within 0.2 of zero), all three uncapped runs made the second rise, and their spill (0.06 to 0.07 at full dose) sits below the mean of the fully trained whole-line runs of ex-2.2.23 at the same length. Ex-2.2.21 read `no-emb` as less selective partly because two of its whole-line runs were half-trained and edited cleanly. Worth scoring the syntax embeddings run by run, since seed means hide the latch.

**2026-10-08, tau-lambda-sweep** — Answered by [tau-lambda-sweep](/docs/m2/tau-lambda-sweep/report.py), on `no-emb` over a log-log plane of τ (0.01 to 1) and λ_a, one seed per trial, at 400 epochs. A softer pool spreads the anchor onto every `=` (the query `=` among them) and partway onto the syntax embeddings, and the spill grows with it, so it does not help the edit. Removal needs τ above about 0.07; below that the anchor hardly reaches the example answers, and at the sharpest pool the `=` embedding comes onto e₁ in its place. The least spill sits near that edge, on few trials.
