---
status: done
tags: [D2.2, anchoring, in-context, ex-2.2.22]
opened: 2026-10-05
closed: 2026-10-07
---
# Drop runs that miss the second rise?

In [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E5) most runs rise quickly to a first plateau of task skill, stay there for some tens of epochs, and then rise again. A few runs never make the second rise within 200 epochs and end well below the others. Those runs pull the seed means of every measurement around, so the seed band is mostly a record of which runs made it. If slow runs all make the second rise eventually, a future experiment could keep the 200-epoch budget and leave out the runs that miss it, as half-trained models, topping up from the next unused seed.

Two things need checking before that policy is safe.

**Do slow runs get there?** Train the runs that missed the second rise for longer (300 or 400 epochs), and ask whether they make it, and whether a run that rises late ends up like one that rises early (task skill, op margin, and the edit criteria). If some never rise, the policy still works, though it is then choosing a kind of seed rather than waiting for a slow one.

**Does the treatment decide who is slow?** It does, at least partly. In ex-2.2.21 and ex-2.2.22 the run at model seed 702 missed the second rise on most conditions that pull the embedding slice, and made it on the slice restrictions and on the control. So slowness depends on the seed and the condition together, and screening seeds on the control would let treatment-made slowness through. Under a per-run gate, the drop count of each condition becomes a measurement in its own right: a condition that sends more runs down the slow path is less reliable to train, and that should be reported beside its scores, never hidden by the top-up.

The gate itself should be on task skill only (for example held-out exact match by epoch 200), fixed before the anchoring results are seen, so it cannot select on the outcome. That matters here because the edit spills so far appear only on runs that rose quickly: hinge and `cap-0.95` at model seed 700, and the replicate run at 703, where the edit takes `darken` down by many times the gate. Leaving out slow runs will probably raise the spill rate that a report sees. That would be the rate for trained models, which is the one we want, but only if the gate was chosen blind to it.

A longer run of the spilling conditions would answer a related question at the same cost: whether slow runs spill once they make the second rise, or whether spilling belongs to some fast runs only. [pull-slice-restrictions-at-more-seeds](./pull-slice-restrictions-at-more-seeds.md) proposes longer training at more seeds for the slice restrictions, so the two could share runs.

## Notes

**2026-10-07, ex-2.2.23** — [Ex-2.2.23](/docs/m2/ex-2.2.23/report.py) answered this. Every run that missed the rise at 200 epochs made it at 400 and ended with the early risers. But the runs a rule would drop are the runs whose edit stays clean: the edit spills on nearly every run that learned the HSV ops. So the decision is no drop rule; training keeps every run, at 400 epochs. Why the edit spills on fully trained runs is the next question.
