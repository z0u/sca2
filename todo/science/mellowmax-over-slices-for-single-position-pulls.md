---
status: open
tags: [D2.2, anchoring, ex-2.2.21]
opened: 2026-10-02
---
# Pool over slices when the pull has one position

The anchor term pools over positions with mellowmax, so the model can choose where along a context the concept sits, and it averages over slices. When the pull acts on one position only, as `query-eq` does in [ex-2.2.21](/docs/m2/ex-2.2.21/report.py), the question of where moves from the position axis to the slice axis, and the term gives the model no such choice there: every slice at that position is pulled alike, the embedding slice included. In `query-eq` the result was the `=` embedding at about 0.96 on e₁ for every op, which makes the anchor at the query `=` unselective from the bottom of the stack up.

So try mellowmax over slices too (perhaps for every label, since which slice holds an abstract op is as unknown as which position does). Leaving the embedding slice out (see [the slice-restriction item](./pull-slice-restrictions-at-more-seeds.md)) is the cheaper fix for the embedding case; pooling over slices is the general one.

## Notes

**2026-10-06, Claude** — The [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) decision keeps every slice in the pull, since restricting the slices by hand made the edit spill (E1), and names pooling over slices as the next idea for where the pull should act. Sandy would like to try it.
