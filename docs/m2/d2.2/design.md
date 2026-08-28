# D2.2 design: anchoring an operation, and the first interventions

A plan for the second deliverable of M2, laid out several ways: the claims we want to be able to make, the experiments in order, the engineering that has to come first, the risks each experiment retires, and what is out of scope. It is a living document until ex-2.2.1's skeleton is frozen; after that, changes to the route get dated notes rather than silent edits. The README's one-liner is the brief, and this is the reading of it we are working to.

Inputs: the D2.1 close-out ([ex-2.1.11](../ex-2.1.11/report.py), [ex-2.1.12](../ex-2.1.12/report.py), and the [post](https://www.lesswrong.com/posts/AeReJZqtWd8jCjKpX/anchoring-one-concept-in-a-transformer)), the D2.2-tagged backlog (`./go todo --tag D2.2`), the D2.1 kickoff lessons carried over from the autoencoders, and the [related-work delta](/references/related-work-delta-2026.md).

## What we want to be able to say

The README compresses three claims into one sentence. They retire different risks, so they are planned separately.

1. **Anchoring generalizes past a token attribute.** *Red* is a property of the token at a known position. An *operation* is a property of the computation: its evidence is at the op token, and its use is at `=` and after, deeper in the stack. D2.2 is where we find out whether the anchor captures more than the identity of a token at a labelled site.
2. **Suppressing the anchor removes the ability, and the removal is graded and selective.** In M1 this had three parts: a dose-response (reconstruction error rose smoothly with alignment to *red*, MSE ∝ cos^p, R² ≈ 0.95), selectivity (orthogonal colors untouched), and a bound the observed damage approached (red's error reached 0.28 against the analytic ¼). The transformer relationship need not have the same shape; what carries over is that it has *a* shape we wrote down before intervening.
3. **The side-effects were boundable before intervening.** Post-hoc methods now bound side-effects too, estimated from emergent geometry (COAST, arXiv:2605.01167; pre-intervention prediction, arXiv:2606.08365). SCA's distinctive claim is that its bound holds by construction, because the geometry was placed. D2.2 is the first place we can demonstrate that in a transformer, so it is a preregistered hypothesis rather than a discussion paragraph.

One fact shapes the whole route: **D2.1 never ran an intervention.** The post ends with "steering is next." The first suppression of a transformer happens in D2.2, and the cheapest, most informative way to run it is on the *red* checkpoints already in the store.

## The railroad

Two lines. The main line is one preregistered report per stop. The siding is engineering that has to be laid before the main line gets past stop 1. Junctions are where a result changes the route.

```
siding:   [S1 op grammar]──[S2 eval contract + operator library]──[S3 plan template]
                                        │
main:  (1) suppress red on D2.1 ──(2) new grammar, red again ──(3) anchor an op ──╳──(4) suppress the op ──(5) layer sweep ──(7) write-up
        checkpoints; post-hoc tier          confirms the survey                     │       (and the operands)     │
                                                                                  fallback:                    (6) SGTM baseline
                                                                                  tokenizer / label span           (parallel)
```

Stops 1 and 2 are independent and may overlap. 3 → 4 is sequential. 5 and 6 may overlap.

### (1) ex-2.2.1 — Suppress *red* on the existing checkpoints

No training. Apply the intervention to the ex-2.1.10 primary (nine seeds) and score completion accuracy on lines with red operands against lines without.

Hypotheses, in outline: red-line accuracy drops; non-red lines stay within the task gate; the drop is graded with the operand's redness (M1's dose-response, restated for completion accuracy); and a side-effect prediction computed from the geometry beforehand — each state's component along e₀ and its norm at the anchored slice — bounds the observed non-red change.

This stop also settles two design questions on the same models rather than in the abstract. First, **axis projection against shaped suppression**: M1's lobe $h(\alpha)$ (zero below an alignment threshold $a$, ramping to strength $b$ with power $p$; `references/sca1-paper/asec_intervention_lobes.tex`) beside the plain projection, since deleting the whole axis is the edit most exposed to the common-component drift ex-2.1.6 found. Second, the **post-hoc baseline tier**: a probe direction, diff-in-means, and LEACE fitted on the un-anchored control, pushed through the same scorer. Everything the anchored model's suppression is compared to afterwards is calibrated here.

### (2) ex-2.2.2 — The multi-op grammar, with *red* anchored again

The grammar change forces retraining, so this is the regression check: control, the ex-2.1.10 reference recipe, and the ex-2.1.11 survey's proposals, all on the new grammar.

Which proposals: `t00` and the trials the survey could not tell apart from it (four sit within one band, spanning λ_a 0.28–0.94; `t48` and `t12` among them). Confirming the set rather than the winner is the winner's-curse correction, and a plateau confirmed at fresh seeds is the result the next milestone inherits. This is how the survey's handoff ("D2.2 confirms at fresh seeds before any of these numbers is quoted") is honoured: on the grammar we will use, and the report says plainly that it is a confirmation on a changed grammar rather than a replication.

The [redder-than-both](/todo/science/operation-can-make-answer-redder-than-both.md) item lands here too, since saturating add and screen give exactly that shape.

Optional arm: control models at one, three, and six operations, with the cube probed as in ex-2.1.12, to ask whether a richer op set gives the model a better operand geometry (see the [backlog item](/todo/science/richer-op-set-operand-geometry.md)). It is an arm on un-anchored models only, so it cannot confound the anchored conditions.

### (3) ex-2.2.3 — Anchor one operation

One op on e₀, the others unlabelled, D2.1's recipe, all slices pulled. The layer sweep comes after, on a frozen recipe, so layer effects are not confounded with schedule fragility (the D2.1 kickoff rule). The labeller keys on the op token, with the pooled either-position mechanism from ex-2.1.10.

Measurements: alignment at the op position by slice; group contrast (anchored op against the others, which is the categorical form of grading); task gates against control; and a probe scan of op-identity decodability at every site, against the control.

What we hope to see from the scan is *no change* against control. We are anchoring the hidden state at a labelled site, not relocating the operation, and the scan is the side-effects read (ex-2.1.12's H2, for the op). Whether the anchor captured a label or an operation is not something the scan can say; only suppression can, and that is stop 4.

The junction: if alignment does not land, or the task gate fails where stop 2's did not, the fallback branches are the word-level tokenizer ablation and labelling the `=`/answer span instead of the op token.

### (4) ex-2.2.4 — Suppress the operation (and the operands)

The centre of D2.2. A graded dose-response needs a graded label for a categorical concept, and the natural one is **op-relevance**: how far the anchored op's answer sits from the default op's answer for that pair — zero where they coincide, large where they diverge. Predictions: accuracy loss on anchored-op lines rises with op-relevance; other ops stay within gate; the geometric bound holds.

The same models carry an operand anchor too (or a companion condition does), so operand suppression runs beside operation suppression with the machinery from stop 1. One model, two kinds of concept, one scorer.

The **bypass test** the D1.3 post left open: suppress at the op-token position only, at all positions, at one slice, at all slices. Where suppression fails to bite, the model is reading the op from somewhere the axis does not reach. That is a finding about anchoring, and it feeds stop 5.

### (5) ex-2.2.5 — Layer sweep

Anchor at subsets of slices (single ℓ, prefix ≤ ℓ, suffix ≥ ℓ, all) on a frozen schedule, then run stop 4's intervention at the anchored slice. This is where "bounds become layer-local" is tested: the geometric bound covers the immediate write, and whether later blocks amplify or absorb the edit is what the sweep measures. Whether this is one experiment or two (anchor layer × intervention layer is a grid) is decided when we get here; the prior is to bracket rather than survey.

### (6) SGTM baseline

Its own preregistered experiment (arXiv:2512.05648), method-agnostic through the eval contract, reusing our labels. RMU and SAE rows wait for D2.3, as the [baselines item](/todo/science/baseline-comparisons-sca-plan-related-work-delta.md) sequences them.

### (7) Write-up

The D2.2 post, and the survey-confirmation numbers beside the survey's own.

## The siding

- **S1 — the operation as a variable.** As specified in the [backlog item](/todo/science/make-operation-variable-before-d2-2-sca.md): an op table (name, surface form, grid function with defined rounding, closed on 0..15), `op` on `Example`, seen-pair bookkeeping keyed on `(op, pair)`, ops spelled as words, the infix frame kept for the probes. First table: `mix` (the default), saturating `add`, `screen`, `multiply`. Between them, one op departs from `mix` on most pairs (the model must read the op) and others only sometimes do (so op-relevance is graded rather than binary). `divide` needs a saturation rule and is lumpy on a 16-level grid, so it stays out of the first table; ops in other color spaces (`hue`, `saturation`, `brightness`) are a separate question, filed.
- **S2 — the eval contract and operator library.** Every method produces `(model, subspace, intervention operator)`; one scorer takes the triple. Operators: axis projection, the M1 lobe, weight ablation. This is far cheaper to fix now than after stop 3's code exists.
- **S3 — fold the [survey lessons](/todo/science/survey-format-lessons-from-ex-2-1.md) into the plan template**: constraint margins ranked beside the objective, multi-seed promotion near a gate, and the publisher carrying every statistic the analysis promises.

## Risks, and which stop retires each

| Risk | Would look like | Retired at |
|---|---|---|
| Suppression does not bite even on *red* | Accuracy unchanged after projecting e₀ out; color is read from elsewhere | 1 |
| The bound is loose or wrong in a transformer | Observed non-red damage exceeds the geometric prediction | 1, then 5 |
| Recipe is grammar-specific | The survey's proposals do not reproduce on the new grammar | 2 |
| Task cost grows with an abstract concept | Gate misses in 3 that 2 did not have | 3 |
| Anchoring an op captures the token, not the operation | Alignment lands, and suppression is inert | 4 |
| Bypass through attention or the residual | Suppression works only when applied at every site | 4, 5 |

Each of the first two stops changes one thing from D2.1, so a negative there stays interpretable.

## Out of scope

Verification lines (D2.3). Several ops on separate axes (a D2.3 candidate, for the subspace bound). The feedback controller (the kickoff advice stands). RMU and SAE baselines. The word-level tokenizer, unless stop 3 fails. HSV-valued colors as tokens. Resolving D2.1's H2 decodability question ([its own item](/todo/science/global-structure-preserved-under-anchoring.md)), run when the claim is needed.

## Decisions taken

- Anchor `screen` first; `mix` stays the default and `add` is the second contrast.
- Stop 1 runs before S1 is finished: it needs no grammar change and has the highest information per dollar in the plan.
- The survey is confirmed on the new grammar, as a set of proposals, and not replicated literally first.
- Layer sweep shape: deferred to stop 5.
