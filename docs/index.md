# Sparse Concept Anchoring in transformers

Sparse Concept Anchoring (SCA) is a training-time method for concept control.
A light geometric regularizer, driven by a small number of noisy labels,
guides a chosen concept toward a known location in representation space —
shaping feature geometry during training rather than reverse-engineering it
afterwards. Because the concept then lives where you put it, suppressing or
ablating it has side effects you can bound from the geometry before running
the intervention.

The first milestone (M1) established the method in autoencoders:
[paper](https://arxiv.org/abs/2512.12469),
[blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts),
[code](https://github.com/z0u/ex-preppy). This site holds the experiment
reports for the second milestone (M2): does SCA transfer to transformers?
We anchor concepts in the residual stream of a small transformer trained on
a synthetic color-mixing task (*red + blue = purple*), where ground truth is
unambiguous, so a negative result stays interpretable. The plan and
deliverables are in the [project README](https://github.com/z0u/sca2).

&nbsp;

## Experiment reports

Each report is a [Marimo](https://marimo.io) notebook that reads durable
results produced by a separately run experiment. Reports are published
automatically, with their figures served from a Hugging Face dataset; the
infrastructure is [mi-ni](https://github.com/z0u/mi-ni).

<!-- These URLs are rewritten to point to the published notebooks -->

### Iteration 0: prep

- [Experiment 2.9.1 redux](./m1/ex-2.9.1/report.py): the M1 headline result
  (delete *red* from a 5D autoencoder), ported from
  [ex-preppy](https://github.com/z0u/ex-preppy) to JAX as an end-to-end
  shakedown of the M2 infrastructure.
- [nGPT scaling](./ngpt-scaling/report.py): a character-level width × depth
  sweep of our simplified nGPT (scalar gains, a residual step fixed at
  1/n_layer), checking that converged loss stays flat across the grid — no depth
  penalty, no width-gated instability — so the backbone the color-mixing
  experiments build on actually scales.
- [Experiment 2.9.2](./m1/ex-2.9.2/report.py): fallback control for deleting
  *red*. Ex-2.9.1's selectivity varies a lot with the seed; this tests the
  SCA paper's "optimal ablation" suggestion against a training-time
  alternative — pin the decoder's response at the reserved anti-anchor
  direction to mid-gray, then redirect the concept there — and finds the
  trained fallback collapses the variance to an analytic bound.
- [Experiment 2.9.3](./m1/ex-2.9.3/report.py): why anchoring fails. Per-step
  trajectories show every failing seed anchors successfully and then breaks
  during the high-LR plateau; an init × data-stream factorial shows the
  failures follow the stream, not the seed; and a schedule sweep finds the
  fix — halve the LR peak, keep the anneal.
- [Experiment 2.9.4](./m1/ex-2.9.4/report.py): closed-loop regularizer weights.
  Replaces the timed anneal with feedback (dual ascent with hysteresis on
  the anchor terms). The mechanism works — no rescue is missed — but it
  causes as many catastrophes as it prevents, its knobs are sharper than the
  LR knob it replaces, and the fallback term already prevents every
  catastrophic failure on its own. A clean negative: the boring fix stands.

### Iteration 1: D2.1, anchoring in a transformer

- [Experiment 2.1.1](./m2/ex-2.1.1/report.py): the color-mixing transformer,
  un-anchored. Defines the synthetic color-mixing language (named colors and
  hex codes denoting the same concepts, with exact integer mixing), sweeps the
  nGPT backbone over width × depth × seed, and builds D2.1's measurement
  apparatus: exact-match completion accuracy on seen, held-out, and unseen
  operand pairs, plus per-layer residual-stream probes for operand and result
  colors. The baseline the anchored runs are compared against.
- [Experiment 2.1.2](./m2/ex-2.1.2/report.py): making composition pay.
  Ex-2.1.1's baseline never solves the held-out named pairs, and its diagnosis
  blamed the corpus: a memorizable named slice, a one-way alias dictionary,
  and hex answers that factorize per channel. A 2 × 2 factorial of grammar
  interventions (reverse alias lines × off-palette named equations) on the
  frozen backbone supplies both missing ingredients — and both train, yet
  their composition never appears: held-out named accuracy stays at zero,
  with the failure now split into a form-rule error (correct value, hex
  spelling) and a value → name translation that never engages mid-equation.
  The position-resolved probes also show the answer is computed just in time,
  channel by channel, with earlier channels evicted — the mix never fully
  exists at any one position. Still un-anchored; the anchored runs inherit
  the richest corpus plus the graded instruments built here (margins, s₂,
  transfer probes), with the open holdout set as their degradation canary.

More reports will appear here as the M2 experiments land.
