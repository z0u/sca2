# Sparse Concept Anchoring in transformers

Sparse Concept Anchoring (SCA) is a training-time method for concept control.
A light geometric regularizer, driven by a small number of noisy labels,
guides a chosen concept toward a known location in representation space.
The geometry is shaped during training, so there is no need to
reverse-engineer it afterwards: the concept lives where you put it, and the
side effects of suppressing or ablating it can be bounded before running the
intervention.

The first milestone (M1) established the method in autoencoders:
[paper](https://arxiv.org/abs/2512.12469),
[blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts),
[code](https://github.com/z0u/ex-preppy). This site holds the experiment
reports for the second milestone (M2), which asks: does SCA transfer to
transformers? We anchor concepts in the residual stream of a small transformer
trained on a synthetic color-mixing task (*red + blue = purple*), where ground
truth is unambiguous, so a negative result stays interpretable. The plan and
deliverables are in the [project README](https://github.com/z0u/sca2).

&nbsp;

## Experiment reports

Each report is a [Marimo](https://marimo.io) notebook that reads durable
results produced by a separately-run experiment. Reports are published
automatically, with their figures served from a Hugging Face dataset; the
infrastructure is [mi-ni](https://github.com/z0u/mi-ni).

<!-- These URLs are rewritten to point to the published notebooks -->

### Iteration 0: prep

- [Experiment 2.9.1 redux](./m1/ex-2.9.1/report.py): the M1 headline result
  (deleting *red* from a 5D autoencoder), ported from
  [ex-preppy](https://github.com/z0u/ex-preppy) to JAX. We ran it as an
  end-to-end test of the M2 infrastructure.
- [nGPT scaling](./ngpt-scaling/report.py): a character-level width × depth
  sweep of our simplified nGPT (scalar gains, a residual step fixed at
  1/n_layer). Converged loss stays flat across the grid — no depth penalty,
  no instability at large widths — so this architecture should be safe to build
  the color-mixing experiments on.
- [Experiment 2.9.2](./m1/ex-2.9.2/report.py): fallback control for deleting
  *red*. Ex-2.9.1's selectivity varies a lot with the seed. This experiment
  tests the SCA paper's "optimal ablation" suggestion against a training-time
  alternative: pin the decoder's response at the reserved anti-anchor
  direction to mid-gray, then redirect the concept there. The trained
  fallback collapses the variance to an analytic bound.
- [Experiment 2.9.3](./m1/ex-2.9.3/report.py): why anchoring fails. Per-step
  trajectories show that every failing seed anchors successfully, then breaks
  during the high-LR plateau. An init × data-stream factorial shows the
  failures follow the stream, not the seed. A schedule sweep finds the fix:
  halve the LR peak and keep the anneal.
- [Experiment 2.9.4](./m1/ex-2.9.4/report.py): closed-loop regularizer weights.
  Replaces the timed anneal with feedback (dual ascent with hysteresis on
  the anchor terms). The mechanism works — no rescue is missed — but it
  causes as many catastrophes as it prevents, and its knobs are sharper than
  the LR knob it replaces. Meanwhile, the fallback term already prevents
  every catastrophic failure on its own. It was a clean negative result, so
  we're keeping the static schedule from ex-2.9.3.

### Iteration 1: D2.1, anchoring in a transformer

- [Experiment 2.1.1](./m2/ex-2.1.1/report.py): the color-mixing transformer,
  un-anchored. Defines the synthetic color-mixing language (named colors and
  hex codes denoting the same concepts, with exact integer mixing), sweeps
  the nGPT backbone over width × depth × seed, and builds D2.1's measurement
  apparatus: exact-match completion accuracy on seen, held-out, and unseen
  operand pairs, plus per-layer residual-stream probes for operand and result
  colors. This is likely the baseline that the anchored runs will be compared
  against.
- [Experiment 2.1.2](./m2/ex-2.1.2/report.py): making composition necessary.
  Ex-2.1.1's baseline never solves the held-out named pairs, and its diagnosis
  pointed to the corpus: a memorizable named slice, a one-way alias dictionary,
  and hex answers that factorize per channel. This experiment tests grammar
  interventions (reverse alias lines and off-palette named equations). Both
  train, but held-out named accuracy stays at zero, with the failure now split
  into a form-rule error (correct value, hex spelling) and a value → name
  translation that never engages mid-equation. Position-resolved probes suggest
  the mix never fully exists at any one position.
- [Experiment 2.1.3](./m2/ex-2.1.3/report.py): names all the way down. Removes
  the hex scaffolding entirely — every color is a single opaque token and the
  only sentences are named mixing equations — and sweeps the vocabulary from
  27 to 4096 grid colors. The model infers the color-space geometry from
  co-occurrence alone: embeddings hold RGB as a decodable linear subspace,
  mixes are computed in value space at the pre-answer position, and guesses
  land near the nearest-name floor even for pair types never seen in training.
  Exact match is non-monotonic in vocabulary size (essentially solved at 216
  colors; neighbor-level near misses at 4096), which informs the vocabulary
  design for the anchored runs.
- [Experiment 2.1.4](./m2/ex-2.1.4/report.py): spelling the names. The
  char-level twin of ex-2.1.3 — corpora identical line for line, but every
  color is an opaque four-letter random string, read and written one character
  at a time. At 216 colors the geometry survives multi-token naming: held-out
  accuracy 0.91 with neighbor-level misses, the mix computed in value space at
  the pre-answer position, and no per-channel eviction during emission (unlike
  hex answers). At 27 colors exact match collapses to zero with confidently
  wrong neighbor answers, even though the neighborhood structure is learned.
  Reading names consumes most of the network's depth, which informs where
  anchors can live.

More reports will appear here as the M2 experiments land.
