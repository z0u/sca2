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

- [Experiment 2.9.1 redux](./m1/ex-2.9.1/report.py): the main M1 result
  (deleting *red* from a 5D autoencoder), ported from
  [ex-preppy](https://github.com/z0u/ex-preppy) to JAX. We ran it as an
  end-to-end test of the M2 infrastructure.
- [nGPT scaling](./ngpt-scaling/report.py): a character-level width × depth
  sweep of our simplified nGPT (scalar gains, a residual step fixed at
  1/n_layer). Converged loss stays flat across the sweep, so this architecture
  seems safe to build the color-mixing experiments on.
- [Experiment 2.9.2](./m1/ex-2.9.2/report.py): fallback control for deleting
  *red*. In ex-2.9.1, selectivity varied a lot with the seed. This experiment
  tests the "optimal ablation" suggestion from the SCA paper against a
  training-time alternative: pin the decoder response at the reserved
  anti-anchor direction to mid-gray, then redirect the concept there. The
  trained fallback collapses the variance to an analytic bound.
- [Experiment 2.9.3](./m1/ex-2.9.3/report.py): exploration of why anchoring
  fails. Per-step trajectories show that every failing seed anchors
  successfully, then breaks during the high-LR plateau. A schedule sweep finds
  the fix: halve the LR peak and keep the anneal.
- [Experiment 2.9.4](./m1/ex-2.9.4/report.py): closed-loop regularizer term
  weights. Replaces the timed anneal with feedback. The mechanism "works", but it
  causes as many problems as it solves.

### Iteration 1: D2.1, anchoring in a transformer

- [Experiment 2.1.1](./m2/ex-2.1.1/report.py): a color-mixing transformer,
  un-anchored. Defines a synthetic color-mixing language (named colors and hex
  codes denoting the same concepts, with exact integer mixing), sweeps the nGPT
  architecture over width × depth × seed, and builds measurement tools:
  exact-match completion accuracy on seen, held-out, and unseen operand pairs,
  plus per-layer residual-stream probes for operand and result colors.
- [Experiment 2.1.2](./m2/ex-2.1.2/report.py): composition.
  The Ex-2.1.1 models never solve the held-out named pairs. This experiment
  tests grammar interventions (reverse alias lines and off-palette named
  equations). Both train, but held-out named accuracy stays at zero.
  Position-resolved probes suggest the mix never fully exists at any one
  position.
- [Experiment 2.1.3](./m2/ex-2.1.3/report.py): word-level tokens. Removes the
  hex values entirely; every color is a single opaque token. The model infers
  the color-space geometry from co-occurrence: embeddings hold RGB as a
  decodable linear subspace, mixes are computed in 3D value space at the
  pre-answer position, and guesses land near the nearest color even for held-out
  pairs. Exact match is non-monotonic in vocabulary size (essentially solved at
  216 colors; near misses at 4096).
- [Experiment 2.1.4](./m2/ex-2.1.4/report.py): multi-token names. The char-level
  twin of Ex-2.1.3: corpora identical line for line, but colors are opaque
  four-letter random strings, read and written one character at a time. At 216
  colors the geometry survives multi-token naming: held-out accuracy 0.91 with
  neighbor-level misses, the mix computed in value space at the pre-answer
  position, and no per-channel eviction during emission (unlike hex answers). At
  27 colors exact match collapses to zero with confidently wrong neighbor
  answers. Reading names consumes most of the network's depth.

- [Experiment 2.1.5](./m2/ex-2.1.5/report.py): disjoint vocabularies. A
  preregistered baseline in which named and hex sublanguages never share a
  line — 140 xkcd names at full 8-bit depth, no aliases, no cross form —
  asking whether the two surface forms converge on one latent geometry.
  Behavior and within-form geometry sections are filled (named held-out 0.667
  over strong nulls; hex 0.996); the cross-form alignment sections are still
  in their preregistered form, with the analysis round in progress.

More reports will appear here as the M2 experiments land.
