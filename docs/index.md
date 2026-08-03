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

- [Experiment 2.9.1 redux](./m1/ex-2.9.1/report.py): a port of the main M1
  result (anchor *red* to one latent axis of a small autoencoder, then zero the
  axis and watch red disappear) from PyTorch to JAX, run as an end-to-end test
  of this repo's infrastructure. The result reproduces, seed sensitivity and
  all.
- [nGPT scaling](./ngpt-scaling/report.py): a character-level width × depth
  sweep of our simplified nGPT (scalar gains, a residual step fixed at
  1/n_layer). Converged loss improves with width and is flat across depth, and
  no condition spikes or stalls, so the architecture seems safe to build the
  color-mixing experiments on.
- [Experiment 2.9.2](./m1/ex-2.9.2/report.py): fallback control for deleting
  *red*. Ablation scores in ex-2.9.1 varied a lot with the seed, so we compare
  optimal ablation, which picks a replacement constant after training, against
  a training-time alternative: one decoder-only loss term that teaches the
  model what to output once the concept has been removed. Fallback control
  gives the intervention a designed response, tight against its analytic bound
  on every seed. Neither remedy touches the other half of the variance, where
  anchoring itself fails.
- [Experiment 2.9.3](./m1/ex-2.9.3/report.py): why anchoring fails. Our guess
  was that some initializations simply cannot be anchored by our schedule, and
  that doesn't hold up: every failing run anchors first and then comes apart
  during the high learning-rate plateau, and which runs fail follows the batch
  stream rather than the init. Halving the peak learning rate removes the
  failures.
- [Experiment 2.9.4](./m1/ex-2.9.4/report.py): closed-loop regularizer term
  weights, replacing the timed anneal with feedback. The mechanism does what it
  was designed to do, rescuing the seeds the static schedule loses, but it
  breaks a similar number of others. We're keeping the static schedule.

### Iteration 1: D2.1, anchoring in a transformer

- [Experiment 2.1.1](./m2/ex-2.1.1/report.py): the un-anchored baseline for M2.
  A small transformer learns a character-level language of color-mixing
  equations, solving the forms it saw in training and unseen hex pairs, and
  color turns out to be linearly decodable from its residual stream — with each
  seed putting *redness* in a different place, which is the problem SCA is
  meant to solve. Held-out *named* pairs sit at zero accuracy, so that eval set
  can't show us degradation later.
- [Experiment 2.1.2](./m2/ex-2.1.2/report.py): making composition necessary.
  The Ex-2.1.1 models never answered a held-out *named* pair, so this
  experiment changes the grammar to make composition the only route: reverse
  alias lines, and named equations whose mix falls off the palette. The model
  picks up both new skills and still won't chain them within one forward pass,
  leaving held-out named accuracy at zero. The off-palette set is a better
  place to look for degradation in the anchored runs.
- [Experiment 2.1.3](./m2/ex-2.1.3/report.py): named colors only. Removes the
  hex codes, leaving one opaque token per color and nothing in the text to say
  that colors are values at all. The model infers the geometry from
  co-occurrence alone: its embeddings hold the RGB cube as a linear subspace,
  it computes mixes in value space just before answering, and its held-out
  guesses land on or beside the right color. Exact match rises and falls with
  vocabulary size, while geometric closeness improves steadily.
- [Experiment 2.1.4](./m2/ex-2.1.4/report.py): spelling the names. Does
  geometry inference need one token per concept? At 216 colors, no. This is the
  character-level twin of Ex-2.1.3, with the same equations but every color
  spelled as an opaque four-letter name. The value subspace, the mix computed
  just before the answer, and the neighbor-shaped misses all survive the
  change, at a small cost in held-out accuracy. Reading names takes most of the
  network's depth, leaving one layer for the mix to live in. The 27-color grid
  turns out to be too coarse to tell us anything.
- [Experiment 2.1.5](./m2/ex-2.1.5/report.py): disjoint vocabularies. Two ways
  of writing the same colors, 140 xkcd names and full 8-bit hex codes, that
  never share a line. Both sublanguages train, and each builds its own linear
  color geometry, but the two geometries stay separate under everything tested:
  narrowing the residual stream until the named form loses resolution, and
  supervising a bridge between the forms. Keeping them apart apparently costs
  the model less than merging them would.
- [Experiment 2.1.6](./m2/ex-2.1.6/report.py): anchoring *red*, the first
  anchored transformer run. One anchor term, with no anti terms and no
  intervention. The anchor cost the task nothing measurable, and it moved the
  residual stream a long way onto the chosen direction. But it moved the whole
  color cube there, rather than *red* in particular: the margin settles at
  about half its threshold, and a tenfold increase in the regularizer weight
  did not improve it. The missing ingredient looks like a repulsive term, which
  M1 had and this experiment deliberately left out.
- [Experiment 2.1.7](./m2/ex-2.1.7/report.py): a repulsive term and a narrower
  pull, giving the first *selective* anchor. Two mechanisms were tested for
  improving anchor selectivity: applying the anchor term to operand 1 alone,
  and adding a repulsive term to clear the target subspace. Both work, the
  narrower pull more than the repulsion, and their effects stack somewhat.
  Together they clear the margin gate at no measurable task cost.
