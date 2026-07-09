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

- [Experiment 2.9.1 redux](./ex-2.9.1/report.py): the M1 headline result
  (delete *red* from a 5D autoencoder), ported from
  [ex-preppy](https://github.com/z0u/ex-preppy) to JAX as an end-to-end
  shakedown of the M2 infrastructure.
- [nGPT scaling](./ngpt-scaling/report.py): a character-level sweep of our
  simplified nGPT that traced a wide-and-deep training failure to a bug in the
  residual step (it added the *raw*, un-normalized sub-module output, so the
  fixed step over-rotated the hidden state as width grew), and validated the
  fix — the nGPT-faithful step toward the *normalized* output — across the
  width × depth grid before the color-mixing experiments.

More reports will appear here as the M2 experiments land.
