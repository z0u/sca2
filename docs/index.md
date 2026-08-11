# Sparse Concept Anchoring in transformers

Sparse Concept Anchoring (SCA) is a training-time method for concept control. A light geometric regularizer, driven by a small number of noisy labels, guides a chosen concept toward a known location in representation space. The geometry is shaped during training, so there is no need to reverse-engineer it afterwards: the concept lives where you put it, and the side effects of suppressing or ablating it can be bounded before running the intervention.

The first milestone (M1) established the method in autoencoders: [paper](https://arxiv.org/abs/2512.12469), [blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts), [code](https://github.com/z0u/ex-preppy). This site holds the experiment reports for the second milestone (M2), which asks: does SCA transfer to transformers? We anchor concepts in the residual stream of a small transformer trained on a synthetic color-mixing task (`red + blue = purple`), where ground truth is unambiguous, so a negative result stays interpretable. The plan and deliverables are in the [project README](https://github.com/z0u/sca2).

&nbsp;

## Experiment reports

Each report is a [Marimo](https://marimo.io) notebook that reads durable results produced by a separately-run experiment. Reports are published automatically, with their figures served from a Hugging Face dataset; the infrastructure is [mi-ni](https://github.com/z0u/mi-ni).

<!-- These URLs are rewritten to point to the published notebooks -->

<details markdown="1"><summary><h3>Iteration 0 (prep)</h3></summary>

These experiments were preparation for the main work: exercising the infrastructure, testing the normalized transformer architecture, and tying off some loose ends from M1.

- [nGPT scaling](./ngpt-scaling/report.py)

    Width × depth sweep of our simplified nGPT to check that it stays well-behaved as it grows. Converged loss improves with width and is flat across depth, and no condition spikes or stalls. The architecture seems safe to build the color-mixing experiments on.

- [M1 2.9.1. Deleting *red*, now in JAX](./m1/ex-2.9.1/report.py)

    We ported the main M1 result from PyTorch to JAX: Anchor *red* to one latent axis of a small autoencoder, then zero the axis and watch red disappear. The result reproduces, seed sensitivity and all, and our new infrastructure holds up.

- [M1 2.9.2. Fallback control](./m1/ex-2.9.2/report.py)

    We compared two remedies for ablation seed-sensitivity: **1.** optimal ablation, which picks a replacement constant after training, and **2.** fallback control, a decoder-only loss term that teaches the model what to output once the concept has been removed. Fallback control worked well. Neither remedy touches the other half of the variance, where anchoring itself fails.

- [M1 2.9.3. When anchoring fails](./m1/ex-2.9.3/report.py)

    Every failing run anchors first and then comes apart during the high learning-rate plateau. Halving the peak learning rate removes the failures.

- [M1 2.9.4. Closed-loop regularizer weights](./m1/ex-2.9.4/report.py)

    We replaced the timed regularizer anneal with a feedback controller, so that each weight climbs while its constraint is being violated and settles back once it is met. It kind of worked but not very well, and it adds complexity.

</details>

### D2.1: anchoring in a transformer

- [2.1.1. Un-anchored color-mixing transformer](./m2/ex-2.1.1/report.py)

    A small transformer learns a character-level language of color-mixing equations, solving the forms it saw in training and unseen hex pairs. Color turns out to be linearly decodable from its residual stream, with each seed putting *redness* in a different place. Held-out *named* pairs sit at zero accuracy.

- [2.1.2. Making composition necessary](./m2/ex-2.1.2/report.py)

    The previous model never answered a held-out *named* pair, so we changed the grammar to make composition the only route: reverse alias lines, and named equations whose mix falls off the palette. The model picks up both new skills and still won't chain them within one forward pass, leaving held-out named accuracy at zero.

- [2.1.3. Named colors only](./m2/ex-2.1.3/report.py)

    No hex codes, just one opaque token per color and nothing in the text to say that colors are values at all. The model infers the geometry from co-occurrence alone: its embeddings hold the RGB cube as a linear subspace, it computes mixes in value space just before answering, and its held-out guesses land on or beside the right color. Exact match rises and falls with vocabulary size, while geometric closeness improves steadily.

- [2.1.4. Spelling the names](./m2/ex-2.1.4/report.py)

    Does geometry inference need one token per concept? Not at 216 colors. We test the same equations but with every color spelled as an opaque four-letter name. There's a small drop in held-out accuracy, but the value subspace survives. Reading names takes most of the network's depth, leaving one layer for the mix to live in. The 27-color grid is too coarse.

- [2.1.5. Disjoint vocabularies and more named colors](./m2/ex-2.1.5/report.py)

    Names and hex codes. Both sublanguages train, and each builds its own linear color geometry. But the two geometries resist being combined, even under pressure from a narrow residual stream. Keeping them apart apparently costs less than merging them would.

- [2.1.6. Anchoring *red* in a transformer](./m2/ex-2.1.6/report.py)

    Our first anchored transformer, with a single attractive term. The anchor moved the activations to the chosen direction, and cost the task nothing measurable. But it moved the whole color cube, rather than *red* in particular.

- [2.1.7. A repulsive term and a narrower pull](./m2/ex-2.1.7/report.py)

    We tested two mechanisms to improve anchor selectivity: **1.** Apply the anchor term only to operand 1 (no other tokens), and **2.** Add a repulsive term to clear the target subspace. Both work, but 1. worked better, and their effects stack somewhat.

- [2.1.8. Repulsion tuning](./m2/ex-2.1.8/report.py)

    We tested anti-subspace schedules (trailing timing and strength) to find an operating point that contains the cube-wide drift without reducing selectivity. Holding the repulsion high for longer contains the drift and keeps the margin, which nothing before this did. It's unclear if the response is well-graded.

- [2.1.9. Softmin sequence pooling](./m2/ex-2.1.9/report.py)

    Anchoring on op1 alone worked best so far, but in natural language we won't know which tokens hold the concept. So we pool over sequences with a soft minimum (mellowmax) and let the pull choose its own position. At the embedding it picks op1 unaided, the operating point stays healthy, and grading improves. It wins no margin though, and only the softest pooling stays graded in every run.

- [2.1.10. A label that doesn't point](./m2/ex-2.1.10/report.py)

    Ex-2.1.9's labels still keyed on op1, so the label itself said where the concept was. Here either operand can trigger the label, and the pooled pull finds the red operand line by line: the weight profiles track the label groups, the operating point survives, and selectivity matches the slot oracle. The soft-τ arms win margin but smear the grading.

- [2.1.11. Ablations and a survey for the D2.1 operating point](./m2/ex-2.1.11/report.py)

    Survey (preregistered search plan; ablations in, search running). The recipe's schedules and weights were inherited piecewise and never tuned together, so flat arms bracket each schedule and a half-length condition trims the budget. The anchor schedule turns out to be replaceable by a constant and 50 epochs is enough, but the repulsion schedule earns its shape at both ends. A Sobol search over what is left (λ_a, τ, repulsion peak and timing) then maps the good region and proposes the operating point D2.2 confirms.
