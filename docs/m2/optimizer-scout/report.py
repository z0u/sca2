# ruff: noqa: B018
# title: Optimizers with no privileged basis

r"""
# Optimizers with no privileged basis

/// tip |
<!-- lede -->
Adam's per-coordinate scaling makes the coordinate axes of the residual stream a privileged basis, and the model parks its syntax tokens one-hot on them. This scout trains the un-anchored control under three optimizers whose update is the same in every orthonormal basis of the residual coordinates, and asks whether the stream comes out isotropic and whether the task is learned as well. TODO: which way it came out.
///

The question comes from the memo of 2026-10-23 on whether the coordinate axes are special (in the project files), and from the discussion after it: the prior is that an isotropic stream would be easier to reason about, and the open question is what it costs. The scout runs twelve new 400-epoch control runs and three learning-rate finder runs, at the two seeds where ex-2.2.23's Adam controls are stored as the reference. No anchored runs: an optimizer that keeps the stream isotropic and learns the task as well is a candidate for an anchored arm later, which would need its own tuning of the term weights.
"""

# %%

r"""
## Observations

- [Does the task suffer? (E1)](#does-the-task-suffer-e1): TODO.
- [Is the stream isotropic? (E2)](#is-the-stream-isotropic-e2): TODO.
- [Where the syntax tokens go (E3)](#where-the-syntax-tokens-go-e3): TODO.

## Scope

This is an exploratory scout, with no preregistration and no gate. Three optimizers, each on the control at model seeds 704 and 705 for the full 400-epoch schedule, at two learning rates: the rate the finder of ex-2.2.17 suggests for that optimizer at seed 704, and a third of it. The finder's steepest-descent suggestion sat above the rate the ladders chose for Adam, so the lower rung is the safer one, and the pair says how sensitive each optimizer is. Twelve runs in all, against ex-2.2.23's two Adam controls at the same seeds, which share the initialization, the batches, and the model. Two seeds separate an optimizer from Adam only where the gap is larger than the seed range of the stored controls (about 0.02 in the task score; see split-anti-seeds' control columns); the scout is for ruling optimizers in or out, not for ranking close ones.

The three optimizers:

- Pooled Adam: Adam with its second moment pooled over the residual axis of each matrix that reads or writes the stream, so the scaling is one number per row (of a reading matrix) or column (of a writing matrix) and the same whichever orthonormal basis the residual coordinates are written in. The gains and the other axis of each matrix are scaled per coordinate as in Adam.
- Muon: each matrix's momentum is orthogonalized (a Newton–Schulz iteration toward the nearest orthogonal matrix) before the step, which is equivariant to a rotation on either side of the matrix; the gains are updated with Adam at the recipe's rate.
- SGD with momentum 0.9, the plain reference, with no per-coordinate scaling at all.

A test in the repo checks that the rotated-basis trick changes nothing for these three and does change Adam.

## The measurements

- The task: the expected exact match of each op before any edit (embedding-lean's measurement) against the Adam control's at the same seed, and the validation loss along training.
- Isotropy, from the basis measurement of rotated-basis: the share of the stream's energy that each coordinate axis holds, sorted, against the same along a random orthonormal basis; the mean kurtosis along each (a privileged basis has a higher kurtosis along its axes than along random directions; a Gaussian has 3); and the cost of projecting out one direction at full dose, for each coordinate axis and for sixteen random directions. The memo's Adam reference is one coordinate at about half of the energy, a kurtosis of 4 to 5 along the axes against 3 along random directions, and a typical axis at a cost of 0.03 against 0.08 to 0.13 for a random direction.
- Where the tokens sit: each token embedding's largest cosine with a coordinate axis. Under Adam the syntax tokens `?`, `,` and `⏎` sit at a cosine near 1 on one coordinate and `=` on another.
"""

# %%

r"""
## Does the task suffer? (E1)

Is the task learned as well without a privileged basis, at either rate? If an optimizer trails Adam by more than the seed range at both rates, it is out for now; if it matches at one rate, it is a candidate.

<!-- Figure: validation loss along training per optimizer and rate, with the Adam control; a table of the task score per op at the end. -->

TODO

TODO
"""

# %%

r"""
## Is the stream isotropic? (E2)

Does the energy spectrum flatten and the kurtosis gap close? The equivariance says the optimizer has no privileged basis, not that the stream is isotropic: the model may still build a spike along some direction, in which case the spectrum along a random basis would show it too, and the removal cost of a random direction would stay above a typical axis's.

<!-- Figure: sorted energy spectra along the coordinate axes and along a random basis, per optimizer, with Adam; the kurtosis per slice; the cost of axes against random directions. -->

TODO

TODO
"""

# %%

r"""
## Where the syntax tokens go (E3)

With no cheap one-hot coordinates, where do the separators and `=` sit? Spread over many coordinates, on a direction of their own that is not an axis, or still one-hot on an axis for a reason the optimizer does not supply.

<!-- Figure or table: the largest cosine of each syntax token's embedding with a coordinate axis, per optimizer, against the Adam control. -->

TODO

TODO
"""

# %%

r"""
## Discussion

TODO
"""
