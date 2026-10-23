# ruff: noqa: B018
# title: Anchoring on an axis the optimizer does not favor

r"""
# Anchoring on an axis the optimizer does not favor

/// tip |
<!-- lede -->
The coordinate axes of the residual stream are a privileged basis, because Adam's per-coordinate scaling is written in them, so e₁ is one of the directions the optimizer makes cheap to hold. This report trains the recipe of record and its control again with Adam's scaling applied in a random rotation of the coordinates, so that e₁ is an ordinary direction, and asks whether the anchor holds, removes, and spills as it did. TODO: which way it came out.
///

The question comes from the memo of 2026-10-23 on whether the coordinate axes are special (in the project files, not on the site). On stored checkpoints it found that every run parks its syntax tokens one-hot on a coordinate of its own, that a coordinate axis is cheaper to project out than a random direction, and, in short training runs, that Adam builds this basis and Adam in a rotated basis builds the same thing on the rotated axes. The anchoring design had taken e₁ for an arbitrary direction. This report runs sixteen new runs: `falloff` (the recipe of record of [split-anti-seeds](../split-anti-seeds/report.py)) and the un-anchored control, at the eight seeds where both are stored, with the optimizer's basis rotated and nothing else changed.
"""

# %%

r"""
## Observations

- [Does the anchor hold? (E1)](#does-the-anchor-hold-e1): TODO.
- [Removal, spill and landing (E2)](#removal-spill-and-landing-e2): TODO.
- [Where the privileged basis went (E3)](#where-the-privileged-basis-went-e3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate. It covers `falloff` and the control at model seeds 704 to 711, trained for 400 epochs with Adam's per-coordinate scaling applied in a random orthogonal rotation of the residual coordinates (one rotation per seed, drawn from the model seed). Each new run pairs with a stored twin at the same seed, split-anti-seeds' `falloff` and ex-2.2.23's 400-epoch control, which share its initialization, batches and label draws and differ in the optimizer's basis alone. The model, the schedules, the anchoring code, the measurements and the dose scoring are those of split-anti-seeds, unchanged.

A difference between a new run and its twin is read against the seed range of the stored arm. Split-anti-seeds found `falloff` within 0.002 of `split-anti` on removal with every seed within the spill criterion, and the loss-rate screen put its loss rate near one in forty, so a loss on one of eight seeds here would be unremarkable on its own. The screen's bound on that rate is about one in nine, which leaves two losses in eight inside chance as well, so a change in the loss rate would show only at three or more; a smaller effect would show in α on the pulled slices, or in removal on the seeds that kept the anchor.
<!-- REVIEW: "two or more would not" read the loss-rate point estimate (1 of 40) as the rate; its one-sided bound is near one in nine, at which two of eight losses have about one chance in five. Verify: the binomial at p = 1/9, n = 8. -->

## The measurements

- Whether the anchor held: α at the example answers at the end of training, per slice, as split-anti-seeds read it; a run that lost the anchor sits near zero on the pulled slices. A run counts as having lost the anchor when the full edit removes less than half of `difference`, net of the control, the rule of split-anti-seeds and the loss-rate screen.
- Removal, spill, and landing: the dose scoring of ex-2.2.22 (the projection at every slice and position, at four doses), net of the control at the same seed: removal as the net drop on `difference` over its clean-to-null span, spill as the largest net drop on another op, and landing as the fraction of the distance to the target null that the full edit covers.
- The task scores of every op before any edit, against the control's.
- The basis measurement, new here: the share of the stream's energy that each direction holds, along the coordinate axes and along the rotated basis; the mean kurtosis along each; and the cost of projecting out one direction at full dose, for e₁, each coordinate axis, each rotated axis, and sixteen random directions. The memo's stored runs give the reference: one coordinate at about half of the energy, a typical axis at a cost of 0.03 and a random direction at 0.08 to 0.13.
"""

# %%

r"""
## Does the anchor hold? (E1)

With the basis rotated, is e₁ still held through training at every seed, and as firmly? If the anchor leans on e₁ being a privileged axis, some seeds should lose it, or α at the example answers should sit lower on the pulled slices than in the stored `falloff` runs. If not, the per-slice α profiles should lie within the stored seed range. One reading to keep apart from the first: in the stored runs the syntax spike sits on a coordinate axis, orthogonal to e₁ by construction, while under the rotation it sits on a rotated axis whose cosine with e₁ is a few tenths at most and known per seed. So an anchor that is harder to hold could mean that the anchor needs a privileged axis, or that the anti term now has to push the syntax spike off e₁; E3 reports that cosine per seed so the reader can see how much of any seed-to-seed difference it accounts for.

<!-- Figure: α at the example answers by slice, the new runs beside the stored falloff runs, one line per seed. -->

TODO

TODO
"""

# %%

r"""
## Removal, spill and landing (E2)

On the seeds that kept the anchor, does the edit remove `difference` as it did, and does it spill as little? The stored `falloff` runs removed four fifths to nine tenths of the op at full dose with every seed within the spill criterion of 0.02; the comparison is paired by seed, against the new control at the same seed.

<!-- Figure: dose curves of removal and spill, new runs beside the stored; a table of removal, spill, landing and the task change per seed. -->

TODO

TODO
"""

# %%

r"""
## Where the privileged basis went (E3)

The check that the manipulation did what it says: the energy spectrum should be flat along the coordinate axes and spiked along the rotated axes, the syntax tokens should sit on a rotated axis, and e₁ should cost what an ordinary direction costs in the stored runs (0.08 to 0.13 in a control, where it touches the spikes) rather than what an ordinary axis costs (0.03). In the anchored runs e₁ holds the anchored op, so its cost there is the removal plus the spill, and the comparison is with the stored `falloff` runs' e₁. The cleaner comparison is within a run: e₁ against that run's own random directions and its rotated axes off the spikes, since the stored reference numbers are means over draws that touch the spikes by varying amounts. The basis measurement also records where each token embedding sits (its largest cosine with a coordinate axis and with a rotated axis), so whether the syntax tokens moved to a rotated axis is read from the published data.

<!-- Figure: sorted energy spectra along the coordinate axes and along the rotated basis; the cost of e₁, of the rotated axes, and of random directions. -->

TODO

TODO
"""

# %%

r"""
## Discussion

TODO
"""
