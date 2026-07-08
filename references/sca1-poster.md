---
channel: GRAM workshop at ICLR
date: 2026-04-26
context: Text-only version of our poster.
---

# Sparse Concept Anchoring for Interpretable and Controllable Neural Representations

Sandy Fraser & Patryk Wielopolski, independent

Current interpretability methods attempt to discover safety-critical concept locations post-hoc. We anchor them during training, enabling reliable intervention with no search and &lt;0.1% labeled data.

<!-- two column mode -->

## Single Anchor

- Anchor only concepts of interest
- Few & noisy labels suffice

![Latent space with single red anchor and no intervention (4D bottleneck). Left: a full colorful hemisphere with red anchored at the north pole, indicated with a solid black triangle labeled (1,0,0,0), showing the complete color distribution (a color wheel). Right: another view of the sphere surface with all colors present and a wavy white horizontal line, indicating a subspace constraint. Below, axis labels show projections of the 4D bottleneck onto two 2D planes.](Fig-1-a.png)

![After suppression of the red vector. Left: the hemisphere with its upper half erased out above a dashed line — only non-red colors (blue, cyan, green) survive in the lower half. Right: the sphere with the remaining activations in the lower region and scattered dots showing where red inputs now project. Below, a line chart shows R-squared=0.99 with MSE scaling as the square of cosine similarity to the red direction, confirming near-perfect suppression.](Fig-1-b.png)

![After ablation of the red dimension — given a single anchor without concept isolation. Left: the hemisphere is almost entirely erased; only a thin horizontal stripe of non-red hues (violet through cyan to yellow-green) survives. Right: a sphere with scattered dots. Below, a line chart shows R-squared=0.37 with a warning symbol, indicating that ablation fails: without repulsion-based isolation the red concept was not cleanly separated, so zeroing the axis impacts untargeted concepts.](Fig-1-c.png)

## Isolation w/ Repulsion

- Clear subspaces for ablation
- No extra labels needed

![Latent space with repulsion-isolated red anchor and no intervention (5D bottleneck). Left: projection onto the (Z4,Z1) plane shows a curved colour surface, like a disc that has been bent 90-degrees, viewed side-on. There is an anchor at the north pole, indicated with a solid black triangle and labeled (1,0,0,0,0). An anti-anchor opposes it at the south pole, indicated by a hollow triangle, and a vertical wavy double line shows that an anti-subspace was applied to the first dimension. The bottom half of the hemisphere is empty. Right: the (Z2,Z3) projection reveals a wide colourful curved surface covering much of the sphere.](Fig-2-a.png)

![After suppression in the repulsion-isolated setting. Left: the (Z4,Z1) projection is now a thin horizontal line of hue variation with no red component; the suppressed top region is entirely empty. Right: the sphere shows a dashed boundary outline with scattered dots for suppressed inputs. Below, a line chart shows R-squared=0.98, confirming near-perfect suppression.](Fig-2-b.png)

![After ablation in the repulsion-isolated setting. Left: only a short thin line of colour remains, representing the hue subspace with the red dimension zeroed out. It looks almost identical to the suppressed latent space (see previous figure). Right: the sphere surface shows a colourful patch (the surviving hue space) with scattered dots. Below, a line chart shows R-squared=0.98, demonstrating that repulsion-based isolation enables reliable ablation — the concept is cleanly separated so zeroing has no impact on untargeted concepts.](Fig-2-c.png)

---

## Targeted inductive biases on activations

$$\mathcal{L}_{\text{total}}(\cdot) = \mathcal{L}_{\text{task}}(\cdot) + \mathcal{L}_{\text{structural}}(\hat{\mathbf{z}}) + \mathcal{L}_{\text{concept}}(\hat{\mathbf{z}}, \ell_{\mathcal{C}})$$

<!-- three column mode -->

**Structural constraints** provide a _geometric basis_ for latent representations. _Normalize_ places activations on the hypersphere, while _separate_ prevents excessive clustering (pairwise). These constraints create structure that supports concept organization and intervention.

![Two conceptual diagrams labelled Normalize and Separate. Normalize: two dots with arrows showing them being projected onto a curved arc (the hypersphere surface), illustrating that all activations are constrained to lie on the sphere. Separate: two dots already on the arc with arrows pushing them apart along the surface, illustrating the pairwise repulsion that prevents activations from clustering.](Fig-3-a.png)

**Attractive regularizers** pull _rare_ labeled samples to predetermined locations. _Anchor_ positions simple linear concepts, while _subspace_ collects multidimensional and cyclic concepts. The network is _free to learn_ optimal representations for others.

![Two conceptual diagrams labelled Anchor and Subspace. Anchor: two dots on the sphere arc with arrows converging toward a single fixed point marked by a solid triangle, showing that labeled samples are attracted to a predetermined point on the sphere. Subspace: a horizontal wavy line representing a subspace, with one dot above and one below it, both with arrows pointing toward the line, showing that samples are attracted to a predetermined subspace (e.g. for cyclic concepts like hue).](Fig-3-b.png)

**Repulsive regularizers** push _all_ samples away from dimensions intended for ablation. _Weights vary_ to a schedule: strong early in training to clear regions while thespace is malleable, and weak later toallow attractive terms to dominate.

![Two conceptual diagrams labelled Anti-anchor and Anti-subspace. Anti-anchor: a hollow triangle at the top of a sphere arc, with two dots on either side and arrows pointing away from the triangle — all samples are repelled from the target axis. Anti-subspace: a double horizontal wavy line with one dot above and one below, both with arrows pointing away from the line — all samples are repelled from the ablation subspace.](Fig-3-c.png)

---

[View paper on arXiv — ID arxiv:2512.12469](https://arxiv.org/abs/2512.12469).

Task: Color autoencoder on RGB color space. Anchored concepts: red (linear), and in some experiments, hue (cyclic). Bottleneck dim ∈ [4,5].
