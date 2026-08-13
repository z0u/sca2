# Related-work delta since M1 (searched 2026-08-13)

M1's related-work section missed gradient routing, and the field has moved in the year since. This file records a four-angle literature sweep (gradient-routing lineage, training-time interpretability, unlearning, post-hoc erasure/steering) so the M2 baseline plan and the D2.4 write-up start from the current field.

Verification status matters: **[V]** = abstract fetched and read; **[S]** = seen only in search snippets — verify before citing. Entries kept to what changes our comparisons or our framing; the raw sweeps found more.

## The must-know finding

**SGTM is now its own paper**: "Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs" (Shilov, Cloud, et al., arXiv:2512.05648, Dec 2025, Anthropic alignment team) [V]. Selective gradient masking so only dedicated parameters update on target-domain examples; improves gradient routing's robustness to label noise. Same logic as SCA — localize during training so later removal is clean — evaluated on LLM capability removal. This is the head-to-head baseline a reviewer will name first. Rejected from ICLR 2026; cite as arXiv. The meta-review's grounds are a preview of what M2 will face: loss-only evaluation, no downstream task metrics, small scale (max 254M), and thin comparison to other pretraining-time mitigations (Korbak 2023 conditional training, Wang 2025 SLUNG).

## Gradient-routing lineage

- **Gradient Routing** (Cloud et al., arXiv:2410.04332, 2024) [V] — the seed paper; data-dependent stop-gradient masks route concepts into designated parameter/stream subregions. Already named in our README as closest work. Rejected from ICLR (OpenReview z1mLNhWFyY); cite as arXiv.
- **GRAM: Modular Pretraining Enables Access Control** (Roland et al., arXiv:2607.08077, Jul 2026) [V] — gradient-routed auxiliary modules, scaled 50M–5B; ablate modules at inference to disable capabilities. Evidence the family scales; module-based rather than direction-based.
- **UNDO: Distillation Robustifies Unlearning** (Lee, Foote, et al., arXiv:2506.06278, NeurIPS 2025 spotlight) [V/S] — a few fine-tuning steps revert almost any unlearning; distilling the unlearned model into a noised student transfers behavior without the latent capability. Cheap in a toy pipeline; arguably the strongest "is it really gone" test available.
- **Clusterability loss** (Golechha et al., arXiv:2502.02470) [V] — independent training-time modularity regularizer (spectral cluster separation); finds modularity without increased task specialization.
- LessWrong follow-ups by the GR team (interpretable latent spaces via GR; GR vs. pretraining filtering) [S] — pull before citing.

## Training-time interpretability ("intrinsic interpretability")

The category now has a name and a survey: **arXiv:2604.16042** [V] organizes it into five paradigms (functional transparency, concept alignment, representational decomposability, explicit modularization, latent sparsity). Checked 2026-08-13: neither SCA nor gradient routing appears in the survey or its GitHub repo (PKU-PILLAR-Group/Survey-Intrinsic-Interpretability-of-LLMs) — SCA belongs in the concept-alignment bucket, gradient routing in explicit modularization. Cite the survey as the category frame; consider a PR/issue to the repo suggesting both.

- **Weight-sparse transformers** (OpenAI, Gao et al., Nov 2025) [S — confirm venue/id] — extreme L0 weight sparsity during pretraining yields circuits ~16× smaller at matched loss. Global sparsity rather than placed concepts.
- **Parity Bottleneck Layers** (Mack et al., arXiv:2607.20652) [V] — algebraically generated feature directions inserted in the forward pass, so features are load-bearing by construction; costs 6.7–9.4× training tokens. Closest in spirit to "known location by construction"; the token-cost trade-off is a comparison axis SCA should report too.
- **RepBend** (Yousefpour et al., arXiv:2504.01550, ACL 2025) [S] — fine-tuning loss enforcing orthogonality between safe/unsafe activation trajectories. Loss-shaped geometry for known semantic classes; separates two classes rather than anchoring a direction.
- **Safe Transformer** (Feng et al., arXiv:2603.06727) [V] — explicit binary safety bit via a discrete bottleneck; a designated-slot contrast to SCA's continuous anchored direction.
- **MoE-X** (Yang et al., arXiv:2503.07639, ICML 2025) [S] and **EMO** (arXiv:2605.06663) [V] — modularity/sparsity during (pre)training; related-work rather than baselines.
- **CB-LLM** (arXiv:2412.07992, ICLR 2025) [S] — concept-bottleneck LLM with concept-level steering and unlearning; the standard CBM-for-LLMs citation.
- Ancestor worth citing outside the date window: **Codebook Features** (arXiv:2310.17230).

## Unlearning (baselines + evaluation standards)

- **Ready2Unlearn** (Duan et al., arXiv:2505.10845) [V] — meta-learning so later unlearning is fast and low-damage. The other "prepare during training" method; method-agnostic, portable to a small model.
- **Deep Ignorance** (O'Brien, Casper, et al., arXiv:2508.06601) [V] — pretraining data filtering builds tamper-resistant safeguards (survives 10k adversarial fine-tuning steps); knowledge still usable in-context. The key "training-time beats post-hoc" evidence; also the caveat template for any SCA robustness claim.
- **Mechanistic Unlearning** (Guo et al., arXiv:2410.12949, ICML 2025) [S] — localize the responsible component via circuit discovery, then edit; localization-informed edits are more relearning-robust with fewer side effects. The closest post-hoc analog of SCA's philosophy — the strongest conceptual foil for the training-time-anchoring pitch.
- **LUNAR** (Shen et al., arXiv:2502.07218, NeurIPS 2025) [V] — redirect forget activations to the model's natural refusal region via a single-matrix edit; tractable in a toy transformer.
- **From Dormant to Deleted** (Siddiqui et al., arXiv:2505.22310) [V] — forget accuracy rebounds ~50%→100% after retain-only fine-tuning; weight-space distance diagnostic is cheap to adopt as an eval.
- **Layered Unlearning** (Qian et al., arXiv:2505.09500) [V] — sequential defense-in-depth unlearning; already validated on synthetic tasks, so its scaffolding is closest to our pipeline's shape.
- **Auditing standards**: ActPert activation-perturbation auditing (arXiv:2505.23270) [V]; obfuscation-vs-removal framing + DF-MCQ (arXiv:2505.02884, EMNLP 2025) [V]; minor low-variance directions carry
  > 88% recoverable forget-information after relearning (arXiv:2605.11685) [S]. The last is a concrete failure mode to test SCA against: does anchoring leave the concept recoverable from off-axis residue?
- RMU refinements: Adaptive RMU (arXiv:2408.06223) [S], LUNAR above, RNA (arXiv:2501.19202) [V]. RMU itself stays the family representative.

## Post-hoc erasure and steering

- **Bounded/predicted side-effects now exist post-hoc** — the two papers closest to SCA's headline claim:
  - **COAST** (Nguyen et al., arXiv:2605.01167, 2026) [V] — formalizes collateral damage of steering (expected squared change in non-target directions) and minimizes it under a target-alignment budget via constrained optimization on the activation hypersphere.
  - **Pre-intervention prediction of SAE steering side effects** (Duan, arXiv:2606.08365, 2026) [V] — predicts side-effects from decoder geometry before intervening; works on small models, weakens at 2B. Framing consequence: "bounded side-effects" alone is no longer a unique claim — the distinctive part is that SCA's bound holds _by construction_ (the geometry was placed), where theirs is estimated from emergent geometry and degrades with scale.
- **Completeness argument, strengthened**:
  - Feature absorption (Chanin et al., arXiv:2409.14507) [V] — monosemantic- looking SAE latents silently fail to fire where they should.
  - SAE interventions unreliable (arXiv:2606.18322, 2026) [V] — suppressing targeted SAE features leaves 75–100% of behavior recoverable from the reconstruction residual.
  - SAEs not canonical (Leask et al., 2025) [S — pin the id]; cyclic ablation / distributed resilience (arXiv:2509.25220) [S].
- **Completeness argument, to address rather than ignore**: Fundamental Limits of Perfect Concept Erasure (arXiv:2503.20098) [V] — perfect erasure is achievable under stated dimensionality/distribution conditions; and erased concepts are often recoverable with small parameter updates (arXiv:2505.16174) [S].
- **SAE steering done well**: SAEs Are Good for Steering If You Select the Right Features (Arad et al., EMNLP 2025, arXiv:2505.20063) [S] — causal output-score filtering gives 2–3× improvement, matching supervised methods. If we run an SAE baseline, this is the selection rule to use, so the baseline is the strong version.
- **LEACE** (arXiv:2306.03819) stays the canonical erasure baseline; nonlinear successors (arXiv:2507.12341, arXiv:2607.03973) [S] exist if a reviewer asks.

## Consequences for the M2 baseline plan

1. **SGTM (2512.05648) is the must-run training-time baseline** — reuses our labels; mechanism swap in the training step.
2. **Post-hoc tier stays as planned** (probe / diff-in-means / LEACE ablation on existing controls; filtered-corpus row) and is still the cheapest.
3. **The bound needs sharper framing**: compare against COAST and pre-intervention prediction on bound tightness, and state the by-construction vs. estimated-from-emergent-geometry distinction.
4. **Adopt the 2025 auditing standards** in our own evals where portable: relearning/rebound checks, ActPert-style perturbation, off-axis (minor-direction) recoverability. These make the "removed vs. masked" claim credible by current standards, and D2.3 should speak their language.
5. **If an SAE baseline runs**, use output-score feature selection (2505.20063) so we compare against the strong version.
6. **D2.4's related work** needs the intrinsic-interpretability survey (2604.16042) as the frame, plus Ready2Unlearn, Mechanistic Unlearning, RepBend, and Parity Bottlenecks as the nearest neighbors in each family.
