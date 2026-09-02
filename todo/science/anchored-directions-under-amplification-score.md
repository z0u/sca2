---
status: open
tags: [anchoring, metrics, M3]
opened: 2026-08-26
---
# Does anchoring make a direction "natural" under the amplification score?

[In Search of Natural Features](https://www.lesswrong.com/posts/SNAKJuN8FdoEaWeFC/in-search-of-natural-features) (LessWrong, 2026) proposes an amplification score: for a candidate direction, measure whether an MLP layer acts as a noise gate — boosting the signal along that direction when it is strong and suppressing it when weak. The argument, building on computation-in-superposition, is that denoising costs the model capacity, so a direction the model actively error-corrects is one it computationally commits to. Gradient-ascending the score over activation space, they find only a handful of maxima per layer (3–6), recurring across GPT-2, Pythia, and Llama.

They solve the discovery problem (the model put its concepts somewhere unknown; detect the load-bearing directions post-hoc), where SCA solves the placement problem (choose the location at training time so no search is needed). That mirror-image raises a question their lens makes measurable: after SCA training, does the model learn to error-correct toward the anchor? Concretely, does the anchored direction appear as a new local maximum of the amplification score, alongside the emergent ones, when it does not in the un-anchored baseline?

Either answer is informative. Yes would be evidence the anchor is computationally integrated — the model builds a noise gate around it — rather than only linearly readable. No would suggest anchored concepts are a different kind of object from emergent features, which matters for predicting how interventions behave and would be worth saying in M3/M4 framing.

This is not a run-scoring metric for M2: our synthetic domain has ground truth, so alignment, probe accuracy, and suppression side-effects measure anchoring quality more sharply than an indirect error-correction score could. It is a side-experiment, closer to M3 (their machinery — MLP Jacobians, whitened data covariance, LayerNorm-free variants — was built for real LMs and would need adapting to our small transformer), or an optional M2 extension if an anchored checkpoint and a baseline are already on hand.

Independent of the experiment, the post is worth a citation in D2.4's related work as a complementary post-hoc approach: further evidence the field finds post-hoc discovery fragile enough to want firmer footing. Read the post in full before citing — the summary above is from one shallow fetch and should be checked against the source.
