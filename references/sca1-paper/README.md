This is the source for our first SCA paper.

```bibtex
@InProceedings{fraser26,
  title      = {{S}parse {C}oncept {A}nchoring for Interpretable and Controllable Neural Representations},
  author     = {Fraser, Sandy and Wielopolski, Patryk},
  abstract   = {We introduce Sparse Concept Anchoring, a method that biases latent space to position a targeted subset of concepts while allowing others to self-organize, using only minimal supervision (in our setting, labels for <0.1\% of examples per anchored concept). Training combines activation normalization, a separation regularizer, and anchor or subspace regularizers that attract rare labeled examples to predefined directions or axis-aligned subspaces. The anchored geometry enables two practical interventions: reversible behavioral steering that projects out a concept's latent component at inference, and permanent removal via targeted weight ablation of anchored dimensions. Experiments on structured autoencoders show selective attenuation of targeted concepts with negligible impact on orthogonal features, and complete elimination with reconstruction error approaching theoretical bounds. Sparse Concept Anchoring therefore provides a practical pathway to interpretable, steerable behavior in learned representations.},
  openreview = {fCyFNits3s},
  software   = {https://github.com/z0u/ex-preppy/tree/a91164f},
  url        = {https://arxiv.org/abs/2512.12469}
}
```
