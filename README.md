# SCA2: Concept control in transformers with Sparse Concept Anchoring

Develop a training-time method for transformers that puts concepts where you can find them, so removal has predictable efficacy and bounded side-effects.

## Project summary

Post-hoc interpretability tools (sparse autoencoders, probes, concept activation vectors) search for concepts after training. If a concept is fragmented across redundant directions, the search can miss some of them, and suppressing the ones you found may leave the rest intact. That is a completeness problem no post-hoc method can rule out.

Sparse Concept Anchoring (SCA) takes a different approach. It adds a light geometric regularizer to the training objective that guides a concept toward a known location (a direction or subspace) using a small number of noisy labels. In effect, it lets you shape feature geometry during training, rather than reverse-engineering it later. Because the concept then lives where you put it, suppressing or ablating it has side-effects you can bound analytically from the geometry before you run the intervention. SCA is currently the only training-time localization method with that property.

This is part of a milestone program.

- M1 anchored concepts in autoencoders (done, [published](https://arxiv.org/abs/2512.12469); [blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts); [source](https://github.com/z0u/ex-preppy))
- **M2 tests whether it transfers to transformers (this project)**
- M3 & M4 carry it to small language models and LLM fine-tunes with real safety targets.

M2 works in a synthetic color-mixing domain with unambiguous ground truth (`red + blue = purple`), in a small transformer:

- D2.1: Does SCA work in a transformer at all? Anchor a concept such as _red_ across the residual stream in the color-mixing task; probe each layer for the anchored concept, and confirm that completion accuracy (predicting the correct result color) matches an un-anchored baseline.
- D2.2: Anchor an abstract _operation_ (e.g. _addition_, rather than a concrete attribute like _redness_); sweep over layers; confirm task performance is intact and that suppression scales as it did in the autoencoders.
- D2.3: Add a verification task (`red + blue = purple TRUE/FALSE`) and test whether suppression can degrade _completion_ while preserving _verification_: the experimental analog for letting a model recognize a behavior without being able to produce it.
- D2.4: Consolidation/outreach/publication.

Why the synthetic domain first, rather than language models straight away? A toy domain keeps a negative result interpretable. In an LLM, a null result at this stage could mean the method failed, or it could mean any of the other things that can go wrong in language-model training. The color domain removes those confounds, so M2 is a clean test of the method itself; M3 and M4 then carry it to natural-language models and real safety targets (sycophancy the lead candidate).

## Working in this repo

Two entrypoints, split by what they touch: `./go` for the repo (deps, checks, reports, the site) and `bin/mini` for experiments (compute, durable results). Both print usage when run bare.

A full pass over one experiment looks like this:

```bash
./go install                                   # once per checkout: deps + git hooks
./go auth --check                              # confirm Modal + HF credentials

# Science loop — experiments run under mini (memoized: re-runs only what changed)
bin/mini run docs/m2/ex-2.1.2/experiment.py --app modal --budget 2h
bin/mini status ex-2.1.2                       # read-only; also: watch, logs, results
bin/mini retry docs/m2/ex-2.1.2/experiment.py  # after a fix — finished tasks are reused

# Report loop — reports read stored results; they never re-run the experiment
./go open docs/m2/ex-2.1.2/report.py           # edit live in marimo
./go preview                                   # export stale reports → local site → :8000
./go check --fix                               # lint/format/types/tests before committing

# Ship
./go publish docs/m2/ex-2.1.2/report.py        # sync the report bundle to the publish tier
git push                                       # open a PR; merging to main deploys the site
```

CI uses the same verbs: the checks from `./go check` (split per step in `lint-check.yml`), and `./go site` (`pr-preview.yml`, `publish-docs.yml`), which assembles the public site from *published* bundles — read-only, never runs a notebook, assets stay on the CDN. `./go preview` is its local sibling: the same site, but built from local exports with assets copied beside the HTML, so it works offline (`--no-serve` to just build). The two are separate verbs on purpose — preview answers "what does my work look like?", site answers "what will the internet see?".

Conventions for the site live in [docs/README.md](docs/README.md); the *why* behind the publishing pipeline in [eng/](eng/README.md); experiment authoring in the `mi-ni` skill.

## Related work

The closest existing method, gradient routing ([Cloud et al., 2024](https://arxiv.org/abs/2410.04332)), also steers where a concept lands during training, but by masking gradients rather than shaping the loss.
