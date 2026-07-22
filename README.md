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

Here's what the experiment side of that loop looks like in practice, by intent. _(Outputs are illustrative — a small two-stage `demo` run: a `prep` step, then a `train_one` fan-out.)_

<details>
<summary><b>Launch &amp; advance</b> — <code>run</code> (the only verb that spends money)</summary>

`run` launches the next stage and returns at once — it doesn't block. Each call advances the DAG by one wake; finished tasks are memo hits, so re-running only does what's left.

```console
$ bin/mini run docs/demo/experiment.py --app modal --budget 2h
demo:
  ◌ train_one  train_one-5b23f9ed87eb  queued  ⧖ queued 1s ago  [fc-01KXYEE4WJ…]
  ◌ train_one  train_one-dbf269543ed8  queued  ⧖ queued 1s ago  [fc-01KXYEE503…]
… suspended — 2 task(s) in flight (re-run to advance)
```

`◌ queued` is launched-but-not-yet-running; `--budget` caps the run's wall-clock, so a forgotten or wedged job tears itself down instead of burning money.
</details>

<details>
<summary><b>Wait for the next event</b> — <code>watch</code> (read-only; Ctrl-C is safe)</summary>

`watch` blocks until the run settles _or_ something needs you — a task fails mid-stage, a worker goes stale or wedged — then returns. It never ticks the DAG, so watching costs nothing and interrupting it leaves the workers running.

```console
$ bin/mini watch demo --timeout 10m
⚠ needs attention: train_one-dbf269543ed8 settled failed
demo  —  running  (3 tasks)
```

The **exit code** names what happened — `0` settled all-done, `1` settled with a failure, `3` needs attention now, `124` timed out with work still in flight — so a script (or the babysitting agent) branches without reading the text. `--json` swaps the live progress bars for one compact summary object:

```console
$ bin/mini watch demo --timeout 10m --json
{"experiment": "demo", "app": "local", "state": "running", "settled": false,
 "counts": {"done": 1, "running": 1, "failed": 1},
 "attention": [{"key": "train_one-dbf269543ed8", "fn": "train_one",
                "state": "failed", "error": "RuntimeError: …"}],
 "outcome": "attention", "reason": "train_one-dbf269543ed8 settled failed"}
```

The full monitor loop — which exit code does what — lives in the `mi-ni` skill's `running.md`.
</details>

<details>
<summary><b>Check in</b> — <code>status --brief</code></summary>

`status` is read-only. `--brief` prints the aggregate state, a count by state, and _only_ the tasks that need a look (failed / stale / wedged / long-queued), so a fifty-task sweep doesn't scroll off the screen:

```console
$ bin/mini status demo --brief
demo  —  failed  (3 tasks)
  2 done · 1 failed
  ✗ train_one  train_one-dbf269543ed8  failed  !! RuntimeError: synthetic mid-stage failure
```

Plain `status` lists every task with its metrics and heartbeat; add `--json` (with or without `--brief`) for the machine-readable form.
</details>

<details>
<summary><b>Recover</b> — <code>logs</code>, then <code>retry</code></summary>

`FAILED` and `CANCELLED` are terminal by design — a plain `run` won't relaunch them. Read the traceback, fix, then `retry` re-runs just the failed tasks (finished ones stay memo hits):

```console
$ bin/mini logs demo train_one-dbf269543ed8
…
RuntimeError: synthetic mid-stage failure

$ bin/mini retry docs/demo/experiment.py --app modal
retrying 1 task(s): train_one-dbf269543ed8
```
</details>

CI uses the same verbs: the checks from `./go check` (split per step in `lint-check.yml`), and `./go site` (`pr-preview.yml`, `publish-docs.yml`), which assembles the public site from *published* bundles — read-only, never runs a notebook, assets stay on the CDN. `./go preview` is its local sibling: the same site, but built from local exports with assets copied beside the HTML, so it works offline (`--no-serve` to just build). The two are separate verbs on purpose — preview answers "what does my work look like?", site answers "what will the internet see?".

Conventions for the site live in [docs/README.md](docs/README.md); the *why* behind the publishing pipeline in [eng/](eng/README.md); experiment authoring in the `mi-ni` skill.

## Related work

The closest existing method, gradient routing ([Cloud et al., 2024](https://arxiv.org/abs/2410.04332)), also steers where a concept lands during training, but by masking gradients rather than shaping the loss.
