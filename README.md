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

Use `./go` for the repo (deps, checks, reports, the site) and `mini` for
experiments (compute, durable results).

A full pass over one experiment looks like this:

```bash
./go install                                   # once per checkout: deps + git hooks
./go auth --check                              # confirm Modal + HF credentials

# Science loop — experiments run under mini (memoized: re-runs only what changed)
mini run docs/m2/ex-2.1.2/experiment.py --app modal --budget 2h
mini status ex-2.1.2                           # read-only; also: watch, logs, results
mini retry docs/m2/ex-2.1.2/experiment.py      # after a fix — finished tasks are reused

# Report loop — reports read stored results; they never re-run the experiment
./go open docs/m2/ex-2.1.2/report.py           # edit live in marimo
./go preview                                   # export stale reports → local site → :8000
./go check --fix                               # lint/format/types/tests before committing

# Ship
./go publish docs/m2/ex-2.1.2/report.py        # sync the report bundle to the publish tier
git push                                       # open a PR; merging to main deploys the site
```

<details>
<summary><b>Launch &amp; advance</b> — <code>mini run</code></summary>

`mini run` launches the next stage and returns at once. Each call advances the DAG by one wake; finished tasks are memo hits, so re-running only does what's left.

```console
$ bin/mini run docs/demo/experiment.py --app modal --budget 2h
demo:
  ◌ train_one  train_one-5b23f9ed87eb  queued  ⧖ queued 1s ago  [fc-01KXYEE4WJ…]
  ◌ train_one  train_one-dbf269543ed8  queued  ⧖ queued 1s ago  [fc-01KXYEE503…]
… suspended — 2 task(s) in flight (re-run to advance)
```

`◌ queued` is launched-but-not-yet-running; `--budget` caps the run's wall-clock.
</details>

<details>
<summary><b>Wait for the next event</b> — <code>mini watch</code></summary>

`mini watch` blocks until the run settles or something needs your attention. It never advances the DAG, so interrupting it leaves the workers running.

```console
$ bin/mini watch demo --timeout 10m
⚠ needs attention: train_one-dbf269543ed8 settled failed
demo  —  running  (3 tasks)
```

Exit codes: `0`: settled all-done, `1`: settled with a failure, `3`: needs attention, `124`: timed out with work still in flight.
</details>

<details>
<summary><b>Check in</b> — <code>mini status --brief</code></summary>

`mini status` is like `watch`, but designed for polling. `--brief` prints the aggregate state, a count by state, and the tasks that need attention (failed / stale / wedged / long-queued):

```console
$ bin/mini status demo --brief
demo  —  failed  (3 tasks)
  2 done · 1 failed
  ✗ train_one  train_one-dbf269543ed8  failed  !! RuntimeError: synthetic mid-stage failure
```

Identical failures collapse to one counted line. A failed task drops off this list once it's re-run (or the function is edited).

Plain `mini status` lists every task with its metrics and heartbeat.
</details>

<details>
<summary><b>Recover</b> — <code>mini logs</code>, <code>mini retry</code></summary>

`FAILED` and `CANCELLED` are terminal: a plain `mini run` won't relaunch them. After fixing the problem `retry` re-runs just the failed tasks (finished ones stay memo hits):

```console
$ bin/mini logs demo train_one-dbf269543ed8
…
RuntimeError: synthetic mid-stage failure

$ bin/mini retry docs/demo/experiment.py --app modal
retrying 1 task(s): train_one-dbf269543ed8
```
</details>

## Related work

The closest existing method, gradient routing ([Cloud et al., 2024](https://arxiv.org/abs/2410.04332)), also steers where a concept lands during training, but by masking gradients rather than shaping the loss.
