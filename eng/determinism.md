# Reproducible GPU runs

*Part of the [engineering notes](./README.md).*

Two tasks with the same code and the same inputs should produce the same result. On a GPU they don't, by default — and because the memo keys a task on the *content* of its inputs, that turns one re-run upstream into a re-run of everything downstream.

## What we measured

150 training steps of the real `train_step` (nGPT, d64-L4, batch 64×128, one seed), three replicas in three separate `single_use_containers=True` L4 containers, `us-east`. The probe hashes the final weights; the flags ride in the container environment.

| `XLA_FLAGS` (beyond the CPU scheduler flag) | distinct digests | seconds |
| --- | --- | --- |
| *(none)* | **3 of 3** | 29.0, 29.8, 29.0 |
| `deterministic_ops` | 1 | 49.6, 79.3, 49.0 |
| `deterministic_ops` + `autotune_level=0` | 1 | 49.3, 83.7, 50.2 |
| all three | 1 | 49.1, 85.3, 48.3 |
| **`deterministic_ops` + `exclude_nondeterministic_ops`** (shipped) | 1 | 82.9, 48.0, 49.0 |

The baseline's three final losses were 4.147422, 4.147426, 4.147430 — a drift in the sixth significant figure, which is invisible in a plot and total for a hash. All four deterministic settings landed on the *same* digest (`6d18d792…`), so there's one canonical answer they agree on rather than four flavours of stable.

One replica per batch runs 1.6–1.7× slower than its siblings, in the deterministic rows only. It was the middle replica three times running and then the first one, so it reads as an unlucky container rather than anything about the flags — but it does mean a run's slowest cell isn't well predicted by its fastest. The cost figure to plan around is the fast replicas: **~1.7×** (29s → 48s) on a cell this size.

### Re-measured at full loop length (2026-09-27)

The table above is 150 steps, so it is mostly startup and compilation. A second probe ran ex-2.2.14's real `train_one` for 30 epochs (2,970 steps), two single-use containers per cell, jax 0.11.0, timing the steady-state step after the first 200 steps:

| GPU | flags | ms / step | first step (compile) | task wall |
| --- | --- | --- | --- | --- |
| L4 | deterministic | 9.3, 11.2 | 11.3s, 10.3s | 85s, 84s |
| L4 | none | 11.6, 10.7 | 7.8s, 7.5s | 86s, 73s |
| T4 | deterministic | 19.3, 17.9 | 7.0s, 6.3s | 86s, 78s |
| T4 | none | 18.2, 20.3 | 7.5s, 8.7s | 79s, 90s |

At this length the flags cost about three seconds of compilation and nothing measurable per step: the two settings overlap within container-to-container spread. The 1.7× above does not carry over to a full run, so it is not a reason to turn the flags off.

The same probe shows what the flags buy. Both deterministic replicas of each cell finished bit-identical, while the two nondeterministic L4 replicas ended at `m_line` 0.867 and 0.888, and the T4 pair at 0.856 and 0.875. Drift in the sixth significant figure at 150 steps grows to the second decimal of a headline metric by 3,000 steps.

It also answers the open question below about GPU models: the deterministic runs agree within a device class and differ across them (`m_line` 0.866 on L4, 0.866 on T4, 0.845 on A10, 0.857 on L40S, one seed throughout). Moving a role to another GPU is a numerics change in the same sense as a jax upgrade.

## Seeds are draws

Determinism makes a run repeatable. It does not make a seed mean the same thing under two conditions. A small change to a hyperparameter sends the runs of one seed apart early in training: in the *fixed-confirm* experiment, the two recipes differed by 0.6% in the slice temperature at epoch 32, and the runs of a seed were already up to 0.43 apart in removal. By epoch 400, removal under the two recipes correlated about 0.2 across eight seeds, and in the sans-diff report two runs at one seed were about as far apart as runs at different seeds. A change of GPU model does the same.

So we keep the flags for what they buy on their own (a stable memo, exact reproductions of a run, and stopped-and-resumed runs that match runs trained straight through), and compare conditions as two samples rather than seed by seed. Pairing by seed is worth it only where runs branch from one stored state and are measured soon after. The rules for reports are in the design checks of the `sci-report` skill, and for the choice of card in the `mi-ni` authoring reference. Decided 2026-10-29, from a paper review of the sans-diff report; the comparison of costs and seed counts is in the project files under `determinism/`.

That view also settles the choice of card. Since about 2026-10-27, L4s on Modal have been scarce: a training task waited a median of 12 to 28 minutes for one, with a tail past an hour, while A10Gs and L40Ss scheduled in seconds. If a card change counts as seed noise, a condition can run on a mix of cards, so training roles list `["L40S", "L4", "A10G"]` and Modal picks the first one free. The mix then depends on when a condition launched, so each report shows the cards per condition, and a close comparison is checked again on the runs of one card alone. Arms that branch from one stored state and are compared in pairs stay on one card. Decided 2026-10-30 with the L4 first; the L40S went ahead of the A10G the same day, when the seed-pack probe found it the cheaper of the two per step, and ahead of the L4 on 2026-10-31, when the scatter-free step and the batch draw on its own thread made it the cheapest per step of the three (`todo/eng/scatter-free-training-step.md`).

## What we set, and why only that

`[tool.mini] env` in `pyproject.toml` carries the CPU scheduler flag plus:

- **`--xla_gpu_deterministic_ops=true`** — the one that does the work. It makes scatter deterministic through XLA's scatter-determinism expander (an optimized rewrite, so it's much cheaper than the fallback). Our model had one big scatter: the token-embedding gradient. Without the flag those accumulate in whatever order the atomics land, which is a different order every run. Since 2026-10-31 the training step has no scatter at all (`todo/eng/scatter-free-training-step.md`), and the flags are still needed (below).
- **`--xla_gpu_exclude_nondeterministic_ops=true`** — the broader guard, for ops the scatter expander doesn't cover. It measured free at our sizes, and it's what keeps a future model that reaches for some other atomically-reducing op from quietly reintroducing drift.

The pair was measured on its own (last row) rather than inferred from the three-flag run, so what's configured is what was tested.

**Autotuning is in effect off.** It's the other classic source of GPU nondeterminism — the algorithm pick comes from measured timings, so a noisy host can pick differently. We left `--xla_gpu_autotune_level` alone, since setting it to 0 changed neither the digest nor the time here. But with either flag above set, XLA's GPU backend skips the timing and takes the first candidate it has for each GEMM and fusion (`RequireDeterminism` and `PickBestResult` in its source; the [XLA determinism guide](https://openxla.org/xla/determinism) says so too), which is why the level made no difference. The seed-pack probe found the untimed picks cost about a fifth of a step on an L40S (`/mnt/project-files/pack-timing/why-packing-doesnt-pay.md`).

### With no scatter left (2026-10-31)

The flags were there for the scatters, so once the step had none, we checked whether it was deterministic without them. The step-speed probe (`docs/m2/step-speed`, dev profile) trained the recipe of record for 40 epochs (about 2,100 steps, on a fifth of its corpus) in separate containers, with the flags and without:

| card | flags | containers | distinct digests | ms per step |
| --- | --- | --- | --- | --- |
| L40S | on | 4 | 1 | 3.0 to 3.4 |
| L40S | off | 6 | 6 | 2.7 to 3.1 |
| L4 | off | 4 | 4 | 6.2 to 7.2 |

Without the flags, every container ended on weights of its own, on both cards. So the scatters were not the only source of drift. The likely remaining one is the autotuner's timed pick, and a fourth round tested it by pinning the picks: JAX's compilation cache in a directory on the Volume, filled by one task and then loaded by three more, each in a container of its own, on the recipe's full corpus and with the batch draw on its own thread:

| card | flags | containers | distinct digests | ms per step |
| --- | --- | --- | --- | --- |
| L40S | on, no cache | 1 | | 2.42 |
| L40S | off, shared cache | 4 | 1 | 2.0 to 2.1 |
| L4 | on, no cache | 1 | | 6.74 |
| L4 | off, shared cache | 3 | 1 | 5.6 to 5.7 |

With the compiled programs shared, the flags-off runs agree bit for bit on both cards, and a step is about a sixth faster than with the flags on. So the flags are not the only way to a reproducible run any more. They stay on for now, because the cache brings questions of its own: a run reproduces only while its compiled program is in the cache, so deleting the cache, or two first compiles racing in one fan-out, would let one memo key map to two numerically different results (the quiet failure the JAX version has, below). The open item is `todo/eng/flags-off-with-a-compilation-cache.md`.

The digest does not hold across GPU *models* (measured in the full-length probe above): determinism is a per-device-class property. Training roles list `["L40S", "L4", "A10G"]`, so a re-run can land on another card and give another digest; the card is recorded on every attempt and returned in training results (see [Seeds are draws](#seeds-are-draws)).

## Why the environment, and not a line of Python

A Modal container is reused across the tasks of a map, and XLA parses `XLA_FLAGS` once, when its backend comes up. So a task that sets the variable on itself only wins if it's the first task on that container — every later one silently inherits whatever the first one had. (This is why `sca/__init__.py`'s scheduler flag was never reliable on Modal; it's still right for the interactive path, where a notebook is one process and imports `sca` before touching jax.)

So it's an apparatus option: `env=`, which `ModalApparatus` turns into a container Secret and `LocalApparatus` overlays onto the task worker's subprocess. `[tool.mini] env` supplies it project-wide — the same shape as `region` — and a role's own `env=` merges over it key by key. It is deliberately **not** a credential channel: the values are recorded on the task record, and `secrets=` is still there for tokens.

## Why this isn't in the memo fingerprint

Flipping the flags changes results without changing code or inputs, so the memo can't see it. We chose not to make it evidence:

- Turning determinism on today would invalidate every DONE record in the project and re-run four published sweeps to reproduce numbers we already have. The existing records are honest about what they are; they just came from the older regime.
- The question this actually needs to answer is "what was *this* number computed under?", which is provenance, not identity. `compute_env` records `XLA_FLAGS` on every attempt (`env.numerics_env`, alongside the GPU model and region), read from the worker's own environment — so a setting that never reached the container reads as absent instead of as what the client meant to send.

If we ever do want a re-run on a flag change, `version=` on the task is the explicit lever, and it lands as a new attempt on the same record.

## The JAX version moves the digest too

The flags are not the only thing outside the fingerprint that changes the number. Measured while upgrading JAX for #73 — same code, same inputs, same seed, same L4, same `XLA_FLAGS`, 150 steps of the real `train_step` on nGPT d64-L4, three single-use containers per version:

| jax / jaxlib | digest | final loss |
| --- | --- | --- |
| 0.10.1 | `e4b7c106…` | 5.531451225280762 |
| 0.11.0 | `884ee00c…` | 5.531452655792236 |

One distinct digest within each version, so 0.11 is every bit as reproducible as 0.10 — it just lands somewhere else, about a part in 4×10⁶ away. That is the same magnitude as the nondeterminism the flags were introduced to remove, which is worth sitting with: the flags buy reproducibility *within* a pinned environment, not across one.

The memo cannot see this. `_is_project_file` excludes site-packages, so no library source reaches the manifest, and `task_key_parts` composes identity from the fn's module-qualified name and an input fingerprint — neither of which mentions a package version. Checked directly: no `jax`, `jaxlib` or version string appears anywhere in a task's key or its evidence. So the failure mode is not a spuriously invalidated key (harmless — just a re-run) but the quiet one: **the same key mapping to a numerically different result after an upgrade.** A DONE record written under 0.10.1 keeps serving its old result, and a task re-run for any other reason silently produces the new one, under the key that already meant the old.

The same reasoning as above still applies — this is provenance, not identity, and `compute_env` is where it belongs. But unlike `XLA_FLAGS`, a version bump arrives with an ordinary dependency upgrade rather than a deliberate flag change, so it wants a deliberate gate. When a numerics-relevant package moves, either accept that records straddle two regimes and say so where the numbers are published, or bump `version=` on the affected tasks to force a clean re-run.

## The gate: detection, not invalidation

`compute_env` records the version of each package in `mini.runs._NUMERICS_PACKAGES` (`jax`, `jaxlib`, `numpy`) on every attempt, as `env.numerics_packages`. That is what makes the straddle *visible*: `numerics_drift` compares a record's versions against the ones installed now, and two readers act on it.

- A **tick** notes every memo hit whose result predates an upgrade, and logs one warning naming what moved (`jax 0.10.1 → 0.11.0`) and how many hits it covers. Said once per answer, per process — a watching driver ticks every few seconds and the answer doesn't change — but a later wake that reaches more drifted hits (a suspended tick only walks the DAG to its suspension point) speaks again with the larger count. A run that straddles an upgrade can carry several recorded versions of one package; the warning names them all (`jax 0.9.0/0.10.1 → 0.11.0`).
- **`mini status`** re-derives the same comparison from the records alone (`⚠ N result(s) computed under different numerics: …`, and a `numerics_drift` object under `--json`). No import, no tick, no fingerprinting, so it still answers weeks later, which is when someone is usually asking where a number came from.

Nothing re-runs on its own, which is the point: the levers stay `version=` and an honest note beside the published numbers. Two things this does not see. A package outside the list moves silently — the list is what a measurement has implicated, and adding a name only affects records written afterwards. And the comparison is against the *driver's* installed set, not the container's; the Modal image is frozen from the same lock, so they agree unless something has been pinned apart deliberately. Records written before this shipped carry no versions and read as quiet rather than as drifted: no evidence, rather than evidence of no change.

## The alternative we didn't take: tolerating small differences

The tempting move is to leave the GPU alone and teach the DAG that 4.147422 and 4.147430 are the same number. It doesn't work, for a reason worth writing down: near-equality isn't an equivalence relation. `a ≈ b` and `b ≈ c` doesn't give `a ≈ c`, so there's no way to bucket values into hashable classes. Concretely:

- **Rounding before hashing** just relocates the problem to the bucket boundaries. Two values a hair apart that straddle one still hash differently, so misses become rarer *and* less predictable — the worst combination for a cache you're trying to trust. It also does nothing for the input that dominates here, which is a checkpoint blob.
- **Keeping the previous result when the new one is within tolerance** does hold the downstream keys still, and it's the only version that's mechanically compatible with content addressing. But it serves a result the current code didn't produce, it needs a tolerance per result *shape* (there isn't a meaningful one for checkpoint bytes), and it makes the DAG's answer depend on run history rather than on inputs — which is the property that makes a memo worth trusting.
- **Keying downstream on the upstream's identity rather than its bytes** — pass the config and seed, stash the weights beside it — is the honest version, and it's the fallback if determinism ever costs too much. It's also a deliberate trade: a genuine change in the upstream result stops invalidating downstream. Worth pairing with recording the checkpoint digest, so a surprise is at least detectable after the fact.

Determinism is simply the cheaper lever: one flag, one place, a few seconds of compilation per training task, and the DAG's equality test keeps meaning what it says.

## Re-measuring

The probe isn't checked in — it needs a GPU, so it can't be a test. It's ~70 lines: map the real `train_step` over N replicas with `single_use_containers=True` and `.w(env={"XLA_FLAGS": ...})`, hash `jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))`, and compare digests across containers. Worth re-running when the model grows a new kind of op, or when a jax upgrade might have changed what the flags cost.
