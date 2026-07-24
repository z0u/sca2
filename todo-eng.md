# Engineering todo

Scratchpad for deferred *infrastructure* work that isn't worth a tracking issue
yet — tooling, storage, publishing, CLI, and the `mini` library. Science
questions and experiment findings live in [`todo-science.md`](./todo-science.md).
When something here grows real, promote it to a GitHub issue and remove it from
this list.

Scratch items sit under _Scratch_; everything below that is the prioritized
index into GitHub issues. Durable design rationale and recorded
decisions live in [`eng/`](./eng/README.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Scratch

- **Function-local imports hide a task's real dependencies from the evidence
  fingerprint (2026-07-24, ex-2.1.5).** Dropping the `as2` landmark should have
  re-run the 24 probe cells, but they memo-hit and served stale 16-column arrays.
  Cause: `eval_one` imports `collect_activations`/`probe_maps`/etc. *inside the
  function* (to keep `main` and the CLI light — they pull jax), and the
  fingerprinter only traces module-level references — so `sca.compute.geometry`,
  and the `LANDMARKS` it reads, never enter `eval_one`'s evidence (`mini explain`
  lists only the experiment-module helpers). Any change to geometry code or the
  landmark scheme is then a silent stale hit. Workaround in place: the eval map is
  tagged `version="lm-<sha1(LANDMARKS)>"`, so a scheme change re-runs the probes.
  Worth considering: (a) trace function-local imports of *project* modules, or at
  least warn when a task body imports untracked project code; (b) a convention to
  fold such deferred-import deps into `version=`/inputs. General hazard — it hits
  any task that defers heavy imports this way.

- **`mini run` can settle a run as `done` while a downstream task still needs a
  later wake — and an expired budget silently reaps it (2026-07-24, ex-2.1.5).**
  Adding `r2_ch` to `eval_one` correctly re-ran all 24 eval cells and (because an
  `Artifact` keys by content, so publish's `input_fp` moved) correctly gave
  `publish_results` a new key — but it took several `mini run` invocations before
  publish actually executed and `ARRAYS_REF` picked up the new stack. Two traps
  compounded, and cost a long misdiagnosis (I first, wrongly, blamed a publish
  memo-hit — `Artifact` content-addressing already handles this):
  (a) the very first re-run was reaped by an *expired* wall-clock deadline stamped
  by a prior invocation ("budget elapsed — settled CANCELLED", 0 launched); a
  plain `run` past an expired budget does nothing until you re-arm with
  `--budget`. (b) publish only becomes runnable after all 24 evals finish, so an
  earlier wake that launched the eval batch exited before advancing to publish,
  yet `status` still read `done` (of the *prior* settled DAG). Worth a
  `status`/`explain` hint that distinguishes "nothing stale" from "stale but
  reaped by expired budget", and a clearer signal when `done` reflects a
  superseded DAG rather than the current one.

- **Single region.** By default, the Modal container region is unspecified.
  Within a single sweep, containers may be placed anywhere in the world, and
  disparate containers have high latency to the shared Volume, Queue, and Dict.
  This can significantly impact training runs and waste GPU time. It's possible
  to specify the region when launching a function; see
  https://modal.com/docs/guide/region-selection.md. Note that doing so increases
  the cost, so analysis of the trade-off is required; perhaps we can change the
  way we do I/O to avoid the need most of the time.

- **Monitoring should compare against expectations, and the tools should do
  the comparing (2026-07-23, ex-2.1.5).** The haiku experiment-monitor
  reported "progressing normally" while 3 of 5 containers ran 15–30× slow:
  its playbook covered settled/failed/wedged but not deviation-from-expected.
  Agent-side fix applied (anomaly-scan section in
  `.claude/agents/experiment-monitor.md`: sibling throughput comparison,
  finish-time-vs-timeout projection, metric trends, "healthy ≠ nothing
  failed"). Tooling half still open, so a small model reads verdicts instead
  of computing them: (a) `train_model` passes loss in the progress *message
  string*, not the `ProgressMessage.metrics` dict, so status shows
  `"metrics": {}` and no tool can check trends — one-line fix in
  `sca/compute/training.py`, but it's memoization evidence, so bundle with
  the next change that re-runs cells; (b) `status --brief` attention flags
  for throughput outliers (vs sibling median) and projected timeout
  overruns, complementing the existing queued-too-long and stale-progress
  flags; (c) a loss-trend flag once (a) lands. #monitoring

- **Synchronous progress emission serializes training on cross-region queue
  puts (diagnosed 2026-07-23, ex-2.1.5).** Containers outside us-east-1 ran
  identical train cells 15–30× slower (92–220 steps/min vs 2,500–3,500), in
  order of distance from us-east — initially misread as a possible CPU
  fallback; AF's I/O hypothesis was right. Mechanism: `train_model` calls
  `emit_progress` every step; `Debouncer`'s leading edge runs the Modal Queue
  `put` synchronously on the training thread (`mini/progress.py` `_do_emit`,
  `mini/_debounce.py`); and once put latency exceeds `emission_interval`
  (`max_containers / 10` = 0.5 s this run), every step re-triggers the leading
  edge, so the loop degrades to one blocking put per step. Implied put
  latencies from steps/min: ~0.28 s us-west1, ~0.38 s eu-south-2, ~0.65 s
  asia-northeast3 — a few RTTs each, i.e. HTTPS without connection reuse.
  Fixes, in order: (1) emit from a background thread with a single-slot
  latest-wins buffer so the training thread never blocks on the network —
  removes the cliff outright; (2) adaptive interval (≥ k × observed put
  latency) as a cheap guard; (3) optionally region-pin workers to the queue's
  home region for locality. Still worth doing for observability regardless:
  accelerator identity in task `env`, and a `status --brief`
  throughput-outlier flag (steps_per_min under ~⅓ of the sibling median for
  the same fn joins the attention list). Not a wedge: progress heartbeats
  stayed fresh throughout, as the watchdog's stale-progress flag is designed
  to check. Knock-on cost: the train role's 1.5 h timeout was sized for
  full-speed cells, so the slowdown turned it into a kill switch — the
  asia-northeast3 cell was killed at step 7,895 of 7,900 and retrained from
  scratch (a fast container redid it in 5 min). Timeouts sized to a multiple
  of expected duration only work if throughput is observable and roughly
  uniform; the background-emit fix restores that assumption.

  Also, should we even be using a Queue? It's useful when calling Apparatus.run
  directly, but when polling/ticking the DAG, it may not be the right container.
  We also have a Dict and maybe that's enough; consider whether it makes sense
  to prevent Queue use when using the `mini.orchestration` path, and maybe even
  remove the Queue altogether (it has other hazards, e.g. it may fill up if
  there's no consumer).

- **Science skill.** We have a fledgeling `science` skill that describes how to
  collaborate on experiment design. There may be old descisions in
  todo-science.md that could be moved there and polished.

- **First-run Modal image build can eat a small `--budget` (observed
  2026-07-20).** In a fresh Modal environment the first launch spends minutes
  building the container image while the task sits `queued`; a `--budget 10m`
  expired during the build and the watch's opportunistic enforcement settled
  the run CANCELLED before any work ran. Harmless-but-confusing: the image is
  cached, so a `retry --budget …` succeeds immediately (that's what happened).
  Options if it bites again: exclude time-in-queue from the budget clock (risky
  — queue time is exactly what the budget guards on a capacity-starved run), or
  just document "size the first run's budget for the image build" in
  running.md. Leaning documentation-only.

- **Document subline.** Describe subline in a skill: what it is, why we might
  use it instead of a token heatmap, and how to use it.

- **Document s_2.** Describe surprise-surprise in a skill: what it is, why we
  might use it instead of surprisal, and how to calculate it. The mean s_2 over
  a sequence would be analogous to perplexity. It's probably more informative
  than perplexity alone, since it would capture the _per-token_ difference from
  what the model anticipated. Note that negative values of s_2 are rare and
  probably uninformative; they suggest the model finds the token to be
  unsurprising.

- **Dark-mode rim on `plot_latent_disc` (opened 2026-07-21).** The disc's
  over-the-data rim is a hard-coded `#0005`, which over the `#111` dark fill is
  effectively invisible — it only reads where data covers it. The new
  `sca.vis.plot_rgb_cube` uses `light_dark("#0005", "#fff4")` instead, so the
  two bounds are now drawn differently. Worth unifying, but changing
  `plot_latent_disc` restyles the published ex-2.9.x figures, so it wants a
  deliberate pass over those rather than a drive-by edit.

- **Blend modes in matplotlib figures (opened 2026-07-21).** matplotlib has no
  `mix-blend-mode` — no compositing operators on artists at all. Where several
  series coincide (e.g. the RGB channels in ex-2.1.4's answer-schedule), the last
  one drawn wins and the rest are hidden. `mini.vis.smooth_step` sidesteps it with
  tapered line widths, which works but encodes an arbitrary draw order in the
  widths. A real multiply/screen is possible: render each series to its own RGBA
  buffer and composite in numpy. Two things to get right if we build it — the
  chrome (axes, grid, text) must be a separate layer that is *not* blended, or
  labels over- and under-expose; and each layer's empty pixels must contribute the
  mode's identity (1 for multiply, 0 for screen) rather than the background color,
  or the background gets blended in once per layer. That second one only shows up
  in dark mode, since empty-over-white happens to equal multiply's identity.
  Subline gets all of this free because SVG has the property natively
  ([`subline.py`](src/subline/subline.py) sets `--blend-mode` and applies it to
  the series paths only) — worth revisiting if a second figure wants it.

- **Slope-capped sublines (opened 2026-07-21).** `Sparkline._create_path_data`
  takes its curve knots from glyph ink bounds, so a ramp is always one inter-glyph
  gap wide however big the jump is. A large step in surprisal therefore renders
  near-vertical, which reads as a discontinuity and gives up the rate-of-change
  cue the smooth step exists for. Deriving the ramp width from the jump height
  instead (cap the on-screen angle, then shrink adjacent ramps so a plateau
  survives) fixes it, but it restyles the published ex-2.1.1/2.1.2 figures, so it
  wants an opt-in parameter and a deliberate pass rather than a drive-by edit.

- **GPU determinism.** Configure GPU runs to use deterministic computation when
  we care about reproducibility (e.g. when we want to refer to a stable
  measurement from a particular seed).

- **Responsive multi-panel figures in reports (opened 2026-07-16).** The
  ex-2.1.1 two-panel *named-pair lattice* was split into two independent
  `themed` figures wrapped in a `.report-figure-row` (inline-block, reflows to
  a stack on narrow screens; matched size via a shared projection + pinned
  limits + full-figure bbox rather than `sharey`). That pattern works when the
  panels carry no shared axis *labels* and share only a scale. Still undecided
  for the remaining wide plots — `accuracy-sweep` (1×4) and `probe-r2` (1×3),
  which shrink illegibly on phones: (a) split like the lattice, but then we
  must manage the shared y-axis label and legend that currently live only on
  the leftmost panel; or (b) keep them single figures and give each a declared
  native width (e.g. `style="--mini-fig-width: 700px"`) that a wrapper turns
  into a `min-width` + horizontal scroll box, so they scroll instead of
  shrinking below legibility (mirrors the `.report-table-scroll` fix). Option
  (b) is less disruptive and generalizes; the open question is where the
  min-width/scroll wrapper lives — a `themed` option, or a CSS class the author
  opts into. Decide before the anchoring reports reuse these figures.

- **Reconsider WandB (or a hosted tracker) at M3/M4 planning.** Removed from M2
  (2026-07-17): it was authenticated and a declared dep but unused — `mini`'s own
  stack covers everything M2 needs (live `emit_metrics`/`watch`, content-addressed
  artifact/checkpoint versioning, git-aware lineage, memoized sweeps, Modal cost).
  The five things a hosted tracker adds and `mini` doesn't — persisted metric
  *time-series* (mini keeps only the latest value per key), an interactive
  live-curve dashboard, cross-run/sweep comparison UI, live GPU/system-utilization
  telemetry, and grouped-hyperparameter views — don't earn their keep on M2's short
  synthetic-domain runs with publication-curated matplotlib figures. They get more
  attractive at M3/M4 (small LMs, then LLM fine-tunes): longer, costlier runs and
  many un-curated runs to compare. Revisit then; if we do, the cheapest first step
  is per-step time-series persistence in `mini` (extend `emit_metrics` past
  last-writer-wins), not necessarily WandB.

- **CLI usability, remaining gaps** (from the 2026-07-14 cold-exploration
  session; the copy-pasteable-hints / sorting / help-text tier shipped — see
  #57 for the running thread):
  - No way to *delete* an experiment's memo state. `mini gc <name>` sweeps only
    stale attempt files/superseded records, so a scratch or renamed experiment's
    DONE records live forever — on Modal too (a `cli-probe` probe experiment now
    sits there as a permanent example). Wants a `mini rm <name>` with the same
    dry-run-by-default posture as gc.
  - `mini ls` reads local launch state only and (alone among the verbs) has no
    `--app` — there's no way to enumerate experiments that exist on Modal; you
    must already know the name. The empty-state hint now says so, but listing
    would be better.
  - `mini results <name>` prints raw result reprs; a sweep with per-step metric
    lists dumps ~120 KB of floats. The new optional `key` arg narrows it, but
    consider truncating long reprs by default and/or `--json`.
  - `mini logs` holds only failure tracebacks (now stated honestly), and the
    Modal `fc-…` ids that `status` prints can't be fed back into any `mini`
    verb — worker stdout/logs need the Modal dashboard.
  - `tests/mini/test_apparatus.py::test_local_apparatus_concurrent` asserts
    3 × 0.1 s sleeps finish < 0.25 s; on a loaded 4-CPU remote container the
    pool takes ~1.9 s, so the test fails on a pristine tree. Loosen the bound
    or gate it on available CPU.

- **Published reports depend on jsDelivr for the marimo runtime.** `marimo export
  html` points ~200 `<script>`/`<link>`/font URLs at
  `cdn.jsdelivr.net/npm/@marimo-team/frontend@<version>/dist`, so a published
  report only renders while that CDN + the pinned version stay up. Not worth doing
  now, but for archival we could self-host `dist/` into each bundle's `_assets/`
  and rewrite the CDN base to a relative path in `clean_docs`/`export_reports`
  (same post-export surgery seam as the show-code shim). Cost: ~a few MB of
  JS/fonts per bundle and a maintenance tie to the marimo version. (The *local*
  half — repointing CDN refs at marimo's bundled `_static/` to browser-check an
  export offline — is done: see the `report-render` skill.)

- **Publish-tier exports go stale on rename.** `export_key` derives from the
  docs-relative path, so moving a notebook orphans its synced bundle: the build
  looks for the new key, skips with a warning, and the site 404s while
  `index.md` still links the page. The `docs/m1/` casualties (ex-2.9.1..4,
  stranded by 31e103e) were moved to their new keys on 2026-07-14;
  `exports/ngpt-sweep` (notebook renamed to ngpt-scaling) is still there as
  pure cruft. Prevention: teach `./go publish` (or the build) to list remote
  export keys and warn on ones with no matching notebook, and/or a `./go
  publish --move old new` verb. Consider folding orphan cleanup into
  `mini gc --store`.

- Cross-experiment lineage is now **auto-detected**: `set_ref` in a task worker
  stamps producer identity onto the ref (via an ambient `producer_context`, so
  the project-shared `Store` stays experiment-agnostic), `get_ref` records the
  resolution on the task record (`upstream_refs`), and the driver rolls both
  into `lineage.upstreams`. `Experiment(deps=[...])` remains for upstreams a run
  doesn't read via a ref. Known gaps: refs written by the interactive
  `Apparatus` (`app.map` in a notebook) or driver-side code are unstamped, and
  a consumer served entirely from memo hits records nothing new — its
  previously-recorded `upstream_refs` persist on the old records, which is
  usually what you want. Pre-existing refs (e.g. the m1 `reports/*` ones) stay
  unstamped until their publish step re-runs, so their report footers are empty
  for now.

- Modal `mem_total_gb` in a task's `env` reads the *host* total from
  `/proc/meminfo` (gvisor shows the whole node, ~186–363 GB), not the container's
  memory limit. Fine as a coarse "what class of machine" signal; if we ever want
  the true per-container cap, read the requested `memory=` from the role config
  instead (or the cgroup limit, if gvisor exposes it).

- `mini.temporal` can't drive feedback control. `DynamicProp.set()` retargets
  mid-flight from the current (value, velocity) state — exactly what a
  controller needs — but experiments consume schedules via `realize_timeline`,
  which bakes the dopesheet into a static per-step array before training, and
  the dopesheet's own keyframes would fight any runtime `set()` calls on the
  same prop. Ex-2.9.4's controller therefore lives inside the training loop
  (duals in the `lax.scan` carry), with the dopesheet still driving the
  non-controlled props. If feedback-driven weights become standard, consider a
  Timeline mode where a prop is declared "controlled": keyframes set its
  *bounds/defaults* and a callback supplies the live value.

- ex-2.9.3's `publish_results` publishes `exemplar-hot`/`exemplar-cool` refs
  (the worst catastrophic run and its cooled-LR rescue) that no report reads.
  Either add the intended before/after rescue figure to the ex-2.9.3 report
  (mirroring ex-2.9.2's exemplar plot) or drop the two `set_ref` calls and the
  `worst`/`rescue` computation.

- ngpt-scaling's sweep cells all `save_checkpoint` to the same shared
  `get_data_dir()`, so the checkpoint file is last-writer-wins across a fan-out.
  Harmless there (cells return their metrics; nothing reads the checkpoints
  back). `train_model` now takes a `checkpoint_dir` for per-cell keying —
  ex-2.1.1 uses it because its eval step reads checkpoints back — but
  ngpt-scaling still writes to the shared default.

- The ex-2.1.1 report refs moved to `reports/m2/ex-2.1.1/*`; the pre-rename
  `reports/ex-2.1.1/*` refs still sit in the store (there's no ref-delete API).
  Harmless clutter, but they pin their artifacts through GC's mark-and-sweep.
  If a ref-delete/rename verb ever lands (eng/gc.md), sweep them. The m1 refs
  (`reports/ex-2.9.*`) predate milestone nesting and stay flat on purpose.

- Remove the remaining mi-ni template *experiments* (`docs/pipeline`,
  `docs/probe`, `docs/acts` — their report notebooks are already gone) once the
  e2e tests that drive them (`tests/mini/test_experiments_e2e.py`) get their own
  fixtures, or once the first real M2 experiments can play that role. Ties into
  #45 (docs rework). (`docs/gpt-sweep` has since become `docs/ngpt-scaling`, a
  real Iteration 0 output rather than a template.)

## Backlog, grouped by what a single dev session should bundle

(M2 *science* backlog, including issue #10, now lives in
[`todo-science.md`](./todo-science.md).)

**Storage/control-plane design.** These stem from the same list in
[`eng/decisions.md`](./eng/decisions.md):

- #38 — publish-tier hardening (private-CAS/public-publish bucket split;
  citable versioned publish via a dataset repo). Only matters once the template
  is used for work that shouldn't be world-readable by default.
- Settled: #46 shipped (gen-fenced `set_ref`/`publish` + `StaleWriteError`,
  PR #56). #37 (implicit cross-experiment dedup + shared working volume) closed
  as not planned — the explicit ref path covers reuse; reopen only if
  identical-prep recompute becomes a real recurring cost.

**Sequence after the above:**

- #15 — GC across the control plane, I/O-plane volume dirs, and the CAS.
  Shipped in two cuts: the local per-experiment control-plane + I/O-plane sweep
  (`mini gc <name>`, PR #49), then the Modal Volume sweep and the CAS
  mark-and-sweep (`mini gc --store`, PR #60). Rationale and safety posture in
  [`eng/gc.md`](./eng/gc.md). Only #38 (bucket split) would
  still reshape the CAS leg; the `mini-hf-cache` Volume (#50) stays out of scope
  (pure cache — `modal volume delete mini-hf-cache` is a safe reset).

**Orthogonal, no code overlap with the above:**

- #45 — docs rework. Touches `docs/`, `README.md`, `eng/`, not `src/mini/`.
  Can run in parallel with anything.
- #57 — CLI DevX: tier 1 shipped (`_load_experiment_or_hint` gives a friendly
  error + the `path` positional documents file-vs-NAME). Anything beyond that
  (e.g. auto-resolving a name to its experiment file) is still open.
