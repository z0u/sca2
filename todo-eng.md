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

- **Teach the skills and review agents to check for tautologies (2026-07-31).**
  Ex-2.1.7's H4(a) scored the containment of ᾱ in conditions carrying a term
  that is *defined* as a penalty on ᾱ, so "the repulsive term lowers ᾱ" was
  never going to be news — the gate could only ever measure whether the weight
  was large enough, not whether the mechanism explains anything. It went
  unnoticed through preregistration and a prereg review round, and was caught
  only when writing up the results. The general shape: a hypothesis whose
  statistic is the quantity a treatment directly optimizes, or a metric that
  shares its definition with the intervention. Two places to fix it. (1) The
  `science` skill should name the failure mode when discussing how to choose a
  scored statistic, with the rule of thumb that a gate is informative only if a
  reader cannot predict its direction from the method section alone. (2) The
  `prereg-reviewer` agent should carry an explicit check — for each hypothesis,
  ask what the treatment optimizes and whether the scored statistic is that
  quantity, a monotone function of it, or independent of it; only the third is
  a real test. `results-reviewer` should carry the mirror check for claims
  built on such a gate. Worth a look at whether earlier reports have the same
  shape. #skills #agents #methodology

- **The watchdog fires during a task's post-loop artifact upload (2026-07-31).**
  Four of ex-2.1.7's 21 training cells aborted with `WatchdogStall` at step
  3300/3300 — training had finished, and the stall was `put(workdir / "model")`
  pushing a checkpoint to the HF store, which took over 300s with eight
  containers uploading at once. The role's `watchdog` covers the gap between
  step emissions, so a long tail *after* the last step reads as a stall no
  matter how healthy the loop was. `watchdog_grace` already solves the
  mirror-image case at the start of a task, so the shape of a fix exists: either
  a matching teardown grace, or have `put` emit liveness while it uploads (it
  knows the byte count, so it could emit real progress). Worked around in
  ex-2.1.7 by sizing the watchdog for the upload rather than the loop, which
  makes the number mean something different per experiment — worth fixing
  properly, since the failure costs a full re-train of a finished run. #storage
  #cli

- **Glossary of preferred terms for the science skill (2026-07-30).** Reports
  have accreted near-synonyms with drifting meanings: "condition" vs "cell"
  (ex-2.1.7 settled on condition for the seven treatments, cell only for
  condition × seed), plus "arm", "rung", "gate", "partial" vs "named contrary
  reading" (ex-2.1.7's H3 distinguishes these: a partial is a weaker pass, a
  contrary reading is a preregistered interpretation of a miss). A short
  glossary in `.claude/skills/science/SKILL.md` would keep future reports and
  reviews from re-negotiating them. Also note the loss-weight symbol scheme
  adopted in ex-2.1.7: λ subscripted by term (λ_a anchor, λ_s̄ anti-subspace;
  bar = repulsive), replacing the one-off μ.

- **`ty` loses a PEP 695 `type` alias when solving widens it to a supertype
  (2026-07-30).** Reproduced on 0.0.49 (our pin) and 0.0.65 (latest):

  ```python
  type Alias = Sequence[int]
  def same[T](x: Sequence[T]) -> T: ...
  def wider[T](x: Iterable[T]) -> T: ...
  same(a)   # int
  wider(a)  # Unknown
  ```

  So `zip`/`list`/`enumerate`/`max` all drop the type argument, while assignment
  and plain iteration keep it. Found via `zip(cast(AxesRow, axes), rungs)` in
  ex-2.1.6, where `ax` came out `Unknown`. A plain `AxesRow: TypeAlias =
  Sequence[Axes]` works everywhere, so `mini.vis.plt` uses that form and pays a
  runtime `Axes` import for it. [ty#1851](https://github.com/astral-sh/ty/issues/1851)
  is closed and its generic-alias example does pass now — this supertype case
  looks separate and unfiled. Worth filing upstream, and re-testing later so the
  aliases can go back to `type`.

- **Bumped `ty` 0.0.49 -> 0.0.63 (2026-07-30).** `exclude-newer = "3 days"`
  in pyproject.toml capped us below 0.0.64/0.0.65. `ty check` is clean at
  0.0.63. Notable in the gap: uv workspace-root discovery, several PEP 695
  generic-type-alias fixes (0.0.63/0.0.64) — didn't fix the supertype-widening
  bug above though — and ongoing inference-performance work every release.

- **Pre-push `ty` check scans nested worktrees (2026-07-29).** Pushing from a
  `.claude/worktrees/<name>` worktree runs `ty check` from the parent checkout,
  which picks up the worktree's copy of `src/mini` as a separate tree with the
  wrong first-party roots — 23 spurious unresolved-import/type errors on a
  Markdown-only change. The hook (or ty config) should exclude
  `.claude/worktrees/`, or run from the repo root of the tree being pushed.

- **Ex-2.1.5 export time: what's left (2026-07-28).** Two fixes landed —
  `baselines.precision_limited_acc` now finds the nearest candidate with a
  matmul, and `geometry.principal_angles` batches over leading dimensions so
  the report's 2000-sample null is one pair of LAPACK calls. Both are
  bit-identical, verified. Interleaved A/B on the same machine: ~85s → ~46s.

  What's left is matplotlib, and it's structural. `@themed` renders each figure
  twice, and `base.mplstyle` sets `figure.constrained_layout.use: True`, so
  every save solves a layout: 9.4s across the 18 saves, with ~13s in
  `get_tightbbox` overall. Measured on a synthetic multi-panel figure, the
  layout engine costs *about as much as drawing the figure again*, and it
  scales with **panel count, not data volume** — +0.05s per panel (0.05s at 1
  panel, 0.56s at 12, 1.23s at 24), and flat from 200 to 20,000 points per
  panel. It has to measure the rendered extent of every tick label, axis label
  and title to allocate margins, then iterate the solve; our figures are
  12-panel grids, so we pay it 12 times over.

  Two things to try, in order of bluntness:
  - `themed_figure_html` passes `bbox_inches="tight"` *on top of* constrained
    layout, which makes `print_figure` draw the figure a second time to measure
    the crop (the profile shows 36 `figure.draw` for 18 saves). Measured ~20%
    off a save with no visible change in what constrained layout already
    produced — the two are solving nearly the same problem.
  - Beyond that it's the engine itself, and the honest options are fewer panels
    per figure or a fixed layout for the grid-shaped figures that don't need
    solving. Both change margins, so eyeball across a few reports rather than
    swapping blind.

  Reproduce with: `python -m cProfile -o p.prof` around a `MINI_EXPORTING=1`
  `runpy` of the notebook, or monkeypatch `DefaultExecutor.execute_cell` for
  per-cell times. Beware absolute timings across container restarts — the box
  this was measured on drifted ~25% between sessions, so A/B interleaved.

- **"Cell" vs "condition" terminology split (2026-07-27).** Report prose now
  says "condition" for one sweep item, reserving "cell" for visual elements
  (heatmap/table cells). The `mini` library still says "cell" throughout
  (`orchestration.py`, `__main__.py` monitor output, docstrings). Decide whether
  to rename the library term to match — it touches CLI output and docs, so it's
  a deliberate rename, not a sweep-through.

- **Published sweeps are one tick away from a full re-run after an evidence-scheme
  change (2026-07-27).** Widening what the fingerprint tracks re-stamps every
  task's evidence, so the next `mini run` re-runs the whole DAG in place, even
  though no experiment code moved. Adding a small step to ex-2.1.5 tripped this:
  the deferred-import tracing from #58/#59 landed after the sweep, so the tick
  re-ran `prepare_corpus` and would have re-trained all 24 cells. Cost is the
  smaller half of the problem; the real one is that a re-trained sweep may not
  reproduce the numbers a published report already quotes (determinism landed
  after that run too), so the report and the store would silently disagree.
  Nothing to fix in the mechanism itself — over-invalidation is the right bias —
  but two things would help. A read-only `mini plan <exp>` that lists what a tick
  *would* launch and why, so the choice to re-run is made before the launch and
  not after; and something that records, per published ref, the evidence the run
  was produced under, so "this report's numbers predate the current scheme" is a
  fact the report can state rather than a thing you rediscover. The workaround
  for now is what `docs/m2/ex-2.1.5/cross_eval.py` does: read the published
  checkpoints from a standalone script and write results back under their own
  ref, leaving the DAG alone.

- **A settled state can land on a successor's attempt, and the reader then
  trusts it (2026-07-26).** `merge_if` on Modal is read-check-write with a
  one-round-trip window (`ModalRecordStore.merge_if`), so a superseded worker
  can merge `state=DONE` onto a record its successor now owns. The record then
  reads `gen=B, state=DONE` while `result-B.pkl` doesn't exist yet, and
  `Ctx._classify` → `store.result` raises `FileNotFoundError` out of the map.
  Cosmetic for progress fields (overwritten a second later); real for the one
  terminal write per task. Long-standing — the previous three-round-trip
  version had a *wider* window.
  Not fixable with `modal.Dict` primitives: `put(skip_if_exists=)` is
  insert-if-absent, which arbitrates *creating* a key (`write_if` already uses
  it for the double-spawn race) but can't compare-and-swap a value. Building
  CAS from it means a lock — 4+ round-trips on the hottest write in the system,
  and a worker dying mid-lock wedges the record forever unless the value
  carries a lease, whose expiry can't be stolen safely without… CAS.
  Cheaper where it counts: don't trust `state=DONE` without the matching
  gen-qualified result. A missing one means "the current attempt hasn't
  finished", which is exactly true, and it covers the race however it arose.
  Care needed — a missing result can also be a swept blob or a Volume commit
  that didn't land, and reading *that* as "still running" suspends the DAG
  forever (`reap_dead` only settles records that say RUNNING), so it probably
  wants a record reset so the next tick relaunches. Touches `keep_stale`,
  `retry`, and the `settled` aggregates; worth its own change.

- **Metric trends know a direction, not a rate (2026-07-27, PR #58 review).**
  `expect_metrics`, wrong-way window counting, and a sample floor on the window
  all shipped; what's left is how coarse the judgement is. A window mean is
  compared without reference to the within-window spread, so a metric with a
  genuinely wide spread can still string together three wrong-way windows by
  chance — if that starts crying wolf, judge the movement against the spread the
  worker already has the samples to compute (a running sum of squares would do
  it, alongside the sum it already keeps). And a direction can't catch a loss that
  is descending far too slowly to reach anything useful inside the budget: that
  reads as perfectly healthy. A projected-final-value flag would be the
  counterpart to the timeout projection.

- **An unresolvable module leaves no evidence and says nothing (2026-07-26).**
  Deferred-import evidence is now symbol-granular, so the blast-radius half of
  this is done. What's left is the failure direction that can actually serve a
  stale hit: if the `sys.path` search doesn't find a module, `_module_index`
  returns `None` and the walk moves on — indistinguishable from the stdlib and
  site-packages, which are *meant* to be skipped. A task importing something the
  driver process can't see would then depend on nothing and cache forever. Fixing
  it means telling "deliberately excluded" from "expected to resolve and didn't",
  which needs a notion of what should have been findable (an installed-distribution
  check, or a project-roots list). A warning would be enough. Related smaller
  assumption: `sys.path` order is taken as stable within a process.

- **Science skill.** We have a fledgeling `science` skill that describes how to
  collaborate on experiment design. There may be old descisions in
  todo-science.md that could be moved there and polished.

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

- **CLI usability, remaining gaps** (from the 2026-07-14 cold-exploration
  session; the copy-pasteable-hints / sorting / help-text tier shipped — see
  #57 for the running thread):
  - No way to *delete* an experiment's memo state. `mini gc <name>` sweeps only
    stale attempt files/superseded records, so a scratch or renamed experiment's
    DONE records live forever — on Modal too (a `cli-probe` probe experiment now
    sits there as a permanent example). Wants a `mini rm <name>` with the same
    dry-run-by-default posture as gc. The manual escape hatch, verified: `bin/modal
    dict delete mini-cp-<name> --yes` plus `bin/modal volume delete <name> --yes`
    clears both planes — which is roughly what `mini rm` would wrap.
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
  `mini gc --store`. Same shape one level down: deleting a figure from a
  notebook leaves its `_assets/<name>-{light,dark}.png` behind, since re-export
  writes into the existing bundle dir without pruning. Harmless locally (the
  HTML stops referencing them) but it ships dead bytes on publish — a prune of
  assets not referenced by the fresh `index.html` would cover both.

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

- The two landmark figures in ex-2.1.5 name the x axis differently: the probe
  heatmap prints the raw keys (`o1s0`, `ae1`, …) rotated 90°, while the trace
  grid below it uses the math labels from `sca.vis_probes.LANDMARK_LABELS`
  ($a_1$, $r_{n-1}$, …). The figures are laid out to be read against each other,
  so the mismatch costs the reader a translation step. Point the heatmap at
  `label_landmarks` too — its panels are wide enough for the full set.

## Backlog, grouped by what a single dev session should bundle

(M2 *science* backlog, including issue #10, now lives in
[`todo-science.md`](./todo-science.md).)

**Storage/control-plane design.** These stem from the same list in
[`eng/decisions.md`](./eng/decisions.md):

- #38 — publish-tier hardening (private-CAS/public-publish bucket split;
  citable versioned publish via a dataset repo). Only matters once the template
  is used for work that shouldn't be world-readable by default. It's also the
  only thing left that would reshape what "CAS" means to
  [`mini gc`](./eng/gc.md) (#15, shipped in two cuts).

**Orthogonal, no code overlap with the above:**

- #45 — docs rework. Touches `docs/`, `README.md`, `eng/`, not `src/mini/`.
  Can run in parallel with anything.
- #57 — CLI DevX: tier 1 shipped (`_load_experiment_or_hint` gives a friendly
  error + the `path` positional documents file-vs-NAME). Anything beyond that
  (e.g. auto-resolving a name to its experiment file) is still open.
