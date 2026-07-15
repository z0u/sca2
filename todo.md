# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Scratch items sit under _Scratch_; everything below that is the prioritized
index into GitHub issues. Durable design rationale and recorded
decisions live in [`eng/`](./eng/README.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Scratch

- **Ex-2.1.2 results (2026-07-15): the `named_holdout` diagnosis was half
  right, and the interesting half failed.** The 2×2 factorial (reverse
  aliases × off-palette named-as-hex, frozen d64-L4) supplied both missing
  ingredients and both *trained* — reverse aliases read out at 1.0 in their
  own frame, and the arithmetic on name + name prompts generalizes to unseen
  off-palette pairs at ≈ 0.92 — yet `named_holdout` stays at exactly 0 in
  every condition. The failure decomposed: in the `open` conditions ~1/3 of
  held-out answers are the *correct mix value in hex form* (form rule learned
  per-pair, not per-value), and the rest are still lookup-neighbor names; the
  name-identity margin (log P(true name) − best other name) sits ≈ −9 nats
  everywhere, so the value → name translation never engages mid-equation even
  though the same mapping is perfect in the `#hex = ` frame. Full analysis
  with figures in `docs/m2/ex-2.1.2/report.py`. Consequences: the anchored
  runs train on the `both` corpus and use `open_holdout` + s₂ as the graded
  canaries (`named_holdout` has no headroom to lose); whether `named_holdout`
  can be made solvable at all in 4 layers is parked — candidates: a denser
  named sub-grid (value-diverse rgb→name supervision *in-frame*; note it
  changes the concept inventory the anchors will label), more depth, or a
  curriculum that interleaves frames. Bonus finding worth carrying into
  anchor design: the answer-schedule probe shows just-in-time computation
  *with eviction* — at the final layer, channel k is decodable (R² ≈ 0.97)
  only at its own emission position, and previously-emitted channels are
  dropped from the deep residual stream — so a "result" concept never fully
  exists at any single position, and anchoring one there would fight the
  model's schedule.

- **s₂ (surprise-surprise) is now a standard metric.** Adopted in ex-2.1.2
  (`answer_calibration`: mean answer nll/entropy/s₂ per eval set per cell) and
  it behaved exactly as designed — snaps from ≈ 0.7 (confidently wrong) to ≈ 0
  in precisely the conditions that train a form, per-set. Caveats live in the
  docstring: it measures calibration, not competence (uniform ignorance also
  scores ≈ 0 — pair with accuracy or raw surprisal), and per-token s₂ is a
  noisy one-sample draw, so aggregate over many positions. Carry into the
  anchored runs as the early-warning dial.

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

- Calibrate the redirect's γ against the model's pre-norm activation scale
  instead of the fixed γ = 1. Ex-2.9.3 found the fixed value silently no-ops
  on ~1 run in 250 (the bias fails to dominate that seed's pre-norm residual,
  so "deleted" red passes through nearly untouched); ex-2.9.2 saw the same
  once. Cheap fix: set γ to a multiple of the ablated row's typical pre-norm
  contribution, measured on the train set after training.

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

- The color-mixing grammar (`sca/data/colors.py`) hardcodes one operation:
  `mix`, spelled `+`. D2.2 anchors *the operation*, which only makes sense once
  the operation is a variable — with a single op there's nothing for the model
  to represent. When extending: add an operation table (name, surface form,
  grid fn with a defined rounding — saturating add/subtract and screen all stay
  closed on 0..15), thread an `op` field through `Example` and key the
  seen-pair bookkeeping on `(op, pair)`. Spell operators as *words*
  (`red mix blue = purple`), not symbols, so the operation concept is
  multi-token like the colors. No need to keep `+` compatible — each experiment
  retrains from scratch and carries its own un-anchored control, so grammars
  may differ across experiments. Probe positions in
  `sca/compute/evaluation.py` assume the infix `a <op> b = ` frame; keep that
  frame.

- The ex-2.1.1 report refs moved to `reports/m2/ex-2.1.1/*`; the pre-rename
  `reports/ex-2.1.1/*` refs still sit in the store (there's no ref-delete API).
  Harmless clutter, but they pin their artifacts through GC's mark-and-sweep.
  If a ref-delete/rename verb ever lands (eng/gc.md), sweep them. The m1 refs
  (`reports/ex-2.9.*`) predate milestone nesting and stay flat on purpose.

- Cheap capacity/superposition proxies for the ex-2.1.x eval step: per-layer
  participation ratio of residual-stream activations (eigenspectrum of the
  covariance — how many effective dimensions the model uses) and pairwise |cos|
  between the fitted probe directions (operand vs result vs redness —
  interference between concepts). Both fall out of arrays `eval_one` already
  computes; they'd let the width sweep read as a compression axis. Full
  superposition accounting (feature dictionary / SAE) is its own experiment,
  after D2.1.2.

- If anchoring a composed concept fails in D2.1.2, a useful ablation is a
  word-level tokenizer variant (one token per color name, hex still
  char-level): it separates "anchoring fails for transformers" from "anchoring
  fails for concepts that don't coincide with an embedding row". Not the
  default — the char-level task is the honest version of what M2 claims.

- ngpt-scaling shows the simplified nGPT (fixed scalar α = 1/n_layer) trains
  flat across the width × depth grid we can afford. Follow-up: confirm the fixed
  scalar gate holds at a genuinely larger size (wider/deeper than 128×12, bigger
  GPU + batch) before leaning on it for M3.

- Remove the remaining mi-ni template *experiments* (`docs/pipeline`,
  `docs/probe`, `docs/acts` — their report notebooks are already gone) once the
  e2e tests that drive them (`tests/mini/test_experiments_e2e.py`) get their own
  fixtures, or once the first real M2 experiments can play that role. Ties into
  #45 (docs rework). (`docs/gpt-sweep` has since become `docs/ngpt-scaling`, a
  real Iteration 0 output rather than a template.)

## Backlog, grouped by what a single dev session should bundle

**M2 science.**

- [sca2#10](https://github.com/z0u/sca2/issues/10) — D2.1 kickoff: carry-over
  lessons and hypothesis queue from ex-2.9.3/2.9.4 (schedule ordering, the
  fallback analog, γ calibration, superposition watch-outs). Read before
  designing the first transformer anchoring experiment.

**Quick wins.** All shipped: #39 and #36 (PR #51), #19 (queued ≠ running,
PR #54), #47 (per-experiment backend memory for `--app`).

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
