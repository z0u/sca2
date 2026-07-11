# Todo

Scratchpad for deferred work that isn't worth a tracking issue yet. When something
here grows real, promote it to a GitHub issue and remove it from this list.

Scratch items sit under _Scratch_; everything below that is the prioritized
index into GitHub issues. Durable design rationale and recorded
decisions live in [`eng/`](./eng/README.md); each open issue
also carries a grounding comment with current file:line refs, so it should be
readable cold without re-deriving code state.

## Scratch

- Calibrate the redirect's α_rd against the model's pre-norm activation scale
  instead of the fixed α_rd = 1. Ex-2.9.3 found the fixed value silently no-ops
  on ~1 run in 250 (the bias fails to dominate that seed's pre-norm residual,
  so "deleted" red passes through nearly untouched); ex-2.9.2 saw the same
  once. Cheap fix: set α_rd to a multiple of the ablated row's typical pre-norm
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

- `test_publish_serves_with_content_type_from_extension` fails: an anonymous
  GET of the *store bucket's* `published/` revision now returns 401 (the
  publish repo `z0u/sca2-pub` still serves 200 anonymously — verified
  2026-07-09). Looks like a stale test assumption from before exports were
  routed to the publish repo (#38); either point the test at the publish
  tier or drop the anonymous-serving assertion for the bucket.

- Sweep cells all `save_checkpoint` to the same shared `get_data_dir()`, so the
  checkpoint file is last-writer-wins across a fan-out. Harmless today (the
  ngpt-scaling cells return their metrics; nothing reads the checkpoints back),
  but key checkpoints by cell label before any experiment resumes from or
  evaluates them.

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
  fallback analog, α_rd calibration, superposition watch-outs). Read before
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
- #57 — CLI DevX: passing a name to `retry`/`run` dies with a raw traceback
  (tick verbs take a file, read verbs a name). Tier 1 (friendly error + help
  text on the `path` positional) is a quick win in `src/mini/__main__.py`.
