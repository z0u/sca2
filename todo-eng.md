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

- **Showcase handout PDFs (opened 2026-07-19).** The showcase notebooks
  (`docs/m2/showcase/`) present as slides via `marimo run` thanks to their
  `layouts/*.slides.json`. Marimo can also render a reveal.js-style PDF deck
  (`marimo export pdf --as=slides`, needs nbconvert + Playwright) — worth
  wiring into `./go` if we ever need a file to attach rather than a URL to
  share. Not needed while the published page + live slides cover both uses.

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

- **Warm-cache reads bypass HF's Xet layers; blob pulls could be faster.**
  `HFStore` pulls blobs via `HfApi.download_bucket_files` straight into our own
  content-addressed warm cache (`store-cache/hf/cas/…`), so a read gets neither
  Xet chunk-level dedup nor huggingface_hub's shared `HF_HOME` cache — every
  blob is a full, non-resumable transfer, and we maintain a second cache that
  misses independently of HF's. The per-op round trip (~2–3s/commit, same on
  reads) is the real floor. Levers, roughly in priority: (1) parallelize
  callers that resolve many refs sequentially — the checkpoint-loading loops in
  reports/`eval` fan out one `store.get` per cell; `Store.get` already threads
  *tree children* (≤8), but not sibling top-level gets; (2) point the warm cache
  at `HF_HOME` (or route large reads through `hf_hub_download`) so the Xet chunk
  cache and resumable/atomic download are in play — the atomic part would also
  have prevented the Ctrl-C'd-export cache-poisoning bug (fixed in
  `_local_blob`, 9af9282) for free. Measure before/after — a timing harness over
  a cold-cache `get` of the ex-2.1.2 checkpoints is the obvious probe. Related:
  #37 (closed) was about *dedup of identical prep*, not read-path caching.

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
