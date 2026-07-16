# Publishing reports to the web

*Part of the [engineering notes](./README.md).*

## Publish is a separate, outward-facing verb

`put` persists; `publish` deliberately exposes. **Public exposure is never a side
effect of persisting a result** — that separation is the point.

A CAS blob at `cas/<sha>` has no extension, so a browser gets `application/octet-stream`.
`publish` does a server-side copy *by xet hash* (instant, no bytes moved) to an
extensioned path; the bucket's resolve URL then sets `Content-Type` from that extension
and serves `Content-Disposition: inline`. So one bucket is both the durable store and a
CDN-backed asset host.

## Two tiers, split by write concurrency (#38)

By default **one bucket** backs both the CAS and the published views. That's fine until
you persist something that shouldn't be world-readable: HF buckets have **no per-prefix
ACL** (public/private is bucket-level only), so a public bucket makes *every* `put`
world-readable, not just what you `publish`. The fix is two stores — and the seam
between them is **write concurrency**, not just visibility:

| Tier | Namespaces | Writer | Concurrency | Backend |
|---|---|---|---|---|
| **Durable store** | `cas/<sha>`, `refs/` | workers | high, parallel | a **bucket** (no git history → concurrent writers never 412) — make it **private** |
| **Publish tier** | `published/`, `exports/` | a driver / CI | low, single-writer | a **dataset repo** (real git history → versioned, citable) — **public** |

The CAS *must* stay a bucket: that's the whole reason buckets were chosen (concurrent
result writers would 412 on a git-backed repo's shared parent). But `publish` and report
export are deliberate, single-writer, driver-side acts — so they can afford a git-backed
**dataset repo**, which buys what a bucket can't:

- a **public** face over a **private** CAS (the two-store split HF's bucket-level ACL forces), and
- **versioned names with history** — a citation pins to `…/resolve/<commit-sha>/published/<path>`, which a mutable bucket path can't guarantee.

So one move settles both halves of #38: point the publish tier at a dataset repo and it's
public *and* citable.

**Cost.** Not a byte more of storage. Xet chunk dedup is **account-wide**, so a blob
published from the private CAS into the public repo references the *same* chunks —
publishing pulls the blob local (the warm cache usually has it) and uploads, but Xet skips
every chunk the CAS already stored, so it's a metadata + git-commit op, not a re-transfer.
The only real deltas: public storage is best-effort quota (vs private's ~100 GB
guaranteed), and each publish is a commit rather than the in-bucket instant by-hash copy.

**Enabling it.** Set `[tool.mini] publish-repo = "<ns>/<repo>"` (or `MINI_PUBLISH_REPO`)
and flip the existing bucket to private. Unset, everything stays in the one bucket — the
seam is opt-in and `HFStore` routes `publish`/`export_*` on `publish_repo` alone, so no
experiment or report code changes. Provisioning is typically just **one** new public
dataset repo; a second public *bucket* is only needed if you want high-churn public assets
that shouldn't be versioned (reports overwrite by name, so they don't accumulate — you
usually don't).

## Reports are a bundle plus a `<base>` switch

A report is one Marimo HTML document plus its heavy assets. Assets are externalized
*at production* and referenced by a **relative** URL (`_assets/<name>`). A single
`<base href>` in the `<head>` decides where those resolve:

- **Opened locally / offline** → no base tag → `_assets/…` resolves to co-located files
  (real PNGs).
- **Published** → `build_site` inserts one `<base href="…/exports/<key>/">` → the *same*
  relative URLs resolve at the bucket CDN.

The `<base>` is what makes this work without **per-URL rewriting**, which matters because
the asset URLs are buried in Marimo's doubly-escaped session JSON where surgical rewrites
are fragile — one tag repoints all of them, including a relative `fetch()`. It's safe
because a fresh `marimo export` contributes *zero* relative URLs of its own (every
framework resource is an absolute CDN URL), so a `<base>` governs only the assets we
introduce. The one caveat: `<base>` also repoints *author-written* relative links, so
`build_site` resolves those to absolute targets (`stray_links` / `rewrite_links`) — a
link to another report becomes its rendered page, a link to a source file its GitHub
source. **Convention: the only relative URLs in a report are its assets.**

**Hotlinking couples the site's figures to HF availability — accepted.** The pages live
on GitHub Pages but every figure resolves through the `<base>` at huggingface.co, so an
HF outage blanks the site's figures while the prose keeps loading (observed July 2026,
a multi-day Hub/CDN outage). The alternative — copying each bundle's `_assets/` into
`_site` so Pages serves everything — was considered and declined: it puts binary assets
on the `gh-pages` branch, whose history grows with every changed figure. (It would cost
no extra CI bandwidth, though: `fetch_export` already pulls each bundle in full — a fact
the build now uses to *verify* bundles instead. See the integrity check below.) Revisit
if HF outages become frequent enough to matter.

**The build refuses to ship a page with holes.** Since the fetched bundle is on disk
anyway, `build_site` checks every `_assets/…` URL a page references against the files
its bundle ships (`missing_assets`); in `--externalize` mode (CI) any miss fails the
build — an incomplete publish stops the deploy (and shows up red on the PR preview)
instead of deploying figure-less pages that look exactly like a CDN outage.

Further decisions:

- **No HTML in Git.** The notebooks (`docs/**/*.py`) are the only source of truth; each
  exports on demand to a bundle synced under `exports/<key>/`. This keeps PRs reviewable
  *and* lets a Cloud agent publish — the original blocker was Git LFS, which a Cloud
  session can't write, so an agent could run an experiment but not publish it.
- **Assets keyed by readable name**, not content hash, so a re-export overwrites in place
  and a report accumulates no orphans (the name is also what a browser "Save as"
  suggests, since the bucket sets no `Content-Disposition`).
- **Publish/build split by trigger.** `./go publish` (authenticated, runs the notebook,
  writes the store) is the heavy half; the CI build is **read-only** — it pulls each
  bundle, resolves links, inserts the `<base>`, and never writes or runs a notebook, so
  a read-only token suffices. When the publish tier is split off (#38) the build reads
  exports straight from the public dataset repo and never touches the CAS, so CI needs
  only `MINI_PUBLISH_REPO` — not the (now private) bucket. `store_for` builds a CAS-less
  store from a `publish-repo` alone for exactly this; the single-bucket default still
  serves the build off `MINI_STORE_BUCKET`. Set whichever your store layout uses.
- **PR previews ride the same read-only build.** Because `./go publish` runs on the
  agent's branch *before* the PR opens, the bundles are already on the publish tier when
  review starts — a preview needs no compute, just HTML assembled around them.
  `pr-preview.yml` runs `./go site` on the PR and deploys to `pr-preview/pr-<n>/` on the
  `gh-pages` branch (torn down on close, linked from a sticky PR comment). That forces
  the production deploy to be branch-based too (`clean-exclude: pr-preview/`): an
  artifact deploy replaces the whole site, so previews couldn't coexist with it. Viewing
  a bundle straight off the dataset repo is *not* an option — HF serves repo HTML as
  `text/plain` behind a sandbox CSP. Branch scoping comes from the pin manifest (next
  bullet): the preview serves each report at the revision pinned on the *PR branch*,
  production at the one pinned on main, off the same mutable `exports/<key>/` names.
- **A publish is staged until its pin lands on main.** The dataset repo is git-backed,
  so every `sync_export` is already an immutable commit; the missing half was making
  the *build* consume one. `./go publish` records each bundle's commit sha in
  [`docs/publish.lock`](/docs/publish.lock) (export key → revision, committed to Git),
  and the build fetches *and* bases the report at that revision —
  `resolve/<sha>/exports/<key>/` — never the mutable head. This is the same
  identity/evidence split the memo store uses: the dataset repo holds evidence (every
  bundle ever published, at every revision), while the identity (which revision the
  site serves) travels with the code. Publishing from a branch therefore deploys
  nothing: production serves main's pins, the PR preview serves the branch's, and
  merging the PR is the promotion — no staging prefix, no HF-side branches, and CI
  keeps its read-only token (the alternative, `pr-<n>` revisions promoted on merge,
  would have needed a write token in CI). Re-publishing unchanged content mints no
  commit (huggingface_hub drops no-op operations and returns the current head), so
  idempotent publishes stay idempotent. Two corollaries: an unpinned report falls back
  to the head with a build warning (how pre-lock bundles behave until republished),
  and the dataset repo's history must never be rewritten (`super_squash_history`) —
  pins and citations resolve through old commits. For the same reason `publish()`
  itself now returns commit-pinned URLs, so a published single file is citable as-is.

  **Pinning is repo-only — a note for backports (mi-ni).** A bucket keeps no history,
  so on the single-bucket default there is nothing to pin: `sync_export` returns no
  revision, `publish.lock` is never written, and the build serves the mutable
  `exports/<key>/` with no unpinned warning (the nudge to pin would be misleading
  there). That's the pre-pin behavior, caveat included: a publish is live the moment
  it syncs, and a PR preview shares production's assets. So the pin machinery is safe
  to carry into template projects without making `publish-repo` a requirement — the
  seam stays opt-in, as #38 chose — but the staging guarantee and PR-preview isolation
  exist only once a project sets one.
- **Provenance is injected at export, not build.** The provenance footer (which
  experiment runs produced the data a report shows) needs the producers stamped on the
  store's refs — state only the authenticated export half can see, and only *while the
  notebook runs* (the resolutions happen inside cells). So the render records each
  `get_ref` into the bundle's `_assets/provenance.json` via the active publisher, and
  the exporter injects the footer into the HTML before the bundle syncs; the read-only
  build stays dumb. Everything in the footer derives from the store's refs — no build
  timestamps — so re-publishing unchanged data produces the same bundle content and
  publishing stays idempotent.
