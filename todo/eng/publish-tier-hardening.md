---
status: open
tags: [publishing, storage]
bundle: storage-control-plane
---
# Publish-tier hardening — private CAS, public publish bucket

Tracked upstream as [mi-ni#38](https://github.com/z0u/mi-ni/issues/38) — split the private CAS from the public publish bucket, and make the publish tier citable and versioned via a dataset repo.

Stems from the same list as [`eng/decisions.md`](../../eng/decisions.md).

Only matters once the template is used for work that shouldn't be world-readable by default. It's also the only thing left that would reshape what "CAS" means to [`mini gc`](../../eng/gc.md) ([mi-ni#15](https://github.com/z0u/mi-ni/issues/15), shipped in two cuts).
