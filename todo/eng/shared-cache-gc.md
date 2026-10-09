---
status: open
tags: [tooling, storage]
opened: 2026-10-09
---

# Garbage-collect the shared render caches

In the project's cloud threads `MINI_CACHE_DIR` puts the weave cache and the bucket's warm cache in the shared folder (`/mnt/project-files/.mini-cache/`), and nothing removes entries from either. `mini gc` leaves `store-cache/` and `lit-cache/` alone, and `lit-cache/assets/` keeps one copy of the figures per key and evidence, so every edit to a plot function adds another. On 2026-10-09 the two held about 3 MB and 5 MB.

A rule that keeps what was read recently (by access or modification time, with a margin of a day or so for sessions that are mid-render) would suit both, since every entry can be rebuilt from the bucket and the code. Worth doing once the folder grows enough to notice.
