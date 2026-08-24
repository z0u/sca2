---
status: open
tags: [D2.1, ex-2.1.12, publishing, figures]
opened: 2026-08-24
priority: high
---
# Land ex-2.1.12 and the D2.1 figure report, or decide not to

[PR #99](https://github.com/z0u/sca2/pull/99) opened as a sketch of lighter grading-chart encodings and grew into two published reports plus a reusable figure artist. None of it is on `main`, and it has sat untouched since 2026-08-19. The branch is `claude/grading-chart-simplifications-rkczlj`.

What lives there and nowhere else:

- **`docs/m2/ex-2.1.12`** — fitted per-channel ridge probes over the D2.1 checkpoints, preregistered, run and published. Its post-hoc result is the part worth rescuing: the bare anchor drops last-slice op2 decodability to ≈ 0.03 against the control's 0.61, the anti-subspace term restores it to 0.32, and the full recipe to 0.60. That is a measured decodability cost of anchoring and a measured recovery, and nothing on `main` records either. The preregistered comparison did not resolve (control seed variance is large — only a ~0.5 R² deficit was catchable), and the off-key demonstration landed: the strict per-value holdout puts every condition, control included, at the fold design's no-information floor.
- **`docs/m2/d2.1`** — a figures-only report for the LessWrong post on D2.1: the recipe one piece at a time, and the primary condition per (slice, position). It declares itself not an experiment.
- **`src/sca/vis_grading.py`** — `GradingCloud`, an artist that re-rasters at the axes' device size at draw time, so nearest-neighbor resampling is the identity and the dither can't moiré against the output grid. The `style-fig` conventions for it are in the same branch.
- **Three backlog items** that exist only there: `channel-probes-on-d21-checkpoints` (done, carrying the results note above), `grading-stepped-at-embedding-slice` (open — the embedding slice at op1 responds as a step rather than a grade, and the slice-mean r² hides it, which matters for any claim quoting r² as "the response is graded"), and `migrate-grading-grid-to-gradingcloud`.

Two hazards while it sits. The number **ex-2.1.12 is already taken**, so an experiment started fresh from `main` would collide on the directory, the index entry and the publish lock. And the grading-figure work has no pointer on `main` at all, so it is easy to start over from scratch — which is what this item exists to prevent.

Deciding not to land it is a fine answer, but then the ex-2.1.12 finding and the two open figure items want copying onto `main` before the branch is dropped.

## Notes

**2026-08-24, housekeeping** — Read the branch rather than the PR description, which still describes the original sketch and understates it by a lot: 22 commits, and `docs/publish.lock` on the branch pins both new reports, so they are exported and the preview built.

On landing it. The merge base is 2026-08-15, so the branch is nine days behind, and `main` has touched `.agents/skills/style-fig/SKILL.md` and `style-py` in that window while the branch rewrites both — expect conflicts there rather than in the reports. CI's "Run all checks" reads failure on the 08-19 head, but the log doesn't explain it on its own: 746 tests passed, the publish check exited clean, and the only error is `./go deadcode` flagging three of ex-2.1.12's preregistration constants (`OFFKEY_CEILING`, `POSITIONS`, `H2_PARTIAL`) — a step marked `continue-on-error` on both the branch and `main`, so it shouldn't be deciding the job. Cheapest first move is to merge `main` in and re-run; a `# noqa` on those three constants clears the finding either way.

Promoted to the shortlist because the decodability result feeds the D2.2 side-effects framing, and because holding a published experiment off `main` is what makes the number collision possible.
