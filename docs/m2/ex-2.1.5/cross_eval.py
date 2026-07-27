"""Cross-form exact match for the bridge cell — a side analysis for H5.

The sweep's eval sets are single-form. Cross pairs have no holdout, because the
form exists to supervise alignment rather than to be graded, so `experiment.py`
never scores them. H5's verdict turns on whether that supervision took: a bridge
that never learned to answer `name + #hex` would say nothing about whether a
bridge aligns the two geometries. This scores it after the fact, reading the
published checkpoints and writing its results back to the store under
`CROSS_REF`, so `report.py` reads it the same way it reads everything else.

It lives outside the experiment DAG on purpose. The sweep's evidence scheme has
moved on since the sweep ran (mini now traces deferred imports), so any tick of
`experiment.py` re-trains all 24 cells — and a re-trained sweep would not
reproduce the numbers the rest of the report already quotes. A plain module,
not a notebook, so the site build ignores it:

    uv run python docs/m2/ex-2.1.5/cross_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from experiment import (
    CKPT_REF,
    CORPUS_SEED,
    CROSS_REF,
    HEX_SUBSET_SEED,
    HOLDOUT_FRAC,
    MAX_ANSWER,
    N_EVAL,
    N_EXAMPLES,
    SEEDS,
    _answer_value,
    _hex_grid_lifted,
    _well_formed,
)
from mini.store import project_store, put, set_ref, store_context
from sca.compute.evaluation import greedy_completions
from sca.compute.model import load_checkpoint
from sca.data import mixed_vocab as mv
from sca.data.tokenizer import CharTokenizer

WORKDIR = Path(".mini/scratch/ex-2.1.5-cross")


def cross_sets(n_names: int, n_hex: int) -> tuple[dict, dict]:
    """Cross-form eval sets, split by whether the (name, hex) pair was trained on.

    Rebuilds the bridge corpus from its seed to recover which pairs were drawn.
    Also returns how much of the cross space training covered, which is what
    makes the unseen split worth reading.
    """
    palette = mv.xkcd_palette(n_names)
    hex_ops = mv.hex_operands(n_hex, HEX_SUBSET_SEED)
    _, named_held = mv.holdout_split(mv.distinct_pairs(palette.values()), CORPUS_SEED, HOLDOUT_FRAC)
    _, hex_held = mv.holdout_split(mv.distinct_pairs(hex_ops), CORPUS_SEED, HOLDOUT_FRAC)
    held_n, held_h = set(named_held), set(hex_held)
    corpus = mv.sample_corpus(
        N_EXAMPLES, CORPUS_SEED, palette, hex_ops,
        lambda a, b: (min(a, b), max(a, b)) in held_n,
        lambda a, b: (min(a, b), max(a, b)) in held_h,
        mv.BRIDGE_WEIGHTS,
    )  # fmt: skip
    # Operand order is randomized per line, so read which side is named off the
    # prompt rather than by value.
    cross = [ex for ex in corpus if ex.prompt.count("#") == 1]
    seen = {(ex.rhs, ex.lhs) if ex.prompt.startswith("#") else (ex.lhs, ex.rhs) for ex in cross}
    unseen = [p for n in palette.values() for h in hex_ops if (p := (n, h)) not in seen]
    pick = lambda ps, s: [ps[i] for i in np.random.default_rng(s).permutation(len(ps))[:N_EVAL]]  # noqa: E731
    sets = {
        "cross_seen": mv.as_form(pick(sorted(seen), CORPUS_SEED + 8), "cross", palette, CORPUS_SEED + 9),
        "cross_unseen": mv.as_form(pick(unseen, CORPUS_SEED + 10), "cross", palette, CORPUS_SEED + 11),
    }
    return sets, {"n_lines": len(cross), "seen": len(seen), "unseen": len(unseen)}


def score(model, tokenizer, exs, grid) -> dict:
    """Greedy exact match with miss geometry, mirroring `experiment.eval_one`."""

    def guess_value(g: str) -> tuple[int, int, int]:
        if not _well_formed(g, False, None):
            return (0, 0, 0)
        r, gr, b = (int(c, 16) for c in g[1:])
        return mv.lift((r, gr, b))

    got = greedy_completions(model, tokenizer, [ex.prompt for ex in exs], MAX_ANSWER)
    well = np.array([_well_formed(g, False, None) for g in got])
    values = np.array([guess_value(g) for g in got], dtype=np.float32) / 255
    results = np.array([ex.result for ex in exs], dtype=np.float32) / 255
    floor = np.linalg.norm(np.array([_answer_value(ex) for ex in exs], dtype=np.float32) / 255 - results, axis=1)
    guess_dist = np.linalg.norm(values - results, axis=1)
    rank = (np.linalg.norm(grid[None] - results[:, None], axis=2) < guess_dist[:, None] - 1e-9).sum(axis=1)
    hits = np.array([g == ex.answer for g, ex in zip(got, exs, strict=True)])
    return {
        "n": len(exs),
        "accuracy": float(hits.mean()),
        "p_malformed": float(1 - well.mean()),
        "guess_dist": float(guess_dist[well].mean()) if well.any() else None,
        "floor_dist": float(floor.mean()),
        "nearest_rate": float((well & (rank == 0)).mean()),
        "rank_hist": np.bincount(np.minimum(rank[well], 6), minlength=7).tolist(),
        "failures": [(ex.prompt, ex.answer, g) for g, ex, h in zip(got, exs, hits, strict=True) if not h][:8],
    }


def main() -> dict:
    sets, n_pairs = cross_sets(140, 216)
    grid = _hex_grid_lifted().astype(np.float32) / 255
    store = project_store()
    cells = []
    for seed in SEEDS:
        label = f"bridge-s{seed}"
        ref = f"{CKPT_REF}/{label}"
        workdir = WORKDIR / label
        if not (workdir / "model").exists():
            art = store.get_refs([ref])[ref]
            if art is None:
                sys.exit(f"No checkpoint published at {ref} — run the sweep first.")
            store.get(art, workdir / "model")
        model, config, _ = load_checkpoint(workdir)
        tokenizer = CharTokenizer(config.tokenizer)
        scored = {name: score(model, tokenizer, exs, grid) for name, exs in sets.items()}
        cells.append({"label": label, "sets": scored})
        for name, s in scored.items():
            print(f"{label} {name:13s} acc={s['accuracy']:.3f} malformed={s['p_malformed']:.3f}")
    out = {"cells": cells, "n_pairs": n_pairs}
    with store_context(store):
        set_ref(CROSS_REF, put(json.dumps(out, indent=2).encode(), name="ex-2.1.5-cross.json"))
    return out


if __name__ == "__main__":
    main()
