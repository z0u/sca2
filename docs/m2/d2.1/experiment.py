"""
Re-keying the D2.1 grading measurement onto each operand in turn.

This is not an experiment: it states no hypothesis, trains nothing, and scores
nothing. It recomputes one published measurement so the figures page beside it
can show every condition's α map keyed by *either* operand's color.

It has to run at all because of where the earlier experiments contracted their
measurement. The α map ex-2.1.6 and ex-2.1.8 publish is already averaged over
each color's 27 closed partners, and that average is what drops op2's identity;
no contraction of what they published brings it back. Ex-2.1.10 widened its own
eval to keep the per-line array, which is why its report can read both margins.
The other two left it on the floor of the eval step.

What they did publish is their checkpoints and their probe sets, and all three
build the probe set identically — every color as op1 against each of the 27
partners whose mix stays on the grid, in palette order. So re-running the
alignment probe over it costs about two seconds a run and lands back on the
array they published, to float32 accumulation noise. `rekey_one` asserts that
before it returns anything.

    bin/mini run docs/m2/d2.1/experiment.py --watch

The companion `report.py` reads the published npz and never runs this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini import Ctx, Experiment, get_data_dir

ARRAYS_REF = "reports/m2/d2.1/arrays"

#: How close a recomputed op1 margin has to sit to the one its experiment
#: published. The probe set and the weights are identical, so the only honest
#: source of disagreement is the order float32 accumulates in — about 1e-4 on a
#: quantity that runs 0 to 1. A real mismatch (wrong checkpoint, drifted probe
#: set) moves it by a lot more than this, so the gap between the two is wide.
ALPHA_ATOL = 5e-4

#: The same comparison one level down, on the per-line array. A margin is a mean
#: over 27 lines, so it averages that accumulation noise down by about √27; a
#: single line carries all of it. Measured worst case over ex-2.1.10's nine runs
#: is ~7e-4, so this leaves a factor of three and still sits far below the
#: signal, which runs to α ≈ 1.
LINES_ATOL = 2e-3


@dataclass(frozen=True)
class Condition:
    """One panel of the progression figure, and where its runs are published."""

    exp: str
    """The experiment that ran it, e.g. `ex-2.1.6`."""
    name: str
    """The condition's name within that experiment's sweep."""
    seeds: int
    """How many seeds it published, labelled `{name}-s0` … `{name}-s{seeds-1}`."""
    title: str
    """The panel title, naming the piece of the recipe this condition adds."""

    @property
    def key(self) -> str:
        """`{exp}/{name}` — how the report and the published npz address it."""
        return f"{self.exp}/{self.name}"


#: The recipe, in the order the progression figure adds it: the un-anchored
#: control and the bare anchor, then the anti-subspace term at its tuned
#: operating point, then sequence-level labels naming either operand.
CONDITIONS: list[Condition] = [
    Condition("ex-2.1.6", "lam0", 3, "un-anchored"),
    Condition("ex-2.1.6", "lam0.1", 3, "+ anchor"),
    Condition("ex-2.1.8", "end90-hold30", 3, "+ anti-subspace"),
    Condition("ex-2.1.10", "either-t100", 9, "+ pooled labels"),
]

PRIMARY = CONDITIONS[-1].key
"""The full recipe — the condition the D2.1 claims are read on."""


def operand_colors(probe_tokens):
    """The (op1, op2) color index of every probe line, read off the lines themselves.

    The probe set is built by walking colors in palette order and emitting one
    line per closed partner, so the op1 column visits each color's token in that
    order and the palette index of a token is just where it first appears. That
    map then reads the op2 column too. Taking it from the tokens rather than
    replaying the closure rule means the index cannot drift from the lines it
    describes: whatever the probe set actually holds is what gets keyed.
    """
    import numpy as np

    op1, op2 = probe_tokens[:, 0], probe_tokens[:, 2]
    _, first = np.unique(op1, return_index=True)
    palette_order = op1[np.sort(first)]
    index = {int(t): i for i, t in enumerate(palette_order)}
    return np.array([index[int(t)] for t in op1]), np.array([index[int(t)] for t in op2])


def rekey_one(exp: str, cond: str, seeds: int) -> dict:
    """Both operand margins of one condition's α map, from its published checkpoints.

    Runs the alignment probe over the experiment's own probe set, then collapses
    the per-line result twice: grouping the lines by op1's color gives back the
    array the experiment published, and grouping them by op2's color gives the
    other margin. Neither is a new measurement — they are two contractions of
    one, and every line goes into each of them exactly once.

    Three checks stand behind that claim, because the two collapses fail in
    different ways. The op1 margin is compared against the published array,
    which covers the checkpoint, the probe set and the forward pass. The bulk
    mean over colors has to agree between the margins, since both average the
    same lines — that is what catches a permutation which drops or repeats one.
    And where the experiment published its per-line array (ex-2.1.10 did), the
    recomputed lines are held against it directly.
    """
    import io

    import numpy as np

    from mini.store import get_many, get_ref, put
    from sca.anchoring import alignment
    from sca.compute.model import load_checkpoint

    workdir = get_data_dir() / "rekey" / f"{exp}-{cond}"
    probes, published_arrays = (get_ref(f"reports/m2/{exp}/{n}") for n in ("probes", "arrays"))
    assert probes is not None, f"{exp} has not published its probe set"
    assert published_arrays is not None, f"{exp} has not published its arrays"
    probe_path, arrays_path = get_many([
        (probes, workdir / "probes.npz"),
        (published_arrays, workdir / "arrays.npz"),
    ])  # fmt: skip
    with np.load(probe_path) as z:
        probe_tokens, weights = z["probe_tokens"], z["weights"]

    c_op1, c_op2 = operand_colors(probe_tokens)
    n_colors = len(weights)
    n_partners = len(probe_tokens) // n_colors
    counts = np.bincount(c_op2, minlength=n_colors)
    assert (counts == n_partners).all(), f"{exp} probe set is ragged at op2: {counts.min()}–{counts.max()}"
    # Mixing is a channel-wise round-half-up mean, so closure is symmetric and
    # each color leads as many lines as it follows. `argsort` regroups the lines
    # by op2 so the same reshape reads the other margin.
    by_op2 = np.argsort(c_op2, kind="stable")

    def collapse(lines: np.ndarray) -> np.ndarray:
        """Average the (slices, lines, positions) array within each group of partners."""
        return lines.reshape(lines.shape[0], n_colors, n_partners, lines.shape[2]).mean(axis=2)

    out: dict[str, Any] = {}
    with np.load(arrays_path) as published:
        for seed in range(seeds):
            label = f"{cond}-s{seed}"
            ckpt = get_ref(f"reports/m2/{exp}/checkpoints/{label}")
            assert ckpt is not None, f"{exp} has not published a checkpoint for {label}"
            get_many([(ckpt, workdir / label / "model")])
            model, _, _ = load_checkpoint(workdir / label)
            cos = alignment(model, probe_tokens)  # (slices, lines, positions)
            a1, a2 = collapse(cos), collapse(cos[:, by_op2])
            np.testing.assert_allclose(a1, published[f"{label}/alpha"], rtol=0, atol=ALPHA_ATOL)
            # Both margins average the same 5832 lines, so their means over colors
            # coincide however the lines were grouped.
            np.testing.assert_allclose(a1.mean(axis=1), a2.mean(axis=1), rtol=0, atol=1e-6)
            if f"{label}/alpha_lines" in published.files:
                np.testing.assert_allclose(cos, published[f"{label}/alpha_lines"], rtol=0, atol=LINES_ATOL)
            # Keyed by seed alone: the publish step prefixes `{exp}/{cond}/`, and
            # repeating the condition name inside that prefix reads as an error.
            out[f"s{seed}/op1"] = a1.astype(np.float32)
            out[f"s{seed}/op2"] = a2.astype(np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return {
        "key": f"{exp}/{cond}",
        "n_runs": seeds,
        "arrays": put(buf.getvalue(), name=f"d2.1-{exp}-{cond}-margins.npz"),
        # Which color sits at each operand slot, per line. A property of the probe
        # set rather than of this condition, and the report needs it to say what a
        # color's partners look like — so it travels with the margins instead of
        # being re-derived there from the closure rule.
        "pairs": np.stack([c_op1, c_op2], axis=1).astype(np.int16),
    }


def publish_margins(results: list[dict]) -> dict:
    """Stack every condition's margins into one npz, keyed `{exp}/{cond}-s{seed}/…`.

    The probe set rides along once, under `probe/pairs`. All four conditions were
    measured on their own experiment's copy of it, so agreement between them is
    worth asserting rather than assuming: it is what lets the four panels be read
    against each other, and what makes one shared index legitimate.
    """
    import io

    import numpy as np

    from mini.store import get, put, set_ref

    pairs = results[0]["pairs"]
    for r in results[1:]:
        assert np.array_equal(r["pairs"], pairs), f"{r['key']} was measured on a different probe set"

    arrays: dict[str, Any] = {"probe/pairs": pairs}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['key'].replace('/', '-')}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['key']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="d2.1-margins.npz"))
    return {"n_conditions": len(results), "n_runs": sum(r["n_runs"] for r in results), "n_arrays": len(arrays)}


def main(ctx: Ctx) -> dict:
    exps = [c.exp for c in CONDITIONS]
    names = [c.name for c in CONDITIONS]
    seeds = [c.seeds for c in CONDITIONS]
    margins = ctx.map(rekey_one, exps, names, seeds, role="cpu")
    return ctx.run(publish_margins, margins, role="cpu")


experiment = Experiment(
    name="d2.1",
    main=main,
    # One condition per task, so each fetches its probe set once and reuses it
    # across that condition's seeds. The heavy part is the checkpoint fetch, not
    # the forward pass: nine slices of a 4-layer, 64-wide model over 5832 lines.
    roles={"cpu": dict(cpu=4, timeout=1800, watchdog_grace=900)},
    deps=["ex-2.1.6", "ex-2.1.8", "ex-2.1.10"],
)
