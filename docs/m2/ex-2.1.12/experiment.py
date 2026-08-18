"""
Fitted channel probes over the D2.1 checkpoints.

Trains nothing. For every condition of the D2.1 progression, this fits ridge
probes for the RGB channels of each operand and the answer at every (slice,
position) site, under the strict per-value holdout of ex-2.1.5, and publishes
the R² maps and the probes' out-of-fold prediction curves.

Two questions motivate it. First, the α maps in `docs/m2/d2.1` show an off-key
tilt that the un-anchored control lacks; fitted probes read each model's own
color representation rather than the anchor axis, so if the tilt is a property
of the probe set it must appear here in every condition, control included, and
be capped by the pairing's information ceiling. Second, nothing published yet
measures whether anchoring costs ordinary color decodability, which is the
bounded-side-effects claim read through a probe.

    bin/mini run docs/m2/ex-2.1.12/experiment.py --watch

The companion `report.py` holds the preregistered hypotheses and reads the
published npz; it never runs this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini import Ctx, Experiment, get_data_dir

ARRAYS_REF = "reports/m2/ex-2.1.12/arrays"

#: Ridge strength, matching the probe scans of ex-2.1.5 and the shared helpers
#: in `sca.compute.evaluation`.
RIDGE_L2 = 1e-2

#: Channel levels of the color grid. The mix of two on-grid colors is the
#: round-half-up mean per channel, so a pair is on the probe set iff its levels
#: match in parity per channel; each color therefore has 3³ = 27 partners.
N_LEVELS = 6

#: The information ceiling at sites that precede op2. The state above the first
#: token is a function of that token alone (causal attention), so an op2-target
#: probe there can read at most E[op2 | op1]. Per channel that predictor is the
#: parity sawtooth (2, 3, 2, 3, 2, 3), whose variance is 0.25 against a level
#: variance of 35/12, so R² ≤ 0.25 / (35/12) ≈ 0.086 whatever the model.
OFFKEY_CEILING = 0.25 / (35 / 12)

#: Token positions of a probe line, in sequence order.
POSITIONS = ("op1", "+", "op2", "=", "ans", "\n")

#: Positions whose state cannot depend on op2: everything before it.
PRE_OP2 = (0, 1)

#: Probe targets, and the position of the token whose color each names.
TARGETS = {"op1": 0, "op2": 2, "ans": 4}


@dataclass(frozen=True)
class Condition:
    """One condition of the D2.1 progression, and where its runs are published."""

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


#: The same four conditions as `docs/m2/d2.1`, in recipe order.
CONDITIONS: list[Condition] = [
    Condition("ex-2.1.6", "lam0", 3, "un-anchored"),
    Condition("ex-2.1.6", "lam0.1", 3, "+ anchor"),
    Condition("ex-2.1.8", "end90-hold30", 3, "+ anti-subspace"),
    Condition("ex-2.1.10", "either-t100", 9, "+ pooled labels"),
]

PRIMARY = CONDITIONS[-1].key
"""The full recipe — the condition H2 scores against the control."""

CONTROL = CONDITIONS[0].key
"""The un-anchored condition — H2's baseline, and H1's model-independence case."""


def line_colors(probe_tokens) -> tuple[Any, Any]:
    """Palette index of every line's (op1, op2, ans) tokens, plus the RGB table.

    The probe set walks colors in palette order at op1, so the palette index of
    a color token is where it first appears in that column; the same map then
    reads the op2 and ans columns. Taken from the tokens rather than replayed
    from the closure rule, so the index cannot drift from the lines it
    describes (as in `docs/m2/d2.1/experiment.py`).
    """
    import numpy as np

    from sca.vis_grading import GRID_RGB

    op1 = probe_tokens[:, TARGETS["op1"]]
    _, first = np.unique(op1, return_index=True)
    index = {int(t): i for i, t in enumerate(op1[np.sort(first)])}
    cols = np.stack([[index[int(t)] for t in probe_tokens[:, p]] for p in TARGETS.values()], axis=1)
    return cols, GRID_RGB[cols]  # (N, 3 roles) indices, (N, 3 roles, 3 ch) RGB in [0, 1]


def strict_predictions(x, y, slots, l2: float = RIDGE_L2):
    """Out-of-fold ridge predictions under the per-value holdout, per channel.

    The protocol of `sca.compute.geometry.strict_r2`: to score channel k at
    level v, every line holding level v in channel k of *any* slot (either
    operand or the answer) leaves the fit together, so the probe has to place
    an unseen level from the others. This local variant returns the held-out
    predictions themselves rather than only their R², because the report draws
    the prediction curves; it stays here rather than amending the shared helper
    so ex-2.1.5's cached evidence is untouched.

    Grouped folds make the fit cheap: one Gram matrix serves every fold, each
    downdated by its own block, as in `ridge_probe_loo`.
    """
    import numpy as np

    x64, y64 = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n, c = x64.shape
    sum_x, gram_xx = x64.sum(0), x64.T @ x64
    pred = np.full_like(y64, np.nan)
    for k in range(y64.shape[1]):
        yk = y64[:, k]
        sum_y, gram_xy = yk.sum(), x64.T @ yk
        for v in np.unique(yk):
            held = np.any([np.abs(s[:, k] - v) < 1e-6 for s in slots], axis=0)
            scored = np.abs(yk - v) < 1e-6
            n_tr = n - int(held.sum())
            if n_tr < c or not scored.any():
                continue
            xh, yh = x64[held], yk[held]
            mx, my = (sum_x - xh.sum(0)) / n_tr, (sum_y - yh.sum()) / n_tr
            xx = gram_xx - xh.T @ xh - n_tr * np.outer(mx, mx)
            xy = gram_xy - xh.T @ yh - n_tr * mx * my
            pred[scored, k] = (x64[scored] - mx) @ np.linalg.solve(xx + l2 * np.eye(c), xy) + my
    assert not np.isnan(pred).any(), "every line should be scored exactly once per channel"
    return pred


def probe_one(exp: str, cond: str, seeds: int) -> dict:
    """Probe maps and prediction margins for every seed of one condition.

    For each run: capture the residual stream over the experiment's own probe
    set, then per (slice, position) site and per target fit strict-holdout
    ridge probes for the target's RGB. Publishes per-channel R² and, for the
    curves, both margins of the out-of-fold predictions — grouped by op1's
    color and by op2's, as the α maps in d2.1 are.
    """
    import io

    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    from mini.store import get_many, get_ref, put
    from sca.compute.model import load_checkpoint

    workdir = get_data_dir() / "probes" / f"{exp}-{cond}"
    probes_ref = get_ref(f"reports/m2/{exp}/probes")
    assert probes_ref is not None, f"{exp} has not published its probe set"
    (probe_path,) = get_many([(probes_ref, workdir / "probes.npz")])
    with np.load(probe_path) as z:
        probe_tokens = z["probe_tokens"]

    cols, rgb = line_colors(probe_tokens)
    n_lines = len(probe_tokens)
    n_colors = cols.max() + 1
    n_partners = n_lines // n_colors
    # The ans column must be the round-half-up channel mean of the operands,
    # which pins the token → palette-index map against the closure rule.
    lv = np.rint(rgb * (N_LEVELS - 1)).astype(int)
    assert (lv[:, 2] == (lv[:, 0] + lv[:, 1] + 1) // 2).all(), f"{exp} probe set is not closed under mixing"
    slots = [rgb[:, i] for i in range(rgb.shape[1])]
    by_op2 = np.argsort(cols[:, 1], kind="stable")
    # `margins` reshapes to (color, partner) on each ordering, so both keyings
    # need balanced, contiguous groups of `n_partners`.
    groups = np.repeat(np.arange(n_colors), n_partners)
    assert (cols[:, 0] == groups).all(), f"{exp} probe lines are not grouped by op1 in palette order"
    assert (cols[by_op2, 1] == groups).all(), f"{exp} probe lines are not balanced over op2"

    def margins(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(keyed by op1, keyed by op2) means over each color's partner group."""
        m = lambda a: a.reshape(n_colors, n_partners, *a.shape[1:]).mean(axis=1)  # noqa: E731
        return m(lines), m(lines[by_op2])

    out: dict[str, Any] = {"probe/pairs": cols[:, :2].astype(np.int16)}
    for seed in range(seeds):
        label = f"{cond}-s{seed}"
        ckpt = get_ref(f"reports/m2/{exp}/checkpoints/{label}")
        assert ckpt is not None, f"{exp} has not published a checkpoint for {label}"
        get_many([(ckpt, workdir / label / "model")])
        model, _, _ = load_checkpoint(workdir / label)
        stream = eqx.filter_jit(model.residual_stream)
        acts = np.concatenate(
            [np.asarray(stream(jnp.asarray(probe_tokens[i : i + 1024]))) for i in range(0, n_lines, 1024)],
            axis=1,
        )  # (slices, N, positions, width)
        n_slices, _, n_pos, _ = acts.shape

        r2 = np.full((n_slices, n_pos, len(TARGETS), 3), np.nan, np.float32)
        m1 = np.full((n_slices, n_colors, n_pos, len(TARGETS), 3), np.nan, np.float32)
        m2 = np.full_like(m1, np.nan)
        for si in range(n_slices):
            for pi in range(n_pos):
                for ti in range(len(TARGETS)):
                    y = rgb[:, ti]
                    pred = strict_predictions(acts[si, :, pi], y, slots)
                    ss_res = ((y - pred) ** 2).sum(0)
                    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
                    r2[si, pi, ti] = 1 - ss_res / ss_tot
                    m1[si, :, pi, ti], m2[si, :, pi, ti] = margins(pred.astype(np.float32))
        out[f"s{seed}/r2_strict_ch"] = r2
        out[f"s{seed}/pred_op1"] = m1
        out[f"s{seed}/pred_op2"] = m2

    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return {
        "key": f"{exp}/{cond}",
        "n_runs": seeds,
        "arrays": put(buf.getvalue(), name=f"ex-2.1.12-{exp}-{cond}-probes.npz"),
    }


def publish_results(results: list[dict]) -> dict:
    """Stack every condition's arrays into one npz under `{exp}/{cond}/…`."""
    import io

    import numpy as np

    from mini.store import get, put, set_ref

    arrays: dict[str, Any] = {}
    pairs = None
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['key'].replace('/', '-')}.npz")
        with np.load(path) as z:
            for name in z.files:
                if name == "probe/pairs":
                    assert pairs is None or np.array_equal(z[name], pairs), (
                        f"{r['key']} was measured on a different probe set"
                    )
                    pairs = z[name]
                else:
                    arrays[f"{r['key']}/{name}"] = z[name]
    assert pairs is not None
    arrays["probe/pairs"] = pairs
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.12-probes.npz"))
    return {"n_conditions": len(results), "n_runs": sum(r["n_runs"] for r in results), "n_arrays": len(arrays)}


def main(ctx: Ctx) -> dict:
    exps = [c.exp for c in CONDITIONS]
    names = [c.name for c in CONDITIONS]
    seeds = [c.seeds for c in CONDITIONS]
    probed = ctx.map(probe_one, exps, names, seeds, role="cpu")
    return ctx.run(publish_results, probed, role="cpu")


experiment = Experiment(
    name="ex-2.1.12",
    main=main,
    # One condition per task. The heavy parts are the checkpoint fetches and
    # the 5 × 6 × 3 grid of grouped-fold ridge fits per run; the models are
    # 4-layer and 64-wide, so everything fits comfortably on CPU.
    roles={"cpu": dict(cpu=4, timeout=3600, watchdog_grace=1800)},
    deps=["ex-2.1.6", "ex-2.1.8", "ex-2.1.10"],
)
