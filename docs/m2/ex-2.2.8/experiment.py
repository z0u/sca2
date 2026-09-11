"""
Survey: tuning the intervention operator on the stored ex-2.2.3 checkpoints.

A scoring-only pass, no training. The D2.2 design chooses the operator for the anchored-op
experiments by tuning the shaped suppression's threshold and ramp and trying M1's repulsion
form on stored runs. Ex-2.2.1 left three operators, none both complete and selective: the plain
projection removes red fully at a non-red cost paid at the syntax positions, the shaped
suppression at M1's threshold removes about half of red at no non-red cost, and the
operand-only projection does both but needs the line's syntax.

This module scores a frozen trial list of operators on every stored seed of ex-2.2.3's adopted
point (`recipe-short`, twenty seeds) and of the survey's proposal `t00` (five seeds), on the
six-op probe lines, through the eval contract. Every trial is published; the report draws the
landscape (red removal against non-red cost) and proposes an operator. Nothing here is a result:
the anchored-op prereg adopts the proposal and scores it at fresh seeds.

    bin/mini run docs/m2/ex-2.2.8/experiment.py --app modal --max-containers 5 --budget 2h
    bin/mini status ex-2.2.8
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex223():
    """Ex-2.2.3's module, loaded by path and left out of `sys.modules`, as `mini.load_experiment` does,
    so its task functions still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.3" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex223", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex223 = _load_ex223()

# Bound by name so a task body never references the module object itself.
ANCHOR_AXIS = ex223.ANCHOR_AXIS
ANSWER_POS = ex223.ANSWER_POS
DECODE_POS = ex223.DECODE_POS
SLICES = ex223.SLICES
OPERAND_POSITIONS = ex223.OPERAND_POSITIONS
OP_NAMES = ex223.OP_NAMES
COMPOSITION = ex223.COMPOSITION
RED_DOSE = ex223.RED_DOSE
NONRED_DOSE = ex223.NONRED_DOSE
RED_ACC_GATE = ex223.RED_ACC_GATE
NONRED_DEFICIT_GATE = ex223.NONRED_DEFICIT_GATE
_Readout = ex223._Readout
_load = ex223._load
_npz = ex223._npz
_probe_ops = ex223._probe_ops
top_quantile = ex223.top_quantile

METRICS_REF = "reports/m2/ex-2.2.8/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.8/arrays"

EX223_CHECKPOINT_REF = ex223.CHECKPOINT_REF
EX223_PROBE_REF = ex223.PROBE_REF
EX223_METRICS_REF = ex223.METRICS_REF
"""What this pass reads: ex-2.2.3's checkpoints and six-op probe set, and its metrics for the reference rows."""

# --- The checkpoints ----------------------------------------------------------------------

CONDITIONS = {
    "recipe-short": [f"recipe-short-s{s}" for s in range(5)]
    + [f"recipe-short-more-s{s}" for s in ex223.ADDENDUM_SEEDS],
    "t00": [f"t00-s{s}" for s in range(5)],
}
"""Ex-2.2.3's stored runs by condition: the adopted point at all twenty seeds (the five frozen seeds and the
fifteen of the addendum), and the survey's proposal `t00`, whose syntax rows carry the axis at more than
twice the recipe's level. The seed of each run is the number after `-s`."""

N_RUNS = sum(len(v) for v in CONDITIONS.values())

# --- The space ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One operator: a family and its parameters, applied at every slice, at every position unless `positions`."""

    name: str
    family: str
    """`projection`, `shaped`, `repulsion-linear`, or `repulsion-bezier`."""
    a: float = 0.0
    """The alignment threshold: states below it are untouched (all families but `projection`)."""
    b: float = 1.0
    """`shaped`: the ceiling on the removed fraction. `repulsion-*`: the landing alignment."""
    p: float = 1.0
    """`shaped`: the ramp's shape; 0 is a step (full removal above the threshold), 1 is linear."""
    positions: tuple[int, ...] | None = None


REFERENCE = (
    Trial("projection", "projection"),
    Trial("operands", "projection", positions=OPERAND_POSITIONS),
)
"""Ex-2.2.3's two projection rows, re-scored here as the survey's anchors: their values must match the stored
ones, and they are the two ends the tuned operators are read between (complete but not selective; selective
but syntax-dependent)."""

SHAPED_A = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SHAPED_P = (0.0, 0.5, 1.0, 2.0)
SHAPED = tuple(Trial(f"shaped-a{a:.1f}-p{p:g}", "shaped", a=a, b=1.0, p=p) for a in SHAPED_A for p in SHAPED_P)
"""M1's shaped suppression over its threshold and ramp, with the ceiling fixed at full removal. `a = 0.5, p = 1`
is the arm ex-2.2.1 and ex-2.2.3 scored. `p = 0` is a thresholded projection: everything at or above the
threshold lands at zero alignment, which is also what `repulsion-linear` does at b = 0, so that edge of the
repulsion family is not repeated."""

LINEAR_AB = tuple((a, b) for a in (0.3, 0.4, 0.5, 0.6, 0.7) for b in sorted({0.2, 0.4, a}) if b <= a)
LINEAR = tuple(Trial(f"linear-a{a:.1f}-b{b:.1f}", "repulsion-linear", a=a, b=b) for a, b in LINEAR_AB)
"""M1's repulsion with the linear (ceiling) mapper: a state at or above the threshold lands at *b*. `a = b` is
the continuous ceiling `min(α, b)`; `b < a` puts a step at the threshold."""

BEZIER_AB = tuple((a, b) for a in (0.0, 0.2, 0.4) for b in (0.2, 0.4, 0.6) if b >= a)
BEZIER = tuple(Trial(f"bezier-a{a:.1f}-b{b:.1f}", "repulsion-bezier", a=a, b=b) for a, b in BEZIER_AB)
"""M1's repulsion with the Bézier mapper, continuous at the threshold: unit slope leaving (a, a), flat arriving
at (1, b). Monotone when b ≥ a + (1 − a)/3; below that the map rises before it settles at *b*."""

TUNED = (*SHAPED, *LINEAR, *BEZIER)
AT_OPERANDS = tuple(
    Trial(f"{t.name}-operands", t.family, a=t.a, b=t.b, p=t.p, positions=OPERAND_POSITIONS) for t in TUNED
)
"""Every tuned operator again at the operand positions only, so the survey can say whether a geometric threshold
does the work of the position mask, or whether the two compound."""

TRIALS: tuple[Trial, ...] = (*REFERENCE, *TUNED, *AT_OPERANDS)
assert len({t.name for t in TRIALS}) == len(TRIALS), "trial names are unique"
N_TRIALS = len(TRIALS)

# --- The objectives -----------------------------------------------------------------------

OBJECTIVE = f"""\
Removal against selectivity, as ex-2.2.3's H4 read them: seed-mean accuracy on the red lines (the dose above \
{RED_DOSE}; lower is more removed) against the seed-mean non-red deficit (the drop in P(answer) on lines whose \
dose is at most {NONRED_DOSE}; lower is more selective), on `mix` as the gated op and on every op beside it. \
Feasible: non-red deficit within {NONRED_DEFICIT_GATE} on every op. Proposed: the feasible trial with the \
lowest red accuracy on `mix`, with the front reported whole."""

NOISE_TRIALS = ("projection",)
"""Where the per-run σ of each objective is read: the reference projection at the adopted point's twenty seeds."""


# --- Scoring ------------------------------------------------------------------------------


def operator_for(trial: Trial, sub):
    from sca.intervention import projection, repulsion, shaped_suppression

    match trial.family:
        case "projection":
            return projection(sub)
        case "shaped":
            return shaped_suppression(sub, a=trial.a, b=trial.b, p=trial.p)
        case "repulsion-linear":
            return repulsion(sub, a=trial.a, b=trial.b, kind="linear")
        case "repulsion-bezier":
            return repulsion(sub, a=trial.a, b=trial.b, kind="bezier")
        case _:
            raise ValueError(trial.family)


def expected_write(trial: Trial, alpha: np.ndarray) -> np.ndarray | None:
    """The write a trial's operator makes on a state of alignment *alpha*, where the contract has it in closed form."""
    from sca.intervention import repulsion_mapper, write_angle

    if trial.family == "projection":
        return write_angle(alpha)
    if trial.family.startswith("repulsion"):
        a = np.maximum(alpha, 0.0)
        m = repulsion_mapper(a, trial.a, trial.b, trial.family.split("-")[1])
        return np.abs(np.arccos(np.clip(m, -1, 1)) - np.arccos(np.clip(a, -1, 1)))
    return None


def score_one(checkpoint, probes, trials: tuple[Trial, ...], condition: str, label: str) -> dict:
    """Score one stored checkpoint under every trial, on every op's probe lines at once.

    The six ops' lines are concatenated so each trial is one forward pass (one compilation), then read per op
    with ex-2.2.3's `_Readout`, so every statistic means what it meant there. Per-run statistics return as the
    result; per-line P(answer) and the decoded answer go to the store.
    """
    from sca.intervention import Subspace, angle_between, apply, projection
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, tok2color = _load({"checkpoint": checkpoint}, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    with np.load(get(probes, workdir / "probes.npz")) as z:
        ops = _probe_ops(z)
        probe = {o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"]) for o in ops}
    reads = {o: _Readout(t, r1, r2, color_ids, tok2color) for o, (t, r1, r2) in probe.items()}
    tokens = np.concatenate([probe[o][0] for o in ops])
    bounds = np.cumsum([0] + [len(probe[o][0]) for o in ops])
    spans = {o: slice(bounds[i], bounds[i + 1]) for i, o in enumerate(ops)}
    n_pos = tokens.shape[1]
    red_pos = np.concatenate([reads[o].red_operand for o in ops])  # the operand carrying the dose, per line

    # --- The clean pass: the reference every trial is read against.
    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = clean.pre[..., ANCHOR_AXIS]  # (L1, N, T)
    c = {o: reads[o](clean.logits[spans[o]]) for o in ops}
    per_op: dict[str, dict[str, Any]] = {
        o: {
            "n": {g: int(m.sum()) for g, m in reads[o].groups.items()},
            "clean": {
                "acc": reads[o].by_group(c[o]["guess"] == reads[o].answer),
                "p_ans": reads[o].by_group(c[o]["p_ans"]),
                "expected_dist": reads[o].by_group(c[o]["expected_dist"]),
                "alpha_q99_nonred": top_quantile(
                    np.abs(alpha_clean[:, spans[o]][:, reads[o].groups["nonred"]]), axis=1
                ).tolist(),
                "alpha_red_operand": _landing(alpha_clean[:, spans[o]], reads[o], red_pos[spans[o]]),
            },
            "trials": {},
        }
        for o in ops
    }
    result: dict[str, Any] = {
        "label": label,
        "condition": condition,
        "seed": int(label.rsplit("-s", 1)[1]),
        "ops": per_op,
    }
    arrays: dict[str, np.ndarray] = {}
    for o in ops:
        arrays[f"{o}/dose"] = reads[o].dose.astype(np.float32)
        arrays[f"{o}/clean/p_ans"] = c[o]["p_ans"].astype(np.float32)
        arrays[f"{o}/clean/guess"] = c[o]["guess"].astype(np.int16)

    # --- Each trial: one pass over every line, the contract's checks, then per-op readings.
    for trial in trials:
        positions = None if trial.positions is None else np.isin(np.arange(n_pos), trial.positions).astype(np.float32)
        out = apply(model, tokens, operator_for(trial, sub), slices=SLICES, positions=positions)
        theta = angle_between(out.pre, out.post)  # (L1, N, T): the write at every site
        alpha_pre = out.pre[..., ANCHOR_AXIS]
        alpha_post = out.post[..., ANCHOR_AXIS]
        np.testing.assert_allclose(alpha_pre[0], alpha_clean[0], rtol=0, atol=0)
        if positions is not None:
            np.testing.assert_allclose(theta[:, :, positions == 0], 0.0, rtol=0, atol=1e-6)
        expected = expected_write(trial, alpha_pre)
        if expected is not None:
            on = np.ones(n_pos, bool) if positions is None else positions > 0
            np.testing.assert_allclose(theta[:, :, on], expected[:, :, on], rtol=0, atol=2e-3)
        disp = angle_between(out.post[-1, :, DECODE_POS], clean.post[-1, :, DECODE_POS])
        for o in ops:
            s, r = spans[o], reads[o]
            read = r(out.logits[s])
            nonred = r.groups["nonred"]
            per_op[o]["trials"][trial.name] = {
                "acc": r.by_group(read["guess"] == r.answer),
                "p_ans": r.by_group(read["p_ans"]),
                "deficit": r.by_group(c[o]["p_ans"] - read["p_ans"]),
                "argmax_dist": r.by_group(read["argmax_dist"]),
                "expected_dist": r.by_group(read["expected_dist"]),
                "offvocab": r.by_group(read["offvocab"]),
                "disp": r.by_group(disp[s]),
                "q99_write_nonred": top_quantile(theta[:, s][:, nonred], axis=1).tolist(),
                "mean_write_nonred": theta[:, s][:, nonred].mean(1).tolist(),
                "alpha_red_operand": _landing(alpha_post[:, s], r, red_pos[s]),
                "composition_red": np.bincount(
                    read["composition"][r.groups["red"]], minlength=len(COMPOSITION)
                ).tolist(),
            }
            arrays[f"{o}/{trial.name}/p_ans"] = read["p_ans"].astype(np.float32)
            arrays[f"{o}/{trial.name}/guess"] = read["guess"].astype(np.int16)
    result["arrays"] = put(_npz(**arrays), name=f"ex-2.2.8-{label}-score.npz")
    return result


def _landing(alpha: np.ndarray, read, red_pos: np.ndarray) -> list[float]:
    """Mean alignment per slice of the dose-carrying operand's state on the red lines: where the operator leaves it."""
    red = read.groups["red"]
    rows = np.arange(alpha.shape[1])[red]
    return alpha[:, rows, red_pos[red]].mean(1).tolist()


# --- Publishing ---------------------------------------------------------------------------


def design() -> dict[str, Any]:
    return {
        "n_runs": N_RUNS,
        "conditions": {k: list(v) for k, v in CONDITIONS.items()},
        "n_trials": N_TRIALS,
        "trials": [asdict(t) for t in TRIALS],
        "reference": [t.name for t in REFERENCE],
        "shaped_grid": {"a": SHAPED_A, "p": SHAPED_P},
        "linear_grid": LINEAR_AB,
        "bezier_grid": BEZIER_AB,
        "objective": OBJECTIVE,
        "noise_trials": NOISE_TRIALS,
        "gates": {"red_acc": RED_ACC_GATE, "nonred_deficit": NONRED_DEFICIT_GATE},
        "dose": {"red": RED_DOSE, "nonred": NONRED_DOSE},
        "slices": SLICES,
        "composition": list(COMPOSITION),
    }


def publish_results(scored: list[dict]) -> dict:
    import json

    from mini.store import get, put, set_ref

    metrics = {"scores": [{k: v for k, v in r.items() if k != "arrays"} for r in scored], "design": design()}
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.8-metrics.json"))
    arrays = {}
    for r in scored:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-score.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.8-arrays.npz"))
    return {"n_runs": len(scored), "n_trials": N_TRIALS}


# --- Orchestration ------------------------------------------------------------------------


def resolve_inputs(labels: list[str]) -> dict:
    """Ex-2.2.3's checkpoints and probe set, by ref, so each scoring task is keyed on the checkpoint's content."""
    from mini.store import get_refs

    refs = get_refs([EX223_PROBE_REF, *(EX223_CHECKPOINT_REF.format(label=lb) for lb in labels)])
    missing = [k for k, v in refs.items() if v is None]
    assert not missing, f"ex-2.2.3 refs not in this store: {missing}"
    return {
        "probes": refs[EX223_PROBE_REF],
        "checkpoints": {lb: refs[EX223_CHECKPOINT_REF.format(label=lb)] for lb in labels},
    }


def main(ctx: Ctx) -> dict:
    labels = [lb for v in CONDITIONS.values() for lb in v]
    inputs = ctx.run(resolve_inputs, labels, role="prep")
    scored = ctx.map(
        score_one,
        [inputs["checkpoints"][lb] for lb in labels],
        [inputs["probes"]] * len(labels),
        [TRIALS] * len(labels),
        [c for c, v in CONDITIONS.items() for _ in v],
        labels,
        role="score",
    )
    return ctx.run(publish_results, scored, role="prep")


experiment = Experiment(
    name="ex-2.2.8",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=900),
        # A hundred operators, each one forward pass over the six ops' 34,992 lines, keeping the stream.
        "score": dict(gpu="L4", timeout=3600),
    },
)
