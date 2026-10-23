"""When the spill appears: the edit, a sham edit and a color probe scored at every record of `pool 1–3, soft τ`.

Soft-pool's `pool 1–3, soft τ` kept the anchor on all eight of its seeds, but only four of its edits stayed within the
spill criterion. A cheap early test for spill would let it (and arms tuned from it) run at more seeds, as the
loss-rate screen did for losing the anchor. Nothing in the stored trajectories up to epoch 64 predicts the end spill,
and every stored run keeps only its end checkpoint, so this experiment trains the condition again at its eight seeds
(704 to 711; training is deterministic, so the end reproduces the stored runs) and measures along training:

- the full edit (e₁ projected out at every position and slice) and the drop it causes on every op, over the whole
  held-out set;
- a sham edit beside it, another basis direction projected out the same way (eight draws), as the baseline the drop
  is netted against, in place of a control run;
- a ridge probe from the state at each color position to the RGB of that color, at slices 2 to 4, on the other ops'
  contexts (and on the anchored op's), scored on held-out contexts: from the whole state, without the e₁ coordinate,
  and from e₁ alone.

The same edit, sham and probe also score the loss-rate screen's epoch-64 checkpoints (`falloff` and `step, soft τ` at
68 seeds, eleven with a known end spill) and ex-2.2.23's controls at 704 to 711, where the sham meets a real control.
At the end, every new run gets the usual dose scoring.

    bin/mini run docs/m2/spill-timing/experiment.py --app modal --max-containers 16 --budget 3h
    bin/mini status spill-timing
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


lr = _load_sibling("loss-rate", "spill_timing_lr")
sp = lr.sp
sas = sp.sas
ex2223 = sp.ex2223
ex2222 = sp.ex2222
ex2221 = sp.ex2221
ex2216 = ex2221.ex2216
ex2218 = ex2221.ex2218

OP_NAMES = sp.OP_NAMES
ANCHORED_OP = sp.ANCHORED_OP
K = sp.K
EPOCHS = sp.EPOCHS
EVERY = sp.EVERY
TRAJ_STRIDE_EPOCHS = sp.TRAJ_STRIDE_EPOCHS

ARM = "pool-123-soft"
"""Soft-pool's arm with the most spill: the pull pooled over slices 1 to 3, both temperatures soft-to-sharp."""
SEEDS = sp.SEEDS
"""704 to 711, the stored runs' seeds, so each new run should reproduce one and its end spill is the reference."""

FULL_DOSE = max(ex2216.DOSE_GAMMAS)
"""The edit scored along training: e₁ removed in full, at every position and slice."""
N_SHAM = 8
"""Basis directions other than e₁ per record. A few of them cost an op far more than the rest, so the report takes
their median."""
SHAM_SEED = 0
"""The draws are fixed: the same directions at every record of every run, so sham drops compare along a run."""
PROBE_SLICES = (2, 3, 4)
"""Where the probe reads: the slices the recipe of record pulls, two of which this arm pulls too."""
STORED_PER_TASK = 16
"""Stored checkpoints scored per task: about three seconds each on a GPU, after a setup and start that take longer."""
RIDGE_L2 = 1e-3
"""The probe's ridge weight, relative to the mean variance of a state coordinate times the training count."""


def color_positions(k: int) -> tuple[list[int], list[int]]:
    """The positions of a context of *k* examples whose state the probe reads, and where its color target sits: each
    example's operands and answer, the query operands, and the query `=`, whose target is the query answer after it.
    """
    pos = [6 * i + j for i in range(k) for j in (0, 2, 4)] + [6 * k, 6 * k + 2, 6 * k + 3]
    return pos, pos[:-1] + [6 * k + 4]


def position_names(k: int) -> list[str]:
    return [f"{n}{i + 1}" for i in range(k) for n in ("a", "b", "c")] + ["qa", "qb", "q="]


def sham_directions(width: int, n: int = N_SHAM, seed: int = SHAM_SEED) -> np.ndarray:
    """*n* basis directions other than e₁, drawn without replacement. A basis direction rather than a random one,
    since the trained models are far from rotation-invariant: on a control, projecting out a random unit direction
    costs every op much more than projecting out any one coordinate does.
    """
    from sca.anchoring import ANCHOR_AXIS

    axes = np.random.default_rng(seed).choice(np.delete(np.arange(width), ANCHOR_AXIS), size=n, replace=False)
    return np.eye(width)[axes]


def _ridge_r2(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    """R² on the held-out half of a ridge fit from *x* to *y*, summed over the target channels."""
    xa, ya, xb, yb = x[train], y[train], x[~train], y[~train]
    mx, my = xa.mean(0), ya.mean(0)
    a = xa - mx
    gram = a.T @ a
    lam = RIDGE_L2 * np.trace(gram) / gram.shape[0]
    w = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), a.T @ (ya - my))
    pred = (xb - mx) @ w + my
    return float(1.0 - ((yb - pred) ** 2).sum() / ((yb - yb.mean(0)) ** 2).sum())


def record_scoring(config, ops: tuple[str, ...], k: int, holdout, workdir):
    """The measurement `train_one` calls at every record (its *record_extra*), and `score_checkpoints` once per checkpoint.

    Returns `model -> dict` with, per op over the whole held-out set: expected exact match on the clean pass
    (`es_clean`), under the full edit (`es_edit`) and under each sham draw (`es_sham`, one row per draw), and the
    standard error of the per-context net drop, mean sham minus edit (`es_net_se`). And the probe's held-out R² by
    context group (the other ops', then the anchored op's), state (whole, without e₁, then e₁ alone), slice, and
    position (`probe_r2`).
    """
    import equinox as eqx
    import jax.numpy as jnp

    from sca.anchoring import ANCHOR_AXIS
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    tokenizer = WordTokenizer(config.tokenizer)
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    ctx = ex2216._post_queries(P, ho, k, tok2color)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
    width = config.model.n_embd
    every = tuple(range(config.model.n_layer + 1))
    mask = ex2216.site_positions("every position", k)
    identity = projection(Subspace.axis(width), 0.0)
    edit = projection(Subspace.axis(width), FULL_DOSE)
    shams = [projection(Subspace.direction(v), FULL_DOSE) for v in sham_directions(width)]
    n_ops = len(ops)
    a_id = ops.index(ANCHORED_OP)
    by_op = [ho.op_ids == o for o in range(n_ops)]

    rgb = np.asarray(list(PALETTE.values()), dtype=np.float64)
    rgb = rgb / rgb.max()
    pos, target_pos = color_positions(k)
    targets = tok2color[ho.tokens[:, target_pos]]  # (N, positions)
    assert (targets >= 0).all(), "every probed position has a color target"
    groups = [ho.op_ids != a_id, ho.op_ids == a_id]
    # Alternate contexts of each group train and test the probe.
    halves = [np.arange(int(g.sum())) % 2 == 0 for g in groups]
    keep = np.delete(np.arange(width), ANCHOR_AXIS)

    @eqx.filter_jit
    def states(model, idx):
        return model.residual_stream(idx)[jnp.asarray(PROBE_SLICES)][:, :, jnp.asarray(pos)]

    def stream(model, tokens: np.ndarray, batch: int = 1024) -> np.ndarray:
        out = []
        for i in range(0, len(tokens), batch):
            chunk = tokens[i : i + batch]
            n = len(chunk)
            if n < batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - n, axis=0)])
            out.append(np.asarray(states(model, jnp.asarray(chunk)))[:, :n])
        return np.concatenate(out, axis=1).astype(np.float64)  # (slices, N, positions, C)

    def match(model, op, slices) -> np.ndarray:
        p = ex2216._color_probs(logits_at(model, ho.tokens, op, slices, eq_role, mask), color_ids)
        return ex2216._match(table, ctx, p)

    def per_op(x: np.ndarray) -> list[float]:
        return [round(float(x[m].mean()), 5) for m in by_op]

    def score(model) -> dict[str, Any]:
        clean, edited = match(model, identity, ()), match(model, edit, every)
        sham = np.stack([match(model, s, every) for s in shams])
        net = sham.mean(0) - edited
        se = [round(float(net[m].std(ddof=1) / np.sqrt(m.sum())), 5) for m in by_op]
        r2 = []
        for g, train in zip(groups, halves, strict=True):
            h = stream(model, ho.tokens[g])
            y = rgb[targets[g]]  # (n, positions, 3)
            r2.append(
                [
                    [
                        [round(_ridge_r2(h[s, :, j][:, cols], y[:, j], train), 4) for j in range(len(pos))]
                        for s in range(len(PROBE_SLICES))
                    ]
                    for cols in (slice(None), keep, [ANCHOR_AXIS])
                ]
            )
        return {
            "es_clean": per_op(clean),
            "es_edit": per_op(edited),
            "es_sham": [per_op(x) for x in sham],
            "es_net_se": se,
            "probe_r2": r2,
        }

    return score


def score_checkpoints(checkpoints: dict, holdout, ops: tuple[str, ...], k: int) -> list[dict]:
    """`record_scoring` once on each stored checkpoint of *checkpoints* (label → checkpoint), in one task, since the
    setup and a container start cost more than the scoring.
    """
    from sca.compute.model import load_checkpoint
    from mini.store import get

    out, score = [], None
    for label, checkpoint in checkpoints.items():
        workdir = ex2221.get_data_dir() / "spill-timing" / label
        get(checkpoint, workdir / "model")
        model, config, _ = load_checkpoint(workdir)
        if score is None:
            score = record_scoring(config, ops, k, holdout, workdir)
        out.append({"label": label} | score(model))
    return out


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/spill-timing"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"
STORED_REF = PREFIX + "/stored-checkpoints"


def label_of(model_seed: int) -> str:
    return sp.label_of(ARM, model_seed)  # the stored runs' labels, so the two meet by label in the report


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """Soft-pool's rows for `pool 1–3, soft τ`: the same config, schedules and seeds, so each run is a stored run's twin."""
    return [r for r in sp.rows_of(meta, epochs, seeds) if r["arm"]["name"] == ARM]


def resolve_stored() -> dict:
    """The loss-rate screen's epoch-64 checkpoints and ex-2.2.23's controls at 704 to 711, by ref."""
    from mini.store import get_ref

    labels = [lr.label_of(a.name, s) for a in lr.ARMS for s in lr.SEEDS + lr.REPRO_SEEDS]
    refs = {lab: lr.CHECKPOINT_REF.format(label=lab) for lab in labels}
    refs |= {f"control-s{s}": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)) for s in SEEDS}
    found = {lab: get_ref(r) for lab, r in refs.items()}
    missing = [lab for lab, a in found.items() if a is None]
    assert not missing, f"not pinned: {missing}"
    return found


def design() -> dict[str, Any]:
    return {
        "experiment": "spill-timing",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arm": ARM,
        "seeds": list(SEEDS),
        "full_dose": FULL_DOSE,
        "n_sham": N_SHAM,
        "sham_seed": SHAM_SEED,
        "probe_slices": list(PROBE_SLICES),
        "probe_positions": position_names(K),
        "probe_groups": ["other ops", ANCHORED_OP],
        "probe_states": ["whole", "without e₁", "e₁ alone"],
        "ridge_l2": RIDGE_L2,
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], dosed: list[dict], stored: list[dict]) -> dict:
    """The trajectories with their record scoring, every checkpoint, the dose scoring with its per-context arrays, and
    the scoring of the stored checkpoints, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="spill-timing-trajectories.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="spill-timing-dose.json"))
    body = {"design": design(), "runs": stored}
    set_ref(STORED_REF, put(json.dumps(body).encode(), name="spill-timing-stored.json"))
    return {"n_runs": len(trained), "n_stored": len(stored)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, n_stored: int | None = None,
        publishes: bool = True) -> dict:  # fmt: skip
    """Resolve ex-2.2.21's corpus condition and the stored checkpoints, train every run with the record scoring,
    dose-score its end, score the stored checkpoints, and publish. The arguments let a smoke run take the same path at
    a fraction of the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    found = ctx.run(resolve_stored, role="prep")
    rows = rows_of(resolved["meta"], epochs, seeds)
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        ["whole"] * n,
        [r["anchor_slices"] for r in rows],
        [""] * n,
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [EVERY] * n,
        [None] * n,
        [True] * n,
        [None] * n,
        [record_scoring] * n,
        role="train",
    )
    dosed = ctx.map(
        ex2222.suppress_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [OP_NAMES] * n,
        [K] * n,
        [True] * n,
        [r["label"] for r in rows],
        role="measure",
    )
    labels = list(found)[:n_stored]
    chunks = [labels[i : i + STORED_PER_TASK] for i in range(0, len(labels), STORED_PER_TASK)]
    scored = ctx.map(
        score_checkpoints,
        [{lab: found[lab] for lab in c} for c in chunks],
        [resolved["holdout"]] * len(chunks),
        [OP_NAMES] * len(chunks),
        [K] * len(chunks),
        role="measure",
    )
    stored = [r for chunk in scored for r in chunk]
    if not publishes:
        return {"labels": [t["label"] for t in trained], "stored": labels}
    return ctx.run(publish, rows, trained, dosed, stored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed for eight epochs (past the five-epoch warm-up) and two stored checkpoints, through the whole DAG."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], n_stored=2, publishes=False)


COMPUTE = sas.COMPUTE

experiment = Experiment(name="spill-timing", main=main, roles=COMPUTE)
