"""Low runs: what the e₁ edit misses in the fixed-label runs that end low, scored on stored checkpoints.

The fixed-labels experiment found that a few runs trained with one label draw held for the run remove `difference`
well below the rest (0.60 to 0.79 against about 0.88), that which runs end low looks like chance (the same set at the
same seed ends low or high with the last bits of the arithmetic), and that the low runs mostly never reach the level of
the others. This experiment reads the stored checkpoints of every run in that experiment, its crossed runs, and the
per-visit runs of the fixed confirmation it compared against, and asks what the low runs hold that the edit cannot
reach: the edit restricted to each slice and position set, the state at the example answers with every direction but
e₁ removed, where the per-example judgement is decodable off e₁, and alignment on the labelled contexts against the
unlabelled ones. The same measurements run along training on the checkpoints each run kept. No new training.

    bin/mini run docs/m2/low-runs/experiment.py --app modal --max-containers 8 --budget 1h
    bin/mini status low-runs
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


fl = _load_sibling("fixed-labels", "low_runs_fl")
fc, ac = fl.fc, fl.ac
lean, ex2222, ex2221 = fc.lean, fc.ex2222, ac.ex2221
ex2216, ex2218 = lean.ex2216, lean.ex2218

OP_NAMES = fl.OP_NAMES
ANCHORED_OP = fl.ANCHORED_OP
K = fl.K
EPOCHS = fl.EPOCHS
NAME = "low-runs"
UNIT = 6
"""Tokens per equation: `a ? b = y ,`, with the query ending in `⏎`."""
ANSWERS = tuple(UNIT * i + 4 for i in range(K))
"""The example answer positions of a completion context."""
QUERY_EQ = ex2216.query_role(K, ex2216.QUERY_EQ)
OPERANDS = tuple(UNIT * i + j for i in range(K + 1) for j in (0, 2))
CONTEXT_LEN = ex2216.context_length(K)
N_SLICES = 5
TRAIN_SHARE = 0.9
"""Training visits the first nine tenths of the corpus (ex-2.2.3's split)."""
N_UNLABELLED = 3000
"""How many unlabelled `difference` training contexts the alignment is measured on, beside every labelled one."""
PROBE_SLICES_END = (1, 2, 3, 4)
PROBE_SLICES_ALONG = (2, 3)
BUDGET_USD = 5
"""Forty-eight end checkpoints and thirty-two runs with fourteen kept checkpoints each, each a few forward passes over
the 14,000 held-out contexts and the labelled contexts, on L40S cards."""


# =============================================================================================
# The runs
# =============================================================================================

PREFIX = "reports/m2/low-runs"
END_REF = PREFIX + "/end"
ALONG_REF = PREFIX + "/along"
ROUTES_REF = PREFIX + "/routes"
CROSS_KEPT_REF = PREFIX + "/cross-kept/{label}"
"""The kept checkpoints of the crossed runs. The fixed-labels experiment published only their end checkpoints, so
these refs were set from its train results (the `kept_checkpoints` artifact of each crossed run) before this
experiment ran."""


def runs() -> list[dict[str, Any]]:
    """Every run this experiment reads: the fixed-labels runs at three rates, their crossed runs, and the per-visit
    runs of the fixed confirmation, with the refs of each run's end checkpoint and (where published) its kept ones.
    """
    out = []
    for rate in fl.RATES:
        for s in fl.SEEDS:
            label = fl.label_of(s, None, rate)
            out.append({"label": label, "kind": f"fixed {rate}", "rate": rate, "model_seed": s, "draw_seed": s,
                        "checkpoint": fl.CHECKPOINT_REF.format(label=label), "kept": fl.KEPT_REF.format(label=label)})  # fmt: skip
    for d in fl.CROSS_SEEDS:
        for s in fl.CROSS_SEEDS:
            label = fl.cross_label_of(d, s)
            out.append({"label": label, "kind": "cross", "rate": fl.DEFAULT_RATE, "model_seed": s, "draw_seed": d,
                        "checkpoint": fl.CROSS_CHECKPOINT_REF.format(label=label), "kept": CROSS_KEPT_REF.format(label=label)})  # fmt: skip
    for s in fc.SEEDS:
        label = fc.label_of(s, None)
        out.append({"label": label, "kind": "per-visit", "rate": fl.DEFAULT_RATE, "model_seed": s, "draw_seed": s,
                    "checkpoint": fc.CHECKPOINT_REF.format(label=label), "kept": fc.KEPT_REF.format(label=label)})  # fmt: skip
    return out


def resolve(rows: list[dict]) -> dict:
    """The corpus condition (held-out set, corpus, labels) and every run's checkpoint artifacts, by ref."""
    from mini.store import get_ref

    base = ex2222.resolve_reused([])
    found = {}
    for r in rows:
        found[r["checkpoint"]] = get_ref(r["checkpoint"])
        if r["kept"] is not None:
            found[r["kept"]] = get_ref(r["kept"])
    missing = [k for k, v in found.items() if v is None]
    assert not missing, f"not published: {missing}"
    return {k: base[k] for k in ("holdout", "corpus", "labels")} | {"artifacts": found}


# =============================================================================================
# The measurements
# =============================================================================================


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve of *score* for the binary *y*: the chance a positive outscores a negative."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _probe_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, ridge: float = 1e-2) -> float:
    """A ridge-regression probe of *y* on *x*, fitted on half the groups (contexts) and scored by AUC on the rest."""
    rng = np.random.default_rng(0)
    g = np.unique(groups)
    rng.shuffle(g)
    train = np.isin(groups, g[: len(g) // 2])
    xc = x - x[train].mean(0)
    w = np.linalg.solve(
        xc[train].T @ xc[train] + ridge * train.sum() * np.eye(x.shape[1]), xc[train].T @ (y[train] - y[train].mean())
    )
    return _auc(xc[~train] @ w, y[~train])


def _mask(positions) -> np.ndarray:
    m = np.zeros(CONTEXT_LEN, dtype=np.float32)
    m[list(positions)] = 1
    return m


def _states(model, tokens: np.ndarray, batch: int = 1024) -> np.ndarray:
    import jax.numpy as jnp

    return np.concatenate([np.asarray(model.residual_stream(jnp.asarray(tokens[i : i + batch]))) for i in range(0, len(tokens), batch)], axis=1)  # fmt: skip


class _Scorer:
    """Expected exact match per held-out context under an operator, as the fixed confirmation scores it."""

    def __init__(self, holdout_path, tok2color, color_ids):
        self.P = ex2216._get_posterior()
        self.table = self.P.build_table(ex2218.table_of(OP_NAMES))
        self.ho = ex2216.load_holdout(holdout_path, K)
        self.ctx = ex2216._post_queries(self.P, self.ho, K, tok2color)
        self.color_ids = color_ids
        self.every = _mask(range(CONTEXT_LEN))
        a_id = OP_NAMES.index(ANCHORED_OP)
        null_post = ex2216._target_null(self.P, self.table, self.ctx, self.ho.posterior, a_id)
        self.null = self.P.expected_match(self.table, self.ctx, null_post)

    def score(self, model, op, slices, mask=None) -> np.ndarray:
        from sca.intervention import logits_at

        positions = (self.every if mask is None else mask) if slices else None
        lg = logits_at(model, self.ho.tokens, op, slices, QUERY_EQ, positions)
        return ex2216._match(self.table, self.ctx, ex2216._color_probs(lg, self.color_ids))

    def per_op(self, v: np.ndarray) -> np.ndarray:
        return np.array([float(v[self.ho.op_ids == o].mean()) for o in range(len(OP_NAMES))])


def _labelled_sets(corpus_path, labels_path, draw_seed: int, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training corpus as contexts, with the indices of the `difference` contexts the fixed draw of *draw_seed*
    labels at *rate*, and a sample of the ones it leaves unlabelled. A per-visit run has no fixed set; its "labelled"
    contexts are those of the draw at its seed, which it never saw as a set.
    """
    from sca.compute.data_pipelines import load_data

    data, _ = load_data(corpus_path)
    with np.load(labels_path) as z:
        op_ids = z["op_ids"]
    ctxs = data.reshape(-1, CONTEXT_LEN)
    n_train = int(len(ctxs) * TRAIN_SHARE)
    difc = np.flatnonzero(op_ids[:n_train] == OP_NAMES.index(ANCHORED_OP))
    rng = np.random.default_rng([draw_seed, fl.DRAW_SALT])
    drawn = rng.random(len(op_ids)) < rate
    return ctxs, difc[drawn[difc]], difc[~drawn[difc]][:N_UNLABELLED]


def measure(model, scorer: _Scorer, tok2color, ctxs, labelled, unlabelled, probe_slices, full: bool) -> dict:
    """Every measurement on one checkpoint. With *full*, the larger set of edits and the per-context scores too."""
    from sca.anchoring import alignment
    from sca.intervention import Subspace, projection

    ho = scorer.ho
    width = model.transformer.wte.shape[1]
    a_id = OP_NAMES.index(ANCHORED_OP)
    dif = ho.op_ids == a_id
    e1 = projection(Subspace.axis(width, 0), 1.0)
    identity = projection(Subspace.axis(width), 0.0)
    clean = scorer.score(model, identity, ())
    pc, pn = scorer.per_op(clean), scorer.per_op(scorer.null)
    span = pc - pn
    out: dict[str, Any] = {"clean": pc.tolist(), "null": pn.tolist(), "edits": {}}

    def edit(name: str, op, slices, mask=None) -> np.ndarray:
        scored = scorer.score(model, op, slices, mask)
        out["edits"][name] = ((pc - scorer.per_op(scored)) / span).tolist()
        return scored

    every = tuple(range(N_SLICES))
    edit("slice 1", e1, (1,))
    edit("slice 2", e1, (2,))
    edit("slices 2 to 4", e1, (2, 3, 4))
    edited = edit("every slice", e1, every)
    edit("every slice, answers", e1, every, _mask(ANSWERS))
    if full:
        for s in (0, 3, 4):
            edit(f"slice {s}", e1, (s,))
        edit("slices 1 to 2", e1, (1, 2))
        edit("slices 3 to 4", e1, (3, 4))
        edit("every slice, not answers", e1, every, _mask([p for p in range(CONTEXT_LEN) if p not in ANSWERS]))
        edit("every slice, query =", e1, every, _mask([QUERY_EQ]))
        # Every direction but e₁ removed at the example answers: what the next block sees of them is e₁ alone.
        rest = np.eye(width, dtype=np.float32)[1:]
        only_e1 = Subspace(basis=rest, dual=rest, mean=np.zeros(width, np.float32))
        for s in (1, 2):
            edit(f"only e1, slice {s}, answers", projection(only_e1, 1.0, renorm=False), (s,), _mask(ANSWERS))
            edit(
                f"only e1 renormalized, slice {s}, answers", projection(only_e1, 1.0, renorm=True), (s,), _mask(ANSWERS)
            )
        # The state removed entirely (every direction, no renormalization) at the example answers, or at the other
        # example positions, so the next block sees nothing of them: where the judgement reaches the reader from.
        eye = np.eye(width, dtype=np.float32)
        zero = projection(Subspace(basis=eye, dual=eye, mean=np.zeros(width, np.float32)), 1.0, renorm=False)
        others = [p for p in range(UNIT * K) if p not in ANSWERS]
        for s in (1, 2):
            edit(f"zero, slice {s}, answers", zero, (s,), _mask(ANSWERS))
            edit(f"zero, slice {s}, other example positions", zero, (s,), _mask(others))
        out["per_context"] = {
            "clean": clean[dif].tolist(),
            "edited": edited[dif].tolist(),
            "null": scorer.null[dif].tolist(),
        }

    # Alignment with e₁ at the example answers and the query `=`, by slice, on four sets of contexts.
    cos = {"held-out difference": alignment(model, ho.tokens[dif]), "held-out other": alignment(model, ho.tokens[~dif]),
           "labelled": alignment(model, ctxs[labelled]), "unlabelled": alignment(model, ctxs[unlabelled])}  # fmt: skip
    out["alpha"] = {
        name: {"answers": c[:, :, list(ANSWERS)].mean((1, 2)).tolist(), "query =": c[:, :, QUERY_EQ].mean(1).tolist()}
        for name, c in cos.items()
    }
    out["alpha_quantiles"] = {
        name: np.quantile(cos[name][3][:, list(ANSWERS)], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() for name in ("labelled", "unlabelled", "held-out difference")
    }  # fmt: skip

    # Whether the per-example judgement (this example fits `difference`) and the op at the query are decodable from
    # the state along e₁ alone, and from every other direction: AUC of the e₁ coordinate, and of a linear probe
    # fitted with e₁ zeroed, on held-out contexts of every op.
    st = _states(model, ho.tokens)
    n_colors = len(scorer.color_ids)
    y = tok2color[ho.tokens[:, list(ANSWERS)]]
    a = tok2color[ho.tokens[:, [UNIT * i for i in range(K)]]]
    b = tok2color[ho.tokens[:, [UNIT * i + 2 for i in range(K)]]]
    fits = (scorer.table.lookup(a * n_colors + b, y)[a_id] > 0).ravel().astype(float)
    groups = np.repeat(np.arange(len(ho.tokens)), K)
    out["probe"] = {}
    for s in probe_slices:
        x = st[s][:, list(ANSWERS)].reshape(-1, width)
        xo = x.copy()
        xo[:, 0] = 0
        out["probe"][f"fits, slice {s}"] = {"e1": _auc(x[:, 0], fits), "off-axis": _probe_auc(xo, fits, groups)}
        xq = st[s][:, QUERY_EQ]
        xqo = xq.copy()
        xqo[:, 0] = 0
        out["probe"][f"op at query, slice {s}"] = {"e1": _auc(xq[:, 0], dif), "off-axis": _probe_auc(xqo, dif.astype(float), np.arange(len(xq)))}  # fmt: skip
    return out


def end_one(checkpoint, holdout, corpus, labels, label: str, draw_seed: int, rate: float) -> dict:
    """The full set of measurements on a run's end checkpoint."""
    from mini import get_data_dir
    from mini.store import get

    workdir = get_data_dir() / NAME / "end" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    scorer = _Scorer(get(holdout, workdir / "holdout.npz"), tok2color, color_ids)
    ctxs, labelled, unlabelled = _labelled_sets(
        get(corpus, workdir / "corpus"), get(labels, workdir / "labels.npz"), draw_seed, rate
    )
    out = measure(model, scorer, tok2color, ctxs, labelled, unlabelled, PROBE_SLICES_END, full=True)
    return {"label": label, "n_labelled": int(len(labelled))} | out


def along_one(kept, checkpoint, holdout, corpus, labels, label: str, draw_seed: int, rate: float, epochs: tuple[int, ...]) -> list[dict]:  # fmt: skip
    """The smaller set of measurements on each checkpoint a run kept, and on its end checkpoint."""
    from mini import get_data_dir
    from mini.store import get

    workdir = get_data_dir() / NAME / "along" / label
    end_model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    scorer = _Scorer(get(holdout, workdir / "holdout.npz"), tok2color, color_ids)
    ctxs, labelled, unlabelled = _labelled_sets(
        get(corpus, workdir / "corpus"), get(labels, workdir / "labels.npz"), draw_seed, rate
    )
    out = []
    for e in epochs:
        model = ex2221.kept_checkpoint(kept, e, workdir / "kept")[0]
        out.append(
            {"label": label, "epoch": e}
            | measure(model, scorer, tok2color, ctxs, labelled, unlabelled, PROBE_SLICES_ALONG, full=False)
        )
    out.append(
        {"label": label, "epoch": EPOCHS}
        | measure(end_model, scorer, tok2color, ctxs, labelled, unlabelled, PROBE_SLICES_ALONG, full=False)
    )
    return out


def _forward_by_slice(model, idx, ops: dict[int, tuple]):
    """`sca.intervention._forward` with its own list of (operator, positions) per slice, applied in turn: the logits
    at every position. *positions* is a (T,) mask, or None for every position.
    """
    import jax.numpy as jnp

    def edit(x, s: int):
        for op, pos in ops.get(s, ()):
            out = op(x)
            x = out if pos is None else jnp.where(pos[None, :, None] > 0, out, x)
        return x

    x = edit(model.embed(idx), 0)
    allowed = model.attention_mask(idx)
    enc = model.transformer.rotary_enc
    for i, block in enumerate(model.transformer.blocks, start=1):
        x = edit(block(x, enc, allowed), i)
    return (x @ model.transformer.readout.T) * model.s_z()


def _score_by_slice(scorer: _Scorer, model, ops: dict[int, tuple], batch: int = 1024) -> np.ndarray:
    """`_Scorer.score` under a per-slice list of edits."""
    import equinox as eqx
    import jax.numpy as jnp

    fwd = eqx.filter_jit(_forward_by_slice)
    tokens = scorer.ho.tokens
    ops = {s: tuple((op, None if pos is None else jnp.asarray(pos)) for op, pos in edits) for s, edits in ops.items()}
    out = []
    for i in range(0, len(tokens), batch):
        chunk = tokens[i : i + batch]
        n = len(chunk)
        if n < batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - n, axis=0)])
        out.append(np.asarray(fwd(model, jnp.asarray(chunk), ops)[:n, QUERY_EQ]))
    lg = np.concatenate(out)
    return ex2216._match(scorer.table, scorer.ctx, ex2216._color_probs(lg, scorer.color_ids))


def routes_one(checkpoint, holdout, label: str) -> dict:
    """The full e₁ edit with, on top of it, the whole state removed at the example answers or at the other example
    positions at one slice: whether what the full edit leaves behind reaches the reader from the answers or from the
    other positions. Shares of the way, as `measure` gives them, with the full edit alone as a check.
    """
    from mini import get_data_dir
    from mini.store import get
    from sca.intervention import Subspace, projection

    workdir = get_data_dir() / NAME / "routes" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    scorer = _Scorer(get(holdout, workdir / "holdout.npz"), tok2color, color_ids)
    width = model.transformer.wte.shape[1]
    e1 = projection(Subspace.axis(width, 0), 1.0)
    eye = np.eye(width, dtype=np.float32)
    zero = projection(Subspace(basis=eye, dual=eye, mean=np.zeros(width, np.float32)), 1.0, renorm=False)
    others = [p for p in range(UNIT * K) if p not in ANSWERS]
    pc = scorer.per_op(_score_by_slice(scorer, model, {}))
    span = pc - scorer.per_op(scorer.null)
    out: dict[str, Any] = {"label": label, "edits": {}}

    def edit(name: str, ops: dict[int, tuple]) -> None:
        out["edits"][name] = ((pc - scorer.per_op(_score_by_slice(scorer, model, ops))) / span).tolist()

    full = dict.fromkeys(range(N_SLICES), ((e1, None),))
    edit("full", full)
    for s in (1, 2):
        for name, positions in (("answers", ANSWERS), ("other example positions", others)):
            edit(f"full and {name} cleared, slice {s}", full | {s: ((e1, None), (zero, _mask(positions)))})
    return out


# =============================================================================================
# The DAG
# =============================================================================================


def design() -> dict[str, Any]:
    return {
        "experiment": NAME,
        "sources": ["fixed-labels", "fixed-confirm"],
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "epochs": EPOCHS,
        "kept": list(fl.KEEP),
        "answers": list(ANSWERS),
        "query_eq": QUERY_EQ,
        "n_unlabelled": N_UNLABELLED,
        "probe_slices_end": list(PROBE_SLICES_END),
        "probe_slices_along": list(PROBE_SLICES_ALONG),
    }


def publish(rows: list[dict], end: list[dict], along: list[list[dict]], routes: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {k: r[k] for k in ("kind", "rate", "model_seed", "draw_seed")} for r in rows}
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in end]}
    set_ref(END_REF, put(json.dumps(body).encode(), name=f"{NAME}-end.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for rs in along for r in rs]}
    set_ref(ALONG_REF, put(json.dumps(body).encode(), name=f"{NAME}-along.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in routes]}
    set_ref(ROUTES_REF, put(json.dumps(body).encode(), name=f"{NAME}-routes.json"))
    return {"n_end": len(end), "n_along": len(along), "n_routes": len(routes)}


def main(ctx: Ctx) -> dict:
    rows = runs()
    resolved = ctx.run(resolve, rows, role="prep")
    arts = resolved["artifacts"]
    n = len(rows)
    shared = ([resolved["holdout"]] * n, [resolved["corpus"]] * n, [resolved["labels"]] * n)
    end = ctx.map(
        end_one,
        [arts[r["checkpoint"]] for r in rows],
        *shared,
        [r["label"] for r in rows],
        [r["draw_seed"] for r in rows],
        [fl.RATES[r["rate"]] for r in rows],
        role="measure",
    )
    kept_rows = [r for r in rows if r["kept"] is not None]
    m = len(kept_rows)
    along = ctx.map(
        along_one,
        [arts[r["kept"]] for r in kept_rows],
        [arts[r["checkpoint"]] for r in kept_rows],
        [resolved["holdout"]] * m,
        [resolved["corpus"]] * m,
        [resolved["labels"]] * m,
        [r["label"] for r in kept_rows],
        [r["draw_seed"] for r in kept_rows],
        [fl.RATES[r["rate"]] for r in kept_rows],
        [tuple(fl.KEEP)] * m,
        role="measure",
    )
    routes = ctx.map(
        routes_one,
        [arts[r["checkpoint"]] for r in rows],
        [resolved["holdout"]] * n,
        [r["label"] for r in rows],
        role="measure",
    )
    return ctx.run(publish, rows, end, along, routes, role="prep")


COMPUTE = {
    "prep": dict(cpu=2, timeout=1800),
    # A few dozen forward passes over the 14,000 held-out contexts per checkpoint, and fourteen checkpoints per run
    # along training; the probes are small ridge fits on the host.
    "measure": dict(gpu="L40S", cpu=2, timeout=3600, watchdog=900, watchdog_grace=600),
}

experiment = Experiment(name=NAME, main=main, roles=COMPUTE)
