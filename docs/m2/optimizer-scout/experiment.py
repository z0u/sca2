"""Optimizers with no privileged basis, on the control: pooled Adam, Muon and SGD at two seeds and three rates.

The coordinate axes of the residual stream are a privileged basis because Adam's per-coordinate scaling is written in
them (found on stored checkpoints after the spill-timing experiment, PR #285). This scout asks what the stream looks like, and how well the
task is learned, under optimizers whose update is the same in every orthonormal basis of the residual coordinates:
`adam-pooled` (Adam with its second moment pooled over the residual axis of each matrix that reads or writes the
stream), `muon` (orthogonalized updates on the matrices, Adam on the gains), and `sgd` with momentum. Each arm first
runs the learning-rate finder of ex-2.2.17 at one seed, then trains the un-anchored control for the full 400-epoch
schedule at the finder's rate, three times it and a third of it, at model seeds 704 and 705, where ex-2.2.23's Adam
controls are stored as the reference. Every run gets embedding-lean's measurements, the basis measurement of
rotated-basis (the energy spectrum and kurtosis along the coordinate axes and along a random basis, and the removal
cost of each axis and of random directions), and a principal-direction measurement (the energy spectrum along the
stream's own principal directions, which sees a spike wherever it sits), and a measurement of how the color embeddings
are spread; the stored Adam controls get these too. No anchored runs: an
optimizer that keeps the stream isotropic and learns the task as well is a candidate for an anchored arm later.

    bin/mini run docs/m2/optimizer-scout/experiment.py --app modal --max-containers 16 --budget 4h
    bin/mini status optimizer-scout
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


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


rb = _load_sibling("rotated-basis", "optimizer_scout_rb")
lean = rb.lean
ex2216 = rb.ex2216
ex2223 = rb.ex2223
ex2222 = rb.ex2222
ex2221 = rb.ex2221
ex2217 = ex2222.ex2217

OP_NAMES = rb.OP_NAMES
K = rb.K
EPOCHS = rb.EPOCHS
EVERY = rb.EVERY
TRAJ_STRIDE_EPOCHS = rb.TRAJ_STRIDE_EPOCHS

SEEDS: tuple[int, ...] = (704, 705)
"""Two of the seeds where ex-2.2.23's 400-epoch Adam controls are stored."""
FINDER_SEED = SEEDS[0]
RATE_FACTORS: tuple[float, ...] = (3.0, 1.0, 1 / 3)
"""Each arm trains at the finder's rate, at three times it and at a third of it: the finder's 300-step suggestion
has sat above the rate the ladders chose for Adam (ex-2.2.17, ex-2.2.19), and for SGD it may sit below the rate a
long run wants (the memo's short SGD runs used 0.3 to 1.0), so the bracket reaches both ways."""
ADAM_LR = ex2221.PEAK_LR
"""The recipe's Adam rate, which `muon` keeps for the gains it updates with Adam."""


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    shown_as: str


ARMS: tuple[Arm, ...] = (
    Arm("adam-pooled", "adam-pooled", "pooled Adam"),
    Arm("muon", "muon", "Muon"),
    Arm("sgd", "sgd", "SGD with momentum"),
)

BUDGET_USD = 8
"""Eighteen 400-epoch control runs at about \\$0.3 each, three finder runs, and the basis and principal-direction
measurements of twenty checkpoints."""


def label_of(arm: str, factor: float, model_seed: int) -> str:
    return f"{arm}-lr{factor:.2g}-s{model_seed}"


def with_kind(config, kind: str, learning_rate: float):
    """*config* with the optimizer *kind* at *learning_rate*; `muon` keeps the recipe's rate for the gains."""
    config = config.model_copy(deep=True)
    config.optimizer.kind = kind
    config.optimizer.learning_rate = learning_rate
    config.optimizer.aux_learning_rate = ADAM_LR if kind == "muon" else None
    return config


def control_rows(meta, epochs: int, seeds: tuple[int, ...]) -> list[dict]:
    """Ex-2.2.21's control rows (ex-2.2.23's controls) at *seeds*."""
    n_seeds = max(seeds) - ex2221.SEED_OFFSET + 1
    arm = (replace(ex2221.arm(ex2221.CONTROL), seeds=n_seeds),)
    rows = [r for r in ex2221.cells(arm, {ex2221.MAIN_KEY: {"meta": meta}}, None, epochs) if r["model_seed"] in seeds]
    assert len(rows) == len(seeds), (len(rows), len(seeds))
    for r in rows:
        r["anti_slices"] = None
    return rows


def rows_of(meta, found: dict[str, float], epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: the control row at the seed, with the arm's optimizer at the finder's rate times the
    factor.
    """
    rows = []
    for a in ARMS:
        for f in RATE_FACTORS:
            for r in control_rows(meta, epochs, seeds):
                lr = found[a.name] * f
                rows.append(
                    r
                    | {
                        "config": with_kind(r["config"], a.kind, lr),
                        "arm": asdict(a),
                        "rate_factor": f,
                        "learning_rate": lr,
                        "label": label_of(a.name, f, r["model_seed"]),
                    }
                )
    return rows


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/optimizer-scout"
FINDER_REF = PREFIX + "/finder"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
BASIS_REF = PREFIX + "/basis"
PRINCIPAL_REF = PREFIX + "/principal"


def principal_one(checkpoint, holdout, label: str, n_ctx: int = 1000) -> dict:
    """The energy spectrum of the stream along its own principal directions, per slice, on *n_ctx* held-out
    contexts: the eigenvalues of the second-moment matrix of the states (uncentered, so the shares compare with the
    basis measurement's, and centered), as shares of the total. A spike shows here whichever direction it sits on,
    where the basis measurement sees it only when it sits on a coordinate axis or a random basis vector. Also the
    cosine of each syntax embedding with the top principal direction at the last slice, and the pairwise cosines
    among the syntax embeddings.
    """
    from mini.store import get

    workdir = get_data_dir() / "principal" / label
    model, tok, _, _ = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(ho.tokens), min(n_ctx, len(ho.tokens)), replace=False))
    stream = np.asarray(model.residual_stream(ho.tokens[pick]))  # (L1, N, T, C)
    shares, shares_centered, tops = [], [], []
    for s in range(stream.shape[0]):
        h = stream[s].reshape(-1, stream.shape[-1]).astype(np.float64)
        for centered, out in ((False, shares), (True, shares_centered)):
            x = h - h.mean(0) if centered else h
            w, v = np.linalg.eigh(x.T @ x / len(x))
            w, v = w[::-1], v[:, ::-1]
            out.append((w / w.sum()).tolist())
            if not centered:
                tops.append(v[:, 0])
    emb = np.asarray(model.transformer.wte, dtype=np.float64)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    syntax = ["?", ",", "\n", "="]
    ids = [tok.stoi[t] for t in syntax]
    gram = emb[ids] @ emb[ids].T
    return {
        "label": label,
        "n_ctx": int(len(pick)),
        "shares": shares,  # (L1, C): the share of the energy on each principal direction, largest first
        "shares_centered": shares_centered,
        "syntax": syntax,
        "syntax_on_top": [float(abs(emb[i] @ tops[-1])) for i in ids],  # at the last slice
        "syntax_gram": gram.tolist(),
    }


PALETTE_REF = PREFIX + "/palette"


def palette_one(checkpoint, label: str) -> dict:
    """How the color token embeddings are spread: the participation ratio of their centered (unit-normalized)
    embeddings, the share of their variance on the top three principal directions, the share a linear map from the
    RGB channels explains, the rank correlation between embedding distance and RGB distance, their mean pairwise
    cosine, and their mean absolute cosine with the separator direction (the mean of the `,` and ⏎ embeddings).
    """
    from scipy.stats import spearmanr

    from sca.data.ops import PALETTE

    workdir = get_data_dir() / "palette" / label
    model, tok, color_ids, _ = ex2216._load_run(checkpoint, workdir)
    table = np.asarray(model.transformer.wte, dtype=np.float64)
    emb = table[color_ids]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    centered = emb - emb.mean(0)
    var = np.linalg.svd(centered, compute_uv=False) ** 2
    rgb = np.array([[int(name[1 + i], 16) for i in range(3)] for name in PALETTE], dtype=np.float64)
    design_matrix = np.c_[rgb, np.ones(len(rgb))]
    beta, *_ = np.linalg.lstsq(design_matrix, centered, rcond=None)
    explained = 1 - float(((centered - design_matrix @ beta) ** 2).sum() / (centered**2).sum())
    cos = emb @ emb.T
    upper = np.triu_indices(len(emb), 1)
    d_rgb = np.linalg.norm(rgb[:, None] - rgb[None], axis=-1)[upper]
    sep = table[[tok.stoi[t] for t in (",", "\n")]].mean(0)
    sep = sep / np.linalg.norm(sep)
    return {
        "label": label,
        "participation": float(var.sum() ** 2 / (var**2).sum()),
        "top3": float(var[:3].sum() / var.sum()),
        "rgb_r2": explained,
        "rgb_rho": float(spearmanr(1 - cos[upper], d_rgb).statistic),
        "mean_cos": float(cos[upper].mean()),
        "cos_separators": float(np.abs(emb @ sep).mean()),
    }


def resolve_controls(seeds: tuple[int, ...]) -> dict:
    """The checkpoints of ex-2.2.23's 400-epoch Adam controls at *seeds*, by ref."""
    from mini.store import get_ref

    refs = {f"adam-s{s}": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)) for s in seeds}
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def design() -> dict[str, Any]:
    return {
        "experiment": "optimizer-scout",
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) for a in ARMS],
        "seeds": list(SEEDS),
        "finder_seed": FINDER_SEED,
        "rate_factors": list(RATE_FACTORS),
        "adam_lr": ADAM_LR,
        "finder": {"range": list(ex2217.FINDER_RANGE), "zooms": ex2217.FINDER_ZOOMS, "steps": ex2217.FINDER_STEPS},
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
        "n_random": rb.N_RANDOM,
    }


def publish(
    rows: list[dict],
    finder: list[dict],
    trained: list[dict],
    scored: list[dict],
    based: list[dict],
    principal: list[dict],
    palette: list[dict],
) -> dict:
    import json

    from mini.store import put, set_ref

    meta = {
        r["label"]: {
            "arm": r["arm"]["name"],
            "model_seed": r["model_seed"],
            "learning_rate": r["learning_rate"],
            "rate_factor": r["rate_factor"],
        }
        for r in rows
    }
    meta |= {
        f"adam-s{s}": {"arm": "adam", "model_seed": s, "learning_rate": ADAM_LR, "rate_factor": 1.0} for s in SEEDS
    }
    set_ref(
        FINDER_REF, put(json.dumps({"design": design(), "runs": finder}).encode(), name="optimizer-scout-finder.json")
    )
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="optimizer-scout-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="optimizer-scout-results.json"))
    basis = {"design": design(), "runs": [meta[b["label"]] | b for b in based]}
    set_ref(BASIS_REF, put(json.dumps(basis).encode(), name="optimizer-scout-basis.json"))
    pca = {"design": design(), "runs": [meta[p["label"]] | p for p in principal]}
    set_ref(PRINCIPAL_REF, put(json.dumps(pca).encode(), name="optimizer-scout-principal.json"))
    colors = {"design": design(), "runs": [meta[c["label"]] | c for c in palette]}
    set_ref(PALETTE_REF, put(json.dumps(colors).encode(), name="optimizer-scout-palette.json"))
    return {"n_runs": len(scored), "n_based": len(based), "n_principal": len(principal), "n_palette": len(palette)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition and the Adam controls, find each arm's rate, train every run, measure it,
    take the basis measurement of every run and of the Adam controls, and publish.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    controls = ctx.run(resolve_controls, seeds, role="prep")
    (base,) = control_rows(resolved["meta"], epochs, (FINDER_SEED,))
    finder = ctx.map(
        ex2217.find_lr,
        [with_kind(base["config"], a.kind, ADAM_LR) for a in ARMS],
        [resolved["corpus"]] * len(ARMS),
        [f"finder-{a.name}" for a in ARMS],
        role="finder",
    )
    found = {a.name: f["best_lr"] for a, f in zip(ARMS, finder, strict=True)}
    rows = rows_of(resolved["meta"], found, epochs, seeds)
    n = len(rows)
    epoch_length = [ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["pull"] for r in rows],
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [r["anti_slices"] for r in rows],
        [None] * n,
        [True] * n,
        role="train",
    )
    scored = ctx.map(
        lean.measure_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [r["label"] for r in rows],
        [False] * n,
        role="measure",
    )
    checkpoints = {t["label"]: t["checkpoint"] for t in trained} | (controls if publishes else {})
    labels = list(checkpoints)
    seeds_of = {r["label"]: r["model_seed"] for r in rows} | {f"adam-s{s}": s for s in seeds}
    based = ctx.map(
        rb.basis_one,
        [checkpoints[label] for label in labels],
        [resolved["holdout"]] * len(labels),
        labels,
        [seeds_of[label] for label in labels],
        role="measure",
    )
    principal = ctx.map(
        principal_one,
        [checkpoints[label] for label in labels],
        [resolved["holdout"]] * len(labels),
        labels,
        role="measure",
    )
    palette = ctx.map(palette_one, [checkpoints[label] for label in labels], labels, role="measure")
    if not publishes:
        return {"found": found, "labels": [s["label"] for s in scored], "based": [b["label"] for b in based]}
    return ctx.run(publish, rows, finder, trained, scored, based, principal, palette, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of every arm at the finder's rate for eight epochs, through the whole DAG."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], publishes=False)


COMPUTE = rb.COMPUTE | {"finder": dict(gpu="L4", timeout=900)}

experiment = Experiment(name="optimizer-scout", main=main, roles=COMPUTE)
