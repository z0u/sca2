"""
Does anchoring leave the rest of the geometry alone? A whole-geometry read over the stored checkpoints.

Scoring only, no training and no gates. Every anchored model so far has been scored for the axis it
was given (alignment, margin, containment) and for the task; whether the *rest* of its color geometry
is the geometry an un-anchored model builds has only been read through per-channel probe R²
(ex-2.1.12 H2), which came back unresolved. This pass reads it as a whole: for each stored run of three
experiments, the residual state of every grid color at a site, the matrix of distances between the
216 colors, and how well that matrix correlates with another run's (representational similarity
analysis, RSA). The yardstick is the spread between un-anchored controls at fresh seeds: an anchored
run whose geometry is as close to a control as controls are to each other has kept the rest of the
space. A second variant drops the anchor axis e₁ first, since that coordinate is where anchoring is
*meant* to move things.

    bin/mini run docs/m2/geometry-rsa/experiment.py --app local
    bin/mini status geometry-rsa
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from mini import Ctx, Experiment, get_data_dir

# --- Sources -------------------------------------------------------------------------------------


class Source(NamedTuple):
    """One condition of a stored experiment: every seed's checkpoint is read."""

    exp: str
    cond: str
    seeds: tuple[int, ...]
    lam: float
    title: str
    group: str = "main"
    """Conditions in one group share a control: `main` pairs against the experiment's control at the same
    length; `long` is ex-2.2.3's 100-epoch pair."""


CHECKPOINT_REF = "reports/m2/{exp}/checkpoints/{label}"
"""Where each source experiment published its checkpoints, one ref per run label."""

SOURCES: tuple[Source, ...] = (
    # D2.1's recipe against its control, the pair the D2.1 post's claim is about.
    Source("ex-2.1.10", "lam0", tuple(range(3)), 0.0, "un-anchored"),
    Source("ex-2.1.10", "either-t100", tuple(range(9)), 0.1, "the D2.1 recipe"),
    # The six-op grammar: the adopted point at twenty seeds, and the survey proposals as heavier anchors.
    Source("ex-2.2.3", "control-short", tuple(range(5)), 0.0, "un-anchored, 50 epochs"),
    Source("ex-2.2.3", "recipe-short", tuple(range(20)), 0.1, "the recipe, 50 epochs"),
    Source("ex-2.2.3", "t12", tuple(range(5)), 0.361, "survey proposal t12"),
    Source("ex-2.2.3", "t48", tuple(range(5)), 0.431, "survey proposal t48"),
    Source("ex-2.2.3", "t00", tuple(range(5)), 0.557, "survey proposal t00"),
    Source("ex-2.2.3", "control", tuple(range(5)), 0.0, "un-anchored, 100 epochs", group="long"),
    Source("ex-2.2.3", "recipe", tuple(range(20)), 0.1, "the recipe, 100 epochs", group="long"),
    # The handover grammar: the candidate and the two reference arms that each undo one of its changes.
    Source("ex-2.2.9", "control", tuple(range(5)), 0.0, "un-anchored"),
    Source("ex-2.2.9", "handover", tuple(range(20)), 0.1, "untied readout, whole-line labeller"),
    Source("ex-2.2.9", "handover-slot", tuple(range(20)), 0.1, "untied readout, slot labeller"),
    Source("ex-2.2.9", "handover-tied", tuple(range(9)), 0.1, "tied readout, whole-line labeller"),
)
"""Every run read, by experiment. Ex-2.2.3's addendum seeds are stored under `recipe-short-more` and
`recipe-more` for seeds 5 to 19 (`label_for`), and read here as one condition each."""

EXPERIMENTS: tuple[str, ...] = tuple(dict.fromkeys(s.exp for s in SOURCES))
N_RUNS = sum(len(s.seeds) for s in SOURCES)


def label_for(src: Source, seed: int) -> str:
    """The stored run label: `{cond}-s{seed}`, except ex-2.2.3's addendum seeds, which live under `-more`."""
    if src.exp == "ex-2.2.3" and src.cond in ("recipe-short", "recipe") and seed >= 5:
        return f"{src.cond}-more-s{seed}"
    return f"{src.cond}-s{seed}"


def control_of(src: Source) -> str:
    """The un-anchored condition an anchored condition is read against."""
    return next(s.cond for s in SOURCES if s.exp == src.exp and s.group == src.group and s.lam == 0.0)


# --- Sites and statistics ------------------------------------------------------------------------

POSITIONS: tuple[str, ...] = ("op1", "op2")
"""The two sites a color's state is read at. At op1 (position 0) the state is a function of the color token
alone, since attention is causal. At op2 (position 2) it depends on the whole prompt so far, and is read as
the mean over every op1 partner under the reference op (`mix`, or `+` on the D2.1 grammar)."""

POSITION_INDEX = {"op1": 0, "op2": 2}
N_SLICES = 5
"""The embedding and the four blocks of the d64-L4 model every source shares."""

VARIANTS: tuple[str, ...] = ("full", "axis")
"""`full` reads the geometry as it is. `axis` drops the anchor axis e₁ (`ANCHOR_AXIS`) from every run's states
first, anchored and control alike, so the coordinate anchoring was asked to move is out of the read and what
is left is the geometry of everything else."""

REDNESS_L2 = 1e-2
"""The ridge on the redness fit, as `sca.compute.geometry` fits its probes: a check on where each run reads
*red* from, not a variant of the geometry read."""

ANCHOR_AXIS = 0
"""e₁: where the anchored conditions were asked to put *red*."""

METRICS_REF = "reports/m2/geometry-rsa/metrics"
ARRAYS_REF = "reports/m2/geometry-rsa/arrays"
STATES_REF = "reports/m2/geometry-rsa/states/{exp}/{label}"
"""The run table (JSON), the pairwise matrices and per-run reads (npz), and each run's raw states."""


def npz_bytes(arrays: dict[str, Any]) -> bytes:
    """A compressed npz of *arrays*, as bytes, for `put`."""
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def rdm(x: np.ndarray) -> np.ndarray:
    """The representational dissimilarity matrix of the rows of *x*, as its upper triangle: the Euclidean
    distance between every pair of colors' states. On unit-norm states this is the chord distance, a
    monotone function of the cosine.
    """
    sq = (x**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
    i, j = np.triu_indices(len(x), k=1)
    return np.sqrt(d2[i, j])


def rsa(a: np.ndarray, b: np.ndarray) -> float:
    """Representational similarity: the Pearson correlation of two RDMs."""
    return float(np.corrcoef(a, b)[0, 1])


def redness_direction(x: np.ndarray, target: np.ndarray, l2: float = REDNESS_L2) -> np.ndarray:
    """The unit direction along which a ridge fit reads the grading target from the states."""
    xc = x - x.mean(0)
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x.shape[1]), xc.T @ (target - target.mean()))
    return w / np.linalg.norm(w)


def without_axis(x: np.ndarray, axis: int = ANCHOR_AXIS) -> np.ndarray:
    """The states with coordinate *axis* dropped."""
    return np.delete(x, axis, axis=1)


def procrustes_disparity(x: np.ndarray, y: np.ndarray) -> float:
    """SciPy's standardized Procrustes disparity: after centering, scaling to unit norm, and the best
    orthogonal map of *x* onto *y*, the sum of squared residuals (0 for two geometries that differ by a
    rotation, a reflection, or a scale).
    """
    from scipy.spatial import procrustes

    return float(procrustes(x, y)[2])


# --- Reading the stores --------------------------------------------------------------------------


def source_store():
    """The store the source checkpoints were published to, for reads alone.

    Every science run is on production, and so is this pass once it runs there, where this is the ambient
    store. Under a storage profile (`MINI_PROFILE=dev`, a rehearsal of this pass) the ambient store is the
    profile's pair, which holds none of the source runs, so the base pair is resolved from the project
    configuration instead, exactly as a production session resolves it. The pass's own results still go
    to the ambient store. No bucket is named here.
    """
    import os

    from mini.store import PROFILE_ENV, project_store

    saved = os.environ.pop(PROFILE_ENV, None)
    try:
        return project_store()
    finally:
        if saved is not None:
            os.environ[PROFILE_ENV] = saved


def resolve_inputs(exp: str, labels: list[str]) -> dict:
    """The source experiment's checkpoint handles by label, so each state task is keyed on the checkpoint's
    content rather than on its name.
    """
    names = [CHECKPOINT_REF.format(exp=exp, label=lb) for lb in labels]
    refs = source_store().get_refs(names)
    missing = [n for n in names if refs[n] is None]
    assert not missing, f"{exp} checkpoints not in the store: {missing}"
    return {lb: refs[n] for lb, n in zip(labels, names, strict=True)}


# --- Tasks ---------------------------------------------------------------------------------------


SEEDS_PER_TASK = 10
"""How many checkpoints one state task reads in turn. The local backend runs every task of a `map` as its
own process at once, so one task per checkpoint would put 65 JAX processes on one box; a chunk keeps the
fan-out to a handful per experiment."""


def chunks(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def states_chunk(ckpts: list, exp: str, cond: str, seeds: list[int], labels: list[str]) -> list[dict]:
    """The residual states of every grid color, from each checkpoint of one chunk of a condition's seeds.

    Every ordered pair of grid colors runs as a three-token prompt `a op b`: positions 0 and 2 depend on
    nothing after them, so the answer and the syntax after op2 are not needed. Returns one record per run,
    with `states` as an npz artifact holding one `(slice, color, width)` array per site, in palette order.
    """
    return [states_one(ckpt, exp, cond, seed, label) for ckpt, seed, label in zip(ckpts, seeds, labels, strict=True)]


def states_one(ckpt, exp: str, cond: str, seed: int, label: str) -> dict:
    """One checkpoint's states, as `states_chunk` describes them."""
    import equinox as eqx
    import jax.numpy as jnp

    from mini.store import put
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE

    workdir = get_data_dir() / "ckpt" / exp / label
    source_store().get(ckpt, workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = WordTokenizer(config.tokenizer)
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    op_word = "mix" if "mix" in tokenizer.stoi else "+"
    n = len(color_ids)
    prompts = np.stack(
        [np.repeat(color_ids, n), np.full(n * n, tokenizer.stoi[op_word]), np.tile(color_ids, n)], axis=1
    ).astype(np.int32)

    stream = eqx.filter_jit(model.residual_stream)
    acts = np.concatenate(
        [np.asarray(stream(jnp.asarray(prompts[i : i + 4096]))) for i in range(0, len(prompts), 4096)], axis=1
    )  # (slices, pairs, 3, width)
    assert acts.shape[0] == N_SLICES, f"{label}: {acts.shape[0]} slices"
    by_pair = acts.reshape(N_SLICES, n, n, 3, -1)  # (slice, a, b, position, width)
    op1 = by_pair[:, :, 0, 0]  # identical over b: the state above the first token is a function of it alone
    assert np.abs(by_pair[:, :, :, 0] - op1[:, :, None]).max() < 1e-5, f"{label}: op1 state varies with op2"
    op2 = by_pair[:, :, :, 2].mean(1)  # (slice, b, width): the color as op2, marginal over op1
    states = {"op1": op1.astype(np.float32), "op2": op2.astype(np.float32)}
    return {
        "exp": exp,
        "cond": cond,
        "seed": seed,
        "label": label,
        "op_word": op_word,
        "states": put(npz_bytes(states), name=f"geometry-rsa-{exp}-{label}-states.npz"),
    }


def compare(exp: str, runs: list[dict]) -> dict:
    """Every pairwise read within one experiment, at every site and in both variants.

    Per (variant, site): the run × run RSA matrix and Procrustes disparity matrix, and each run's RSA to the
    RGB cube itself. Per site, two checks on the anchor axis: the share of the states' variance that e₁
    carries (what `axis` removes), and the cosine between e₁ and the ridge-fitted direction *red* is read
    from, which says whether the fit finds red where the anchor put it.
    """
    from mini.store import get_many, put
    from sca.colorcube import sim_to_red
    from sca.data.ops import TOP, colors

    grid = np.asarray(colors(), dtype=float) / TOP
    target = sim_to_red(grid, power=1.5).astype(np.float64)
    cube = rdm(grid)

    workdir = get_data_dir() / "states" / exp
    paths = get_many([(r["states"], workdir / f"{r['label']}.npz") for r in runs])
    states: dict[str, np.ndarray] = {}
    for r, p in zip(runs, paths, strict=True):
        with np.load(p) as z:
            for pos in POSITIONS:
                states[f"{r['label']}/{pos}"] = z[pos].astype(np.float64)

    n = len(runs)
    out: dict[str, np.ndarray] = {}
    for pos in POSITIONS:
        for si in range(N_SLICES):
            x = [states[f"{r['label']}/{pos}"][si] for r in runs]
            dirs = [redness_direction(xi, target) for xi in x]
            out[f"cos_e1/{pos}/{si}"] = np.array([abs(d[ANCHOR_AXIS]) for d in dirs])
            out[f"e1_share/{pos}/{si}"] = np.array([float(xi.var(0)[ANCHOR_AXIS] / xi.var(0).sum()) for xi in x])
            for variant in VARIANTS:
                xv = x if variant == "full" else [without_axis(xi) for xi in x]
                rdms = [rdm(xi) for xi in xv]
                m_rsa = np.eye(n)
                m_proc = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        m_rsa[i, j] = m_rsa[j, i] = rsa(rdms[i], rdms[j])
                        m_proc[i, j] = m_proc[j, i] = procrustes_disparity(xv[i], xv[j])
                out[f"{variant}/rsa/{pos}/{si}"] = m_rsa
                out[f"{variant}/procrustes/{pos}/{si}"] = m_proc
                out[f"{variant}/cube_rsa/{pos}/{si}"] = np.array([rsa(d, cube) for d in rdms])

    return {
        "exp": exp,
        "runs": [{k: r[k] for k in ("exp", "cond", "seed", "label", "op_word")} for r in runs],
        "arrays": put(npz_bytes(out), name=f"geometry-rsa-{exp}-arrays.npz"),
    }


def publish_results(compared: list[dict], all_runs: list[dict]) -> dict:
    """The run table under `METRICS_REF`, every experiment's matrices under `ARRAYS_REF` (keys prefixed by
    experiment), and each run's raw states under its own ref.
    """
    import json

    from mini.store import get, put, set_ref

    arrays: dict[str, np.ndarray] = {}
    for c in compared:
        path = get(c["arrays"], get_data_dir() / "publish" / f"{c['exp']}.npz")
        with np.load(path) as z:
            arrays |= {f"{c['exp']}/{k}": z[k] for k in z.files}
    set_ref(ARRAYS_REF, put(npz_bytes(arrays), name="geometry-rsa-arrays.npz"))
    metrics: dict[str, Any] = {
        "experiments": {c["exp"]: c["runs"] for c in compared},
        "sources": [s._asdict() for s in SOURCES],
        "positions": POSITIONS,
        "n_slices": N_SLICES,
        "variants": VARIANTS,
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="geometry-rsa-metrics.json"))
    for r in all_runs:
        set_ref(STATES_REF.format(exp=r["exp"], label=r["label"]), r["states"])
    return {"n_runs": len(all_runs), "n_experiments": len(compared), "n_arrays": len(arrays)}


def main(ctx: Ctx) -> dict:
    compared, all_runs = [], []
    for exp in EXPERIMENTS:
        srcs = [s for s in SOURCES if s.exp == exp]
        cells = [(s, seed, label_for(s, seed)) for s in srcs for seed in s.seeds]
        labels = [lb for _, _, lb in cells]
        inputs = ctx.run(resolve_inputs, exp, labels, role="cpu")
        tasks = [(s, chunk) for s in srcs for chunk in chunks(list(s.seeds), SEEDS_PER_TASK)]
        chunked = ctx.map(
            states_chunk,
            [[inputs[label_for(s, seed)] for seed in chunk] for s, chunk in tasks],
            [exp] * len(tasks),
            [s.cond for s, _ in tasks],
            [chunk for _, chunk in tasks],
            [[label_for(s, seed) for seed in chunk] for s, chunk in tasks],
            role="states",
        )
        runs = [r for part in chunked for r in part]
        compared.append(ctx.run(compare, exp, runs, role="cpu"))
        all_runs.extend(runs)
    return ctx.run(publish_results, compared, all_runs, role="cpu")


# Each state task is up to ten forward passes over 46,656 three-token prompts; each compare is 216 × 216
# distance matrices and a few thousand 216 × 64 Procrustes fits. CPU work, minutes per task. The thread
# caps keep a handful of concurrent JAX processes from each claiming every core of one box.
THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

experiment = Experiment(
    name="geometry-rsa",
    main=main,
    roles={
        "states": dict(cpu=2, timeout=3600, watchdog_grace=1800, env=THREAD_ENV),
        "cpu": dict(cpu=2, timeout=1800, watchdog_grace=900, env=THREAD_ENV),
    },
)
