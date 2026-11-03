"""The post-hoc comparison: the anchored edit against directions fitted after training, on stored checkpoints.

The preregistration is `report.py` beside this file; this module carries its design as data. No new training. For
each run (a control of the anneal confirmation, or an anchored run of the fixed confirmation, at seeds 712 to 719):

1. `fit_one` accumulates the moments of the states over the fit set at every slice and site, makes every
   closed-form fit (difference in means, ridge probe, LEACE, INLP) from them, and refits slice by slice under the
   edits at the slices before (sequential refit).
2. `search_one` fits the optimized direction by gradient descent, one per target, weight, and slice set.
3. `axes_one` scores every operator form along fifteen fixed coordinate axes on the tuning set: the baseline the
   tuning frontier is netted of.
4. `grid_one` scores every configuration of every method on the tuning set or on the test set.
5. `select` takes the frontier on the tuning set and reads the two points of the selection rule off it.
6. `finish_one` scores what the hypotheses and the E sections need at the selected configurations: the permuted
   fits, the median along the other 63 axes, the label ladder, the per-context arrays for grading and landing, the
   write and its bound, a fresh probe, and the cosine with e₁.

`smoke` runs every stage on two stored runs outside the comparison (ex-2.2.23's control and split-anti-seeds'
`falloff` run at seed 704), with a thinned grid, and records the time of each kind of pass, which gives the cost
of the full run.

    bin/mini run docs/m2/post-hoc-compare/experiment.py --app modal --max-containers 12 --budget 1h
    bin/mini status post-hoc-compare
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from dataclasses import asdict, dataclass
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


fc = _load_sibling("fixed-confirm", "post_hoc_fc")
ac = fc.ac
ex2223 = ac.ex2223
sas = _load_sibling("split-anti-seeds", "post_hoc_sas")
rb = _load_sibling("reference-baseline", "post_hoc_rb")
ex2222, ex2221 = ac.ex2222, ac.ex2221
ex2216, ex2218 = fc.lean.ex2216, fc.lean.ex2218

NAME = "post-hoc-compare"
OP_NAMES = ac.OP_NAMES
ANCHORED_OP = ac.ANCHORED_OP
D = OP_NAMES.index(ANCHORED_OP)
K = ac.K
RHO = ex2222.RHO
CUBE_RATE = ex2222.CUBE_RATE
UNIT = 6
CONTEXT_LEN = ex2216.context_length(K)
ANSWERS = tuple(UNIT * i + 4 for i in range(K))
QUERY_EQ = ex2216.query_role(K, ex2216.QUERY_EQ)
N_SLICES = ex2222.N_SLICES
SEEDS: tuple[int, ...] = ac.SEEDS
SMOKE_SEED = 704

# =============================================================================================
# The design
# =============================================================================================

FIT_PER_OP = 40_000
TUNE_PER_OP = 2_000
FIT_SEED, TUNE_SEED = 9_001, 9_002
"""Seeds of the fresh draws, far from the corpus seed and its held-out and probe branches."""
SPILL_CRITERION = 0.02
N_TUNE_AXES = 15
AXES_SEED = 2026
N_PERMUTED = 15
N_PERMUTED_SEARCH = 5
"""Permuted fits of the optimized direction: fewer, since each is a gradient search."""
LADDER = (1.0, 0.5, 0.1, 0.04, 0.02)
"""The label rates of E3: every `difference` context of the fit set, then a fixed half, tenth, twenty-fifth, and
fiftieth of them (nested draws)."""
LADDER_SEED = 11
DOSES = (0.25, 0.5, 0.75, 1.0)
REPLACEMENTS = ("zero", "mean")
SITES = ("pooled", "answers", "query =", "query operands")
# The query operands joined the sites after the smoke test (the open decision of the preregistration): on the control
# at 704 a probe for the op scored best there, at slices 3 and 4.
SLICE_SETS: tuple[tuple[int, ...], ...] = (*((s,) for s in range(N_SLICES)), (1, 2, 3), tuple(range(N_SLICES)))
INLP_RANKS = (2, 4, 8)
RIDGE_L2 = 1e-2
LEACE_L2 = 1e-6
SEARCH_WEIGHTS = (0.3, 1.0, 3.0, 10.0)
"""The weight on keeping the answers of the other ops where they were, in the loss of the optimized direction."""
SEARCH_STEPS = 300
SEARCH_BATCH = 512
SEARCH_LR = 1e-2
METHODS = ("dim", "ridge", "leace", "inlp", "optimized")
STANDARD = ("dim", "ridge", "leace", "inlp")
RANK1 = ("dim", "ridge", "leace")
TARGETS = {"dim": ("binary",), "ridge": ("binary", "graded"), "leace": ("binary", "graded"),
           "inlp": ("binary",), "optimized": ("binary", "graded")}  # fmt: skip
H4_SITES = ("pooled", "answers")
WRITE_QUANTILE = 99
N_WRITE_CTX = 2_000
"""Contexts of the other ops the write and its bound are measured on."""
PROBE_L2 = 1e-2

# --- Target columns of the moments ----------------------------------------------------------------------------
# One column per target the closed-form fits read: the binary label (the generating op is `difference`), the graded
# one (the posterior on `difference`), fifteen permuted copies of each, and the thinned copies of the label ladder.

COL_BINARY, COL_GRADED = 0, 1
COL_PERM_BINARY = 2
COL_PERM_GRADED = COL_PERM_BINARY + N_PERMUTED
COL_LADDER_BINARY = COL_PERM_GRADED + N_PERMUTED
COL_LADDER_GRADED = COL_LADDER_BINARY + len(LADDER) - 1
N_COLS = COL_LADDER_GRADED + len(LADDER) - 1


def target_col(target: str, perm: int | None = None, rung: int = 0) -> int:
    """The column of *target*, permuted copy *perm*, or ladder rung *rung* (0 is every label)."""
    if perm is not None:
        return (COL_PERM_BINARY if target == "binary" else COL_PERM_GRADED) + perm
    if rung:
        return (COL_LADDER_BINARY if target == "binary" else COL_LADDER_GRADED) + rung - 1
    return COL_BINARY if target == "binary" else COL_GRADED


def site_masks() -> np.ndarray:
    """The positions of each site, a row per site in `SITES` order."""
    masks = np.zeros((len(SITES), CONTEXT_LEN), np.float32)
    masks[0] = 1
    masks[1, list(ANSWERS)] = 1
    masks[2, QUERY_EQ] = 1
    masks[3, [UNIT * K, UNIT * K + 2]] = 1
    return masks


# --- Configurations -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One point of the grid: how a direction is fitted, where it acts, and how much it removes."""

    method: str
    target: str
    site: str
    slices: tuple[int, ...]
    refit: bool
    positions: str  # "every" or "sites"
    dose: float
    replacement: str
    rank: int = 1
    weight: float = 0.0  # the optimized direction only

    @property
    def form(self) -> tuple:
        """The operator form a coordinate axis is scored under for this configuration's baseline."""
        return (self.slices, self.position_site, self.dose, self.replacement)

    @property
    def position_site(self) -> str:
        """The site whose positions the edit touches."""
        return "pooled" if self.positions == "every" else self.site

    @property
    def fit_key(self) -> tuple:
        return (self.method, self.target, self.site, self.rank, self.weight, self.slices if self.refit or self.method == "optimized" else None)  # fmt: skip


def config_of(d: dict) -> Config:
    """A configuration from its stored form (JSON turns the slice tuple into a list)."""
    return Config(**(d | {"slices": tuple(d["slices"])}))  # ty: ignore[invalid-argument-type]


def configs(method: str) -> list[Config]:
    """Every configuration of *method* in the grid."""
    out = []
    ranks = INLP_RANKS if method == "inlp" else (1,)
    weights = SEARCH_WEIGHTS if method == "optimized" else (0.0,)
    sites = ("pooled",) if method == "optimized" else SITES
    for target, site, rank, weight in itertools.product(TARGETS[method], sites, ranks, weights):
        positions = ("every",) if site == "pooled" else ("every", "sites")
        for slices in SLICE_SETS:
            refits = (False,) if len(slices) == 1 or method == "optimized" else (False, True)
            for refit, pos, dose, rep in itertools.product(refits, positions, DOSES, REPLACEMENTS):
                out.append(Config(method, target, site, slices, refit, pos, dose, rep, rank, weight))
    return out


def forms() -> list[tuple]:
    """Every operator form of the grid: what the coordinate-axis baseline is scored under."""
    return sorted({c.form for m in METHODS for c in configs(m)}, key=str)


def thin(cs: list[Config], extra: int = 0, seed: int = 0) -> list[Config]:
    """A subset of *cs* that visits every level of every factor at least once, greedily, plus *extra* at random:
    the smoke test's grid.
    """
    levels = lambda c: {(k, str(v)) for k, v in asdict(c).items()}  # noqa: E731
    need = set().union(*(levels(c) for c in cs))
    out: list[Config] = []
    while need:
        best = max(cs, key=lambda c: len(levels(c) & need))
        out.append(best)
        need -= levels(best)
    rest = [c for c in cs if c not in out]
    rng = np.random.default_rng(seed)
    out += [rest[i] for i in rng.choice(len(rest), size=min(extra, len(rest)), replace=False)]
    return out


def tune_axes(width: int = 64) -> tuple[int, ...]:
    """The fifteen coordinate axes other than e₁ the tuning baseline is taken along, fixed before any run."""
    return tuple(int(a) for a in np.sort(np.random.default_rng(AXES_SEED).choice(np.arange(1, width), N_TUNE_AXES, replace=False)))  # fmt: skip


# =============================================================================================
# The runs
# =============================================================================================

PREFIX = "reports/m2/post-hoc-compare"
RESULTS_REF = PREFIX + "/results"
ARRAYS_REF = PREFIX + "/arrays/{label}"
SMOKE_REF = PREFIX + "/smoke"
# The selections over part of the grid (`RESTRICTED`), added after the full run: H4 restricts its fitters to two
# sites, and `select` had taken the frontier over all four.
RESTRICTED_REF = PREFIX + "/restricted"


def runs(smoke: bool = False) -> list[dict[str, Any]]:
    """The controls and the anchored runs, with the ref of each checkpoint. For the smoke test, the pair at 704."""
    if smoke:
        s = SMOKE_SEED
        return [
            {"label": f"control-s{s}", "kind": "control", "seed": s,
             "ref": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", ac.EPOCHS, s))},
            {"label": f"falloff-s{s}", "kind": "anchored", "seed": s,
             "ref": sas.CHECKPOINT_REF.format(label=sas.label_of("falloff", s))},
        ]  # fmt: skip
    controls = [{"label": f"control-s{s}", "kind": "control", "seed": s,
                 "ref": ac.CHECKPOINT_REF.format(label=ac.label_of(s, None, "control"))} for s in SEEDS]  # fmt: skip
    anchored = [{"label": fc.label_of(s, None), "kind": "anchored", "seed": s,
                 "ref": fc.CHECKPOINT_REF.format(label=fc.label_of(s, None))} for s in SEEDS]  # fmt: skip
    return controls + anchored


# =============================================================================================
# The sets
# =============================================================================================


def _draw(table, per_op: int, seed: int, exclude: set[bytes], tokenizer_stoi: dict) -> dict:
    """*per_op* fresh completion contexts of every op from the training generator (with its replacement op noise),
    none of them in *exclude*, with the op, the posterior over ops, and the query pair of each.
    """
    from sca.data import ops as grammar
    from sca.data.incontext import encode_corpus
    from sca.data.incontext import op_ids as op_ids_of

    P = ex2216._get_posterior()
    post_table = P.build_table(table)
    color_index = {c: i for i, c in enumerate(grammar.colors())}
    op_index = {o.name: i for i, o in enumerate(table)}
    # A little over the count, so that the contexts dropped as duplicates leave enough.
    contexts = ex2216._sample_by_op(table, K, RHO, CUBE_RATE, 0.0, int(per_op * 1.02) + 10, seed)
    tokens = encode_corpus(contexts, tokenizer_stoi).reshape(len(contexts), CONTEXT_LEN).astype(np.int32)
    op_ids = op_ids_of(contexts, table)
    keep, seen = [], set(exclude)
    count = dict.fromkeys(range(len(table)), 0)
    for i, row in enumerate(tokens):
        key = row.tobytes()
        if key in seen or count[int(op_ids[i])] >= per_op:
            continue
        seen.add(key)
        count[int(op_ids[i])] += 1
        keep.append(i)
    assert all(v == per_op for v in count.values()), count
    keep = np.array(keep)
    post = np.concatenate([
        P.posterior(post_table, ex2216._post_contexts(P, [contexts[i] for i in keep[j : j + 20_000]], color_index, op_index), RHO, CUBE_RATE)
        for j in range(0, len(keep), 20_000)
    ])  # fmt: skip
    return {"tokens": tokens[keep], "op_ids": op_ids[keep].astype(np.int8), "posterior": post.astype(np.float32)}


def prepare_sets(holdout) -> dict:
    """The fit set and the tuning set, drawn fresh from the training generator, with no context in the test set
    (ex-2.2.21's held-out contexts) or in both. Each comes with its op, its posterior over ops, and, for the fit
    set, the target columns of the moments (`target_col`): the binary and graded targets, fifteen permutations of
    each across contexts, and the nested thinnings of the label ladder.
    """
    from mini.store import get, put
    from sca.data.named_colors import WordTokenizer

    workdir = get_data_dir() / NAME / "sets"
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    table = ex2218.table_of(OP_NAMES)
    from sca.config import TokenizerConfig
    from sca.data.incontext import vocabulary

    stoi = WordTokenizer(TokenizerConfig(vocabulary=sorted(vocabulary()))).stoi
    test = {row.tobytes() for row in ho.tokens.astype(np.int32)}
    fit = _draw(table, FIT_PER_OP, FIT_SEED, test, stoi)
    tune = _draw(table, TUNE_PER_OP, TUNE_SEED, test | {r.tobytes() for r in fit["tokens"]}, stoi)

    binary = (fit["op_ids"] == D).astype(np.float32)
    graded = fit["posterior"][:, D].astype(np.float32)
    z = np.zeros((len(binary), N_COLS), np.float32)
    z[:, COL_BINARY], z[:, COL_GRADED] = binary, graded
    rng = np.random.default_rng(FIT_SEED)
    for p in range(N_PERMUTED):
        perm = rng.permutation(len(binary))
        z[:, COL_PERM_BINARY + p], z[:, COL_PERM_GRADED + p] = binary[perm], graded[perm]
    # The ladder thins the labels of `difference` contexts only: an unlabelled one counts among the others, with a
    # zero target, as an unlabelled context does for the anchor.
    u = np.random.default_rng(LADDER_SEED).random(len(binary))
    for r, rate in enumerate(LADDER[1:]):
        drop = (binary > 0) & (u >= rate)
        z[:, COL_LADDER_BINARY + r] = np.where(drop, 0.0, binary)
        z[:, COL_LADDER_GRADED + r] = np.where(drop, 0.0, graded)

    def pack(s: dict, **extra) -> bytes:
        return ex2216._npz(tokens=s["tokens"], op_ids=s["op_ids"], posterior=s["posterior"], **extra)

    stats = {
        "n_fit": int(len(fit["tokens"])),
        "n_tune": int(len(tune["tokens"])),
        "n_fit_difference": int(binary.sum()),
        "ladder_counts": [int((z[:, target_col("binary", rung=r)] > 0).sum()) for r in range(len(LADDER))],
    }
    return {
        "fit": put(pack(fit, z=z), name=f"{NAME}-fit.npz"),
        "tune": put(pack(tune), name=f"{NAME}-tune.npz"),
        "stats": stats,
    }


@dataclass
class Scored:
    """A set of completion contexts ready to score: tokens, ops, and what the expected exact match needs."""

    tokens: np.ndarray
    op_ids: np.ndarray
    posterior: np.ndarray
    ctx: Any
    table: Any
    null: np.ndarray
    color_ids: np.ndarray

    def device_tokens(self):
        """The tokens on the device, copied once per set rather than once per pass."""
        import jax

        if "_tokens" not in self.__dict__:
            self.__dict__["_tokens"] = jax.device_put(self.tokens)
        return self.__dict__["_tokens"]

    def device_color_ids(self):
        import jax

        if "_color_ids" not in self.__dict__:
            self.__dict__["_color_ids"] = jax.device_put(self.color_ids)
        return self.__dict__["_color_ids"]

    def match(self, logits: np.ndarray) -> np.ndarray:
        return ex2216._match(self.table, self.ctx, ex2216._color_probs(logits, self.color_ids))

    def per_op(self, v: np.ndarray) -> np.ndarray:
        return np.array([float(v[self.op_ids == o].mean()) for o in range(len(OP_NAMES))])


def scored_set(path, tok2color: np.ndarray, color_ids: np.ndarray, holdout: bool = False) -> Scored:
    """A set of `prepare_sets` (or the held-out set, with *holdout*) as `Scored`."""
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    if holdout:
        ho = ex2216.load_holdout(path, K)
        tokens, op_ids, post = ho.tokens, ho.op_ids, ho.posterior
    else:
        with np.load(path) as z:
            tokens, op_ids, post = z["tokens"], z["op_ids"], z["posterior"]
    n_colors = len(color_ids)
    a, b = tok2color[tokens[:, UNIT * K]], tok2color[tokens[:, UNIT * K + 2]]
    empty = np.zeros((len(a), 0), np.int64)
    ctx = P.Contexts(op_ids.astype(np.int64), empty, empty, (a * n_colors + b).astype(np.int64))
    null_post = ex2216._target_null(P, table, ctx, post.astype(np.float64), D)
    null = P.expected_match(table, ctx, null_post)
    return Scored(tokens.astype(np.int32), op_ids.astype(np.int64), post, ctx, table, null, color_ids)


# =============================================================================================
# The fits
# =============================================================================================


def _closed_form(method: str, m, col: int, rank: int):
    from sca import posthoc

    match method:
        case "dim":
            return posthoc.dim_fit(m, col)
        case "ridge":
            return posthoc.ridge_fit(m, col, RIDGE_L2)
        case "leace":
            return posthoc.leace_fit(m, col, LEACE_L2)
        case "inlp":
            return posthoc.inlp_fit(m, col, rank, RIDGE_L2)
    raise ValueError(method)


def _mean_of_others(m) -> np.ndarray:
    """The mean state of the contexts of the other ops, from moments (the binary column splits them off)."""
    return m.class_means(COL_BINARY)[1]


def fit_keys() -> list[tuple]:
    """(method, target, site, rank) for every closed-form fit."""
    return [(m, t, s, r) for m in STANDARD for t in TARGETS[m] for s in SITES for r in (INLP_RANKS if m == "inlp" else (1,))]  # fmt: skip


def refit_keys(smoke: bool) -> list[tuple]:
    """The fits that get a sequential refit: all of them, or for the smoke test the first of each method."""
    keys = fit_keys()
    return [next(k for k in keys if k[0] == m) for m in STANDARD] if smoke else keys


def fit_one(checkpoint, fit_set, label: str, smoke: bool = False) -> dict:
    """Every closed-form fit of one run, from moments over the fit set.

    `clean[(method, target, site, rank, column)][slice]` is the fit on the clean states (every target column, for
    the permuted baselines and the ladder too), and `refit[(method, target, site, rank, slices)][slice]` the fit
    at each slice of a multi-slice set made on the states as they arrive under the full edits at the slices
    before it (at every position, zero replacement), with the target columns of the plain fit. Beside them, the
    mean state of the other ops at each slice and site, clean and under each refit's edits (for mean ablation),
    and the time a pass over the fit set takes.
    """
    import pickle

    from mini.store import get, put
    from sca.posthoc import SliceEdit, stream_moments

    workdir = get_data_dir() / NAME / "fit" / label
    model, _, _, _ = ex2216._load_run(checkpoint, workdir)
    width = model.transformer.wte.shape[1]
    with np.load(get(fit_set, workdir / "fit.npz")) as z:
        tokens, cols = z["tokens"], z["z"]
    masks = site_masks()
    t0 = time.perf_counter()
    clean = stream_moments(model, tokens, SliceEdit.identity(N_SLICES, width, CONTEXT_LEN), masks, cols)
    pass_s = time.perf_counter() - t0

    keys = fit_keys()
    columns = lambda t: [
        target_col(t),
        *(target_col(t, perm=p) for p in range(N_PERMUTED)),
        *(target_col(t, rung=r) for r in range(1, len(LADDER))),
    ]  # noqa: E731
    fits: dict = {}
    for (method, target, site, rank), sl in itertools.product(keys, range(N_SLICES)):
        m = clean[sl][SITES.index(site)]
        for col in columns(target):
            fits[(method, target, site, rank, col, sl)] = _closed_form(method, m, col, rank)
    others = {(sl, site): _mean_of_others(clean[sl][i]) for sl in range(N_SLICES) for i, site in enumerate(SITES)}

    refit: dict = {}
    refit_others: dict = {}
    n_passes = 1
    multi = [s for s in SLICE_SETS if len(s) > 1]
    for (method, target, site, rank), slices in itertools.product(refit_keys(smoke), multi):
        col = target_col(target)
        done = {slices[0]: fits[(method, target, site, rank, col, slices[0])]}
        for sl in slices[1:]:
            edit = SliceEdit.build(done, N_SLICES, width, 1.0, np.ones(CONTEXT_LEN, np.float32), rank)
            moments = stream_moments(model, tokens, edit, masks, cols[:, [COL_BINARY, col]] if col != COL_BINARY else cols[:, [COL_BINARY]])  # fmt: skip
            n_passes += 1
            m = moments[sl][SITES.index(site)]
            done[sl] = _closed_form(method, m, len(m.sz) - 1, rank)
            for j, s in enumerate(SITES):
                refit_others[(method, target, site, rank, slices, sl, s)] = _mean_of_others(moments[sl][j])
        refit[(method, target, site, rank, slices)] = done
    body = {"fits": fits, "others": others, "refit": refit, "refit_others": refit_others,
            "moments_mean": {(sl, s): clean[sl][i].mean for sl in range(N_SLICES) for i, s in enumerate(SITES)}}  # fmt: skip
    return {
        "label": label,
        "fits": put(pickle.dumps(body), name=f"{NAME}-fits-{label}.pkl"),
        "seconds_per_fit_pass": pass_s,
        "n_fit_passes": n_passes,
        "seconds_total": time.perf_counter() - t0,
        "n_fit_contexts": int(len(tokens)),
    }


def search_group(checkpoint, fit_set, fits, label: str, jobs: list[tuple]) -> list[dict]:
    """Optimized directions for one run, one per job `(target, slices, weight, perm)`: a unit vector per edited
    slice, fitted by Adam on the fit set from the difference in means (clean, pooled). With a *perm*, the labels are
    permuted across contexts first (the permuted-label baseline), each context keeping its own target null. The
    jobs share one task, so the run, the fit set, and its target nulls are loaded once.
    """
    import pickle

    from mini.store import get
    from sca.posthoc import optimize_directions, search_data

    workdir = get_data_dir() / NAME / "search" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    width = model.transformer.wte.shape[1]
    s = scored_set(get(fit_set, workdir / "fit.npz"), tok2color, color_ids)
    with np.load(get(fit_set, workdir / "fit.npz")) as z:
        cols = z["z"]
    with open(get(fits, workdir / "fits.pkl"), "rb") as f:
        body = pickle.load(f)
    P = ex2216._get_posterior()
    null_post = ex2216._target_null(P, s.table, s.ctx, s.posterior.astype(np.float64), D)
    q_null = np.concatenate([
        P.predictive(s.table, P.Contexts(s.ctx.true_op[i : i + 20_000], s.ctx.ex_pair[i : i + 20_000],
                                         s.ctx.ex_color[i : i + 20_000], s.ctx.query_pair[i : i + 20_000]),
                     null_post[i : i + 20_000]).astype(np.float16)
        for i in range(0, len(s.tokens), 20_000)
    ])  # fmt: skip
    out = []
    for target, slices, weight, perm in jobs:
        w = cols[:, target_col(target, perm=perm)] if perm is not None else cols[:, target_col(target)]
        init = np.zeros((N_SLICES, width), np.float32)
        for sl in slices:
            init[sl] = body["fits"][("dim", "binary", "pooled", 1, COL_BINARY, sl)].basis[0]
        t0 = time.perf_counter()
        v, losses = optimize_directions(
            model, search_data(s.tokens, q_null, w, 1.0 - w), color_ids, tuple(slices), QUERY_EQ, weight, init,
            steps=SEARCH_STEPS, batch_size=SEARCH_BATCH, lr=SEARCH_LR, seed=0 if perm is None else 100 + perm,
        )  # fmt: skip
        out.append({"label": label, "target": target, "slices": list(slices), "weight": weight, "perm": perm,
                    "v": v.tolist(), "losses": losses[:: max(1, SEARCH_STEPS // 30)],
                    "seconds_per_step": (time.perf_counter() - t0) / SEARCH_STEPS})  # fmt: skip
    return out


# =============================================================================================
# Scoring
# =============================================================================================


class _Fits:
    """The fits of one run, and the edits of its configurations."""

    def __init__(self, body: dict, searched: list[dict], width: int):
        self.body, self.width = body, width
        self.searched = {(r["target"], tuple(r["slices"]), r["weight"], r["perm"]): np.asarray(r["v"], np.float32)
                         for r in searched}  # fmt: skip

    def subspaces(self, c: Config, perm: int | None = None, rung: int = 0) -> dict:
        from sca.intervention import Subspace

        if c.method == "optimized":
            v = self.searched[(c.target, c.slices, c.weight, perm)]
            return {sl: Subspace.direction(v[sl]) for sl in c.slices}
        if c.refit and perm is None and rung == 0:
            return dict(self.body["refit"][(c.method, c.target, c.site, c.rank, c.slices)])
        col = target_col(c.target, perm=perm, rung=rung)
        return {sl: self.body["fits"][(c.method, c.target, c.site, c.rank, col, sl)] for sl in c.slices}

    def edit(self, c: Config, perm: int | None = None, rung: int = 0, subs: dict | None = None):
        from sca.posthoc import SliceEdit, with_mean

        subs = self.subspaces(c, perm, rung) if subs is None else subs
        if c.replacement == "mean":
            subs = {sl: with_mean(sub, self.others(c, sl)) for sl, sub in subs.items()}
        positions = site_masks()[SITES.index(c.position_site)]
        rank = max(c.rank, max(s.basis.shape[0] for s in subs.values()))
        return SliceEdit.build(subs, N_SLICES, self.width, c.dose, positions, rank_of(rank))

    def others(self, c: Config, sl: int) -> np.ndarray:
        if c.refit and c.method != "optimized" and sl != c.slices[0]:
            return self.body["refit_others"][(c.method, c.target, c.site, c.rank, c.slices, sl, c.position_site)]
        return self.body["others"][(sl, c.position_site)]


def rank_of(k: int) -> int:
    """The rank a program is compiled for: 1, or the next of 2, 4, 8, so INLP shares three programs."""
    return 1 if k == 1 else min(r for r in INLP_RANKS if r >= k)


def axis_config(form: tuple, axis: int, width: int) -> tuple[Config, Any]:
    """A configuration of *form* along coordinate axis *axis*, and its subspace."""
    from sca.intervention import Subspace

    slices, site, dose, rep = form
    c = Config("axis", "binary", site, slices, False, "every" if site == "pooled" else "sites", dose, rep)
    return c, Subspace.axis(width, axis)


def _load_common(checkpoint, split_path, label: str, sub: str, holdout: bool):
    workdir = get_data_dir() / NAME / sub / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    return model, scored_set(split_path(workdir), tok2color, color_ids, holdout=holdout), workdir


def _per_op_scores(model, s: Scored, edit) -> np.ndarray:
    from sca.posthoc import color_probs_at

    p = color_probs_at(model, s.device_tokens(), edit, QUERY_EQ, s.device_color_ids())
    return s.per_op(ex2216._match(s.table, s.ctx, p))


def _forward_seconds(model, s: Scored, edit, n: int = 10) -> float:
    """The device time of one forward pass over *s*, with no copies to or from the host."""
    from sca.posthoc import _color_logprobs

    tokens, cid = s.device_tokens(), s.device_color_ids()
    _color_logprobs(model, tokens, edit, QUERY_EQ, cid).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        out = _color_logprobs(model, tokens, edit, QUERY_EQ, cid)
    out.block_until_ready()
    return (time.perf_counter() - t0) / n


def axes_one(checkpoint, tune_set, fits, label: str, smoke: bool = False) -> dict:
    """The drop per op under every operator form of the grid along each of the fifteen tuning axes, on the tuning
    set: the baseline the tuning frontier is netted of (its median over the axes). Mean ablation along an axis
    replaces the coordinate with that of the mean state of the other ops on the fit set.
    """
    import pickle

    from mini.store import get
    from sca.intervention import Subspace
    from sca.posthoc import SliceEdit, with_mean

    model, s, workdir = _load_common(checkpoint, lambda w: get(tune_set, w / "tune.npz"), label, "axes", False)
    width = model.transformer.wte.shape[1]
    with open(get(fits, workdir / "fits.pkl"), "rb") as f:
        body = pickle.load(f)
    clean = _per_op_scores(model, s, SliceEdit.identity(N_SLICES, width, CONTEXT_LEN))
    axes = tune_axes(width)
    fs = forms()
    if smoke:
        fs = fs[:: max(1, len(fs) // 12)]
        axes = axes[:2]
    out = {}
    t0 = time.perf_counter()
    for form in fs:
        slices, site, dose, rep = form
        drops = []
        for a in axes:
            sub = Subspace.axis(width, a)
            subs = {sl: (with_mean(sub, body["others"][(sl, site)]) if rep == "mean" else sub) for sl in slices}
            edit = SliceEdit.build(subs, N_SLICES, width, dose, site_masks()[SITES.index(site)])
            drops.append(clean - _per_op_scores(model, s, edit))
        out[str(form)] = np.median(drops, axis=0).tolist()
    n = len(fs) * len(axes)
    return {"label": label, "clean": clean.tolist(), "null": s.per_op(s.null).tolist(), "median_drop": out,
            "axes": list(axes), "seconds_per_pass": (time.perf_counter() - t0) / max(n, 1), "n_passes": n}  # fmt: skip


def grid_one(checkpoint, split, fits, searched: list[dict], label: str, holdout: bool, smoke: bool = False) -> list[dict]:  # fmt: skip
    """Expected exact match per op under every configuration of each method (in the order of `METHODS`), on the
    tuning set or (with *holdout*) the test set, with the clean pass and the target null. One task per run and
    set: a task spends some 30 s starting before its first pass.
    """
    import pickle

    from mini.store import get
    from sca.posthoc import SliceEdit

    name = "test.npz" if holdout else "tune.npz"
    model, s, workdir = _load_common(checkpoint, lambda w: get(split, w / name), label, "grid", holdout)
    width = model.transformer.wte.shape[1]
    with open(get(fits, workdir / "fits.pkl"), "rb") as f:
        f_ = _Fits(pickle.load(f), [r for r in searched if r["perm"] is None], width)
    identity = SliceEdit.identity(N_SLICES, width, CONTEXT_LEN)
    clean = _per_op_scores(model, s, identity).tolist()
    null = s.per_op(s.null).tolist()
    forward_s = _forward_seconds(model, s, identity)
    out = []
    for method in METHODS:
        cs = smoke_grid(method) if smoke else configs(method)
        t0 = time.perf_counter()
        scores = [_per_op_scores(model, s, f_.edit(c)).tolist() for c in cs]
        out.append({"label": label, "method": method, "holdout": holdout, "configs": [asdict(c) for c in cs],
                    "clean": clean, "null": null, "scores": scores,
                    "seconds_per_pass": (time.perf_counter() - t0) / max(len(cs), 1), "seconds_forward": forward_s})  # fmt: skip
    return out


def smoke_grid(method: str) -> list[Config]:
    """The smoke test's thinned grid: every level of every factor at least once, and a few more, among the
    configurations whose fits the smoke test makes (the refits of `refit_keys`, the searches of `search_plan`).
    """
    refit = set(refit_keys(True))
    searched = set(search_plan(True))
    cs = [
        c for c in configs(method)
        if (not c.refit or (c.method, c.target, c.site, c.rank) in refit)
        and (c.method != "optimized" or (c.target, c.slices, c.weight) in searched)
    ]  # fmt: skip
    return thin(cs, extra=6)


# =============================================================================================
# Selection
# =============================================================================================


def frontier(removal: np.ndarray, spill: np.ndarray) -> np.ndarray:
    """Indices of the configurations no other beats on both removal (higher) and spill (lower), by spill."""
    order = np.lexsort((-removal, spill))
    out, best = [], -np.inf
    for i in order:
        if removal[i] > best:
            out.append(i)
            best = removal[i]
    return np.array(out, dtype=int)


def read_point(removal: np.ndarray, spill: np.ndarray, front: np.ndarray, at: float, on: str) -> dict:
    """A point of the selection rule on the frontier *front*: where it crosses *at* in spill (*on* = "spill") or
    in removal (*on* = "removal"), as the two bracketing configurations and the weight on the second. Linear
    interpolation between them. Past either end in spill the end configuration is taken alone; a frontier that
    never reaches *at* in removal has no point (`matched` false).
    """
    x = spill[front] if on == "spill" else removal[front]
    if on == "spill":
        if x[0] > at:
            return {"configs": [int(front[0])], "weights": [1.0], "matched": True, "clipped": "none within"}
        if x[-1] <= at:
            return {"configs": [int(front[-1])], "weights": [1.0], "matched": True, "clipped": "all within"}
    elif x[-1] < at:
        return {"configs": [], "weights": [], "matched": False, "clipped": "not reached"}
    elif x[0] >= at:
        return {"configs": [int(front[0])], "weights": [1.0], "matched": True, "clipped": "at the first"}
    j = int(np.searchsorted(x, at, side="right"))
    lo, hi = j - 1, j
    t = float((at - x[lo]) / (x[hi] - x[lo])) if x[hi] != x[lo] else 0.0
    return {"configs": [int(front[lo]), int(front[hi])], "weights": [1.0 - t, t], "matched": True, "clipped": None}


def removal_spill(scores: np.ndarray, clean: np.ndarray, null: np.ndarray, baseline: np.ndarray) -> tuple:
    """Removal and spill of each configuration: per-op drops net of *baseline* (configs × ops)."""
    net = (clean[None] - scores) - baseline
    removal = net[:, D] / (clean[D] - null[D])
    spill = np.delete(net, D, axis=1).max(axis=1)
    return removal, spill, net


def select(grids: list[dict], axes: dict, anchored_removal: float) -> dict:
    """Per method, the frontier on the tuning set (net of the median over the fifteen axes) and the two points of
    the selection rule: removal at the spill criterion and spill at matched removal.
    """
    out = {}
    for g in grids:
        cs = [config_of(c) for c in g["configs"]]
        scores, clean, null = np.asarray(g["scores"]), np.asarray(g["clean"]), np.asarray(g["null"])
        base = np.array([axes["median_drop"].get(str(c.form), [0.0] * len(OP_NAMES)) for c in cs])
        removal, spill, _ = removal_spill(scores, clean, null, base)
        front = frontier(removal, spill)
        out[g["method"]] = {
            "frontier": front.tolist(),
            "removal": removal.tolist(),
            "spill": spill.tolist(),
            "at_criterion": read_point(removal, spill, front, SPILL_CRITERION, "spill"),
            "at_matched": read_point(removal, spill, front, anchored_removal, "removal"),
            "missing_baseline": int(sum(str(c.form) not in axes["median_drop"] for c in cs)),
        }
    return out


# The frontiers taken over part of the grid: H4 restricts each fitter to the fits on the pooled positions or at the
# example answers, and the pooled fits alone are what a pipeline with no role labels could use.
RESTRICTED = {"h4": ("pooled", "answers"), "pooled": ("pooled",)}


def select_sites(grids: list[dict], axes: dict, anchored_removal: float, sites: tuple[str, ...]) -> dict:
    """`select` over the configurations fitted at *sites* alone, with the indices of the whole grid."""
    sub, keep = [], {}
    for g in grids:
        idx = [i for i, c in enumerate(g["configs"]) if c["site"] in sites]
        keep[g["method"]] = idx
        sub.append(g | {"configs": [g["configs"][i] for i in idx], "scores": [g["scores"][i] for i in idx]})
    out = select(sub, axes, anchored_removal)
    for m, sel in out.items():
        idx = keep[m]
        n = len(grids[[g["method"] for g in grids].index(m)]["configs"])
        for key in ("removal", "spill"):
            full = [None] * n
            for j, i in enumerate(idx):
                full[i] = sel[key][j]
            sel[key] = full
        sel["frontier"] = [idx[j] for j in sel["frontier"]]
        for p in ("at_criterion", "at_matched"):
            sel[p]["configs"] = [idx[j] for j in sel[p]["configs"]]
    return out


def restricted_one(checkpoint, holdout, fits, searched: list[dict], tune_grids: list[dict], axes: dict,
                   selection: dict, label: str, anchored: bool, anchored_removal: float) -> dict:  # fmt: skip
    """The restricted selections of `RESTRICTED` (H4 on the anchored runs only), with the two test-set baselines
    at each bracketing configuration that `finish_one` did not already score: the permuted-label median and the
    median along the other 63 axes, as `finish_one` takes them.
    """
    import pickle

    from mini.store import get
    from sca.posthoc import SliceEdit, with_mean

    model, s, workdir = _load_common(checkpoint, lambda w: get(holdout, w / "test.npz"), label, "restricted", True)
    width = model.transformer.wte.shape[1]
    with open(get(fits, workdir / "fits.pkl"), "rb") as f:
        f_ = _Fits(pickle.load(f), searched, width)
    clean = _per_op_scores(model, s, SliceEdit.identity(N_SLICES, width, CONTEXT_LEN))
    by_method = {g["method"]: [config_of(c) for c in g["configs"]] for g in tune_grids}
    out: dict[str, Any] = {"selections": {}, "baselines": {}}
    for name, sites in RESTRICTED.items():
        if name == "h4" and not anchored:
            continue
        sels = select_sites(tune_grids, axes, anchored_removal, sites)
        out["selections"][name] = sels
        for method, sel in sels.items():
            done = set(_chosen(selection[method]))
            for i in _chosen(sel):
                if i in done or str(i) in out["baselines"].get(method, {}):
                    continue
                c = by_method[method][i]
                if method == "optimized":
                    perms = [p for p in range(N_PERMUTED_SEARCH) if (c.target, c.slices, c.weight, p) in f_.searched]
                else:
                    perms = list(range(N_PERMUTED))
                permuted = [(clean - _per_op_scores(model, s, f_.edit(c, perm=p))).tolist() for p in perms]
                axis_drops = []
                for a in range(1, width):
                    c_ax, sub = axis_config(c.form, a, width)
                    subs = {sl: (with_mean(sub, f_.others(c_ax, sl)) if c.replacement == "mean" else sub)
                            for sl in c.slices}  # fmt: skip
                    edit = SliceEdit.build(subs, N_SLICES, width, c.dose, site_masks()[SITES.index(c.position_site)])
                    axis_drops.append(clean - _per_op_scores(model, s, edit))
                out["baselines"].setdefault(method, {})[str(i)] = {
                    "config": asdict(c), "permuted_drops": permuted,
                    "axis_median_drop": np.median(axis_drops, axis=0).tolist()}  # fmt: skip
    return {"label": label} | out


def publish_restricted(rows: list[dict], restricted: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    set_ref(RESTRICTED_REF, put(json.dumps({"runs": restricted}).encode(), name=f"{NAME}-restricted.json"))
    return {"runs": len(rows), "ref": RESTRICTED_REF}


# =============================================================================================
# What the hypotheses and the E sections read at the selected configurations
# =============================================================================================


def _chosen(sel: dict) -> list[int]:
    """The configurations a method's two points use."""
    return sorted({i for p in ("at_criterion", "at_matched") for i in sel[p]["configs"]})


def _main_config(sel: dict) -> int:
    """The configuration the E sections read for a method: the bracketing configuration of removal at the spill
    criterion that lies within it (the lower-spill one), or the end configuration where the point is clipped.
    """
    return sel["at_criterion"]["configs"][0]


def finish_one(checkpoint, holdout, fit_set, fits, searched: list[dict], tune_grids: list[dict],
               selection: dict, label: str, anchored: bool, smoke: bool = False) -> dict:  # fmt: skip
    """At the configurations each method selected: the permuted-label baseline and the median along the other
    63 axes on the test set (per op), the label ladder (E3), the per-context arrays for grading, landing, and the
    confusion matrices (E2), the write and its bound (E4), a fresh probe (E5), and the cosine with e₁ (H4). On an
    anchored run, the same for the edit along e₁ at every slice and position (the anchor).
    """
    import pickle

    from mini.store import get, put
    from sca.intervention import Subspace
    from sca.posthoc import SliceEdit, with_mean

    model, s, workdir = _load_common(checkpoint, lambda w: get(holdout, w / "test.npz"), label, "finish", True)
    width = model.transformer.wte.shape[1]
    with open(get(fits, workdir / "fits.pkl"), "rb") as f:
        body = pickle.load(f)
    f_ = _Fits(body, searched, width)
    identity = SliceEdit.identity(N_SLICES, width, CONTEXT_LEN)
    clean = _per_op_scores(model, s, identity)
    by_method = {g["method"]: [config_of(c) for c in g["configs"]] for g in tune_grids}
    every_axis = list(range(1, width))[: (4 if smoke else width - 1)]
    n_perm = 2 if smoke else N_PERMUTED
    out: dict[str, Any] = {"label": label, "clean": clean.tolist(), "null": s.per_op(s.null).tolist(), "methods": {}}
    arrays: dict[str, np.ndarray] = {}
    passes, t0 = 0, time.perf_counter()

    def scores(edit) -> np.ndarray:
        nonlocal passes
        passes += 1
        return _per_op_scores(model, s, edit)

    for method, sel in selection.items():
        cs = by_method[method]
        res: dict[str, Any] = {"configs": {}}
        for i in _chosen(sel):
            c = cs[i]
            if method == "optimized":
                perms = [p for p in range(N_PERMUTED_SEARCH) if (c.target, c.slices, c.weight, p) in f_.searched]
            else:
                perms = list(range(n_perm))
            permuted = [(clean - scores(f_.edit(c, perm=p))).tolist() for p in perms]
            axis_drops = []
            for a in every_axis:
                c_ax, sub = axis_config(c.form, a, width)
                subs = {
                    sl: (with_mean(sub, f_.others(c_ax, sl)) if c.replacement == "mean" else sub) for sl in c.slices
                }
                axis_drops.append(clean - scores(SliceEdit.build(subs, N_SLICES, width, c.dose, site_masks()[SITES.index(c.position_site)])))  # fmt: skip
            res["configs"][i] = {"config": asdict(c), "permuted_drops": permuted,
                                 "axis_median_drop": np.median(axis_drops, axis=0).tolist()}  # fmt: skip
        main = cs[_main_config(sel)]
        edit = f_.edit(main)
        lg = _logits(model, s, edit)
        eem = s.match(lg)
        arrays[f"{method}/eem"] = eem.astype(np.float32)
        arrays[f"{method}/p_difference"] = ex2216._color_probs(lg[s.op_ids == D], s.color_ids).astype(np.float16)
        # E3: the label ladder, for the standard fitters (the optimized direction reads the target null, not labels).
        if method != "optimized":
            ladder = []
            for rung in range(len(LADDER)):
                if main.refit:
                    subs = _refit(model, fit_set, workdir, f_, main, rung, width)
                    ladder.append((clean - scores(f_.edit(main, subs=subs))).tolist())
                else:
                    ladder.append((clean - scores(f_.edit(main, rung=rung))).tolist())
            res["ladder_drops"] = ladder
        res["write"] = _write_and_bound(model, s, edit, identity)
        res["probe"] = _probe(model, s, edit)
        res["cosine_e1"] = _cosines(f_, main, width)
        res["main_config"] = asdict(main)
        out["methods"][method] = res

    if anchored:
        e1 = SliceEdit.build({sl: Subspace.axis(width) for sl in range(N_SLICES)}, N_SLICES, width, 1.0, np.ones(CONTEXT_LEN, np.float32))  # fmt: skip
        lg = _logits(model, s, e1)
        arrays["anchor/eem"] = s.match(lg).astype(np.float32)
        arrays["anchor/p_difference"] = ex2216._color_probs(lg[s.op_ids == D], s.color_ids).astype(np.float16)
        out["anchor"] = {"drop": (clean - s.per_op(s.match(lg))).tolist(), "write": _write_and_bound(model, s, e1, identity),
                         "probe": _probe(model, s, e1)}  # fmt: skip
    lg = _logits(model, s, identity)
    arrays["clean/eem"] = s.match(lg).astype(np.float32)
    arrays["clean/p_difference"] = ex2216._color_probs(lg[s.op_ids == D], s.color_ids).astype(np.float16)
    out["probe_clean"] = _probe(model, s, identity)
    out["probe_roles"] = _probe(model, s, identity, roles=True)
    out["arrays"] = put(ex2216._npz(**arrays), name=f"{NAME}-{label}.npz")
    out["seconds"] = time.perf_counter() - t0
    out["n_passes"] = passes
    return out


def _logits(model, s: Scored, edit) -> np.ndarray:
    from sca.posthoc import logits_at

    return logits_at(model, s.tokens, edit, QUERY_EQ)


def _refit(model, fit_set, workdir, f_: _Fits, c: Config, rung: int, width: int) -> dict:
    """The sequential refit of *c* with the labels of ladder rung *rung*."""
    from mini.store import get
    from sca.posthoc import SliceEdit, stream_moments

    with np.load(get(fit_set, workdir / "fit.npz")) as z:
        tokens, cols = z["tokens"], z["z"]
    col = target_col(c.target, rung=rung)
    done = {c.slices[0]: f_.body["fits"][(c.method, c.target, c.site, c.rank, col, c.slices[0])]}
    for sl in c.slices[1:]:
        edit = SliceEdit.build(done, N_SLICES, width, 1.0, np.ones(CONTEXT_LEN, np.float32), c.rank)
        m = stream_moments(model, tokens, edit, site_masks(), cols[:, [COL_BINARY, col]])[sl][SITES.index(c.site)]
        done[sl] = _closed_form(c.method, m, 1, c.rank)
    return done


def _write_and_bound(model, s: Scored, edit, identity) -> dict:
    """The write (the turn the edit makes on the states as they arrive under the edits before) and its bound (the
    turn it makes on the clean states), the 99% quantile over the contexts of the other ops, per slice and role.
    """
    import jax.numpy as jnp

    from sca.intervention import angle_between
    from sca.posthoc import pre_post

    other = np.flatnonzero(s.op_ids != D)[:N_WRITE_CTX]
    pre, post = pre_post(model, s.tokens[other], edit)
    clean, _ = pre_post(model, s.tokens[other], identity)
    write = angle_between(post, pre)  # (L1, N, T)
    bound = np.stack(
        [angle_between(np.asarray(edit(jnp.asarray(clean[sl]), sl)), clean[sl]) for sl in range(len(clean))]
    )
    role = np.arange(CONTEXT_LEN) % UNIT
    q = lambda a: [
        [float(np.percentile(a[sl][:, role == r], WRITE_QUANTILE)) for r in range(UNIT)] for sl in range(len(a))
    ]  # noqa: E731
    return {"write": q(write), "bound": q(bound)}


def probe_sites(roles: bool) -> list[tuple[str, list[int]]]:
    """Where the probe reads: the query `=` and the example answers, or with *roles* every role of an example
    (pooled over the examples) and every position of the query, for the open decision on where the control holds
    the op.
    """
    if not roles:
        return [("query =", [QUERY_EQ]), ("answers", list(ANSWERS))]
    example = [(f"example role {j}", [UNIT * i + j for i in range(K)]) for j in range(UNIT)]
    return example + [(f"query position {p - UNIT * K}", [p]) for p in range(UNIT * K, CONTEXT_LEN)]


def _probe(model, s: Scored, edit, roles: bool = False) -> dict:
    """AUC of a fresh ridge probe for the binary target on the edited states of the test set, per slice, at the
    query `=` and at the example answers (or at every role, `probe_sites`), fitted on half the contexts and scored
    on the rest. A site with several positions in a context is pooled, each position a sample of its context.
    """
    from sca.posthoc import pre_post

    rng = np.random.default_rng(0)
    pick = rng.choice(len(s.tokens), size=min(7_000, len(s.tokens)), replace=False)
    _, post = pre_post(model, s.tokens[pick], edit)
    y = (s.op_ids[pick] == D).astype(float)
    half = rng.random(len(pick)) < 0.5
    out = {}
    for sl in range(post.shape[0]):
        for site, pos in probe_sites(roles):
            x = post[sl][:, pos].reshape(-1, post.shape[-1]).astype(np.float64)
            yy, hh = np.repeat(y, len(pos)), np.repeat(half, len(pos))
            xc = x - x[hh].mean(0)
            w = np.linalg.solve(xc[hh].T @ xc[hh] + PROBE_L2 * hh.sum() * np.eye(x.shape[1]), xc[hh].T @ (yy[hh] - yy[hh].mean()))  # fmt: skip
            out[f"{site}, slice {sl}"] = _auc(xc[~hh] @ w, yy[~hh])
    return out


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cosines(f_: _Fits, c: Config, width: int) -> list[float | None]:
    """|cos| between e₁ and the clean fit for the target and site of *c* at each of slices 1 to 3 (for INLP, the
    largest cosine between e₁ and the span: the length of e₁ projected onto it).
    """
    out = []
    for sl in (1, 2, 3):
        if c.method == "optimized":
            v = f_.searched[(c.target, c.slices, c.weight, None)][sl]
            out.append(float(abs(v[0])) if sl in c.slices else None)
            continue
        sub = f_.body["fits"][(c.method, c.target, c.site, c.rank, target_col(c.target), sl)]
        out.append(float(np.linalg.norm(sub.basis[:, 0])))
    return out


# =============================================================================================
# The DAG
# =============================================================================================


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2222.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def anchored_removal(smoke: bool) -> float:
    """The seed mean of the anchored removal at full dose on the test set, net of the median over the other 63
    axes of each run, as fixed-effects publishes it: the removal a fitter is matched to. For the smoke test, the
    `falloff` run at 704 alone (from reference-baseline's axis scoring).
    """
    import json

    from mini.store import get, get_ref

    ref = fc.AXES_REF if not smoke else rb.AXES_REF
    art = get_ref(ref)
    assert art is not None, f"not published: {ref}"
    body = json.loads(get(art, get_data_dir() / NAME / "anchored" / ref.replace("/", "_")).read_text())
    rows = body["runs"]
    want = {fc.label_of(s, None) for s in SEEDS} if not smoke else {f"falloff-s{SMOKE_SEED}"}
    vals = []
    for r in rows:
        if r["label"] not in want:
            continue
        drop = np.asarray(r["drop"])
        net = drop[0] - np.median(drop[1:], axis=0)  # (doses, ops)
        vals.append(net[-1, D] / (r["clean"][D] - r["null"][D]))
    assert vals, f"no anchored runs in {ref}"
    return float(np.mean(vals))


def design() -> dict[str, Any]:
    return {
        "experiment": NAME,
        "ops": list(OP_NAMES),
        "anchored_op": ANCHORED_OP,
        "k": K,
        "seeds": list(SEEDS),
        "fit_per_op": FIT_PER_OP,
        "tune_per_op": TUNE_PER_OP,
        "spill_criterion": SPILL_CRITERION,
        "doses": list(DOSES),
        "sites": list(SITES),
        "slice_sets": [list(s) for s in SLICE_SETS],
        "inlp_ranks": list(INLP_RANKS),
        "search_weights": list(SEARCH_WEIGHTS),
        "search_steps": SEARCH_STEPS,
        "n_permuted": N_PERMUTED,
        "n_permuted_search": N_PERMUTED_SEARCH,
        "ladder": list(LADDER),
        "tune_axes": list(tune_axes()),
        "n_configs": {m: len(configs(m)) for m in METHODS},
        "n_forms": len(forms()),
    }


def publish(rows: list[dict], sets: dict, fitted: list[dict], searched: list[list[dict]], axes: list[dict],
            tune: list[list[dict]], test: list[list[dict]], selections: list[dict], finished: list[dict],
            anchored: float, ref: str) -> dict:  # fmt: skip
    import json

    from mini.store import put, set_ref

    body = {
        "design": design(),
        "sets": sets["stats"],
        "anchored_removal": anchored,
        "runs": [
            r | {"fit": {k: v for k, v in f.items() if k != "fits"}, "search": s, "axes": a, "tune": t,
                 "test": te, "selection": sel, "finish": {k: v for k, v in fin.items() if k != "arrays"}}
            for r, f, s, a, t, te, sel, fin in zip(rows, fitted, searched, axes, tune, test, selections, finished, strict=True)
        ],
    }  # fmt: skip
    set_ref(ref, put(json.dumps(body).encode(), name=f"{NAME}-{ref.rsplit('/', 1)[-1]}.json"))
    for r, fin in zip(rows, finished, strict=True):
        set_ref(ARRAYS_REF.format(label=("smoke-" if ref == SMOKE_REF else "") + r["label"]), fin["arrays"])
    return {"runs": len(rows), "ref": ref}


def search_plan(smoke: bool) -> list[tuple]:
    """(target, slices, weight) for every optimized direction; the smoke test fits two."""
    plan = [(t, s, w) for t in TARGETS["optimized"] for s in SLICE_SETS for w in SEARCH_WEIGHTS]
    return [("binary", (1, 2, 3), 1.0), ("graded", (2,), 3.0)] if smoke else plan


def run(ctx: Ctx, smoke: bool = False) -> dict:
    rows = runs(smoke)
    n = len(rows)
    resolved = ctx.run(resolve, [r["ref"] for r in rows], role="prep")
    ckpt = [resolved["checkpoints"][r["ref"]] for r in rows]
    holdout = resolved["holdout"]
    sets = ctx.run(prepare_sets, holdout, role="sets")
    anchored = ctx.run(anchored_removal, smoke, role="prep")
    labels = [r["label"] for r in rows]
    fitted = ctx.map(fit_one, ckpt, [sets["fit"]] * n, labels, [smoke] * n, role="fit")

    plan = search_plan(smoke)
    # One task per run: a task spends about 20 s starting and computing the target nulls before its first step,
    # more than a search of 300 steps takes.
    jobs = [(i, [(t, sl, w, None) for t, sl, w in plan]) for i in range(n)]
    found = ctx.map(
        search_group,
        [ckpt[i] for i, _ in jobs],
        [sets["fit"]] * len(jobs),
        [fitted[i]["fits"] for i, _ in jobs],
        [labels[i] for i, _ in jobs],
        [js for _, js in jobs],
        role="search",
    )
    searched = [[r for (i, _), rs in zip(jobs, found, strict=True) if i == j for r in rs] for j in range(n)]

    axes = ctx.map(axes_one, ckpt, [sets["tune"]] * n, [f["fits"] for f in fitted], labels, [smoke] * n, role="score")
    grid_jobs = [(i, h) for i in range(n) for h in (False, True)]
    grids = ctx.map(
        grid_one,
        [ckpt[i] for i, _ in grid_jobs],
        [holdout if h else sets["tune"] for _, h in grid_jobs],
        [fitted[i]["fits"] for i, _ in grid_jobs],
        [searched[i] for i, _ in grid_jobs],
        [labels[i] for i, _ in grid_jobs],
        [h for _, h in grid_jobs],
        [smoke] * len(grid_jobs),
        role="score",
    )
    tune = [g for (_, h), g in zip(grid_jobs, grids, strict=True) if not h]
    test = [g for (_, h), g in zip(grid_jobs, grids, strict=True) if h]
    selections = ctx.map(select, tune, axes, [anchored] * n, role="prep")

    # The permuted-label baseline of the optimized direction: a fresh search per permutation, at the configurations
    # its two points use.
    perm_jobs = []
    for i in range(n):
        sel = selections[i]["optimized"]
        cs = [tune[i][METHODS.index("optimized")]["configs"][k] for k in _chosen(sel)]
        chosen = sorted({(c["target"], tuple(c["slices"]), c["weight"]) for c in cs})
        perms = range(2 if smoke else N_PERMUTED_SEARCH)
        perm_jobs.append((i, [(t, sl, w, p) for t, sl, w in chosen for p in perms]))
    permuted = ctx.map(
        search_group,
        [ckpt[i] for i, _ in perm_jobs],
        [sets["fit"]] * len(perm_jobs),
        [fitted[i]["fits"] for i, _ in perm_jobs],
        [labels[i] for i, _ in perm_jobs],
        [js for _, js in perm_jobs],
        role="search",
    )
    for (i, _), rs in zip(perm_jobs, permuted, strict=True):
        searched[i] = [*searched[i], *rs]

    finished = ctx.map(
        finish_one,
        ckpt,
        [holdout] * n,
        [sets["fit"]] * n,
        [f["fits"] for f in fitted],
        searched,
        tune,
        selections,
        labels,
        [r["kind"] == "anchored" for r in rows],
        [smoke] * n,
        role="score",
    )
    ref = SMOKE_REF if smoke else RESULTS_REF
    published = ctx.run(publish, rows, sets, fitted, searched, axes, tune, test, selections, finished, anchored, ref, role="prep")  # fmt: skip
    if smoke:
        return published
    restricted = ctx.map(
        restricted_one,
        ckpt,
        [holdout] * n,
        [f["fits"] for f in fitted],
        searched,
        tune,
        axes,
        selections,
        labels,
        [r["kind"] == "anchored" for r in rows],
        [anchored] * n,
        role="score",
    )
    return published | {"restricted": ctx.run(publish_restricted, rows, restricted, role="prep")}


def main(ctx: Ctx) -> dict:
    return run(ctx, smoke=False)


def smoke(ctx: Ctx) -> dict:
    """Every stage on the pair at 704 with a thinned grid; its timings give the cost of the full run."""
    return run(ctx, smoke=True)


STAGE = "main"
"""Which entry point `experiment` runs: the smoke test until its cost is approved, then the full run."""

COMPUTE = {
    "prep": dict(cpu=2, timeout=1800),
    # 294,000 contexts sampled and their posteriors computed on the host.
    "sets": dict(cpu=4, memory=16_384, timeout=10_800),
    # Moments over the fit set at every slice and site, once clean and once per step of each sequential refit.
    # A GPU container left idle for Modal's default minute after its task is billed for it, and no later task
    # reuses it, so the GPU roles let go of it at once.
    "fit": dict(gpu="L40S", cpu=4, memory=16_384, timeout=7200, scaledown_window=2),
    "search": dict(gpu="L40S", cpu=2, memory=16_384, timeout=7200, scaledown_window=2),
    "score": dict(gpu="L40S", cpu=2, memory=16_384, timeout=7200, scaledown_window=2),
}

experiment = Experiment(name=NAME, main=smoke if STAGE == "smoke" else main, roles=COMPUTE)
