"""Experiment 2.1.9: a pull that finds its own position.

Ex-2.1.7's largest single effect was narrowing the pull from the four-position
prompt span to op1 alone — but that is a position oracle this synthetic
language happens to supply and natural language will not. Here we replace the
per-position mean in the anchor term with a soft minimum (mellowmax) over each
labeled line's prompt span, so the loss asks that the span align *somewhere*
rather than everywhere, and the pull is free to concentrate wherever alignment
is cheapest. The anti-subspace term at ex-2.1.8's operating point is what
should make "cheapest" coincide with "the position that carries the concept":
aligning a shared syntax token helps only the labeled lines' anchor term while
every line pays repulsion for it.

The testbed, labels, measurements, anchor schedule and anti-subspace schedule
are ex-2.1.8's, unchanged; only the pooling over span positions moves.

    bin/mini run docs/m2/ex-2.1.9/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.9
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import PROMPT_SPAN, softmin_weights
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from mini import Ctx, Experiment, get_data_dir

# --- Testbed, carried from ex-2.1.8 unchanged -------------------------------

GRID = "v216"  # the ex-2.1.3 cell: 216 colors, one word each, d64-L4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set
N_PROBE = 27  # probe lines per color: all closed partners, exhaustive

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand — the ceiling of the label distribution."""

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
TRAJ_STRIDE = 50  # steps between alignment measurements during training
BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
LABEL_P = REDNESS**8 * RED_RATE
"""P(line labeled red) for each of the 216 colors, as a function of its first operand."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: the weight $u_c$ every alignment statistic scores with."""

SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""M1's ablation response shape mapped to alignment; see ex-2.1.6 for the derivation."""

SCORING_LAMBDA = 0.1
"""Anchor peak for every cell — ex-2.1.6's scoring rung, held fixed throughout."""

SPAN = PROMPT_SPAN
"""The four span roles the pooled pull chooses among: op1, `+`, op2, `=`."""

ROLES = ["op1", "+", "op2", "="]
"""Display names for the span roles, in position order."""

# --- The operating point, from ex-2.1.8 -------------------------------------

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet, unchanged."""

ANTI_ANNEAL_END = 90.0
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.8's `end90-hold30`: the anti-subspace anneal runs to epoch 90 and
settles at 0.30 of the anchor weight — the first condition in M2 to hold ᾱ
near the control while keeping the margin. Fixed in every anchored cell here."""

# --- The pooling ------------------------------------------------------------

TAUS = [0.010, 0.030, 0.100]
"""The mellowmax temperature grid, calibrated from ex-2.1.8's stored alignments.

The leading span position takes 0.6–0.8 of the softmin weight at τ ≈ 0.01–0.03
under both reference profiles (the un-anchored control and the span pull at the
operating point), so the grid brackets that band: 0.01 sits at its sharp edge
(winner-take-most), 0.03 in the middle, and 0.1 one step toward the mean. The
τ = ∞ arm closes the family at the span mean. The calibration figure in
`report.py` derives this from the published arrays.
"""

LEAD_BAND = (0.6, 0.8)
"""The target range for the leading position's softmin weight, per the design note."""


def mellowmax_min(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """The soft minimum −τ·log(mean exp(−x/τ)) along *axis*.

    The 1/n inside the log is what makes τ = ∞ the exact mean (plain
    log-sum-exp diverges instead), and τ = 0 the exact min. This is the
    mellowmax of Asadi & Littman (2017), applied to a minimum.
    """
    if np.isinf(tau):
        return x.mean(axis=axis)
    z = -x / tau
    m = z.max(axis=axis, keepdims=True)
    return -tau * (np.log(np.mean(np.exp(z - m), axis=axis)) + np.squeeze(m, axis=axis))


# --- Conditions -------------------------------------------------------------


def _pooled(tau: float) -> dict:
    return {"name": f"pool-t{tau * 1000:03.0f}", "lam": SCORING_LAMBDA, "tau": tau, "span": SPAN}


CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "tau": np.inf, "span": SPAN},
    {"name": "span-mean", "lam": SCORING_LAMBDA, "tau": np.inf, "span": SPAN},
    *[_pooled(t) for t in TAUS],
    {"name": "op1-oracle", "lam": SCORING_LAMBDA, "tau": np.inf, "span": 1},
]
"""Six conditions × 3 seeds. `span-mean` re-runs ex-2.1.8's `end90-hold30`
through the pooled code path at τ = ∞, so the H2 contrast differs only in τ
and the reproduction check validates the new path. `op1-oracle` is ex-2.1.7's
op1-only pull at the new operating point: the ceiling the oracle buys,
measured through identical code. Every anchored cell carries the anti-subspace
term at the operating point."""

POOLED_NAMES = [_pooled(t)["name"] for t in TAUS]
MEAN_ARM = "span-mean"
ORACLE_ARM = "op1-oracle"

PRIMARY = "pool-t030"
"""P, the condition H2–H4 score, fixed before the run. Chosen because its τ
sits at (or, on the deeper slices, just below) the soft edge of the 0.6–0.8
calibration band on both reference profiles — the conservative end of the
target range, and a criterion independent of every gated statistic, so no
gate is scored on a condition selected by that same gate's statistic. The
other two pooled arms are the τ sweep: reported beside P, not gated."""

# --- Statistics shared with the report --------------------------------------


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span: the mean over slices of the max-over-span margin, for one run's alignment map.

    *alpha* is (slices, colors, positions) on the probe set; *weights* is
    LABEL_W. The max over span roles is applied to every condition and to the
    control alike, so the maximum-of-noise inflation it invites is absorbed by
    the control subtraction the hypotheses specify.
    """
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def alpha_span(alpha: np.ndarray) -> float:
    """ᾱ_span: the layer-mean of the max-over-span mean alignment over all colors.

    The containment statistic ᾱ reads at op1 only; this is the same quantity
    with the same max-over-span treatment as `pooled_margin`, so drift parked
    on the other span roles — which ᾱ at op1 cannot see — is scored too.
    """
    return float(alpha[:, :, :SPAN].mean(axis=1).max(axis=1).mean())


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with midranks for ties. Exploratory only here: the ρ grading
    track is dropped from the gates — see the grading note in `report.py`.
    """
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


def r2_sim(alpha: np.ndarray) -> float:
    """Squared Pearson correlation (r²) between a per-color response and SIM_TARGET.

    M1's proportionality score. Written r² in this report to keep it apart
    from the readability probe's held-out coefficient of determination, which
    earlier reports also wrote as R².
    """
    return float(np.corrcoef(alpha, SIM_TARGET)[0, 1] ** 2)


# --- Gates ------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means). As ex-2.1.8."""

POOL_GAIN_GATE = 0.15
POOL_GAIN_PARTIAL = 0.09
"""H2(a): the pooled margin gain over the span mean, m_span(P) − m_span(span-mean).

A difference of two three-seed condition means; at ex-2.1.6's seed-mean margin
noise of 0.032 that difference carries about 0.032·√2 ≈ 0.045, so the gate is
~3.3× noise and the partial ~2×. A gain in [PARTIAL, GATE) scores H2 as
partial when (b) also holds. For scale, the op1 oracle bought +0.22 at op1
over the span pull in ex-2.1.7.
"""

SYNTAX_DRIFT_GATE = 0.10
"""H2(b): the required drop in ᾱ_span from the span-mean arm to P.

Ex-2.1.8's operating point has ᾱ_span ≈ 0.37 against 0.03 in the control
(3-seed spreads ~0.01), so the gate is small against the headroom and large
against noise (a difference of three-seed means carries ~0.016 at the 0.02
control spread).
"""

MEAN_ALIGN_GATE = 0.1
"""H3(a): ceiling on ᾱ, the mean alignment over all 216 colors at op1. Carried
from ex-2.1.8, whose operating point meets it at 0.087."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
RETENTION_STAT = "min"
"""H3(b): retention on the m_span trajectory — for every run whose running
maximum reaches `RETENTION_FLOOR`, the final value is at least `RETENTION_GATE`×
that maximum. The quoted statistic is the minimum across seeds
(`RETENTION_STAT`, settled in ex-2.1.8). The values are ex-2.1.8's H4 gates
under clearer names, since retention scores H3 here."""

GRADE_R2_DROP = 0.10
"""H3(c): the largest grading loss tolerated, R²(span-mean) − R²(P).

Relative rather than absolute: ex-2.1.8 found no condition reaching the 0.8
absolute gate (the operating point sits at 0.77) and left where a realistic
bar sits to a later grid, so this experiment scores "pooling doesn't cost
response shape" instead. The ρ track is dropped entirely — ex-2.1.8's ceiling
analysis put its best reachable value alongside containment at 0.59–0.82
against a 0.8 gate.
"""

LEAD_GATE = 0.4
"""H4(a): the leading softmin weight that counts as concentration. Uniform over
four roles is 0.25, and seed spread on a label-weighted mean weight is small,
so 0.4 is well clear of both."""

READ_GAIN_GATE = 0.05
"""H4(b): mean over the position-aware slices of Σ_t π̄(l,t)·R²(l,t) − mean_t R²(l,t) —
how much better the weighted positions read op1's redness than the span
average. Zero for uniform weights or a flat readability profile, by
construction."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors, carried from ex-2.1.6: same architecture, same measurement."""

# --- Published references ---------------------------------------------------

EX218_REFERENCE = {
    "lam0": {"m_op1": -0.03, "alpha_op1": 0.008, "m_span": 0.015, "alpha_span": 0.034},
    "end90-hold30": {
        "m_op1": 0.61,
        "alpha_op1": 0.087,
        "retention": 0.97,
        "r2": 0.77,
        "m_span": 0.64,
        "alpha_span": 0.37,
    },
}
"""Ex-2.1.8's control and operating point: means of per-run statistics,
recomputed from that experiment's published metrics and arrays (2026-08-05),
including the two pooled statistics defined above, which ex-2.1.8 itself did
not report. Per-run first, then the mean — the same convention results are
scored under, since both pooled statistics take a max over roles and computing
them on a seed-mean map would suppress the max-of-noise inflation the control
subtraction is there to absorb. The `span-mean` arm here re-runs
`end90-hold30`, so these are its reproduction targets; the calibration cell in
`report.py` re-derives them from the store rather than trusting this dict."""

EX217_OP1_GAIN = 0.22
"""Ex-2.1.7's op1-only margin gain over the span pull at the old schedule — the
scale H2(a)'s gate is read against."""

# --- Result refs ------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.9/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.9/arrays"
EVALS_REF = "reports/m2/ex-2.1.9/evals"
PROBE_REF = "reports/m2/ex-2.1.9/probes"
CKPT_REF = "reports/m2/ex-2.1.9/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.9/geometry"

EX218_METRICS_REF = "reports/m2/ex-2.1.8/metrics"
EX218_ARRAYS_REF = "reports/m2/ex-2.1.8/arrays"
"""Ex-2.1.8's published results, read by the τ calibration cell and the
reproduction check."""

__all__ = ["experiment"]


# --- The DAG ----------------------------------------------------------------


def prepare_corpus(grid: str, red_rate: float, n_examples: int, holdout_frac: float) -> dict:
    """Build the corpus, the eval sets, the alignment probe set, and the label map.

    Ex-2.1.8's prep step verbatim, down to the corpus seed, so the control cell
    here and the published controls there see the same lines. The design
    constants arrive as arguments rather than being read off the module, so
    changing one re-runs this step instead of silently reusing a corpus built
    under the old value.
    """
    import numpy as np

    from sca import baselines
    from sca.colorcube import redness as colorcube_redness
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, dump_example_sets, mix
    from mini.store import put

    levels = GRIDS[grid]
    palette = nc.grid_palette(levels)
    names = {v: k for k, v in palette.items()}
    corpus = nc.sample_corpus(n_examples, CORPUS_SEED, levels, holdout_frac)

    words = [w for ex in corpus for w in nc.as_words(ex)]
    tokenizer_config = TokenizerConfig(vocabulary=[*nc.SYNTAX, *palette])
    tokenizer = nc.WordTokenizer(tokenizer_config)
    tokens = np.asarray(tokenizer.encode_words(words), dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=sum(map(len, words)),
        sources=[DatasetMetadata(title=f"named-only color corpus ({grid})", fixes=[], total_chars=len(words))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    held = nc.holdout_test(levels, CORPUS_SEED, holdout_frac)
    seen = sorted({p for ex in corpus if (p := ex.pair) is not None})
    holdout = [p for p in nc.closed_pairs(levels) if held(*p)]
    opens = nc.open_pairs(levels)
    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, r: [pairs[i] for i in r.permutation(len(pairs))[:N_EVAL]]  # noqa: E731
    sets = {
        "named_seen": [nc.make_example(a, b, names, rng) for a, b in pick(seen, rng)],
        "named_holdout": [nc.make_example(a, b, names, rng) for a, b in pick(holdout, rng)],
        "open": [nc.make_example(a, b, names, rng) for a, b in pick(opens, rng)],
    }

    # The alignment probe set: every color as op1, against each of its closed
    # partners. Exhaustive, so there is no sampling seed and every cell is
    # measured on identical lines.
    colors = list(palette.values())
    on_grid = set(levels)
    probe_lines, partners = [], []
    for c in colors:
        closed = [b for b in colors if all(v in on_grid for v in mix(c, b))]
        partners.append(len(closed))
        for b in closed:
            probe_lines.append([names[c], "+", names[b], "=", names[mix(c, b)], "\n"])
    assert len(set(partners)) == 1, f"probe set is ragged: {sorted(set(partners))}"
    assert partners[0] == N_PROBE, f"expected {N_PROBE} partners per color, got {partners[0]}"
    probe_tokens = np.array([tokenizer.encode_words(line) for line in probe_lines], dtype=np.int32)

    # P(labeled) by token id: the label draw reads the first operand alone. The
    # affinity is recomputed here from the palette, and checked against the
    # design constants the report draws its figures from.
    rgb = np.array(list(palette.values()), dtype=float) / (N_LEVELS - 1)
    affinity = colorcube_redness(rgb) ** 8 * red_rate
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(affinity, LABEL_P, rtol=1e-12, atol=0)
    label_p = np.zeros(tokenizer.vocab_size, dtype=np.float64)  # the tokenizer's, which counts the pad token
    for name, p in zip(palette, affinity, strict=True):
        label_p[tokenizer.stoi[name]] = p
    weights = affinity / affinity.sum()

    # Model-free references for the distance rows (they never read the model).
    vocab_rgb = np.array(colors)
    train_dists = baselines.distances(vocab_rgb, [ex.result for ex in sets["named_seen"]])
    blind = baselines.blind_index(train_dists)
    nulls = {}
    for set_name, exs in sets.items():
        dists = baselines.distances(vocab_rgb, [ex.result for ex in exs])
        shell = baselines.shell_mask(levels, vocab_rgb, [ex.result for ex in exs])
        nulls[set_name] = {
            "blind": baselines.blind_stats(dists, blind),
            "k2": baselines.k_nearest_stats(dists, 2),
            "chance_dist": float(dists.mean()),
            "floor_dist": float(dists.min(axis=1).mean()),
            "neighborhood_exact": baselines.neighborhood_exact_null(shell),
        }

    stats = {
        "n_colors": len(palette),
        "n_seen_distinct": len(seen),
        "n_holdout": len(holdout),
        "n_open": len(opens),
        "n_partners": partners[0],
        "total_tokens": int(len(tokens)),
        "eval_n": {k: len(v) for k, v in sets.items()},
        "label_rate": float(affinity.mean()),
        "nulls": nulls,
        "blind_index": blind,
    }
    return {
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.9-{grid}-evals.json"),
        "probes": put(
            _npz(probe_tokens=probe_tokens, label_p=label_p, weights=weights, redness=colorcube_redness(rgb)),
            name=f"ex-2.1.9-{grid}-probes.npz",
        ),
    }


def _npz(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only the anchor pooling differs between cells."""
    from sca.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=BLOCK,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=256,
            n_layer=4,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=BATCH, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SCHEDULER,
        seed=seed,
    )


def build_sweep(prep: dict) -> list[tuple]:
    """(config, anchor, anti, tau, condition, label) cells; cheap and deterministic per wake.

    Both schedules arrive as plain dicts of every keyframe, so each cell's memo
    key carries the whole shape of its pull and its repulsion — and now also
    its pooling: `tau` rides in the anchor dict, so the τ sweep is visible to
    the memo key the same way the schedules are. Every anchored cell carries
    the anti-subspace term at ex-2.1.8's operating point; the control gets
    `anti=None`, the same path every control since ex-2.1.6 ran down.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    cells = []
    for c in CONDITIONS:
        anchor = {
            "peak": c["lam"],
            "warmup_epochs": SCHEDULER.warmup_epochs,
            "anneal_start": ANNEAL_START,
            "anneal_end": ANNEAL_END,
            "floor": ANNEAL_FLOOR,
            "span": c["span"],
            "tau": float(c["tau"]),
        }
        anti = (
            {
                "lam": c["lam"],
                "peak_ratio": ANTI_PEAK_RATIO,
                "hold_ratio": ANTI_HOLD_RATIO,
                "anneal_end": ANTI_ANNEAL_END,
                "anchor_anneal_start": ANNEAL_START,
                "anchor_anneal_end": ANNEAL_END,
                "floor": ANNEAL_FLOOR,
            }
            if c["lam"] > 0
            else None
        )
        for seed in SEEDS:
            config = _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed), tc)
            cells.append((config, anchor, anti, float(c["tau"]), c["name"], f"{c['name']}-s{seed}"))
    return cells


def _with_tokenizer(config, tokenizer_config):
    config.tokenizer = tokenizer_config.model_copy()
    return config


def train_one(config, anchor: dict, anti: dict | None, grid: str, traj_stride: int, probes, label: str) -> dict:
    """Train one cell with its pooled pull and fixed repulsion, recording the trajectory.

    The trajectory now carries m_span and the softmin weight profile beside
    m_op1 (`sca.compute.training.train_anchored`), so the exploratory question
    of *when* concentration happens is a stored summary rather than a re-run.
    """
    import numpy as np

    from sca.anchoring import AnchorSpec, AntiSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, label_p, weights = z["probe_tokens"], z["label_p"], z["weights"]

    # One line per color for the trajectory. At op1 the partner cannot matter;
    # at the other span roles it can, so the trajectory's span statistics read
    # one fixed partner per color where the endpoint eval averages all 27 —
    # cheap and consistent within a run, which is what retention compares.
    first_of_color = probe_tokens[:: len(probe_tokens) // len(weights)]

    _, metrics, traj = train_anchored(
        config,
        get_data_dir() / "corpora" / grid,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=label_p,
        probe_tokens=first_of_color,
        probe_weights=weights,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: v.tolist() for k, v in traj.items()},
        "checkpoint": put(workdir / "model", name=f"ex-2.1.9-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes, grid: str, tau: float, condition: str, label: str) -> dict:
    """Behavior, the alignment map, leakage, and the softmin weight profile for one cell.

    Ex-2.1.8's eval with the pooled statistics added: m_span and ᾱ_span from
    the same alignment map (via `pooled_margin` / `alpha_span`), and the weight
    profile π̄ — softmin weights at the cell's τ per probe line, averaged over
    each color's 27 partners, then label-affinity-weighted over colors.
    """
    import numpy as np

    from sca.anchoring import ANCHOR_AXIS, alignment, margin
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS, load_example_sets
    from sca.data.named_colors import GRIDS, WordTokenizer, grid_palette
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = WordTokenizer(config.tokenizer)

    palette = grid_palette(GRIDS[grid])
    names = list(palette)
    name_idx = {n: i for i, n in enumerate(names)}
    color_ids = np.array([tokenizer.stoi[n] for n in names])
    vocab_rgb = np.array(list(palette.values()), dtype=np.float32) / (N_LEVELS - 1)

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, weights, red = z["probe_tokens"], z["weights"], z["redness"]

    # --- Behavior: one teacher-forced pass per eval set, read at the pre-answer position.
    sets, arrays = {}, {}
    for set_name, exs in load_example_sets(get(evals, workdir / "evals.json").read_bytes()).items():
        logp = _answer_logprobs(model, tokenizer, exs)[:, color_ids]
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        dists = np.linalg.norm(vocab_rgb[None] - result[:, None], axis=2)  # (N, V)
        guess = logp.argmax(axis=1)
        guess_dist = dists[np.arange(len(exs)), guess]
        p = np.exp(logp)
        floor = dists.min(axis=1)
        stats: dict = {
            "n": len(exs),
            "guess_dist": float(guess_dist.mean()),
            "exp_dist": float((p / p.sum(1, keepdims=True) * dists).sum(1).mean()),
            "floor_dist": float(floor.mean()),
            "nearest_acc": float((guess_dist <= floor + 1e-9).mean()),
            "p_offvocab": float((1 - p.sum(1)).mean()),
        }
        if exs[0].answer:  # closed sets: the true answer is a vocabulary name
            true_idx = np.array([name_idx[ex.answer] for ex in exs])
            stats["accuracy"] = float((guess == true_idx).mean())
            stats["nll"] = float(-logp[np.arange(len(exs)), true_idx].mean())
            stats["failures"] = [
                (ex.prompt, ex.answer, names[g]) for ex, g, t in zip(exs, guess, true_idx, strict=True) if g != t
            ][:8]
        sets[set_name] = stats
        arrays[f"logp/{set_name}"] = logp.astype(np.float32)

    # --- Alignment: cos(h, e₀) per layer × position, averaged over each color's partners.
    n_partners = len(probe_tokens) // len(weights)
    cos = alignment(model, probe_tokens)  # (L1, C*P, T)
    alpha = cos.reshape(cos.shape[0], len(weights), n_partners, cos.shape[2]).mean(axis=2)  # (L1, C, T)
    m = margin(alpha, weights)  # (L1, T)
    arrays["alpha"] = alpha.astype(np.float32)
    arrays["margin"] = m.astype(np.float32)

    # --- The weight profile: what the pull chose, at this cell's τ. Weights are
    #     computed per probe line (pooling is per line in the loss too), then
    #     averaged over partners and label-affinity-weighted over colors.
    x = 1.0 - cos[:, :, :SPAN]
    w_line = softmin_weights(x, tau)  # (L1, C*P, SPAN)
    w_color = w_line.reshape(cos.shape[0], len(weights), n_partners, SPAN).mean(axis=2)
    pi = np.einsum("lct,c->lt", w_color, weights)
    np.testing.assert_allclose(pi.sum(axis=-1), 1.0, rtol=0, atol=1e-5)  # a mean of simplex weights stays on it
    arrays["pi"] = pi.astype(np.float32)

    # --- Leakage: redness from the residual stream at op1, with the anchor axis removed.
    #     At op1 a color's state is context-free, so the 216 colors are the whole sample.
    states = _op1_states(model, probe_tokens[::n_partners])  # (L1, C, D)
    off_axis = np.delete(states, ANCHOR_AXIS, axis=2)
    leak = [_cv_ridge_r2(off_axis[d], red) for d in range(off_axis.shape[0])]
    on_axis = [float(np.corrcoef(states[d, :, ANCHOR_AXIS], red)[0, 1] ** 2) for d in range(states.shape[0])]

    return {
        "label": label,
        "condition": condition,
        "tau": tau,
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "traj": trained["traj"],
        "sets": sets,
        "m_op1": float(m[:, 0].mean()),
        "m_op1_by_layer": m[:, 0].tolist(),
        "m_by_position": m.mean(axis=0).tolist(),
        "m_span": pooled_margin(alpha, weights),
        "alpha_mean_op1": float(alpha[:, :, 0].mean()),
        "alpha_span": alpha_span(alpha),
        "pi": pi.tolist(),
        "leak_r2": leak,
        "axis_r2": on_axis,
        "arrays": put(_npz(**arrays), name=f"ex-2.1.9-{label}-arrays.npz"),
    }


def _answer_logprobs(model, tokenizer, exs) -> "np.ndarray":
    """Full-vocabulary log-probabilities at the pre-answer position, per example."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    forward = eqx.filter_jit(model.__call__)
    seq = np.array([tokenizer.encode_words(ex.prompt.split()) for ex in exs])
    out = [
        np.asarray(jax.nn.log_softmax(forward(jnp.asarray(seq[i : i + 256]))[:, -1], axis=-1))
        for i in range(0, len(seq), 256)
    ]
    return np.concatenate(out)


def _op1_states(model, lines) -> "np.ndarray":
    """The residual stream at op1 for one line per color: (L1, C, D)."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    stream = eqx.filter_jit(model.residual_stream)
    return np.asarray(stream(jnp.asarray(lines)))[:, :, 0, :]


def _span_states(model, lines, batch_size: int = 1024) -> "np.ndarray":
    """The residual stream at every span role, batched: (L1, N, SPAN, D)."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    stream = eqx.filter_jit(model.residual_stream)
    chunks = [
        np.asarray(stream(jnp.asarray(lines[i : i + batch_size])))[:, :, :SPAN, :]
        for i in range(0, len(lines), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


def _cv_ridge_r2(x, y, n_folds: int = 5, penalties=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> float:
    """Held-out R² for a ridge probe, with the penalty chosen inside each training fold.

    No color is scored by a probe that saw it, and no penalty is picked by
    looking at the number reported. The split is a fixed interleave rather than
    a draw, so the estimate has no seed.
    """
    import numpy as np

    from sca.compute.evaluation import ridge_probe

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1, 1)
    fold = np.arange(len(x)) % n_folds
    preds = np.empty_like(y)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        inner = np.arange(tr.sum()) % n_folds
        best = max(
            penalties,
            key=lambda l2: float(
                np.mean([ridge_probe(x[tr][inner != j], y[tr][inner != j], x[tr][inner == j], y[tr][inner == j], l2)[2]
                         for j in range(n_folds)])
            ),
        )  # fmt: skip
        w, b, _ = ridge_probe(x[tr], y[tr], x[te], y[te], best)
        preds[te] = x[te] @ w + b
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _loo_color_r2(x, y_color, n_partners: int, penalties=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> float:
    """Held-out R² for a ridge probe over per-line states, leaving one color out at a time.

    Rows arrive grouped by color, *n_partners* lines each, and every line of a
    color shares its target (the redness of op1), so the held-out unit has to
    be the color: at op1 the partner lines of a color are identical states, and
    a per-row split would test a probe on rows it trained on. The penalty is
    chosen one level up — once per profile entry, by an interleaved 5-group CV
    over the same colors — so the held-out fits share a penalty that every
    color had a say in, while no fit is tested on a color it trained on.
    """
    import numpy as np

    from sca.compute.evaluation import ridge_probe

    x = np.asarray(x, dtype=np.float64)
    y = np.repeat(np.asarray(y_color, dtype=np.float64), n_partners).reshape(-1, 1)
    n_colors = len(y_color)
    color = np.repeat(np.arange(n_colors), n_partners)
    group = color % 5  # fixed interleave over colors: no seed
    best = max(
        penalties,
        key=lambda l2: float(
            np.mean([ridge_probe(x[group != g], y[group != g], x[group == g], y[group == g], l2)[2] for g in range(5)])
        ),
    )
    preds = np.empty_like(y)
    for c in range(n_colors):
        tr, te = color != c, color == c
        w, b, _ = ridge_probe(x[tr], y[tr], x[te], y[te], best)
        preds[te] = x[te] @ w + b
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def readability_profile(checkpoints: dict, probes, grid: str) -> dict:
    """R²(l, t): how well each (slice, role) of the *control* reads op1's redness.

    The reference profile H4(b) scores the weight profile against, computed for
    each un-anchored seed over the full probe set (per-line states, one color
    held out at a time — `_loo_color_r2`). At the embedding every role but op1
    should carry nothing: a token's state there cannot depend on another
    position, which is the fact H4(a)'s prediction rests on.
    """
    import numpy as np

    from sca.colorcube import redness as colorcube_redness
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS
    from sca.data.named_colors import GRIDS, grid_palette
    from mini.store import get

    palette = grid_palette(GRIDS[grid])
    rgb = np.array(list(palette.values()), dtype=np.float64) / (N_LEVELS - 1)
    red = colorcube_redness(rgb)

    workdir = get_data_dir() / "readability"
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["probe_tokens"]
    n_partners = len(probe_tokens) // len(rgb)

    out: dict[str, list] = {}
    for label, ckpt in checkpoints.items():
        get(ckpt, workdir / label / "model")
        model, _, _ = load_checkpoint(workdir / label)
        h = _span_states(model, probe_tokens)  # (L1, C*P, SPAN, D)
        out[label] = [[_loo_color_r2(h[d, :, t], red, n_partners) for t in range(SPAN)] for d in range(h.shape[0])]
    return out


def publish_results(results: list[dict], stats: dict, readability: dict, checkpoints: dict, evals, probes) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), the eval and probe sets."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: (None if k == "tau" and np.isinf(v) else v) for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {
        "cells": cells,
        "corpus_stats": stats,
        "readability": readability,
        # JSON has no ∞, so the τ = ∞ arms carry null here; the report reads the
        # grid off the experiment module, where the value is np.inf.
        "conditions": [{**c, "tau": None if np.isinf(c["tau"]) else c["tau"]} for c in CONDITIONS],
        "design": {
            "seeds": SEEDS,
            "scoring_lambda": SCORING_LAMBDA,
            "span": SPAN,
            "taus": TAUS,
            "lead_band": LEAD_BAND,
            "primary": PRIMARY,
            "pooled": POOLED_NAMES,
            "mean_arm": MEAN_ARM,
            "oracle_arm": ORACLE_ARM,
            "retention_stat": RETENTION_STAT,
            "anti": {
                "peak_ratio": ANTI_PEAK_RATIO,
                "anneal_end": ANTI_ANNEAL_END,
                "hold_ratio": ANTI_HOLD_RATIO,
            },
            "anneal": {"start": ANNEAL_START, "end": ANNEAL_END, "floor": ANNEAL_FLOOR},
            "traj_stride": TRAJ_STRIDE,
            "noise": {"per_seed": NOISE_PER_SEED, "seed_mean": NOISE_SEED_MEAN},
            "ex218_reference": EX218_REFERENCE,
            "ex217_op1_gain": EX217_OP1_GAIN,
            "gates": {
                "task": TASK_GATE,
                "pool_gain": POOL_GAIN_GATE,
                "pool_gain_partial": POOL_GAIN_PARTIAL,
                "syntax_drift": SYNTAX_DRIFT_GATE,
                "mean_align": MEAN_ALIGN_GATE,
                "retention_floor": RETENTION_FLOOR,
                "retention": RETENTION_GATE,
                "grade_r2_drop": GRADE_R2_DROP,
                "lead": LEAD_GATE,
                "read_gain": READ_GAIN_GATE,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.9-metrics.json"))
    set_ref(EVALS_REF, evals)
    set_ref(PROBE_REF, probes)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    for label, profile in readability.items():
        arrays[f"{label}/readability"] = np.asarray(profile, dtype=np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.9-arrays.npz"))
    return {"n_cells": len(results)}


def measure_geometry(checkpoints: dict, probes, grid: str) -> dict:
    """The shape of the color cloud, and what is special about redness off the anchor axis.

    Carried from ex-2.1.7/8, where it is the mechanism picture behind
    containment: if the repulsion is doing its job, the centroid of the 216 op1
    states sits off the anchor and the cloud keeps its extent. The leakage
    probe is repeated for the other color statistics because redness is a
    function of color and the task needs the color cube; greenness, blueness
    and the raw channels beside it tell "red survived" apart from "color
    survived".
    """
    import numpy as np

    from sca.anchoring import ANCHOR_AXIS
    from sca.colorcube import redness as colorcube_redness
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS
    from sca.data.named_colors import GRIDS, grid_palette
    from mini.store import get

    palette = grid_palette(GRIDS[grid])
    rgb = np.array(list(palette.values()), dtype=np.float64) / (N_LEVELS - 1)
    roll = lambda k: colorcube_redness(np.roll(rgb, -k, axis=1))  # noqa: E731 — same formula, other corner
    targets = {
        "redness": roll(0),
        "greenness": roll(1),  # [r,g,b] -> [g,b,r], so the same formula reads green
        "blueness": roll(2),
        "R": rgb[:, 0],
        "G": rgb[:, 1],
        "B": rgb[:, 2],
        "sim^1.5": SIM_TARGET.astype(np.float64),
    }

    workdir = get_data_dir() / "explore"
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["probe_tokens"]
    per_color = probe_tokens[:: len(probe_tokens) // len(rgb)]

    out: dict[str, dict] = {}
    for label, ckpt in checkpoints.items():
        get(ckpt, workdir / label / "model")
        model, _, _ = load_checkpoint(workdir / label)
        h = _op1_states(model, per_color).astype(np.float64)  # (L1, C, D)
        off = np.delete(h, ANCHOR_AXIS, axis=2)
        on = h[:, :, ANCHOR_AXIS]
        centre = h.mean(axis=1)  # (L1, D)
        centred = h - centre[:, None]
        out[label] = {
            "mean_norm": np.linalg.norm(centre, axis=1).tolist(),
            "spread": (np.einsum("lcd,lcd->l", centred, centred) / h.shape[1]).tolist(),
            "centre_dot_anchor": (centre[:, ANCHOR_AXIS] / np.linalg.norm(centre, axis=1)).tolist(),
            "off_axis_r2": {t: [_cv_ridge_r2(off[d], y) for d in range(off.shape[0])] for t, y in targets.items()},
            "on_axis_r2": {
                t: [float(np.corrcoef(on[d], y)[0, 1] ** 2) for d in range(on.shape[0])] for t, y in targets.items()
            },
        }
    return out


def publish_geometry(geometry: dict) -> dict:
    """Publish the geometry pass under its own ref, beside the metrics."""
    import json

    from mini.store import put, set_ref

    set_ref(GEOMETRY_REF, put(json.dumps(geometry, indent=2).encode(), name="ex-2.1.9-geometry.json"))
    return {"n_cells": len(geometry)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, RED_RATE, N_EXAMPLES, HOLDOUT_FRAC, role="prep")
    configs, anchors, antis, taus, conditions, labels = zip(*build_sweep(prep), strict=True)
    n = len(labels)
    trained = ctx.map(
        train_one,
        configs,
        anchors,
        antis,
        [GRID] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        labels,
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [prep["evals"]] * n,
        [prep["probes"]] * n,
        [GRID] * n,
        taus,
        conditions,
        labels,
        role="eval",
    )
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    controls = {label: ckpts[label] for label in ckpts if label.startswith("lam0")}
    readability = ctx.run(readability_profile, controls, prep["probes"], GRID, role="probe")
    summary = ctx.run(
        publish_results, evaled, prep["stats"], readability, ckpts, prep["evals"], prep["probes"], role="prep"
    )
    geometry = ctx.run(measure_geometry, ckpts, prep["probes"], GRID, role="eval")
    ctx.run(publish_geometry, geometry, role="prep")

    def by_condition(fn) -> dict:
        return {
            c["name"]: round(float(np.mean([fn(r) for r in evaled if r["condition"] == c["name"]])), 3)
            for c in CONDITIONS
        }

    return {
        **summary,
        "holdout_accuracy": by_condition(lambda r: r["sets"]["named_holdout"]["accuracy"]),
        "m_span": by_condition(lambda r: r["m_span"]),
        "alpha_span": by_condition(lambda r: r["alpha_span"]),
        "alpha_mean_op1": by_condition(lambda r: r["alpha_mean_op1"]),
    }


experiment = Experiment(
    name="ex-2.1.9",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the probe set is trivial beside it.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens. Both regularizer terms read every residual
        # slice, and the trajectory adds a probe pass every 50 steps (now over
        # the span rather than op1 alone — same forward, wider read).
        #
        # The watchdog is sized for the gap *after* the last step, not between
        # steps: the loop emits at ~450 steps/min, so a wedged run is caught in
        # seconds either way, but the checkpoint upload that follows emits no
        # step progress and took over 300s in ex-2.1.7 when eight containers
        # pushed to the store at once (todo-eng).
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # Three teacher-forced eval sets, one probe pass, and 5 × 5-fold ridge probes on 216 rows.
        # The geometry step runs the same probes over all 18 checkpoints, so it needs longer.
        "eval": dict(gpu="L4", timeout=2700),
        # The readability profile: 20 (slice, role) ridge probes per control
        # seed, each with 216 leave-one-color-out fits on 5 832 lines. CPU-bound
        # linear algebra; the forward pass is small enough not to need a GPU.
        "probe": dict(cpu=4, timeout=2700),
    },
)
