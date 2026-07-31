"""Experiment 2.1.7: a repulsive term and a narrower pull for the anchored transformer.

Ex-2.1.6 found that a bare anchor term buys alignment without selectivity, and
left two candidate mechanisms confounded: no repulsive term constrained the
cube-wide drift, and three of the four pulled positions cannot see which color
sits at op1. This experiment separates them with a 2×2 factorial at the
ex-2.1.6 scoring rung — {bare, + anti-subspace} × {prompt-span pull, op1-only
pull} — plus a schedule-timing arm and a weight-ceiling arm. See `report.py`
for the preregistration.

The testbed, labels, measurements, and anchor schedule are ex-2.1.6's,
unchanged. The anti-subspace term and its schedule are M1/ex-2.9.1's, with the
weight expressed relative to the anchor peak and the keyframes mapped onto our
100 epochs by fraction of training.

    bin/mini run docs/m2/ex-2.1.7/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.7
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import (
    LINE_TOKENS,
    PROMPT_SPAN,
    anchor_weight as _anchor_weight,
    anti_subspace_weight as _anti_subspace_weight,
)
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule
from mini import Ctx, Experiment, get_data_dir

# --- Testbed, carried from ex-2.1.6 unchanged -------------------------------

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

# --- The factorial ----------------------------------------------------------

SCORING_LAMBDA = 0.1
"""Anchor peak for every factorial cell — ex-2.1.6's scoring rung, for comparability."""

CEILING_LAMBDA = 1.0
"""The weight-ceiling arm: one rung above ex-2.1.6's swept maximum of 0.3."""

SPAN_FULL = PROMPT_SPAN  # op1, `+`, op2, `=` — ex-2.1.6's pull
SPAN_OP1 = 1  # op1 alone: every pulled position sees the labeled color

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet: anti-subspace 0.25 against an anchor peak of 0.1."""

ANTI_HOLD_RATIO = 0.03
"""μ/λ from `ANTI_ANNEAL_END` on — M1's 0.003 against 0.1, the ratio ex-2.1.6's discussion cites."""

ANTI_ANNEAL_END = 50.0
"""Epoch at which μ reaches its hold ratio — M1's step 750 of 1500, mapped by fraction of training."""

ANTI_ANNEAL_END_LATE = 90.0
"""The timing arm: the same anneal stretched to meet the end-of-training anneal."""

CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "span": SPAN_FULL, "anti": False},
    {"name": "span-bare", "lam": SCORING_LAMBDA, "span": SPAN_FULL, "anti": False},
    {"name": "span-anti", "lam": SCORING_LAMBDA, "span": SPAN_FULL, "anti": True},
    {"name": "op1-bare", "lam": SCORING_LAMBDA, "span": SPAN_OP1, "anti": False},
    {"name": "op1-anti", "lam": SCORING_LAMBDA, "span": SPAN_OP1, "anti": True},
    {
        "name": "span-anti-late",
        "lam": SCORING_LAMBDA,
        "span": SPAN_FULL,
        "anti": True,
        "anti_anneal_end": ANTI_ANNEAL_END_LATE,
    },
    {"name": "span-anti-hi", "lam": CEILING_LAMBDA, "span": SPAN_FULL, "anti": True},
]
"""`lam0` re-runs the control and `span-bare` re-runs ex-2.1.6's λ=0.1 cell, so every
comparison in this report is between cells that went through identical code."""

FACTORIAL = ["span-bare", "span-anti", "op1-bare", "op1-anti"]
PRIMARY = "span-anti"
"""The primary scoring cell: ex-2.1.6's pull with M1's repulsive company — the
condition the ex-2.1.6 discussion named as the next experiment."""

# --- Schedules --------------------------------------------------------------


def anchor_weight(epoch, *, peak: float = SCORING_LAMBDA, anneal_start: float = ANNEAL_START) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*: ramp, hold, anneal to a floor. As ex-2.1.6."""
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=SCHEDULER.warmup_epochs,
        anneal_start=anneal_start,
        anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def anti_subspace_weight(epoch, *, lam: float = SCORING_LAMBDA, anneal_end: float = ANTI_ANNEAL_END) -> np.ndarray:
    """The anti-subspace weight μ at (fractional) *epoch*.

    M1/ex-2.9.1's shape: full strength from step 0 — before the anchor has
    ramped in — then a minimum-jerk anneal to the hold ratio by *anneal_end*.
    The end-of-training anneal is the anchor's, applied to both terms, so the
    ratio between them is constant from *anneal_end* to the end.

    This is the schedule the run itself uses, reached through the same library
    function, so the figure in the report cannot drift from the training loop.
    """
    return _anti_subspace_weight(
        epoch,
        lam=lam,
        peak_ratio=ANTI_PEAK_RATIO,
        hold_ratio=ANTI_HOLD_RATIO,
        anneal_end=anneal_end,
        anchor_anneal_start=ANNEAL_START,
        anchor_anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def learning_rate(epoch, *, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The ex-2.1.3 LR schedule, evaluated at (fractional) *epoch*. As ex-2.1.6."""
    schedule = configure_schedule(SCHEDULER, peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))


# --- Gates ------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means)."""

MARGIN_GATE = 0.5
"""H2(a): the layer-mean margin at op1, ex-2.1.6's threshold unchanged."""

GRADE_RHO_GATE = 0.8
GRADE_R2_GATE = 0.8
"""H2(b): the two grading tracks, ex-2.1.6's thresholds unchanged."""

CONTROL_MARGIN_GATE = 0.1
"""H2(c): |m_op1| in the control."""

MAIN_EFFECT_GATE = 0.1
"""H3: the smallest margin main effect scored as real — ~3× the noise on a main effect.

A main effect is a difference of two two-condition means, so its noise is about
the same as a single three-seed mean (`NOISE_SEED_MEAN`), and 0.1 / 0.032 ≈ 3.
"""

MEAN_ALIGN_GATE = 0.1
"""H4(a): ceiling on the mean alignment over all colors at op1 in the anti conditions.

Ex-2.1.6's bare term reached 0.53 at this rung, against 0.008 in its control
(seed spread ≈ 0.02) — this statistic averages over 216 colors and 5 layers,
so an unanchored cloud sits near zero rather than at the 1/8 spread of a
single cosine. Working repulsion should push the mean to near-control levels:
0.1 is a fifth of the bare value and still five noise units above control, so
a working mechanism passes comfortably and a merely-weakened runaway does not.
"""

MEAN_ALIGN_PARTIAL = 0.25
"""H4(a) partial: the earlier halving-level gate, kept as a named partial.

Clearing 0.25 without reaching `MEAN_ALIGN_GATE` reads as repulsion weakening
the runaway without containing it.
"""

H4_FLOOR = 0.2
H4_RETENTION = 0.8
"""H4(b): trajectory retention, ex-2.1.6's H4 gate unchanged (see its noise analysis)."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors carried from ex-2.1.6 — same architecture, same measurement."""

# --- Published refs ---------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.7/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.7/arrays"
EVALS_REF = "reports/m2/ex-2.1.7/evals"
PROBE_REF = "reports/m2/ex-2.1.7/probes"
CKPT_REF = "reports/m2/ex-2.1.7/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.7/geometry"  # preregistered here (post-hoc in ex-2.1.6)

# --- Statistics shared with the report --------------------------------------


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with midranks for ties: Pearson correlation of the rank vectors."""
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


def r2_sim(alpha: np.ndarray) -> float:
    """Pearson R² between a per-color response and SIM_TARGET — M1's proportionality score."""
    return float(np.corrcoef(alpha, SIM_TARGET)[0, 1] ** 2)


LINES_PER_BATCH = BLOCK * BATCH / LINE_TOKENS
"""Lines a batch carries; batch composition is ex-2.1.6's, unchanged."""

__all__ = ["experiment"]


# --- The DAG ----------------------------------------------------------------


def prepare_corpus(grid: str, red_rate: float, n_examples: int, holdout_frac: float) -> dict:
    """Build the corpus, the eval sets, the alignment probe set, and the label map.

    Ex-2.1.6's prep step verbatim, down to the corpus seed, so the control cell
    here and the published control there see the same lines. The design
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
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.7-{grid}-evals.json"),
        "probes": put(
            _npz(probe_tokens=probe_tokens, label_p=label_p, weights=weights, redness=colorcube_redness(rgb)),
            name=f"ex-2.1.7-{grid}-probes.npz",
        ),
    }


def _npz(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only the anchor terms differ between cells."""
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
    """(config, anchor, anti, condition, label) cells; cheap and deterministic per wake.

    Both schedules arrive as plain dicts of every keyframe rather than a peak
    plus module constants, so each cell's memo key carries the whole shape of
    its pull and its repulsion. `anti` is None for the bare conditions, which is
    what the training loop reads as "no repulsive term".
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
        }
        anti = (
            {
                "lam": c["lam"],
                "peak_ratio": ANTI_PEAK_RATIO,
                "hold_ratio": ANTI_HOLD_RATIO,
                "anneal_end": c.get("anti_anneal_end", ANTI_ANNEAL_END),
                "anchor_anneal_start": ANNEAL_START,
                "anchor_anneal_end": ANNEAL_END,
                "floor": ANNEAL_FLOOR,
            }
            if c["anti"]
            else None
        )
        for seed in SEEDS:
            config = _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed), tc)
            cells.append((config, anchor, anti, c["name"], f"{c['name']}-s{seed}"))
    return cells


def _with_tokenizer(config, tokenizer_config):
    config.tokenizer = tokenizer_config.model_copy()
    return config


def train_one(config, anchor: dict, anti: dict | None, grid: str, traj_stride: int, probes, label: str) -> dict:
    """Train one cell with its two regularizer schedules, recording the trajectory."""
    import numpy as np

    from sca.anchoring import AnchorSpec, AntiSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, label_p, weights = z["probe_tokens"], z["label_p"], z["weights"]

    # One line per color for the trajectory: at op1 the partner cannot matter,
    # so the cheap probe reads the same number as the exhaustive one.
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
        "checkpoint": put(workdir / "model", name=f"ex-2.1.7-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes, grid: str, condition: str, label: str) -> dict:
    """Behavior, the alignment map, and off-axis leakage for one cell. As ex-2.1.6."""
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

    # --- Leakage: redness from the residual stream at op1, with the anchor axis removed.
    #     At op1 a color's state is context-free, so the 216 colors are the whole sample.
    states = _op1_states(model, probe_tokens[::n_partners])  # (L1, C, D)
    off_axis = np.delete(states, ANCHOR_AXIS, axis=2)
    leak = [_cv_ridge_r2(off_axis[d], red) for d in range(off_axis.shape[0])]
    on_axis = [float(np.corrcoef(states[d, :, ANCHOR_AXIS], red)[0, 1] ** 2) for d in range(states.shape[0])]

    return {
        "label": label,
        "condition": condition,
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "traj": trained["traj"],
        "sets": sets,
        "m_op1": float(m[:, 0].mean()),
        "m_op1_by_layer": m[:, 0].tolist(),
        "m_by_position": m.mean(axis=0).tolist(),
        "alpha_mean_op1": float(alpha[:, :, 0].mean()),
        "leak_r2": leak,
        "axis_r2": on_axis,
        "arrays": put(_npz(**arrays), name=f"ex-2.1.7-{label}-arrays.npz"),
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


def measure_geometry(checkpoints: dict, probes, grid: str) -> dict:
    """The shape of the color cloud, and what is special about redness off the anchor axis.

    Exploratory in ex-2.1.6 and preregistered here, because it is the mechanism
    picture behind H4(a): if the repulsive term is doing its job, the centroid
    of the 216 op1 states sits off the anchor and the cloud keeps its extent.
    `mean_norm` is how concentrated the states are (1 = all identical) and
    `spread` is what extent they keep, so a cloud whose centroid swung onto the
    anchor keeps its spread while a collapsing one does not.

    The leakage probe is repeated for the other color statistics for the same
    reason as in ex-2.1.6: redness is a function of color and the task needs the
    color cube, so a high off-axis redness score may say only that the cube
    survived. Running greenness, blueness and the raw channels beside it is what
    tells those apart.
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

    set_ref(GEOMETRY_REF, put(json.dumps(geometry, indent=2).encode(), name="ex-2.1.7-geometry.json"))
    return {"n_cells": len(geometry)}


def publish_results(results: list[dict], stats: dict, checkpoints: dict, evals, probes) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), the eval and probe sets."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: v for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {
        "cells": cells,
        "corpus_stats": stats,
        "conditions": CONDITIONS,
        "design": {
            "seeds": SEEDS,
            "scoring_lambda": SCORING_LAMBDA,
            "ceiling_lambda": CEILING_LAMBDA,
            "factorial": FACTORIAL,
            "primary": PRIMARY,
            "spans": {"full": SPAN_FULL, "op1": SPAN_OP1},
            "anti": {
                "peak_ratio": ANTI_PEAK_RATIO,
                "hold_ratio": ANTI_HOLD_RATIO,
                "anneal_end": ANTI_ANNEAL_END,
                "anneal_end_late": ANTI_ANNEAL_END_LATE,
            },
            "anneal": {"start": ANNEAL_START, "end": ANNEAL_END, "floor": ANNEAL_FLOOR},
            "traj_stride": TRAJ_STRIDE,
            "gates": {
                "task": TASK_GATE,
                "margin": MARGIN_GATE,
                "grade_rho": GRADE_RHO_GATE,
                "grade_r2": GRADE_R2_GATE,
                "control_margin": CONTROL_MARGIN_GATE,
                "main_effect": MAIN_EFFECT_GATE,
                "mean_align": MEAN_ALIGN_GATE,
                "mean_align_partial": MEAN_ALIGN_PARTIAL,
                "h4_floor": H4_FLOOR,
                "h4_retention": H4_RETENTION,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.7-metrics.json"))
    set_ref(EVALS_REF, evals)
    set_ref(PROBE_REF, probes)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.7-arrays.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, RED_RATE, N_EXAMPLES, HOLDOUT_FRAC, role="prep")
    configs, anchors, antis, conditions, labels = zip(*build_sweep(prep), strict=True)
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
        conditions,
        labels,
        role="eval",
    )
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    summary = ctx.run(publish_results, evaled, prep["stats"], ckpts, prep["evals"], prep["probes"], role="prep")
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
        "m_op1": by_condition(lambda r: r["m_op1"]),
        "alpha_mean_op1": by_condition(lambda r: r["alpha_mean_op1"]),
    }


experiment = Experiment(
    name="ex-2.1.7",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the probe set is trivial beside it.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens. Both regularizer terms read every residual
        # slice, so a step costs more than ex-2.1.3's, and the trajectory adds a
        # probe pass every 50.
        #
        # The watchdog is sized for the gap *after* the last step, not between
        # steps: the loop emits at ~450 steps/min, so a wedged run is caught in
        # seconds either way, but the checkpoint upload that follows emits no
        # step progress and took over 300s when eight containers pushed to the
        # store at once.
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # Three teacher-forced eval sets, one probe pass, and 5 × 5-fold ridge probes on 216 rows.
        # The geometry step runs the same probes over all 21 checkpoints, so it needs longer.
        "eval": dict(gpu="L4", timeout=2700),
    },
)
