"""Ex-2.2.17: a scout on the plateau in ex-2.2.16's center control.

Ex-2.2.16's d64-L4 control at the center corpus condition (`k3-r0.3`, unanchored) scored 0.27 held-out expected
exact match against a Bayes ceiling of 0.52, its validation loss plateauing near 2.40 through the first half of
training and falling again only as the cosine schedule decayed its learning rate. Two readings compete: the
peak learning rate (0.01, ex-2.2.14's `_make_config`) is too high for the task, or the model sits on a
pre-in-context-learning plateau and needs more steps. This scout trains three arms of the same unanchored
control at fresh seeds — `long` (three times the length at the same peak learning rate), `long-low` (three
times the length at a lower peak learning rate), and `low` (the same length at the lower peak learning rate) —
and records the held-out expected exact match and calibration KL against training step, not just at the end,
so the two readings can be told apart.

Round 1 found that more steps help most and that the lower rate helps only with them, with every curve still
rising until the schedule wound down. Round 2 trains for eight times the length with ex-2.2.16's newline mask
and a warmup of fixed length, sweeping the peak learning rate at one seed each, after a learning-rate finder
on the same config sets the range (`ACTIVE_ROUNDS` holds round 2 back until the finder has run).

In round 2 the three HSV ops rose steeply once the decaying learning rate passed below about 0.004, while
the rates that never got that low (or got there too late) stayed on the plateau. Round 3 keeps the rate near
that level on purpose: a warmup-stable-decay schedule, written as a dopesheet, warms up to 0.00316, eases to
0.0025, holds there until 85% of the run, and then anneals to the same floor as the cosine. It trains at
three seeds, beside two more seeds of the cosine at the same peak, paired by model seed.

Round 3 ended level with the cosine, its curves flat through the hold and rising steeply in the final anneal.
Round 4 steps the rate down a staircase (0.0021, 0.001, 0.0001, then 1e-5 for the last 11% of the run) at the
same three seeds, to see what each drop is worth and whether the curves still climb at the lowest rate.

All three schedules ended near 0.45 at eight times, the seeds differing more than the schedules. Round 5 trains
each schedule for sixteen times the length, at one seed first, to see whether more steps move that level.
They did not (a gain of 0.002 to 0.006), so round 6 trains one seed of the wider d128-L4 model at eight times, and round 7 one seed of a deeper
d64-L6 model, since hsvmix (the op with the longest chain of steps) has stayed furthest below its ceiling.

    bin/mini run docs/m2/ex-2.2.17/experiment.py --app modal --max-containers 9 --budget 3h --keep-stale-done

`--keep-stale-done` since round 3: its schedule touched `SchedulerConfig` and the scheduler, which the code
fingerprint of every earlier run follows, though neither changes what those runs computed.
    bin/mini status ex-2.2.17
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.incontext import context_length


def _load_ex2216():
    """Ex-2.2.16's module, by path and left out of `sys.modules` (the `_load_ex2214` pattern that module itself
    uses), so this module's task bodies still cloudpickle by value for a remote worker. Reused for the grammar,
    the corpus condition, the recipe (`Condition229`, `schedules`, `_make_config`), and the end-of-training
    measurements (`eval_one` and its helpers). Loaded once here, at module import time, rather than inside a
    task body: ex-2.2.16 hit a bug loading a sibling by path from inside one.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.16" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2217_ex2216", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2216 = _load_ex2216()

# --- What is inherited, unchanged --------------------------------------------------------------------------

CENTRE = ex2216.CENTRE
"""The corpus condition every arm trains on: ex-2.2.16's center, `k3-r0.3`, reused rather than regenerated."""

CORPUS_KEY = ex2216.cond_key(*CENTRE)

EPOCHS = ex2216.EPOCHS
"""The length ex-2.2.16 trained the center control at (`low` and the base of `long`/`long-low`)."""

PEAK_LR = 0.01
"""The peak learning rate `_make_config` sets by default (Adam), which `long` keeps and `low`/`long-low` cut."""

LOW_LR = 0.003
"""The lower peak learning rate `low` and `long-low` train at."""

_baseline_config = ex2216._make_config(64, 0, 1, *ex2216.model_dims(ex2216.MODEL))
assert _baseline_config.optimizer.learning_rate == PEAK_LR, "PEAK_LR should match _make_config's own default"

SEED_OFFSET = 600
"""Ex-2.2.16 trained the center control at model seeds 500-502; every model seed here is fresh."""

SEEDS = 3

N_TRAJ_POINTS = 50
"""Target trajectory points per run (about every 2% of training): the stride is sized per arm so a run three
times as long still gets about this many records, rather than three times as many."""

N_TRAJ_EEM_PER_OP = 200
"""Held-out contexts per op the trajectory's expected-exact-match and calibration reads are taken on (2,200 in
all): the first this many of ex-2.2.16's `HOLDOUT_CONTEXTS` per op, a fixed subset held constant across every
record and every run."""


@dataclass(frozen=True)
class Arm:
    """One arm of the scout: how long it trains, at what peak learning rate, and (from round 2) with or without
    the newline mask and a warmup fixed in length.
    """

    name: str
    epoch_mult: int
    """The multiple of `EPOCHS` this arm trains for."""
    peak_lr: float
    note: str = ""
    seeds: int = SEEDS
    mask: bool = False
    """The newline mask of ex-2.2.16's `control-mask`: attention does not cross a line break."""
    warmup_epochs: float | None = None
    """A warmup of fixed length in epochs; `None` keeps ex-2.2.16's rule, a tenth of the run."""
    round: int = 1
    first_seed: int = 0
    """The first seed index; a promoted arm starts after the seeds an earlier round already trained."""
    lr_sheet: str | None = None
    """A dopesheet for the learning rate, as a multiple of `peak_lr` over the whole run (round 3); `None` keeps
    the warmup-and-cosine schedule."""
    model: str | None = None
    """A model size of the form `d<embd>-L<layer>` (round 6); `None` keeps ex-2.2.16's center control, d64-L4."""


# --- Round 1: training length against peak learning rate ----------------------------------------------------

ROUND_1: tuple[Arm, ...] = (
    Arm("long", 3, PEAK_LR, "three times ex-2.2.16's steps at its peak LR: the training-length reading alone"),
    Arm("long-low", 3, LOW_LR, "three times the steps at the lower peak LR: both readings at once"),
    Arm("low", 1, LOW_LR, "ex-2.2.16's own length at the lower peak LR: the learning-rate reading alone"),
)

# --- Round 2: a learning-rate sweep at eight times the length, with the newline mask -------------------------

LONG_MULT = 8
"""Round 2 trains for eight times ex-2.2.16's length (about 105,600 steps): round 1's curves were still rising
at three times, and leveled off only as the learning rate decayed."""

WARMUP_EPOCHS = EPOCHS * ex2216.ex2214.ex229.ex223.WARMUP_FRAC
"""Round 2 warms up over ex-2.2.16's own warmup (5 epochs, about 1,320 steps) whatever the run length, where the
default rule would stretch it to a tenth of the run (about 10,560 steps at eight times)."""

SWEEP_LRS: tuple[float, ...] = tuple(round(PEAK_LR * 10 ** (-i / 4), 5) for i in range(7))
"""Peak learning rates a quarter-decade apart, from ex-2.2.16's 0.01 down to 0.00032: one seed each, since
neighboring rates act as replicates of a smooth curve. Two more seeds follow at the rates worth a closer look."""

ROUND_2: tuple[Arm, ...] = (
    *(
        Arm(f"sweep-{lr:g}", LONG_MULT, lr, "one seed of the masked sweep", 1, True, WARMUP_EPOCHS, 2)
        for lr in SWEEP_LRS
    ),
    Arm(
        f"sweep-nomask-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "the sweep rate nearest `long-low`, without the mask: the mask effect at eight times",
        1,
        False,
        WARMUP_EPOCHS,
        2,
    ),
)

ACTIVE_ROUNDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
"""The rounds whose arms train. Round 2 joins once the learning-rate finder has run, since the finder can move
the sweep range."""

# --- Round 3: hold the learning rate where the HSV ops rose -------------------------------------------------

HOLD_LR = 0.0025
"""The rate the round-3 schedule holds: inside the band (0.0022 to 0.004) where round 2's HSV ops rose."""

WSD_SHEET = f"""STEP,PHASE,ACTION,lr
0,Warmup,,0.01
125,Ease,,1
1070,Hold,,{HOLD_LR / SWEEP_LRS[2]:.4f}
8500,Anneal,,{HOLD_LR / SWEEP_LRS[2]:.4f}
10000,,,0.01
"""
"""The round-3 schedule, keyed in hundredths of a percent of the run and as a multiple of the peak rate: warm up
over 1.25% of the run (the 5 epochs of round 2), ease to the hold rate by about step 11,000, hold until 85%, and
anneal to the cosine floor (1% of peak). Keyframes interpolate with minimum jerk, an ease-in-out much like the
half cosine."""

ROUND_3: tuple[Arm, ...] = (
    Arm(
        f"wsd-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "warmup-stable-decay, holding at the rate where round 2's HSV ops rose",
        3,
        True,
        WARMUP_EPOCHS,
        3,
        lr_sheet=WSD_SHEET,
    ),
    Arm(
        f"sweep-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "two more seeds of the masked cosine at the same peak, the comparison for `wsd`",
        3,
        True,
        WARMUP_EPOCHS,
        3,
        first_seed=1,
    ),
)

# --- Round 4: a staircase down to 1e-5 ---------------------------------------------------------------------

STAIRS = (0.0021, 0.001, 0.0001, 0.00001)
"""The rates the round-4 schedule holds in turn. The first sits a little below round 3's hold, still inside the
band where the HSV ops rose; the rest step down a decade or so each, ending at a third of the cosine floor."""

STAIRS_SHEET = "STEP,PHASE,ACTION,lr\n0,Warmup,,0.01\n125,Ease,,1\n" + "".join(
    f"{start},{phase},,{lr / SWEEP_LRS[2]:.5f}\n{end},,,{lr / SWEEP_LRS[2]:.5f}\n"
    for (start, end, phase), lr in zip(
        ((1100, 4500, "Hold"), (5200, 6500, "Step"), (7200, 8200, "Step"), (8900, 10000, "Step")), STAIRS, strict=True
    )
)
"""The round-4 schedule, keyed like `WSD_SHEET`: the same warmup, an ease to the first rate by 11% of the run,
held until 45% (past the latest HSV rise in rounds 2 and 3, at 40%); then anneals of 7% of the run to each lower
rate, held until 65%, 82%, and the end. The last hold (11% of the run, about 11,600 steps) shows whether the
curves still climb at 1e-5."""

ROUND_4: tuple[Arm, ...] = (
    Arm(
        f"stairs-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "a staircase of holds at 0.0021, 0.001, 0.0001, and 1e-5: what each drop in rate is worth",
        3,
        True,
        WARMUP_EPOCHS,
        4,
        lr_sheet=STAIRS_SHEET,
    ),
)

# --- Round 5: twice the length, under each schedule ----------------------------------------------------------

LONGER_MULT = 16
"""Round 5 trains for sixteen times ex-2.2.16's length (about 211,200 steps), at one seed per schedule first."""

ROUND_5: tuple[Arm, ...] = tuple(
    replace(
        a,
        name=f"{a.name.split('-')[0]}{LONGER_MULT}x-{SWEEP_LRS[2]:g}",
        epoch_mult=LONGER_MULT,
        seeds=1,
        round=5,
        first_seed=0,
        note=f"{a.note}; at sixteen times the length",
    )
    for a in (
        next(b for b in ROUND_2 if b.name == f"sweep-{SWEEP_LRS[2]:g}"),
        ROUND_3[0],
        ROUND_4[0],
    )
)
"""The cosine, `wsd`, and `stairs` at sixteen times, one seed each (seed 0, so each pairs with its eight-times
run). The sheets stretch with the run, so their holds double in length, and so does their warmup (2.5% of the
run); the cosine keeps its warmup of 5 epochs."""

# --- Round 6: a wider model -----------------------------------------------------------------------------------

ROUND_6: tuple[Arm, ...] = (
    Arm(
        f"d128-sweep-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "the masked cosine at eight times, on ex-2.2.16's larger control (d128-L4): is the level near 0.45 capacity?",
        1,
        True,
        WARMUP_EPOCHS,
        6,
        model=ex2216.LARGE_MODEL,
    ),
)
"""Rounds 2 to 5 all ended near 0.45 whatever the schedule or length, each op group short of its own ceiling by a
similar amount. One seed of the wider model (seed 0, pairing with `sweep-0.00316-s0`) tests capacity first."""

# --- Round 7: a deeper model ----------------------------------------------------------------------------------

DEEP_MODEL = "d64-L6"
"""Two more blocks at the baseline width, so round 7 changes depth alone."""

ROUND_7: tuple[Arm, ...] = (
    Arm(
        f"d64L6-sweep-{SWEEP_LRS[2]:g}",
        LONG_MULT,
        SWEEP_LRS[2],
        "the masked cosine at eight times, on a deeper d64-L6 model: does depth lift hsvmix?",
        1,
        True,
        WARMUP_EPOCHS,
        7,
        model=DEEP_MODEL,
    ),
)
"""hsvmix has stayed near 0.19 against a ceiling of 0.28 in every run so far, and it chains the most steps
(convert both operands to HSV, mix with a wrapping hue, map back to a word). One seed at seed 0 pairs with
`sweep-0.00316-s0` and `d128-sweep-0.00316-s0`, so width and depth each differ from the baseline in one way."""

ARMS: tuple[Arm, ...] = tuple(
    a for a in ROUND_1 + ROUND_2 + ROUND_3 + ROUND_4 + ROUND_5 + ROUND_6 + ROUND_7 if a.round in ACTIVE_ROUNDS
)

# --- The learning-rate finder, ahead of round 2 --------------------------------------------------------------

FINDER_RANGE = (1e-5, 1.0)
"""The finder's first sweep of peak learning rates, on a log scale: well below and well above anything trained."""

FINDER_ZOOMS = 3
FINDER_STEPS = 300
"""Steps per zoom level. Each level restarts from the same initial model, so a level is a short run of its own
at a rising learning rate."""

FINDER_ARMS: tuple[tuple[str, bool], ...] = (("finder-mask", True), ("finder-nomask", False))
"""The finder runs on the round-2 config with and without the newline mask."""
N_RUNS = sum(a.seeds for a in ARMS)


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


# =============================================================================================
# The DAG
# =============================================================================================


def resolve_refs(key: str) -> dict:
    """Ex-2.2.16's published corpus condition *key*: the corpus, labels, holdout, and probes artifacts a
    training run needs, and the corpus's tokenizer and token count (the latter sizes the trajectory stride).
    Resolving these by ref rather than rebuilding the corpus records ex-2.2.16 as this experiment's lineage.
    """
    from sca.compute.data_pipelines import load_data
    from mini.store import get, get_ref

    corpus = get_ref(ex2216.CORPUS_REF.format(key=key))
    labels = get_ref(ex2216.LABELS_REF.format(key=key))
    holdout = get_ref(ex2216.HOLDOUT_REF.format(key=key))
    probes = get_ref(ex2216.PROBES_REF.format(key=key))
    assert corpus and labels and holdout and probes, f"ex-2.2.16 has not published corpus condition {key!r}"
    _, meta = load_data(get(corpus, get_data_dir() / "resolve" / "corpus"))
    return {"corpus": corpus, "labels": labels, "holdout": holdout, "probes": probes, "meta": meta}


def epoch_length_of(total_tokens: int, config) -> int:
    """Optimizer steps per epoch on the resolved corpus, at *config*'s batch size and window: what
    `train_anchored` itself computes internally, so the trajectory stride can be sized in whole epochs without
    loading the corpus a second time.
    """
    from sca.data.batches import batches_per_epoch

    return batches_per_epoch(int(config.data.train_split * total_tokens), config.data, config.model)


def traj_stride_for(epochs: int, epoch_length: int, n_points: int) -> int:
    """The `traj_stride` that gives a run of *epochs* epochs about *n_points* trajectory records."""
    return max(1, round(epochs * epoch_length / n_points))


def cells(arms: tuple[Arm, ...], resolved: dict, seeds: int = SEEDS) -> list[dict]:
    """One row per run: the config (the arm's epoch count and peak learning rate) and the un-anchored schedule,
    over the center control's corpus condition, at fresh model seeds. Reuses ex-2.2.16's `_make_config`,
    `Condition229`, and `schedules`, as its own `cells` does.
    """
    from sca.config import ModelConfig
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    vocab = align(resolved["meta"].tokenizer_config.vocab_size, 64)
    rows = []
    for a in arms:
        n_embd, n_layer = ex2216.model_dims(a.model or ex2216.MODEL)
        epochs = EPOCHS * a.epoch_mult
        for seed in range(a.first_seed, min(a.seeds, seeds)):
            model_seed = SEED_OFFSET + seed
            config = ex2216._make_config(vocab, model_seed, epochs, n_embd, n_layer)
            config.tokenizer = resolved["meta"].tokenizer_config.model_copy()
            config.model.block_size = ex2216.BLOCK
            config.model.tie_embeddings = False
            config.optimizer.learning_rate = a.peak_lr
            if a.mask:
                config.model = ModelConfig.model_validate(
                    config.model.model_dump() | {"line_mask_token": WordTokenizer(config.tokenizer).stoi["\n"]}
                )
            if a.warmup_epochs is not None:
                config.scheduler.warmup_epochs = a.warmup_epochs
            if a.lr_sheet is not None:
                config.scheduler.lr_sheet = a.lr_sheet
            base = ex2216.Condition229(
                a.name, 1, a.name, lam=0.0, tau=ex2216.TAU, epochs=epochs, ops=ex2216.OP_NAMES, n_lines=ex2216.N_LINES
            )
            anchor, anti = ex2216.schedules(base)
            assert anti is None, "every arm is unanchored"
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "arm": a.name,
                    "epochs": epochs,
                    "peak_lr": a.peak_lr,
                    "mask": a.mask,
                    "warmup_epochs": config.scheduler.warmup_epochs,
                    "round": a.round,
                    "lr_sheet": a.lr_sheet,
                    "model": a.model or ex2216.MODEL,
                    "seed": seed,
                    "model_seed": model_seed,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    corpus,
    labels,
    probes,
    holdout,
    k: int,
    traj_stride: int,
    n_traj_eem: int,
    label: str,
) -> dict:
    """Train one unanchored run of the center control, recording every *traj_stride* steps, beside the loss
    `train_anchored`'s own trajectory already carries: the held-out expected exact match and the calibration
    KL from the Bayes predictive, overall and per op, on a fixed subset of *n_traj_eem* held-out contexts per
    op. Scored the way ex-2.2.16's `eval_one` scores a final checkpoint (`_color_probs`, `_match`,
    `_post_queries`, `logits_at`), on the model as it stands rather than a saved checkpoint.
    """
    from sca.anchoring import AnchorSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    newline_id = tokenizer.stoi["\n"]
    anchored_op_id = ex2216.OP_NAMES.index(ex2216.ANCHORED_OP)

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=anchored_op_id,
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]  # (n_ops * N_TRAJ_PROBE, context_tokens(k))

    per_op = len(probe_tokens) // ex2216.N_OPS
    weights = np.concatenate([np.full(per_op, 1.0 if o == ex2216.ANCHORED_OP else 0.0) for o in ex2216.OP_NAMES])
    weights = weights / weights.sum()

    # --- The fixed EEM/calibration probe: the first n_traj_eem held-out contexts of every op --------------
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    sub = np.concatenate([np.flatnonzero(ho.op_ids == o)[:n_traj_eem] for o in range(ex2216.N_OPS)])
    sub_tokens, sub_op_ids, sub_posterior = ho.tokens[sub], ho.op_ids[sub], ho.posterior[sub]
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    sub_holdout = ex2216.Holdout(
        sub_tokens,
        sub_op_ids,
        sub_posterior,
        np.empty(0),
        np.empty((0, 0), dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )
    P = ex2216._get_posterior()
    table = P.build_table(ex2216.TABLE)
    probe_ctx = ex2216._post_queries(P, sub_holdout, k, tok2color)
    identity = projection(Subspace.axis(config.model.n_embd), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)

    traj_extra: dict[str, list] = {"eem": [], "eem_per_op": [], "kl": [], "kl_per_op": []}

    def on_record(_index: int, model) -> None:
        p = ex2216._color_probs(logits_at(model, sub_tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, probe_ctx, p)
        kl = P.kl(P.predictive(table, probe_ctx, sub_posterior), p)
        traj_extra["eem"].append(float(eem.mean()))
        traj_extra["eem_per_op"].append(ex2216._per_op(eem, sub_op_ids))
        traj_extra["kl"].append(float(kl.mean()))
        traj_extra["kl_per_op"].append(ex2216._per_op(kl, sub_op_ids))

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=None,
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        crop=ex2216.CROP_POLICY,
        anchor_slices=None,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=newline_id,
        min_line_tokens=context_length(k),
        on_record=on_record,
    )

    epoch_train_loss = np.asarray([m.train_loss for m in metrics], dtype=np.float64)
    epoch_axis = np.arange(1, len(epoch_train_loss) + 1, dtype=np.float64)
    train_loss = np.interp(np.asarray(traj["epoch"], dtype=np.float64), epoch_axis, epoch_train_loss).tolist()

    return {
        "label": label,
        "traj": {
            "step": traj["step"].tolist(),
            "epoch": traj["epoch"].tolist(),
            "lr": traj["lr"].tolist(),
            "val_loss": traj["val_loss"].tolist(),
            "train_loss": train_loss,
        }
        | traj_extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.17-{label}-ckpt"),
    }


def find_lr(config, corpus, label: str) -> dict:
    """Run the progressive learning-rate finder (`utils.lr_finder`) on *config*'s model and corpus: Adam at a
    learning rate rising on a log scale over `FINDER_STEPS` steps, from a fresh model at every zoom level, with
    the weights re-projected after every step as training does. Returns the suggestion and every level, raw
    losses included, as plain data.
    """
    from dataclasses import asdict
    from typing import cast

    import jax.random as jr

    from sca.compute.data_pipelines import load_data
    from sca.data.batches import batches_per_epoch, sample_batches, split_data
    from sca.model import LanguageModel, build_model
    from sca.training.loop import loss_fn
    from sca.training.optimizer import configure_optimizer
    from utils.lr_finder.lr_finder import lr_finder_search
    from mini.store import get

    data, _ = load_data(get(corpus, get_data_dir() / "finder" / label / "corpus"))
    train_data, _ = split_data(data, config.data.train_split)
    epoch_length = batches_per_epoch(len(train_data), config.data, config.model)
    model = build_model(config.model, key=jr.key(config.seed))
    rng = np.random.default_rng(config.seed)

    def batches():
        while True:
            yield from sample_batches(train_data, config.data, config.model, epoch_length, rng)

    best, finder_config, history = lr_finder_search(
        model,
        lambda m, x, y, _key: loss_fn(m, x, y),
        lambda learning_rate: configure_optimizer(model, config.optimizer, learning_rate),
        batches(),
        start_lr=FINDER_RANGE[0],
        end_lr=FINDER_RANGE[1],
        num_zooms=FINDER_ZOOMS,
        steps_per_zoom=FINDER_STEPS,
        constrain=lambda m: cast(LanguageModel, m).normalize_weights(),
        key=jr.key(config.seed + 1),
    )
    return {
        "label": label,
        "best_lr": float(best),
        "config": asdict(finder_config),
        "history": [asdict(h) for h in history],
    }


# --- Publishing ------------------------------------------------------------------------------

TRAJ_REF = "reports/m2/ex-2.2.17/trajectories"
METRICS_REF = "reports/m2/ex-2.2.17/metrics"
CHECKPOINT_REF = "reports/m2/ex-2.2.17/checkpoints/{label}"
EVAL_REF = "reports/m2/ex-2.2.17/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.17/eval-arrays/{label}"
FINDER_REF = "reports/m2/ex-2.2.17/lr-finder"
DETAIL_REF = "reports/m2/ex-2.2.17/holdout-detail"
SCORE_REF = "reports/m2/ex-2.2.17/answers/{label}"


def design() -> dict[str, Any]:
    """The design constants a report (and stage C) reads beside the results."""
    return {
        "experiment": "ex-2.2.17",
        "question": "whether ex-2.2.16's center control plateau reflects too high a peak learning rate, too "
        "little training, or both",
        "corpus_condition": CORPUS_KEY,
        "arms": [asdict(a) for a in ARMS],
        "n_runs": N_RUNS,
        "seed_offset": SEED_OFFSET,
        "epochs_base": EPOCHS,
        "peak_lr": PEAK_LR,
        "low_lr": LOW_LR,
        "n_traj_points": N_TRAJ_POINTS,
        "n_traj_eem_per_op": N_TRAJ_EEM_PER_OP,
        "long_mult": LONG_MULT,
        "warmup_epochs": WARMUP_EPOCHS,
        "sweep_lrs": list(SWEEP_LRS),
        "active_rounds": list(ACTIVE_ROUNDS),
        "hold_lr": HOLD_LR,
        "wsd_sheet": WSD_SHEET,
        "stairs": list(STAIRS),
        "stairs_sheet": STAIRS_SHEET,
        "longer_mult": LONGER_MULT,
        "finder": {"range": list(FINDER_RANGE), "zooms": FINDER_ZOOMS, "steps": FINDER_STEPS},
        "block": ex2216.BLOCK,
        "model": ex2216.MODEL,
        "ops": list(ex2216.OP_NAMES),
    }


def publish_results(trained: list[dict], rows: list[dict]) -> dict:
    """The trajectories (JSON, one entry per run), the design and every run's row (JSON), and every end
    checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    traj = {t["label"]: t["traj"] for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.17-trajectories.json"))
    row_meta = [{k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows]
    set_ref(
        METRICS_REF, put(json.dumps({"design": design(), "runs": row_meta}).encode(), name="ex-2.2.17-metrics.json")
    )
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    return {"n_runs": len(trained)}


def publish_evaluation(rows: list[dict], evaled: list[dict]) -> dict:
    """The end-of-training evaluation (JSON, one entry per run), and every run's per-context arrays under its
    own ref. Mirrors ex-2.2.16's `publish_evaluation`, without the suppression pass this scout has no arms for.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    body = {"design": design(), "runs": [slim(r) for r in evaled]}
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.17-eval.json"))
    return {"n_evaled": len(evaled)}


# --- Scoring: the answer distributions behind the evaluation -------------------------------------------------


def scored(row: dict) -> bool:
    """The runs the answer scoring covers: every masked run at the peak learning rate of rounds 3 to 7, whatever
    its schedule, length, or model, and round 2's run at that rate (seed 0 of the cosine).
    """
    return row["mask"] and row["peak_lr"] == SWEEP_LRS[2] and row["round"] >= 2


def holdout_detail(holdout, k: int) -> dict:
    """What the published held-out contexts leave out, recovered by drawing them again: where each shown answer
    came from (the true op, replacement op noise, or a random grid color), which op produced it, the query pair,
    and the Bayes predictive over the grid at the query `=`. The draw is ex-2.2.16's own (`_sample_by_op` at the
    condition's holdout seed), checked token for token against the published contexts.
    """
    from sca.config import TokenizerConfig
    from sca.data.incontext import encode_corpus, vocabulary
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from mini.store import get, put

    workdir = get_data_dir() / "detail"
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    n_per_op = len(ho.op_ids) // ex2216.N_OPS
    seed = ex2216.HOLDOUT_SEED + [(kk, r) for kk, r in ex2216.GRAMMAR_CONDITIONS].index(CENTRE)
    contexts = ex2216._sample_by_op(ex2216.TABLE, k, CENTRE[1], ex2216.CUBE_RATE, 0.0, n_per_op, seed)
    tokenizer = WordTokenizer(TokenizerConfig(vocabulary=sorted(vocabulary())))
    tokens = encode_corpus(contexts, tokenizer.stoi).reshape(ho.tokens.shape)
    assert (tokens == ho.tokens).all(), "the redrawn contexts differ from the published ones"

    op_index = {name: i for i, name in enumerate(ex2216.OP_NAMES)}
    code = {"true": 0, "noise": 1, "cube": 2}
    source = np.array([[code[e.source] for e in c.examples] for c in contexts], dtype=np.int8)
    drawn = np.array([[op_index.get(e.drawn_op, -1) for e in c.examples] for c in contexts], dtype=np.int8)
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(len(tokenizer.stoi), -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    P = ex2216._get_posterior()
    q_ctx = ex2216._post_queries(P, ho, k, tok2color)
    predictive = P.predictive(P.build_table(ex2216.TABLE), q_ctx, ho.posterior)
    arrays = ex2216._npz(
        op_ids=ho.op_ids,
        posterior=ho.posterior,
        ceiling=ho.ceiling,
        source=source,
        drawn_op=drawn,
        query_pair=q_ctx.query_pair,
        predictive=predictive.astype(np.float16),
    )
    return {
        "arrays": put(arrays, name="ex-2.2.17-holdout-detail.npz"),
        "noise_counts": np.bincount((source == 1).sum(1), minlength=k + 1).tolist(),
    }


def score_one(checkpoint, holdout, k: int, label: str) -> dict:
    """The answer distribution over the grid at the query `=` of every held-out context (float16), the input to
    the report's scoring by noise and posterior and its answer clouds. Computed as `ex2216.eval_one` computes it.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    workdir = get_data_dir() / "score" / label
    model, _, color_ids, _ = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    identity = projection(Subspace.axis(model.transformer.wte.shape[1]), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
    p = ex2216._color_probs(logits_at(model, ho.tokens, identity, (), eq_role), color_ids)
    return {"label": label, "arrays": put(ex2216._npz(p=p.astype(np.float16)), name=f"ex-2.2.17-answers-{label}.npz")}


def publish_scoring(detail: dict, answers: list[dict]) -> dict:
    """The held-out detail and every scored run's answer distributions, each under its own ref."""
    from mini.store import set_ref

    set_ref(DETAIL_REF, detail["arrays"])
    for a in answers:
        set_ref(SCORE_REF.format(label=a["label"]), a["arrays"])
    return {"n_scored": len(answers), "noise_counts": detail["noise_counts"]}


def publish_finder(found: list[dict]) -> dict:
    """The finder results (JSON, one entry per finder arm)."""
    import json

    from mini.store import put, set_ref

    body = {"design": design(), "runs": found}
    set_ref(FINDER_REF, put(json.dumps(body).encode(), name="ex-2.2.17-lr-finder.json"))
    return {"best_lr": {f["label"]: f["best_lr"] for f in found}}


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx, arms: tuple[Arm, ...], seeds: int, n_traj_points: int, n_traj_eem: int
) -> tuple[list[dict], list[dict], dict]:
    """Resolve the center control's corpus condition, then train *arms*: the whole DAG, with the seeds, the
    trajectory point target, and the EEM probe size as arguments so a short prototype runs the same code (the
    smoke test overrides all four, plus the arm set itself).
    """
    resolved = ctx.run(resolve_refs, CORPUS_KEY, role="prep")
    rows = cells(arms, resolved, seeds)
    epoch_length = epoch_length_of(resolved["meta"].total_tokens, rows[0]["config"])
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [CENTRE[0]] * n,
        [traj_stride_for(r["epochs"], epoch_length, n_traj_points) for r in rows],
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    return trained, rows, resolved


def finder_configs(resolved: dict) -> list[tuple[str, Any]]:
    """The finder arms' configs: the round-2 config (without its learning rate, which the finder sweeps) with and
    without the newline mask, at the first fresh model seed.
    """
    probe = Arm("finder", LONG_MULT, PEAK_LR, seeds=1, warmup_epochs=WARMUP_EPOCHS, round=2)
    return [(label, cells((replace(probe, mask=mask),), resolved, 1)[0]["config"]) for label, mask in FINDER_ARMS]


def main(ctx: Ctx) -> dict:
    trained, rows, resolved = run(ctx, ARMS, SEEDS, N_TRAJ_POINTS, N_TRAJ_EEM_PER_OP)
    probes = finder_configs(resolved)
    found = ctx.map(
        find_lr, [c for _, c in probes], [resolved["corpus"]] * len(probes), [lbl for lbl, _ in probes], role="finder"
    )
    finder = ctx.run(publish_finder, found, role="prep")
    published = ctx.run(publish_results, trained, rows, role="prep")
    n = len(trained)
    evaled = ctx.map(
        ex2216.eval_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [CENTRE[0]] * n,
        [r["label"] for r in rows],
        role="eval",
    )
    detail = ctx.run(holdout_detail, resolved["holdout"], CENTRE[0], role="prep")
    picks = [i for i, r in enumerate(rows) if scored(r)]
    answers = ctx.map(
        score_one,
        [trained[i]["checkpoint"] for i in picks],
        [resolved["holdout"]] * len(picks),
        [CENTRE[0]] * len(picks),
        [rows[i]["label"] for i in picks],
        role="eval",
    )
    return (
        published
        | finder
        | ctx.run(publish_evaluation, rows, evaled, role="prep")
        | ctx.run(publish_scoring, detail, answers, role="prep")
    )


COMPUTE = {
    # One ref resolution and a small corpus-metadata read.
    "prep": dict(cpu=2, timeout=600),
    # 13,200 steps for `low`, 39,600 for `long` and `long-low` (about 14 minutes each), and 105,600 for round 2
    # (about 36 minutes at round 1's pace), and 211,200 for round 5 (up to about 90 minutes when containers
    # share a host); the watchdog covers the checkpoint upload.
    "train": dict(gpu="L4", timeout=3 * 3600, watchdog=900, watchdog_grace=900),
    # 900 finder steps with no trajectory reads: a minute or two, most of it compilation and the corpus download.
    "finder": dict(gpu="L4", timeout=900),
    # Forward passes only, over 22,000 held-out contexts, as ex-2.2.16's eval role.
    "eval": dict(gpu="L4", timeout=900),
}

experiment = Experiment(name="ex-2.2.17", main=main, roles=COMPUTE)
