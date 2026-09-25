"""
An RGB-distance readout beside expected exact match: a re-score of ex-2.2.11's stored runs.

Scoring only, no training and no gates. Every removal and selectivity number in D2.2 is expected exact match,
which gives no partial credit: a guess one grid step from the answer scores the same as a guess of black. This
pass runs each stored checkpoint over the same probe lines and the same operators as ex-2.2.11, and measures
how far the model's answer sits from the line's raw answer on the color grid (`sca.answer_distance`), for the
greedy guess and for the mean and median of the model's distribution, beside the exact match. Per run it keeps
per-group means, a kept/lost split of the removal lines, and a histogram of the greedy distance; per line it
keeps the guess and the mean under every pass.

    bin/mini run docs/m2/answer-distance/experiment.py --app local
    bin/mini status answer-distance
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, NamedTuple

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex2211():
    """Ex-2.2.11's module, loaded by path and left out of `sys.modules` (as ex-2.2.11 loads ex-2.2.9), so the
    task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.11" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2211", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2211 = _load_ex2211()

# --- Sources -------------------------------------------------------------------------------------

SOURCE = "ex-2.2.11"


class Condition(NamedTuple):
    """One condition of the source experiment: every stored seed is read."""

    cond: str
    seeds: tuple[int, ...]
    title: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("control", tuple(range(5)), "un-anchored"),
    Condition("handover", tuple(range(20)), "untied readout, whole-line labeller"),
    Condition("handover-slot", tuple(range(20)), "untied readout, slot labeller"),
    Condition("handover-tied", tuple(range(9)), "tied readout, whole-line labeller"),
)
"""The four conditions of ex-2.2.11: the candidate, the two arms that each undo one of its changes, and the
control. The stored labels are `{cond}-s{seed}`."""

N_RUNS = sum(len(c.seeds) for c in CONDITIONS)

SRC_PROBES_REF = ex2211.PROBE_REF
SRC_CHECKPOINT_REF = ex2211.CHECKPOINT_REF
"""Where ex-2.2.11 published the probe lines every run was scored on, and per run its final checkpoint."""

METRICS_REF = "reports/m2/answer-distance/metrics"
ARRAYS_REF = "reports/m2/answer-distance/arrays"
"""The run table (JSON) and the arrays (npz): per run `{label}/summary`, `{label}/split`, and `{label}/hist`;
per op the line constants under `lines/{op}/…`; and for `LINE_OPS` on `LINE_CONDS` the per-line guess and mean
under `{label}/{op}/{pass}/…`."""


def label_for(c: Condition, seed: int) -> str:
    return f"{c.cond}-s{seed}"


# --- What is measured ---------------------------------------------------------------------------

ANCHOR_AXIS = ex2211.ANCHOR_AXIS
SLICES = tuple(ex2211.SLICES)
ANSWER_POS = ex2211.ex229.ANSWER_POS
OP_NAMES: tuple[str, ...] = tuple(ex2211.OP_NAMES)

PASSES: tuple[str, ...] = ("clean", "projection", "operands")
"""The clean forward pass, then ex-2.2.11's two projection operators: the whole line at every slice, and the
operand positions alone. The shaped operator is left out: it is not scored in ex-2.2.11's report."""

OPERAND_POSITIONS: tuple[int, ...] = ex2211.OPERATOR_SPEC["operands"][1] or ()
assert OPERAND_POSITIONS, "the operands operator names its positions"

GROUPS: tuple[str, ...] = (
    "all",
    "red",
    "nonred",
    "nonred_excl",
    "red_answer",
    "removal",
    "removal_op1",
    "removal_op2",
    "removal_zero",
    "sv",
)
"""The line groups of ex-2.2.11's `HueReadout` that the summary keeps: every line; the red and non-red lines;
the non-red lines without and with a red answer; the removal lines (red, hue move at least `FAR_MOVE`), by the
slot the red operand sits in, and under ex-2.2.9's to-zero rule; and the saturation-and-value lines."""

STATS: tuple[str, ...] = ("eem", "hit", "greedy", "mean", "median", "expected")
"""Per line, then averaged over a group: expected exact match; whether the greedy guess is one of the line's
possible answers (a hit); and the distance in grid steps from the raw answer of the greedy guess, of the mean
of the model's distribution, of its channel-wise median, and in expectation over the distribution."""

SPLITS: tuple[str, ...] = ("kept", "lost", "never")
"""A line under a pass is *kept* when the pass's greedy guess is a hit, *lost* when the clean guess was a hit
and the pass's is not, and *never* when the clean guess already missed. Under the clean pass nothing is lost."""

SPLIT_STATS: tuple[str, ...] = ("n", "greedy", "mean")
"""Per split: the line count, and the mean greedy and mean-of-distribution distances."""

N_BINS = 10
"""The greedy distance rounded to the nearest step falls in 0..9 (the grid's far corner is 5√3 ≈ 8.7 steps)."""

LINE_OPS: tuple[str, ...] = ("hue-hsv", "darken")
LINE_CONDS: tuple[str, ...] = ("control", "handover")
"""The ops and conditions whose per-line guess and mean are published, for the cube figures: `hue-hsv`, the op
whose removal lines keep the most exact match, and `darken`, the near miss of ex-2.2.11."""

BATCH = 2048
"""Lines per forward pass. The last batch is padded to this size so each pass compiles once per checkpoint."""


def npz_bytes(arrays: dict[str, Any]) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


# --- Reading the stores --------------------------------------------------------------------------


def source_store():
    """The store the source runs were published to, for reads alone.

    On production this is the ambient store. Under a storage profile (`MINI_PROFILE=dev`) the ambient store is
    the profile's pair, which holds none of the source runs, so the base pair is resolved from the project
    configuration instead, as a production session resolves it. This pass's own results still go to the
    ambient store. No bucket is named here.
    """
    import os

    from mini.store import PROFILE_ENV, project_store

    saved = os.environ.pop(PROFILE_ENV, None)
    try:
        return project_store()
    finally:
        if saved is not None:
            os.environ[PROFILE_ENV] = saved


def resolve_inputs(labels: list[str]) -> dict:
    """The source experiment's artifacts by label, so each task is keyed on content rather than on a name."""
    ck = {lb: SRC_CHECKPOINT_REF.format(label=lb) for lb in labels}
    names = [SRC_PROBES_REF, *ck.values()]
    refs = source_store().get_refs(names)
    missing = [n for n in names if refs[n] is None]
    assert not missing, f"{SOURCE} artifacts not in the store: {missing}"
    return {"probes": refs[SRC_PROBES_REF], "checkpoints": {lb: refs[n] for lb, n in ck.items()}}


def load_probes(probes, workdir) -> dict[str, dict[str, np.ndarray]]:
    keys = ("tokens", "r1", "r2", "q_idx", "q_p", "move", "hue_move")
    with np.load(source_store().get(probes, workdir / "probes.npz")) as z:
        return {o: {k: z[f"{o}/{k}"] for k in keys} for o in ex2211._probe_ops(z)}


def readout(f: dict[str, np.ndarray], color_ids: np.ndarray, tok2color: np.ndarray):
    """Ex-2.2.11's line groups for one op's probe lines."""
    return ex2211.HueReadout(
        f["tokens"], f["r1"], f["r2"], f["q_idx"], f["q_p"], f["hue_move"], f["move"], color_ids, tok2color
    )


def palette_maps(vocab_size: int, stoi: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The palette's token ids, and the token → palette-index map (−1 off the palette)."""
    color_ids = np.array([stoi[n] for n in ex2211.ex229.PALETTE])
    tok2color = np.full(vocab_size, -1)
    tok2color[color_ids] = np.arange(len(color_ids))
    return color_ids, tok2color


# --- Tasks ---------------------------------------------------------------------------------------


def line_table(probes, ckpt) -> dict:
    """What is fixed per line, the same for every run: the raw answer, the two floors, chance, the dose, and
    the group masks. The groups need the palette's token ids, which come from one run's tokenizer; every run
    of the source shares it.
    """
    from mini.store import put
    from sca import answer_distance as ad
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer

    workdir = get_data_dir() / "lines"
    probe = load_probes(probes, workdir)
    source_store().get(ckpt, workdir / "model")
    _, config, _ = load_checkpoint(workdir)
    tok = WordTokenizer(config.tokenizer)
    color_ids, tok2color = palette_maps(tok.vocab_size, tok.stoi)
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for op, f in probe.items():
        read = readout(f, color_ids, tok2color)
        raw = ad.raw_answer(f["q_idx"], f["q_p"])
        fl = ad.floors(f["q_idx"], f["q_p"])
        arrays |= {
            f"{op}/raw": raw.astype(np.float32),
            f"{op}/floor_mode": fl["mode"].astype(np.float32),
            f"{op}/floor_draw": fl["draw"].astype(np.float32),
            f"{op}/chance": ad.chance(raw).astype(np.float32),
            f"{op}/dose": read.dose.astype(np.float32),
            f"{op}/on_grid": f["q_p"].max(axis=1) >= 1.0 - 1e-9,
        }
        arrays |= {f"{op}/group/{g}": read.groups[g] for g in GROUPS}
        counts[op] = {g: int(read.groups[g].sum()) for g in GROUPS}
    return {"n_lines": counts, "arrays": put(npz_bytes(arrays), name="answer-distance-lines.npz")}


def measure(
    p: np.ndarray, q_idx: np.ndarray, q_p: np.ndarray, raw: np.ndarray, support: np.ndarray
) -> dict[str, np.ndarray]:
    """`STATS` per line from color mass *p* `(N, 216)` in palette order, against the answer distribution
    (`q_idx`, `q_p`), its raw answer, and its support mask `(N, 216)`. Also the greedy guess and the mean.
    """
    from sca import answer_distance as ad

    guess = p.argmax(axis=1)
    mean = ad.mean(p)
    return {
        "eem": (np.take_along_axis(p, np.maximum(q_idx, 0), axis=1) * q_p).sum(1),
        "hit": support[np.arange(len(p)), guess].astype(float),
        "greedy": ad.distance(ad.GRID[guess], raw),
        "mean": ad.distance(mean, raw),
        "median": ad.distance(ad.median(p), raw),
        "expected": ad.expected_distance(p, raw),
        "guess": guess,
        "mean_rgb": mean,
    }


def tabulate(per_line: dict[str, np.ndarray], splits: tuple[np.ndarray, ...], groups: dict[str, np.ndarray]):
    """One pass's group means of `STATS`, its `SPLITS` table, and its greedy-distance histogram, over `GROUPS`."""
    summary = np.full((len(GROUPS), len(STATS)), np.nan, np.float32)
    split = np.full((len(GROUPS), len(SPLITS), len(SPLIT_STATS)), np.nan, np.float32)
    hist = np.zeros((len(GROUPS), N_BINS), np.int32)
    steps = np.clip(np.rint(per_line["greedy"]).astype(int), 0, N_BINS - 1)
    for i_g, g in enumerate(GROUPS):
        m = groups[g]
        summary[i_g] = [per_line[s][m].mean() if m.any() else np.nan for s in STATS]
        hist[i_g] = np.bincount(steps[m], minlength=N_BINS)
        for i_s, sm in enumerate(splits):
            mm = m & sm
            split[i_g, i_s] = [mm.sum(), *(per_line[s][mm].mean() if mm.any() else np.nan for s in SPLIT_STATS[1:])]
    return summary, split, hist


def score_chunk(ckpts: list, probes, labels: list[str]) -> list[dict]:
    """Each checkpoint over every op's probe lines under every pass: per line, the model's color mass gives
    the greedy guess, the mean and median of its distribution, and the distances of `STATS`; per group, their
    means; per group and pass, the kept/lost split and the greedy-distance histogram.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from mini.store import put
    from sca import answer_distance as ad
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.intervention import Subspace, _forward, projection
    from sca.model.ngpt import NGPT

    src = source_store()
    # One directory per chunk: the local backend runs every chunk at once, and two workers fetching the probe
    # file to one path can hand one of them a half-written archive.
    workdir = get_data_dir() / "score" / labels[0]
    probe = load_probes(probes, workdir)
    n_pos = next(iter(probe.values()))["tokens"].shape[1]
    operand_mask = jnp.asarray(np.isin(np.arange(n_pos), OPERAND_POSITIONS).astype(np.float32))

    def make_passes(sub):
        def answer_lp(m, t, operator, slices, pos):
            _, _, logits = _forward(m, t, operator, slices, pos)
            return jax.nn.log_softmax(logits[:, ANSWER_POS - 1], axis=-1)

        return {
            "clean": eqx.filter_jit(lambda m, t: answer_lp(m, t, projection(sub), (), None)),
            "projection": eqx.filter_jit(lambda m, t: answer_lp(m, t, projection(sub), SLICES, None)),
            "operands": eqx.filter_jit(lambda m, t: answer_lp(m, t, projection(sub), SLICES, operand_mask)),
        }

    def run(fn, model, tokens: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(tokens), BATCH):
            t = tokens[i : i + BATCH]
            n = len(t)
            if n < BATCH:
                t = np.concatenate([t, np.zeros((BATCH - n, n_pos), t.dtype)])
            out.append(np.asarray(fn(model, jnp.asarray(t)))[:n])
        return np.concatenate(out)

    results = []
    for ckpt, lb in zip(ckpts, labels, strict=True):
        src.get(ckpt, workdir / lb / "model")
        model, config, _ = load_checkpoint(workdir / lb)
        assert isinstance(model, NGPT), f"{lb}: {type(model).__name__} is not an nGPT"
        tok = WordTokenizer(config.tokenizer)
        color_ids, tok2color = palette_maps(tok.vocab_size, tok.stoi)
        passes = make_passes(Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS))

        summary = np.full((len(OP_NAMES), len(PASSES), len(GROUPS), len(STATS)), np.nan, np.float32)
        split = np.full((len(OP_NAMES), len(PASSES), len(GROUPS), len(SPLITS), len(SPLIT_STATS)), np.nan, np.float32)
        hist = np.zeros((len(OP_NAMES), len(PASSES), len(GROUPS), N_BINS), np.int32)
        lines: dict[str, np.ndarray] = {}
        for i_op, op in enumerate(OP_NAMES):
            f = probe[op]
            read = readout(f, color_ids, tok2color)
            raw = ad.raw_answer(f["q_idx"], f["q_p"])
            support = np.zeros((len(raw), 216), bool)
            np.put_along_axis(support, np.maximum(f["q_idx"], 0), f["q_p"] > 0, axis=1)
            hit_clean = None
            for i_pass, name in enumerate(PASSES):
                p = np.exp(run(passes[name], model, f["tokens"])[:, color_ids])  # (N, 216), palette order
                per_line = measure(p, f["q_idx"], f["q_p"], raw, support)
                kept = per_line["hit"] > 0
                hit_clean = kept if hit_clean is None else hit_clean
                summary[i_op, i_pass], split[i_op, i_pass], hist[i_op, i_pass] = tabulate(
                    per_line, (kept, hit_clean & ~kept, ~hit_clean), read.groups
                )
                lines |= {
                    f"{op}/{name}/guess": per_line["guess"].astype(np.int16),
                    f"{op}/{name}/mean": per_line["mean_rgb"].astype(np.float16),
                    f"{op}/{name}/eem": per_line["eem"].astype(np.float16),
                }
        results.append(
            {
                "label": lb,
                "summary": put(
                    npz_bytes({"summary": summary, "split": split, "hist": hist}), name=f"answer-distance-{lb}.npz"
                ),
                "lines": put(npz_bytes(lines), name=f"answer-distance-{lb}-lines.npz"),
            }
        )
    return results


def publish_results(lines: dict, scored: list[dict]) -> dict:
    """Every run's summary arrays, the line constants, and the per-line arrays of `LINE_OPS` on `LINE_CONDS`,
    under `ARRAYS_REF`; the run table and every axis name under `METRICS_REF`.
    """
    import json

    from mini.store import get_many, put, set_ref
    from sca import answer_distance as ad

    by_label = {label_for(c, s): (c, s) for c in CONDITIONS for s in c.seeds}
    workdir = get_data_dir() / "publish"
    merged: dict[str, np.ndarray] = {}
    with np.load(get_many([(lines["arrays"], workdir / "lines.npz")])[0]) as z:
        merged |= {f"lines/{k}": z[k] for k in z.files}
    paths = get_many([(r["summary"], workdir / f"{r['label']}.npz") for r in scored])
    for r, p in zip(scored, paths, strict=True):
        with np.load(p) as z:
            merged |= {f"{r['label']}/{k}": z[k] for k in z.files}
    keep = [r for r in scored if by_label[r["label"]][0].cond in LINE_CONDS]
    paths = get_many([(r["lines"], workdir / f"{r['label']}-lines.npz") for r in keep])
    for r, p in zip(keep, paths, strict=True):
        with np.load(p) as z:
            merged |= {f"{r['label']}/{k}": z[k] for k in z.files if k.split("/")[0] in LINE_OPS}
    set_ref(ARRAYS_REF, put(npz_bytes(merged), name="answer-distance-arrays.npz"))

    metrics: dict[str, Any] = {
        "source": SOURCE,
        "conditions": [c._asdict() for c in CONDITIONS],
        "runs": [
            {"label": r["label"], "cond": by_label[r["label"]][0].cond, "seed": by_label[r["label"]][1]} for r in scored
        ],
        "ops": OP_NAMES,
        "passes": PASSES,
        "groups": GROUPS,
        "stats": STATS,
        "splits": SPLITS,
        "split_stats": SPLIT_STATS,
        "n_bins": N_BINS,
        "line_ops": LINE_OPS,
        "line_conds": LINE_CONDS,
        "n_lines": lines["n_lines"],
        "chance": ad.CHANCE,
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="answer-distance-metrics.json"))
    return {"n_runs": len(scored), "n_arrays": len(merged)}


SEEDS_PER_TASK = 6
"""How many checkpoints one scoring task reads in turn: the local backend runs every task of a `map` as its own
process at once, so a chunk keeps the fan-out to a handful of JAX processes."""


def chunks(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main(ctx: Ctx) -> dict:
    labels = [label_for(c, s) for c in CONDITIONS for s in c.seeds]
    inputs = ctx.run(resolve_inputs, labels, role="cpu")
    lines = ctx.run(line_table, inputs["probes"], inputs["checkpoints"][labels[0]], role="cpu")
    parts = chunks(labels, SEEDS_PER_TASK)
    scored = ctx.map(
        score_chunk,
        [[inputs["checkpoints"][lb] for lb in part] for part in parts],
        [inputs["probes"]] * len(parts),
        parts,
        role="states",
    )
    return ctx.run(publish_results, lines, [r for part in scored for r in part], role="cpu")


# Each scoring task is up to six checkpoints, three forward passes each over the eleven probe sets: about a
# quarter of a minute per checkpoint on CPU. The thread caps keep a handful of concurrent JAX processes from
# each claiming every core of one box.
THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

experiment = Experiment(
    name="answer-distance",
    main=main,
    roles={
        "states": dict(cpu=2, timeout=3600, watchdog_grace=1800, env=THREAD_ENV),
        "cpu": dict(cpu=2, timeout=1800, watchdog_grace=900, env=THREAD_ENV),
    },
)
