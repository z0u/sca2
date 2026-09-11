"""
Pilot: where the syntax rows' axis component works, and three ways to train it out.

A scouting run, in the sense of the science skill: no gates, no verdicts, a record of what
we ran and what we saw. Every anchored model so far carries the anchor axis on the embedding
rows of the op words and `=` (ex-2.2.2's E8, ex-2.2.3's E2), and that component is what a
full-position projection pays for on the non-red lines. nGPT ties the readout to the
embedding, so the same row is read twice: as the residual stream's starting state at slice 0,
and as the logit of that token at every position that predicts it. Three mechanisms could put
the component there, and the pilot has one arm for each:

- the anchor's direct pull on the labelled span at slice 0, which the op word and `=` sit in
  (`blocks-only`: both anchoring terms skip the embedding slice, the design's Prep C);
- the tied readout, which raises the next-token logit after a red state by leaning the
  row toward the axis (`untied`: a second table for the readout, initialised as a copy);
- the blocks reading the component from the stream, which no table change removes
  (`rows-clean`: the tied table with the syntax rows held off the axis after every step,
  so the model has to find a solution that does without it).

Before any of that trains, Part A prices the component on ex-2.2.3's stored checkpoints:
strip it from the syntax rows on the input side only, the output side only, or both, and
read the per-position next-token accuracy and the full-position projection's cost. Part B
trains the three arms at ex-2.2.3's recipe, plus the first two under the whole-line labeller
(ex-2.2.6's `line` keying, span 6) that the handover proposes to adopt, and reads Part A's
table on every run. Every task function but the training step and the row scorer is
ex-2.2.3's, loaded from its module unchanged; the report reads production's twenty
`recipe-short` seeds as the reference.

    bin/mini run docs/m2/ex-2.2.7/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.7
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Literal

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
Condition = ex223.Condition
REDNESS = ex223.REDNESS
PALETTE = ex223.PALETTE
ANSWER_POS = ex223.ANSWER_POS
ANCHOR_AXIS = ex223.ANCHOR_AXIS
TRAJ_STRIDE = ex223.TRAJ_STRIDE
SLICES = ex223.SLICES
_npz = ex223._npz
_make_config = ex223._make_config
_load = ex223._load
_probe_ops = ex223._probe_ops
_Readout = ex223._Readout
prepare_corpus = ex223.prepare_corpus
eval_one = ex223.eval_one
score_one = ex223.score_one

METRICS_REF = "reports/m2/ex-2.2.7/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.7/arrays"
TRAJ_REF = "reports/m2/ex-2.2.7/trajectories"
PROBE_REF = "reports/m2/ex-2.2.7/probes"
CHECKPOINT_REF = "reports/m2/ex-2.2.7/checkpoints/{label}"

EX223_METRICS_REF = ex223.METRICS_REF
EX223_ARRAYS_REF = ex223.ARRAYS_REF
"""Production's `recipe-short` (twenty seeds with the addendum) and `t00`: the reference arms."""
EX223_CHECKPOINT_REF = ex223.CHECKPOINT_REF
"""Where Part A's stored checkpoints come from."""

# --- Part A: the stored checkpoints -----------------------------------------------------------

STORED = {"recipe-short": 5, "t00": 5, "control-short": 5}
"""Ex-2.2.3 arms whose stored checkpoints Part A scores, and how many seeds of each: the recipe (the
reference), the adopted point (twice the recipe's row component), and the un-anchored control (the floor)."""
STORED_LABELS = [f"{c}-s{s}" for c, seeds in STORED.items() for s in range(seeds)]

SYNTAX_WORDS = (*ex223.OP_NAMES, "=", "\n")
"""The rows Part A strips and the `rows-clean` arm holds clean: every word of the grammar that is not a
color. The pad row is left alone; it is never live."""

Strip = Literal["clean", "input", "output", "both", "input-control"]
STRIPS: tuple[Strip, ...] = ("clean", "input", "output", "both", "input-control")
"""Which side of the tied table loses the syntax rows' axis component: neither, the embedding only (the
stream's starting state), the readout only (the next-token logits), or both. `input-control` moves the
embedding rows by the same amount as `input` but along a random direction off the axis, so an input-side
cost can be read as the axis component's rather than as any disturbance of a syntax row's."""
CONTROL_SEED = 0
N_PRED = 5
"""Next-token predictions per six-token line: positions 0 to 4 predict positions 1 to 5."""

# --- Part B: the arms -----------------------------------------------------------------------

Keying = Literal["either", "line"]
EPOCHS = ex223.EPOCHS_SHORT
PROMPT = ex223.SPAN
WHOLE = 6
BLOCK_SLICES = tuple(s for s in SLICES if s > 0)
"""Every residual slice but the embedding: where `blocks-only` anchors."""


@dataclass(frozen=True)
class Arm:
    """One way of keeping the axis off the syntax rows, under one labeller. Everything else is the recipe."""

    name: str
    seeds: int
    title: str
    fix: Literal["blocks-only", "untied", "rows-clean"]
    keying: Keying = "either"
    span: int = PROMPT

    @property
    def condition(self) -> Condition:
        return Condition(self.name, self.seeds, self.title, lam=ex223.SCORING_LAMBDA, epochs=EPOCHS)

    @property
    def tie(self) -> bool:
        return self.fix != "untied"

    @property
    def slices(self) -> tuple[int, ...] | None:
        return BLOCK_SLICES if self.fix == "blocks-only" else None

    @property
    def clean(self) -> bool:
        return self.fix == "rows-clean"


BLOCKS_ONLY = Arm("blocks-only", 3, "both terms skip the embedding slice (Prep C)", "blocks-only")
UNTIED = Arm("untied", 3, "a readout table of its own, from a copy of the embedding", "untied")
ROWS_CLEAN = Arm("rows-clean", 3, "tied table; the syntax rows held off the axis every step", "rows-clean")
BLOCKS_ONLY_LINE = Arm(
    "blocks-only-line", 2, "blocks-only, under the whole-line labeller", "blocks-only", "line", WHOLE
)
UNTIED_LINE = Arm("untied-line", 2, "untied, under the whole-line labeller", "untied", "line", WHOLE)
ARMS = (BLOCKS_ONLY, UNTIED, ROWS_CLEAN, BLOCKS_ONLY_LINE, UNTIED_LINE)
REFERENCE = "recipe-short"
"""Production's arm the three fixes are read against: tied, every slice anchored, no row constraint."""
N_RUNS = sum(a.seeds for a in ARMS)

# --- The probe table under line keying (ex-2.2.6) --------------------------------------------


def line_keyed_probes(probes, vocabulary: list[str], per_slot_rate: float) -> dict:
    """Ex-2.2.3's probe table with `line_p` recomputed for a labeller whose answer slot draws too.

    Ex-2.2.6's patch, unchanged: P(labelled) becomes 1 − (1 − p1)(1 − p2)(1 − p3), with p3 the answer's
    rate, and the operand-only version stays beside it as `line_p_either`.
    """
    from mini.store import get, put

    workdir = get_data_dir() / "probes"
    stoi = {w: i for i, w in enumerate(vocabulary)}
    tok2color = np.full(len(vocabulary), -1)
    for i, name in enumerate(PALETTE):
        tok2color[stoi[name]] = i
    with np.load(get(probes, workdir / "probes.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    for op in [k[: -len("/tokens")] for k in arrays if k.endswith("/tokens")]:
        r3 = REDNESS[tok2color[arrays[f"{op}/tokens"][:, ANSWER_POS]]]
        p1, p2, p3 = (r**8 * per_slot_rate for r in (arrays[f"{op}/r1"], arrays[f"{op}/r2"], r3))
        arrays[f"{op}/line_p_either"] = arrays[f"{op}/line_p"]
        arrays[f"{op}/line_p"] = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
        arrays[f"{op}/r3"] = r3
    return {"probes": put(_npz(**arrays), name="ex-2.2.7-probes-line.npz")}


# --- Part A: the row scorer -----------------------------------------------------------------


def stored_checkpoints(labels: list[str]) -> dict[str, dict]:
    """Ex-2.2.3's stored checkpoints by label, as the `trained` dicts `_load` reads.

    Resolved inside a step so the run records the producing experiment as an upstream.
    """
    from mini.store import get_refs

    refs = get_refs([EX223_CHECKPOINT_REF.format(label=lb) for lb in labels])
    missing = [lb for lb in labels if refs[EX223_CHECKPOINT_REF.format(label=lb)] is None]
    if missing:
        raise RuntimeError(f"no stored checkpoint for {missing}; is this the production profile?")
    return {lb: {"label": lb, "checkpoint": refs[EX223_CHECKPOINT_REF.format(label=lb)]} for lb in labels}


def _strip_rows(table, rows: np.ndarray, control: bool = False):
    """The table with the axis component zeroed on *rows*, and those rows back at unit length.

    With *control*, the component is moved rather than removed: each row gets the same amount along a
    random unit direction orthogonal to the axis, so the row turns by about the same angle.
    """
    import jax.numpy as jnp

    comp = table[rows, ANCHOR_AXIS][:, None]
    cleaned = table[rows].at[:, ANCHOR_AXIS].set(0.0)
    if control:
        u = np.random.default_rng(CONTROL_SEED).normal(size=(len(rows), table.shape[1]))
        u[:, ANCHOR_AXIS] = 0.0
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        cleaned = cleaned + comp * jnp.asarray(u, dtype=cleaned.dtype)
    cleaned = cleaned / jnp.maximum(jnp.linalg.norm(cleaned, axis=1, keepdims=True), 1e-12)
    return table.at[rows].set(cleaned)


def _stripped(model, strip: Strip, rows: np.ndarray):
    """The model with the syntax rows' axis component removed on the named side(s) of the table.

    `input` edits the embedding and keeps the original table as the readout (which unties a tied model for
    the scoring pass); `output` does the reverse; `both` edits each table the model has.
    """
    wte, head = model.transformer.wte, model.transformer.lm_head
    readout = wte if head is None else head
    match strip:
        case "clean":
            return model
        case "input":
            return model.with_tables(wte=_strip_rows(wte, rows), readout=readout)
        case "input-control":
            return model.with_tables(wte=_strip_rows(wte, rows, control=True), readout=readout)
        case "output":
            return model.with_tables(wte=wte, readout=_strip_rows(readout, rows))
        case "both":
            return model.with_tables(
                wte=_strip_rows(wte, rows), readout=None if head is None else _strip_rows(head, rows)
            )
    raise ValueError(strip)


def rows_one(trained: dict, probes, condition: str, seed: int, label: str) -> dict:
    """The row table of one checkpoint, and what stripping the syntax rows on either side of it costs.

    Returns the axis component of every embedding row (and of every readout row, when the model has a table
    of its own), then for each strip condition and op: the next-token accuracy and mean P(next token) at
    each of the five predicting positions, on the red, non-red and all probe lines, on the clean pass; and
    the answer accuracy and P(answer) deficit under the full-position projection, read the same way. The
    clean pass says which side the component works on; the projection says whether a model with rows
    stripped by fiat pays the non-red cost.
    """
    from sca.intervention import Subspace, apply, projection
    from mini.store import get

    workdir = get_data_dir() / "rows" / label
    model, tokenizer, color_ids, tok2color = _load(trained, workdir)
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    syntax = np.array([tokenizer.stoi[w] for w in SYNTAX_WORDS])
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe = {o: (z[f"{o}/tokens"], z[f"{o}/r1"], z[f"{o}/r2"]) for o in _probe_ops(z)}

    def row_table(table) -> dict[str, float]:
        comp = np.asarray(table[:, ANCHOR_AXIS])
        # The table is padded past the vocabulary; the pad word ('') and the padding rows are skipped.
        return {w: float(comp[i]) for i in range(len(comp)) if (w := tokenizer.itos.get(i, ""))}

    head = model.transformer.lm_head
    out: dict = {
        "label": label,
        "condition": condition,
        "seed": seed,
        "tied": head is None,
        "rows": row_table(model.transformer.wte),
        "rows_readout": None if head is None else row_table(head),
        "strip": {},
    }
    for strip in STRIPS:
        edited = _stripped(model, strip, syntax)
        per_op: dict[str, dict] = {}
        for op, (tokens, r1, r2) in probe.items():
            read = _Readout(tokens, r1, r2, color_ids, tok2color)
            clean = apply(edited, tokens, projection(sub), slices=())
            lp = np.asarray(_log_softmax(clean.logits[:, :N_PRED]))  # (N, 5, V)
            nxt = tokens[:, 1 : N_PRED + 1]
            rows = np.arange(len(tokens))[:, None]
            acc = (lp.argmax(-1) == nxt).astype(float)  # (N, 5)
            p_next = np.exp(lp[rows, np.arange(N_PRED)[None], nxt])
            c = read(clean.logits)
            proj = read(apply(edited, tokens, projection(sub), slices=SLICES).logits)
            per_op[op] = {
                "acc": {g: acc[m].mean(0).tolist() for g, m in read.groups.items() if g != "redder"},
                "p_next": {g: p_next[m].mean(0).tolist() for g, m in read.groups.items() if g != "redder"},
                "projection": {
                    "acc": read.by_group(proj["guess"] == read.answer),
                    "deficit": read.by_group(c["p_ans"] - proj["p_ans"]),
                },
            }
        out["strip"][strip] = per_op
    return out


def _log_softmax(logits):
    import jax

    return jax.nn.log_softmax(logits, axis=-1)


# --- Part B: training -----------------------------------------------------------------------


def cells(arms: tuple[Arm, ...], prep: dict, probes_by_keying: dict[str, object]) -> list[dict]:
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    rows = []
    tc = prep["meta"].tokenizer_config
    stoi = WordTokenizer(tc).stoi
    syntax = tuple(int(stoi[w]) for w in SYNTAX_WORDS)
    for arm in arms:
        anchor, anti = ex223.schedules(arm.condition)
        anchor = anchor | {"span": arm.span}
        for seed in range(arm.seeds):
            config = _make_config(align(tc.vocab_size, 64), seed, EPOCHS)
            config.tokenizer = tc.model_copy()
            config.model = config.model.model_copy(update={"tie_embeddings": arm.tie})
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "keying": arm.keying,
                    "slices": arm.slices,
                    "clean_rows": syntax if arm.clean else None,
                    "condition": arm.name,
                    "seed": seed,
                    "label": f"{arm.name}-s{seed}",
                    "corpus": prep["corpus"],
                    "evals": prep["evals"],
                    "probes": probes_by_keying[arm.keying],
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    anti: dict | None,
    corpus,
    traj_stride: int,
    probes,
    keying: Keying,
    slices: tuple[int, ...] | None,
    clean_rows: tuple[int, ...] | None,
    label: str,
) -> dict:
    """Ex-2.2.6's training step with the two new options of `train_anchored`; the readout's tying rides in
    `config.model`, the pull's span in `anchor`.
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights, line_p = z["mix/tokens"], z["slot_p"], z["weights"], z["mix/line_p"]
    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()
    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=LabelSpec(p=slot_p, keying=keying, pull="span"),
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        anchor_slices=slices,
        clean_rows=clean_rows,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    keep = ("epoch", "m_line", "m_op1", "m_span", "alpha_op1", "val_loss", "weight", "anti_weight")
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: traj[k].tolist() for k in keep if k in traj},
        "checkpoint": put(workdir / "model", name=f"ex-2.2.7-{label}-ckpt"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict:
    return {
        "n_runs": N_RUNS,
        "arms": [asdict(a) | {"tie": a.tie, "slices": a.slices, "clean": a.clean} for a in ARMS],
        "reference": REFERENCE,
        "stored": STORED,
        "strips": list(STRIPS),
        "syntax_words": list(SYNTAX_WORDS),
        "epochs": EPOCHS,
        "prompt_span": PROMPT,
        "whole_span": WHOLE,
        "red_dose": ex223.RED_DOSE,
        "nonred_dose": ex223.NONRED_DOSE,
    }


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("arrays", "traj", "val_loss", "train_loss")}


def publish_results(
    trained: list[dict], evaled: list[dict], scored: list[dict], rowed: list[dict], corpus_stats: dict, probes
) -> dict:
    import json

    from mini.store import get, put, set_ref

    metrics = {
        "runs": [_slim(r) for r in evaled],
        "scores": [_slim(r) for r in scored],
        "rows": rowed,
        "corpus": corpus_stats,
        "design": design(),
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.7-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.7-trajectories.json"))
    set_ref(PROBE_REF, probes)
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    arrays = {}
    for r in evaled + scored:
        kind = "eval" if "per_op" in r else "score"
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}-{kind}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{kind}/{name}": z[name] for name in z.files}
    set_ref(ARRAYS_REF, put(_npz(**arrays), name="ex-2.2.7-arrays.npz"))
    return {"n_runs": len(evaled), "n_rowed": len(rowed), "holdout_em": {r["label"]: r["holdout_em"] for r in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def main(ctx: Ctx) -> dict:
    prep = ctx.run(
        prepare_corpus,
        ex223.OP_NAMES,
        ex223.N_LINES,
        ex223.CORPUS_SEED,
        ex223.HOLDOUT_FRAC,
        ex223.N_PROBE,
        ex223.PROBE_SEED,
        ex223.PER_SLOT_RATE,
        ex223.RED_RATE,
        role="prep",
    )
    lined = ctx.run(
        line_keyed_probes,
        prep["probes"],
        list(prep["meta"].tokenizer_config.vocabulary),
        ex223.PER_SLOT_RATE,
        role="prep",
    )
    stored = ctx.run(stored_checkpoints, STORED_LABELS, role="prep")

    rows = cells(ARMS, prep, {"either": prep["probes"], "line": lined["probes"]})
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["corpus"] for r in rows],
        [TRAJ_STRIDE] * n,
        [r["probes"] for r in rows],
        [r["keying"] for r in rows],
        [r["slices"] for r in rows],
        [r["clean_rows"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [r["evals"] for r in rows],
        [r["probes"] for r in rows],
        [r["anchor"]["tau"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="eval",
    )
    scored = ctx.map(
        score_one,
        trained,
        [r["probes"] for r in rows],
        [r["condition"] for r in rows],
        [r["seed"] for r in rows],
        [r["label"] for r in rows],
        role="score",
    )

    # Part A on the stored checkpoints and on every new run alike.
    a_labels = STORED_LABELS + [r["label"] for r in rows]
    a_trained = [stored[lb] for lb in STORED_LABELS] + list(trained)
    a_probes = [prep["probes"]] * len(STORED_LABELS) + [r["probes"] for r in rows]
    a_conditions = [lb.rsplit("-s", 1)[0] for lb in STORED_LABELS] + [r["condition"] for r in rows]
    a_seeds = [int(lb.rsplit("-s", 1)[1]) for lb in STORED_LABELS] + [r["seed"] for r in rows]
    rowed = ctx.map(rows_one, a_trained, a_probes, a_conditions, a_seeds, a_labels, role="score")

    return ctx.run(publish_results, trained, evaled, scored, rowed, prep["stats"], lined["probes"], role="prep")


experiment = Experiment(
    name="ex-2.2.7",
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=900),
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        "eval": dict(gpu="L4", timeout=1200),
        "score": dict(gpu="L4", timeout=1800),
    },
)
