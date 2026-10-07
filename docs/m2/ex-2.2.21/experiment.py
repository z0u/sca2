"""Ex-2.2.21: the in-context grammar pilot, again — round 2 of the D2.2 quick route, on the reworked recipe.

Ex-2.2.16 stopped at its first rule: the control got a little under halfway from the floor to the Bayes ceiling, so
its anchored arms were trained and evaluated but never scored. Ex-2.2.17 to ex-2.2.20 reworked the grammar and the
recipe (the newline mask, a lower peak rate, the seven-op set, 200 epochs), and the control now reaches about nine
tenths of the way. This pilot trains the anchored arms again on that recipe and scores the rules ex-2.2.16 left
open, with the changes the report argues for: the corpus rule becomes a regression check, three label variants move
the pull toward the query `=` as references, and the hinge and operator rules become one exploratory analysis of
where the anchor can be edited, which chooses no arm.

The design constants above the DAG were frozen with the report at commit 2d84d2b. The DAG below them prepares the
seven-op corpus condition (the corpus, held-out set, and probe set ex-2.2.18 built and ex-2.2.19 trained on, plus
the arrays the label variants read) and a verification corpus beside it; trains every arm on ex-2.2.19's recipe;
then evaluates every run and runs the suppression pass on `SUPPRESSION_ARMS`. After the results were in, a confusion
pass on the same runs was added (post hoc): what the model answers in place of the anchored op once it is edited out.

    bin/mini run docs/m2/ex-2.2.21/experiment.py --app modal --max-containers 12 --budget 4h
    bin/mini status ex-2.2.21
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.incontext import context_length


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.20
    use), so this module's task bodies still cloudpickle by value for a remote worker.
    """
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


ex2219 = _load_sibling("ex-2.2.19", "ex2221_ex2219")
ex2218 = ex2219.ex2218
ex2217 = ex2219.ex2217
ex2216 = ex2219.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2219.OP_SET
OP_NAMES: tuple[str, ...] = OP_SET.ops
"""The seven-op set (`no-four`) of ex-2.2.18: table A+ less `screen`, `multiply`, `hsvmix`, and `exclusion`."""
N_OPS = len(OP_NAMES)

ANCHORED_OP = ex2216.ANCHORED_OP
"""`difference`, as in ex-2.2.14 to ex-2.2.16. Its closest partner in table A+, `exclusion`, is not in the set."""
assert ANCHORED_OP in OP_NAMES

CENTRE: tuple[int, float] = ex2216.CENTRE
"""Three examples at ρ = 0.3, the only corpus condition. The report rechecks the posterior on `difference` contexts
on the seven-op table."""

CUBE_RATE = ex2216.CUBE_RATE
MIDDLE_BAND = ex2216.MIDDLE_BAND

MODEL = ex2216.MODEL
EPOCHS = 200
"""The length ex-2.2.19 adopted for the seven-op set, at the peak rate, warmup, and schedule of ex-2.2.17's recipe;
ex-2.2.20 kept the plain schedule."""
assert EPOCHS in ex2219.SCOUT_EPOCHS

PEAK_LR = ex2219.PEAK_LR
WARMUP_EPOCHS = ex2219.WARMUP_EPOCHS
NEWLINE_MASK = True
"""Every arm, the controls included: attention does not cross a line break. Ex-2.2.16's two mask arms are now the
recipe, so they leave the arm list."""

CROP_POLICY = ex2216.CROP_POLICY
LABEL_RATE = ex2216.LABEL_RATE
HOLDOUT_CONTEXTS = ex2216.HOLDOUT_CONTEXTS
VERIFY_RATE = ex2216.VERIFY_RATE
PREFIX_THRESHOLD = ex2216.PREFIX_THRESHOLD
HINGE_CAP = ex2216.HINGE_CAP
HINGE_SOFTNESS = ex2216.HINGE_SOFTNESS
"""The anchor recipe of ex-2.2.14 (λ_a, τ, and the anti-subspace schedule, each as a fraction of training) and the
label and hinge settings of ex-2.2.16, unchanged."""

# --- The arms ----------------------------------------------------------------------------------------------

LABEL_VARIANTS: tuple[tuple[str, str], ...] = (
    *ex2216.LABEL_VARIANTS,
    ("prompt", "(e) the context up to and including the query `=`: the query answer and the line break left out"),
    ("query-eq", "(f) the query `=` alone"),
    ("every-eq", "(g) every `=`, the examples' and the query's: the positions whose next token depends on the op"),
)
"""Ex-2.2.16's whole-line label and its four variants, then three new ones. Ex-2.2.16's anchored arms put most of their
alignment on the answer positions and very little at the query `=`, whose state predicts the answer (the report has
the measurement). Variant (e) leaves the query answer out of the pull, to see where the pooled term settles without
it; variant (f) pulls only the query `=`, a position oracle that shows what an anchor there would allow; variant (g) pulls every `=`, the four positions whose next token is an answer
and so depends on the op, to see whether an anchor spread over the examples' `=` settles there too. (The first
example's `=` has no evidence before it, so its pull asks for the op before the context shows it.) All three are
references for E2, and not labels round 3 could adopt as they stand."""

PRIMARY = "anchor-whole"
CONTROL = "control"


@dataclass(frozen=True)
class Arm:
    """One arm of the pilot: what it trains and why it is here."""

    name: str
    group: str
    """Which question the arm serves: `control`, `label`, `site`, or `verify`."""
    anchored: bool = False
    label: str = "whole"
    hinge: bool = False
    verify: bool = False
    seeds: int = 3
    note: str = ""


SEEDS_REFERENCE = 5
"""The control and the whole-line arm, which every comparison runs through, get five seeds; the others get three,
as in ex-2.2.16. A comparison with the control then pairs five seeds with three or five, which narrows the seed band
by a tenth (five against three) or a fifth (five against five) over three against three, for four more runs."""

ARMS: tuple[Arm, ...] = (
    Arm(CONTROL, "control", seeds=SEEDS_REFERENCE, note="the un-anchored control"),
    Arm(PRIMARY, "label", anchored=True, seeds=SEEDS_REFERENCE, note="the whole-line label, the primary"),
    *(
        Arm(f"anchor-{label}", "label", anchored=True, label=label, note=desc)
        for label, desc in ex2216.LABEL_VARIANTS
        if label != "whole"
    ),
    Arm("anchor-prompt", "site", anchored=True, label="prompt", note=LABEL_VARIANTS[-3][1]),
    Arm("anchor-query-eq", "site", anchored=True, label="query-eq", note=LABEL_VARIANTS[-2][1]),
    Arm("anchor-every-eq", "site", anchored=True, label="every-eq", note=LABEL_VARIANTS[-1][1]),
    Arm("anchor-hinge", "site", anchored=True, hinge=True, note="the whole-line pull capped by a hinge"),
    Arm("control-verify", "verify", verify=True, note="the control with verification lines"),
    Arm("anchor-verify", "verify", anchored=True, verify=True, note="the whole-line arm with verification lines"),
)
"""Every arm trains at the center condition on the seven-op set. The control is the reference for all of them, and
the whole-line arm for the label, site, and verification arms."""

N_RUNS = sum(a.seeds for a in ARMS)
assert N_RUNS == 40

SITE_ARMS: tuple[str, ...] = (PRIMARY, "anchor-hinge")
"""The candidates of E2, the arms round 3 could adopt as they stand: the whole-line label, as the M3-shaped labeller,
then the same label with the hinge, a training setting M3 could also use. E2 describes them and chooses neither."""

SITE_REFERENCES: tuple[str, ...] = ("anchor-prompt", "anchor-query-eq", "anchor-every-eq")
"""Scored by E2 beside the candidates: they show where the anchor settles when the pull
leaves out the query answer, what an anchor at the query `=` would allow, and whether one spread over every `=`
lands at the query `=`."""

SUPPRESSION_ARMS: tuple[str, ...] = (*SITE_ARMS, *SITE_REFERENCES, CONTROL)
"""The arms the scoring-only suppression pass runs on: the candidates and references of E2, and the
control, whose damage under the same edit is subtracted."""

POST_HOC_SUPPRESSION_ARMS: tuple[str, ...] = ("anchor-no-emb",)
"""Added to the suppression pass after the results were in (post hoc, not a candidate of E2): the whole-line pull
off the embedding slice gave the best task score of the label arms and kept the anchor, so the report asks whether
its anchor can be edited as well as the whole-line one."""


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


SEED_OFFSET = 700
"""Fresh model seeds 700 onward; no earlier experiment has used them."""

# --- The rules -----------------------------------------------------------------------------------------------

REGRESSION_REF = "e200-lr0.00316"
"""The ex-2.2.19 runs the control is checked against: the four seeds at 200 epochs and the recipe peak rate."""

REGRESSION_TOL = ex2219.SHORTFALL_TOL
"""H1 (a): the control reproduces the recipe when its seed-mean held-out expected exact match is within this of the
seed mean of the ex-2.2.19 runs, either way: 0.015, the tolerance ex-2.2.19 gated length on."""

SEED_BAND_SD = ex2216.SEED_BAND_SD
"""The seed band between two arms at n₁ and n₂ seeds is `SEED_BAND_SD · σ · √(1/n₁ + 1/n₂)`, with σ the seed standard
deviation pooled over the two arms: ex-2.2.16's definition, generalized to unequal seed counts."""

MARGIN_KEEP = ex2216.MARGIN_KEEP
"""S1: a label variant "holds the anchor" when its seed-mean op margin is at least this share of the whole-line
margin."""

DOSE_GAMMAS = ex2216.DOSE_GAMMAS
REPULSION_LANDINGS = ex2216.REPULSION_LANDINGS
REPULSION_THRESHOLD = ex2216.REPULSION_THRESHOLD
REFLECT_GAMMA = ex2216.REFLECT_GAMMA
EDIT_SITES: tuple[str, ...] = (*ex2216.EDIT_SITES, "example answers")
SELECTIVITY_GATE = ex2216.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2216.GRADING_MIN_DAMAGE
"""The suppression pass and the two criteria of ex-2.2.16's operator rule (e), unchanged: the operators, their dose
axes, ex-2.2.16's three sites, a selectivity gate of 0.02 on each other op net of the control, and full-dose damage at least
half the way to the target null, net of the control."""

GRADE_DIP = 0.01
"""E2: the net drop on the anchored op may dip by at most this between adjacent doses and still count as grading, as
ex-2.2.1 and ex-2.2.2 allowed (they used 0.02, on larger drops). Half the selectivity gate, and about a quarter of a
dose step of the whole-line projection in ex-2.2.16."""

TASK_COST_TOL = 0.01
"""H1 (b): the whole-line arm may fall short of the control by at most this in seed-mean held-out expected exact
match. A little wider than the seed band of five seeds against five (about 0.009), so a miss is a cost the comparison
resolves, and narrower than the 0.015 of (a)."""

SCORED_SITES: tuple[str, ...] = ("query =", "every position")
"""E2 scores an operator against its criteria at either of these sites. Ex-2.2.16 scored only every position; the query `=` is added
because its state predicts the answer, and `query-eq` and `every-eq` put the anchor there."""

REPORTED_SITES: tuple[str, ...] = ("query ?", "example answers")
"""Sites the suppression pass also edits, reported with no gate. The example answers are new: ex-2.2.16's anchored
arms put much of their alignment there, and the query may read the op back from them."""


def cost_per_run(epochs: int = EPOCHS) -> float:
    """Dollars of L4 time for one run, from ex-2.2.19's unanchored runs; the anchored step costs about the same."""
    return ex2219.cost_per_run(epochs)


# =============================================================================================
# The DAG
# =============================================================================================
#
# Ex-2.2.16's tasks hard-code its eleven-op table (`TABLE`, `N_OPS`, the anchored op's index), so the tasks here
# are its `train_one`, `eval_one`, and `suppress_one` over an op set passed in, built from its helpers. The corpus
# is ex-2.2.18's `no-four` condition, rebuilt from the same seeds as ex-2.2.19 and ex-2.2.20 rebuilt it.

K, RHO = CENTRE
N_LINES = ex2216.N_LINES
N_TRAJ_PROBE = ex2216.N_TRAJ_PROBE
N_TRAJ_POINTS = ex2218.N_TRAJ_POINTS
N_TRAJ_EEM_PER_OP = ex2218.N_TRAJ_EEM_PER_OP
"""The held-out and trajectory sizes of ex-2.2.18 and ex-2.2.19: about fifty trajectory points per run, each scoring
the first 200 held-out contexts of every op, with the op margin and the leans on the probe set beside them."""

CORPUS_SEED = ex2218.CORPUS_SEED + 3 * ex2218.OP_SETS.index(OP_SET)
"""Ex-2.2.18's seed for the `no-four` condition: the corpus at this seed, the held-out set at +1, the probes at +2."""
VERIFY_SEED = 222_100
"""The verification corpus, which no earlier experiment built on the seven-op set; its held-out set at +1."""

MAIN_KEY = f"{ex2216.cond_key(K, RHO)}-{OP_SET.name}"
VERIFY_KEY = f"{MAIN_KEY}-verify"

ROLE_PULLS: dict[str, tuple[int, ...]] = {
    "prompt": tuple(range(ex2216.query_role(K, ex2216.QUERY_EQ) + 1)),
    "query-eq": (ex2216.query_role(K, ex2216.QUERY_EQ),),
    "every-eq": tuple(6 * i + ex2216.QUERY_EQ for i in range(K + 1)),
}
"""The roles each new label variant pulls in a labelled context of `K` examples (roles as in the report's Conditions:
`a ? b = y ,` per example, then `a ? b = y ⏎`): `prompt` every role up to and including the query `=`, `query-eq`
that one, `every-eq` the `=` of every example and of the query. Each goes to `LabelSpec` as its `prefix_ok` array
under `variant="prefix"`: a per-token boolean that narrows a drawn context to these positions, so the draw, the crop
policy, and the pooled term are the ones every other variant uses."""
assert ROLE_PULLS["prompt"] == tuple(range(22)) and ROLE_PULLS["every-eq"] == (3, 9, 15, 21)

EXAMPLE_ANSWERS = "example answers"


def example_answer_roles(k: int) -> tuple[int, ...]:
    return tuple(6 * i + 4 for i in range(k))


SYNTAX_TOKENS: tuple[tuple[str, str], ...] = (("?", "?"), ("=", "="), (",", ","), ("⏎", "\n"))
"""E1: the syntax tokens whose embedding alignment is reported, by display name and vocabulary word."""

PREFIX = "reports/m2/ex-2.2.21"
LABELS_REF = PREFIX + "/labels/{key}"
CORPUS_REF = PREFIX + "/corpus/{key}"
HOLDOUT_REF = PREFIX + "/holdout/{key}"
PROBES_REF = PREFIX + "/probes/{key}"
METRICS_REF = PREFIX + "/metrics"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
EVAL_REF = PREFIX + "/eval"
EVAL_ARRAYS_REF = PREFIX + "/eval-arrays/{label}"
SUPPRESSION_REF = PREFIX + "/suppression"
SUPPRESSION_ARRAYS_REF = PREFIX + "/suppression-arrays/{label}"
CONFUSION_REF = PREFIX + "/confusion"


def prepare_condition(
    ops: tuple[str, ...],
    k: int,
    rho: float,
    verify_rate: float,
    cube_rate: float,
    n_lines: int,
    seed: int,
    holdout_n: int,
    probe_n: int,
    key: str,
) -> dict:
    """One corpus condition over the op set *ops*: ex-2.2.18's `prepare_op_set` (same seeds, so the same corpus,
    held-out set, and probes) with the label arrays of ex-2.2.16's `prepare_corpus_condition` (`prefix_ok`,
    `sample_prob`, and on a verification corpus the `FALSE`-candidate `loss_mask`), plus one boolean per new label
    variant (`ROLE_PULLS`), true at the roles it pulls on every completion line. The arrays read by the posterior
    are filled on the anchored op's contexts only; the role arrays on every completion line, since the labeller
    draws only on the anchored op's contexts anyway.
    """
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.incontext import encode_corpus, line_boundaries, sample_corpus, vocabulary
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    P = ex2216._get_posterior()
    table = ex2218.table_of(ops)
    a_id = ops.index(ANCHORED_OP)
    corpus = sample_corpus(
        n_lines, seed, k, rho, op_table=table, cube_rate=cube_rate, rounding=ex2216.ROUNDING, verify_rate=verify_rate
    )
    tokenizer_config = TokenizerConfig(vocabulary=sorted(vocabulary()))
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = encode_corpus(corpus, tokenizer.stoi)
    newline_id = tokenizer.stoi["\n"]
    context_op = op_ids_of(corpus, table)
    context_len = np.array([c.n_tokens for c in corpus], dtype=np.int32)
    line_starts = line_boundaries(tokens, newline_id)
    color_index = {c: i for i, c in enumerate(grammar.colors())}
    op_index = {o.name: i for i, o in enumerate(table)}
    post_table = P.build_table(table)

    # --- The posterior-driven label arrays (variants c and d), as ex-2.2.16 builds them -------------------
    anchored_idx = np.flatnonzero(context_op == a_id)
    prefix_ok = np.zeros(len(tokens), dtype=bool)
    sample_prob = np.zeros(len(tokens), dtype=np.float32)
    batch = ex2216._post_contexts(P, [corpus[i] for i in anchored_idx], color_index, op_index)
    prefix_post = P.prefix_posteriors(post_table, batch, rho, cube_rate)[:, :, a_id]
    post_after = np.concatenate([np.full((len(anchored_idx), 1), 1.0 / len(table)), prefix_post], axis=1)
    raw = ex2216.LABEL_RATE / post_after[:, k].mean() * post_after[:, k]
    scaled = np.clip(raw, 0.0, 1.0)
    for local_i, ctx_i in enumerate(anchored_idx):
        start, length = int(line_starts[ctx_i]), int(context_len[ctx_i])
        at_role = np.minimum(np.arange(length) // 6, k)
        prefix_ok[start : start + length] = post_after[local_i, at_role] >= PREFIX_THRESHOLD
        sample_prob[start : start + length] = scaled[local_i]

    # --- The role arrays of the new variants (e to g), on every completion line --------------------------
    role = np.arange(len(tokens)) - np.repeat(line_starts, context_len)
    completion = np.repeat(context_len == context_length(k), context_len)
    pulls = {f"pull_{name}": completion & np.isin(role, roles) for name, roles in ROLE_PULLS.items()}

    loss_mask = None
    if verify_rate > 0:
        loss_mask = np.zeros(len(tokens), dtype=bool)
        for i, c in enumerate(corpus):
            if c.verify is not None and not c.verify.verdict:
                loss_mask[int(line_starts[i]) + k * 6 + 4] = True

    holdout = ex2216._sample_by_op(table, k, rho, cube_rate, verify_rate, holdout_n, seed + 1)
    ho_batch = ex2216._post_contexts(P, holdout, color_index, op_index)
    ho_post = P.posterior(post_table, ho_batch, rho, cube_rate)
    ho_ceiling = P.expected_match(post_table, ho_batch, ho_post)
    ho_floor = P.floor(post_table, ho_batch)
    ho_op = op_ids_of(holdout, table)
    ho_verdict = np.array([(c.verify.verdict if c.verify is not None else -1) for c in holdout], dtype=np.int8)
    done = ho_verdict < 0

    probes = ex2216._sample_by_op(table, k, rho, cube_rate, 0.0, probe_n, seed + 2)
    probe_tokens = encode_corpus(probes, tokenizer.stoi).reshape(len(table) * probe_n, context_length(k))

    n_chars = sum(len(w) for c in corpus for w in c.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"in-context grammar corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)

    def per_op(x: np.ndarray) -> list[float]:
        return [float(x[done & (ho_op == o)].mean()) for o in range(len(ops))]

    return {
        "key": key,
        "k": k,
        "ops": list(ops),
        "meta": meta,
        "newline_id": newline_id,
        "stats": {
            "key": key,
            "ops": list(ops),
            "k": k,
            "rho": rho,
            "verify_rate": verify_rate,
            "cube_rate": cube_rate,
            "n_lines": n_lines,
            "seed": seed,
            "total_tokens": int(len(tokens)),
            "n_anchored_contexts": int(len(anchored_idx)),
            "sample_prob_clipped": int((raw > 1).sum()),
            "sample_prob_mean": float(scaled.mean()),
            "ceiling": float(ho_ceiling[done].mean()),
            "floor": float(ho_floor[done].mean()),
            "ceiling_per_op": per_op(ho_ceiling),
            "floor_per_op": per_op(ho_floor),
            "posterior_anchored_per_op": per_op(ho_post[:, a_id]),
            "pull_roles": {name: list(roles) for name, roles in ROLE_PULLS.items()},
        },
        "corpus": put(corpus_dir, name=f"ex-2.2.21-{key}-corpus"),
        "labels": put(
            ex2216._npz(
                op_ids=context_op,
                context_len=context_len,
                prefix_ok=prefix_ok,
                sample_prob=sample_prob,
                **pulls,
                **({"loss_mask": loss_mask} if loss_mask is not None else {}),
            ),
            name=f"ex-2.2.21-{key}-labels.npz",
        ),
        "holdout": put(
            ex2216._npz(
                tokens=encode_corpus(holdout, tokenizer.stoi),
                op_ids=ho_op,
                context_len=np.array([c.n_tokens for c in holdout], dtype=np.int32),
                posterior=ho_post,
                ceiling=ho_ceiling,
                verify_verdict=ho_verdict,
            ),
            name=f"ex-2.2.21-{key}-holdout.npz",
        ),
        "probes": put(ex2216._npz(tokens=probe_tokens), name=f"ex-2.2.21-{key}-probes.npz"),
    }


def check_corpus(stats: dict) -> dict:
    """The rebuilt `no-four` condition has the token count and held-out ceiling ex-2.2.19 published, so it is the
    corpus and held-out set the control is checked against (H1 (a)).
    """
    import json
    import tempfile
    from pathlib import Path

    from mini.store import get, get_ref

    art = get_ref(ex2219.EVAL_REF)
    assert art is not None, f"{ex2219.EVAL_REF} is not published"
    published = json.loads(get(art, Path(tempfile.mkdtemp()) / "eval.json").read_text())["op_set"]
    assert published["total_tokens"] == stats["total_tokens"], "rebuilt corpus differs in length"
    assert abs(published["ceiling"] - stats["ceiling"]) < 1e-9, f"{stats['ceiling']} vs {published['ceiling']}"
    return {"ceiling": published["ceiling"], "total_tokens": published["total_tokens"]}


# --- Training ---------------------------------------------------------------------------------------


def label_variant(
    label: str, n_layer: int
) -> tuple[Literal["whole", "latter", "prefix", "sampled"], tuple | None, str]:
    """The `LabelSpec.variant`, the training step's `anchor_slices`, and the labels array `LabelSpec.prefix_ok`
    reads (empty for none), for one label short name. `no-emb` is the whole-line pull off the embedding slice, as in
    ex-2.2.16; the three new variants are `prefix` over their role arrays.
    """
    if label == "no-emb":
        return "whole", tuple(range(1, n_layer + 1)), ""
    if label in ("whole", "latter", "sampled"):
        return label, None, ""
    if label == "prefix":
        return "prefix", None, "prefix_ok"
    if label in ROLE_PULLS:
        return "prefix", None, f"pull_{label}"
    raise ValueError(f"unknown label variant {label!r}")


def recipe_config(meta, model_seed: int, epochs: int):
    """Ex-2.2.19's config for the seven-op set at *epochs* and the recipe peak rate, at *model_seed*: built by its
    own `rows_of`, so the control differs from its `e200-lr0.00316` runs in the seed alone.
    """
    (row,) = ex2219.rows_of([(epochs, PEAK_LR, model_seed - ex2219.SEED_OFFSET)], meta)
    config = row["config"]
    assert config.seed == model_seed and config.scheduler.warmup_epochs == WARMUP_EPOCHS
    assert config.model.line_mask_token is not None and NEWLINE_MASK
    return config, row["anchor"]


def cells(arms: tuple[Arm, ...], preps: dict[str, dict], seeds: int | None = None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run, in the order of *arms*: the config, both schedules, the labeller's variant, and the corpus
    condition it trains on. *seeds* caps every arm's seed count and *epochs* sets the length, so a smoke run goes
    through the same code at a fraction of the size.
    """
    rows = []
    for a in arms:
        key = VERIFY_KEY if a.verify else MAIN_KEY
        prep = preps[key]
        n_layer = ex2216.model_dims(MODEL)[1]
        variant, anchor_slices, pull = label_variant(a.label, n_layer)
        base = ex2216.Condition229(
            a.name, 1, a.name, lam=(ex2216.LAM if a.anchored else 0.0), tau=ex2216.TAU, epochs=epochs, ops=OP_NAMES,
            n_lines=N_LINES,
        )  # fmt: skip
        anchor, anti = ex2216.schedules(base)
        if a.hinge:
            anchor = anchor | {"hinge": (HINGE_CAP, HINGE_SOFTNESS)}
        for seed in range(a.seeds if seeds is None else min(a.seeds, seeds)):
            config, unanchored = recipe_config(prep["meta"], SEED_OFFSET + seed, epochs)
            assert a.anchored or anchor == unanchored, "the control is ex-2.2.19's run"
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "variant": variant,
                    "anchor_slices": anchor_slices,
                    "pull": pull,
                    "corpus_key": key,
                    "arm": a.name,
                    "seed": seed,
                    "model_seed": SEED_OFFSET + seed,
                    "epochs": epochs,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    anti: dict | None,
    variant: Literal["whole", "latter", "prefix", "sampled"],
    anchor_slices,
    pull: str,
    ops: tuple[str, ...],
    corpus,
    labels,
    probes,
    holdout,
    k: int,
    traj_stride: int,
    n_traj_eem: int,
    label: str,
) -> dict:
    """Train one run over the op set *ops*: ex-2.2.16's context-keyed anchor (crop policy `whole`, the label variant
    and slices given), recording every *traj_stride* steps both ex-2.2.16's trajectory on the probe set (the op
    margin over the whole context, the first-operand lean, and the trailing-fragment lean, at the last block) and
    ex-2.2.18's on the first *n_traj_eem* held-out contexts of every op (expected exact match and calibration KL).
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec, alignment
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    a_id = ops.index(ANCHORED_OP)
    final = config.model.n_layer

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=a_id,
            label_rate=ex2216.LABEL_RATE,
            variant=variant,
            prefix_ok=(z[pull] if variant == "prefix" else None),
            sample_prob=(z["sample_prob"] if variant == "sampled" else None),
            context_len=z["context_len"],
        )
        loss_mask = z["loss_mask"] if "loss_mask" in z.files else None
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]
    per_op = len(probe_tokens) // n_ops
    weights = np.concatenate([np.full(per_op, 1.0 if o == ANCHORED_OP else 0.0) for o in ops])
    weights = weights / weights.sum()

    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    sub = np.concatenate([np.flatnonzero(ho.op_ids == o)[:n_traj_eem] for o in range(n_ops)])
    sub_tokens, sub_op_ids, sub_post = ho.tokens[sub], ho.op_ids[sub], ho.posterior[sub]
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    sub_ho = ex2216.Holdout(
        sub_tokens,
        sub_op_ids,
        sub_post,
        np.empty(0),
        np.empty((0, 0), dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    probe_ctx = ex2216._post_queries(P, sub_ho, k, tok2color)
    identity = projection(Subspace.axis(config.model.n_embd), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)

    extra: dict[str, list] = {
        k_: [] for k_ in ("m_context", "op1_lean", "fragment_lean", "eem", "eem_per_op", "kl", "kl_per_op")
    }

    def on_record(_index: int, model) -> None:
        cos = alignment(model, probe_tokens)  # (L1, N, T)
        extra["m_context"].append(ex2216.context_margin(cos, weights))
        extra["op1_lean"].append(float(cos[final, :, 0].mean()))
        total, count = 0.0, 0
        for s in ex2216.fragment_starts(k):
            fc = alignment(model, np.ascontiguousarray(probe_tokens[:, s:]))
            total += float(fc[final].sum())
            count += fc.shape[1] * fc.shape[2]
        extra["fragment_lean"].append(total / count)
        p = ex2216._color_probs(logits_at(model, sub_tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, probe_ctx, p)
        kl = P.kl(P.predictive(table, probe_ctx, sub_post), p)
        extra["eem"].append(float(eem.mean()))
        extra["eem_per_op"].append([float(eem[sub_op_ids == o].mean()) for o in range(n_ops)])
        extra["kl"].append(float(kl.mean()))
        extra["kl_per_op"].append([float(kl[sub_op_ids == o].mean()) for o in range(n_ops)])

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=(AntiSpec(**anti) if anti is not None else None),
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        probe_line_w=weights,
        crop=CROP_POLICY,
        anchor_slices=anchor_slices,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=tokenizer.stoi["\n"],
        min_line_tokens=context_length(k),
        loss_mask=loss_mask,
        on_record=on_record,
    )
    epoch_train_loss = np.asarray([m.train_loss for m in metrics], dtype=np.float64)
    epoch_axis = np.arange(1, len(epoch_train_loss) + 1, dtype=np.float64)
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": epoch_train_loss.tolist(),
        "traj": {
            kk: np.asarray(traj[kk]).tolist()
            for kk in ("step", "epoch", "lr", "weight", "anti_weight", "val_loss")
            if kk in traj
        }
        | {"train_loss": np.interp(np.asarray(traj["epoch"], dtype=np.float64), epoch_axis, epoch_train_loss).tolist()}
        | extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.21-{label}-ckpt"),
    }


# --- Evaluation and the suppression pass ----------------------------------------------------------


def _per_op(x: np.ndarray, op_ids: np.ndarray, n_ops: int) -> list[float | None]:
    return [float(x[op_ids == o].mean()) if (op_ids == o).any() else None for o in range(n_ops)]


def eval_one(checkpoint, holdout, verify_holdout, ops: tuple[str, ...], k: int, label: str) -> dict:
    """Ex-2.2.16's `eval_one` over the op set *ops*, on the completion contexts of *holdout* (the main condition's
    held-out set for every arm, the verification arms included, so every comparison is on the same contexts): the
    task and calibration measurements, the alignment by op, slice, and role, the op margin, the alignment at the
    query against the posterior on the anchored op, and the two leans. Beside them, the alignment of the syntax
    embeddings at the embedding slice (E1), and the mean posterior on the anchored op per op (E2). On the
    verification lines of *verify_holdout*, if given: verification accuracy (S2).
    """
    from sca.anchoring import alignment
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "eval" / label
    model, tokenizer, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = ops.index(ANCHORED_OP)
    ctx = ex2216._post_queries(P, ho, k, tok2color)
    final = len(model.transformer.blocks)
    q_role, eq_role = ex2216.query_role(k, ex2216.QUERY_Q), ex2216.query_role(k, ex2216.QUERY_EQ)

    identity = projection(Subspace.axis(model.transformer.wte.shape[1]), 0.0)
    p = ex2216._color_probs(logits_at(model, ho.tokens, identity, (), eq_role), color_ids)
    eem = ex2216._match(table, ctx, p)
    floor = P.floor(table, ctx)
    kl = P.kl(P.predictive(table, ctx, ho.posterior), p)
    post_a = ho.posterior[:, a_id]
    task = {
        name: {"all": float(x.mean()), "per_op": _per_op(x, ho.op_ids, n_ops)}
        for name, x in (("eem", eem), ("ceiling", ho.ceiling), ("floor", floor), ("kl", kl))
    }
    task["color_mass"] = {"all": float(p.sum(1).mean())}
    task["posterior_anchored"] = {"all": float(post_a.mean()), "per_op": _per_op(post_a, ho.op_ids, n_ops)}

    cos = alignment(model, ho.tokens)  # (L1, N, T)
    anchored = ho.op_ids == a_id
    weights = anchored / anchored.sum()
    by_slice = np.einsum("n,lnt->lt", weights, cos) - cos.mean(axis=1)
    role_mean = np.stack([cos[:, ho.op_ids == o].mean(axis=1) for o in range(n_ops)])  # (ops, L1, T)
    syntax_ids = np.array([[tokenizer.stoi[w]] for _, w in SYNTAX_TOKENS], dtype=np.int32)
    syntax = alignment(model, syntax_ids)[0, :, 0]  # the embedding slice, one token alone

    at_query = cos[:, :, [q_role, eq_role]].transpose(1, 0, 2)  # (N, L1, 2)
    evidence = {
        "edges": list(ex2216.EVIDENCE_BINS),
        "anchored": ex2216._binned(post_a[anchored], at_query[anchored], ex2216.EVIDENCE_BINS),
        "other": ex2216._binned(post_a[~anchored], at_query[~anchored], ex2216.EVIDENCE_BINS),
    }

    fragments, total, count = [], 0.0, 0
    for s in ex2216.fragment_starts(k):
        fc = alignment(model, np.ascontiguousarray(ho.tokens[:, s:]))
        total += float(fc[final].sum())
        count += fc.shape[1] * fc.shape[2]
        fragments.append(
            {
                "start": s,
                "lean": float(fc[final].mean()),
                "lean_by_slice": fc.mean(axis=(1, 2)).tolist(),
                "profile": fc.mean(axis=1).tolist(),
            }
        )
    leans = {
        "op1": float(cos[final, :, 0].mean()),
        "op1_by_slice": cos[:, :, 0].mean(axis=1).tolist(),
        "fragment": total / count,
        "fragments": fragments,
    }

    verify = None
    if verify_holdout is not None:
        vh = ex2216.load_holdout(get(verify_holdout, workdir / "verify-holdout.npz"), k)
        t_id, f_id = tokenizer.stoi["TRUE"], tokenizer.stoi["FALSE"]
        mark = ex2216.query_role(k, ex2216.VERDICT_MARK)
        lg = logits_at(model, vh.verify_tokens, identity, (), mark).astype(np.float64)
        right = (lg[:, t_id] > lg[:, f_id]) == vh.verdict
        lg -= lg.max(axis=1, keepdims=True)
        pv = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
        verify = {
            "n": int(len(right)),
            "accuracy": float(right.mean()),
            "accuracy_per_op": _per_op(right.astype(float), vh.verify_op_ids, n_ops),
            "accuracy_by_verdict": {"TRUE": float(right[vh.verdict].mean()), "FALSE": float(right[~vh.verdict].mean())},
            "verdict_mass": float(np.where(vh.verdict, pv[:, t_id], pv[:, f_id]).mean()),
        }

    arrays = put(
        ex2216._npz(
            op_ids=ho.op_ids.astype(np.int8),
            posterior=ho.posterior.astype(np.float32),
            query_pair=ctx.query_pair,
            eem=eem.astype(np.float32),
            ceiling=ho.ceiling.astype(np.float32),
            floor=floor.astype(np.float32),
            kl=kl.astype(np.float32),
            p=p.astype(np.float16),
            align_query=at_query.astype(np.float16),
        ),
        name=f"ex-2.2.21-{label}-eval.npz",
    )
    return {
        "label": label,
        "k": k,
        "ops": list(ops),
        "n": int(len(ho.tokens)),
        "n_per_op": [int((ho.op_ids == o).sum()) for o in range(n_ops)],
        "roles": {"query ?": q_role, "query =": eq_role, EXAMPLE_ANSWERS: list(example_answer_roles(k))},
        "task": task,
        "margin": {
            "value": ex2216.context_margin(cos, weights),
            "by_slice": by_slice.max(axis=1).tolist(),
            "role_by_slice": by_slice.argmax(axis=1).tolist(),
        },
        "alignment": role_mean.tolist(),
        "syntax_embeddings": {name: float(v) for (name, _), v in zip(SYNTAX_TOKENS, syntax, strict=True)},
        "evidence": evidence,
        "leans": leans,
        "verify": verify,
        "arrays": arrays,
    }


def site_positions(site: str, k: int) -> np.ndarray:
    """`ex2216.site_positions`, with the example answers added."""
    if site != EXAMPLE_ANSWERS:
        return ex2216.site_positions(site, k)
    mask = np.zeros(context_length(k), dtype=np.float32)
    mask[list(example_answer_roles(k))] = 1
    return mask


def suppress_one(checkpoint, holdout, ops: tuple[str, ...], k: int, label: str) -> dict:
    """Ex-2.2.16's `suppress_one` over the op set *ops* and the sites `EDIT_SITES`: each operator at each dose and
    site, at every slice, scored on the held-out completion contexts of every op. Per op: expected exact match on
    the clean pass, under each edit, and under the target null, the KL from the target-null predictive, and the
    mean posterior on the anchored op (E2 plots each other op's net drop against it). Per context, to the store.
    """
    from sca.intervention import Subspace, logits_at
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "suppress" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = ops.index(ANCHORED_OP)
    ctx = ex2216._post_queries(P, ho, k, tok2color)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))

    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    q_null = P.predictive(table, ctx, null_post)
    null_eem = P.expected_match(table, ctx, null_post)

    edits = ex2216.suppression_edits(sub)
    p = ex2216._color_probs(logits_at(model, ho.tokens, edits[0][2], (), eq_role), color_ids)
    clean_eem, clean_kl = ex2216._match(table, ctx, p), P.kl(q_null, p)

    records, eem, kl_null = [], [], []
    for site in EDIT_SITES:
        mask = site_positions(site, k)
        for operator, dose, edit in edits:
            p = ex2216._color_probs(logits_at(model, ho.tokens, edit, every, eq_role, mask), color_ids)
            e, d = ex2216._match(table, ctx, p), P.kl(q_null, p)
            eem.append(e)
            kl_null.append(d)
            records.append(
                {
                    "operator": operator,
                    "dose": dose,
                    "site": site,
                    "eem": _per_op(e, ho.op_ids, n_ops),
                    "drop": _per_op(clean_eem - e, ho.op_ids, n_ops),
                    "kl_null": _per_op(d, ho.op_ids, n_ops),
                }
            )
    post_a = ho.posterior[:, a_id]
    arrays = put(
        ex2216._npz(
            op_ids=ho.op_ids.astype(np.int8),
            posterior_anchored=post_a.astype(np.float32),
            clean_eem=clean_eem.astype(np.float32),
            null_eem=null_eem.astype(np.float32),
            clean_kl_null=clean_kl.astype(np.float32),
            eem=np.stack(eem).astype(np.float32),  # (site × edit, N), in the order of `edits`
            kl_null=np.stack(kl_null).astype(np.float32),
        ),
        name=f"ex-2.2.21-{label}-suppression.npz",
    )
    return {
        "label": label,
        "k": k,
        "ops": list(ops),
        "slices": list(every),
        "posterior_anchored": _per_op(post_a, ho.op_ids, n_ops),
        "clean": {"eem": _per_op(clean_eem, ho.op_ids, n_ops), "kl_null": _per_op(clean_kl, ho.op_ids, n_ops)},
        "null": {"eem": _per_op(null_eem, ho.op_ids, n_ops)},
        "edits": records,
        "arrays": arrays,
    }


def _answered_as(table, ctx, p: np.ndarray) -> np.ndarray:
    """`(n, n_ops)`: per context, the mass of *p* on the answers each op gives on the query pair, weighted as
    `_match` weights the true op. Ops that share an answer on a pair both take its mass, so a row can sum past 1.
    """
    from dataclasses import replace

    return np.stack(
        [ex2216._match(table, replace(ctx, true_op=np.full_like(ctx.true_op, o)), p) for o in range(table.n_ops)],
        axis=1,
    )


def confuse_one(checkpoint, holdout, ops: tuple[str, ...], k: int, label: str) -> dict:
    """Post hoc, added after the results were in: what the model answers in place of the anchored op once it is
    edited out. The full projection at each scored site, beside the clean pass and the two Bayes references (the
    posterior, and the target null with the anchored op removed). Each is returned as a matrix over the held-out
    completion contexts: row i, column j is the mean mass on the answers of op j in the contexts of op i.
    """
    from sca.intervention import Subspace, logits_at
    from mini.store import get

    n_ops = len(ops)
    workdir = get_data_dir() / "confuse" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = ops.index(ANCHORED_OP)
    ctx = ex2216._post_queries(P, ho, k, tok2color)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))
    full = next(e for o, g, e in ex2216.suppression_edits(sub) if o == "projection" and g == max(DOSE_GAMMAS))

    def matrix(p: np.ndarray) -> list[list[float]]:
        m = _answered_as(table, ctx, p)
        return [m[ho.op_ids == o].mean(axis=0).tolist() for o in range(n_ops)]

    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    return {
        "label": label,
        "ops": list(ops),
        "bayes": matrix(P.predictive(table, ctx, ho.posterior)),
        "null": matrix(P.predictive(table, ctx, null_post)),
        "clean": matrix(ex2216._color_probs(logits_at(model, ho.tokens, full, (), eq_role), color_ids)),
        "edits": {
            site: matrix(
                ex2216._color_probs(
                    logits_at(model, ho.tokens, full, every, eq_role, site_positions(site, k)), color_ids
                )
            )
            for site in SCORED_SITES
        },
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.21",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "op_set": OP_SET.name,
        "arms": [asdict(a) for a in ARMS],
        "n_runs": N_RUNS,
        "seed_offset": SEED_OFFSET,
        "centre": list(CENTRE),
        "cube_rate": CUBE_RATE,
        "epochs": EPOCHS,
        "peak_lr": PEAK_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "newline_mask": NEWLINE_MASK,
        "label_rate": LABEL_RATE,
        "verify_rate": VERIFY_RATE,
        "holdout_contexts": HOLDOUT_CONTEXTS,
        "pull_roles": {name: list(roles) for name, roles in ROLE_PULLS.items()},
        "final_slice": ex2216.FINAL_SLICE,
        "offsets": {"query ?": ex2216.QUERY_Q, "query =": ex2216.QUERY_EQ, "verdict mark": ex2216.VERDICT_MARK},
        "syntax_tokens": [name for name, _ in SYNTAX_TOKENS],
        "evidence_bins": list(ex2216.EVIDENCE_BINS),
        "middle_band": list(MIDDLE_BAND),
        "suppression": {
            "arms": list(SUPPRESSION_ARMS),
            "post_hoc_arms": list(POST_HOC_SUPPRESSION_ARMS),
            "sites": list(EDIT_SITES),
            "scored_sites": list(SCORED_SITES),
            "reported_sites": list(REPORTED_SITES),
            "gammas": list(DOSE_GAMMAS),
            "landings": list(REPULSION_LANDINGS),
            "threshold": REPULSION_THRESHOLD,
            "kind": ex2216.REPULSION_KIND,
            "reflect_gamma": REFLECT_GAMMA,
            "selectivity_gate": SELECTIVITY_GATE,
            "grading_min_damage": GRADING_MIN_DAMAGE,
            "grade_dip": GRADE_DIP,
        },
        "rules": {
            "regression_ref": REGRESSION_REF,
            "regression_tol": REGRESSION_TOL,
            "task_cost_tol": TASK_COST_TOL,
            "seed_band_sd": SEED_BAND_SD,
            "margin_keep": MARGIN_KEEP,
            "hinge_cap": HINGE_CAP,
        },
    }


def publish(
    rows: list[dict],
    preps: dict[str, dict],
    trained: list[dict],
    evaled: list[dict],
    suppressed: list[dict],
    confused: list[dict],
):
    """Every ref the report reads: the metrics (design and corpus statistics), the trajectories, the eval, the
    suppression pass, and the post hoc confusion pass (JSON, one record per run with its arm and seed), every per-context array, every checkpoint,
    and every corpus condition's corpus, labels, held-out set, and probes.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {k: r[k] for k in ("label", "arm", "seed", "model_seed", "epochs", "corpus_key")} for r in rows}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for key, p in preps.items():
        set_ref(CORPUS_REF.format(key=key), p["corpus"])
        set_ref(LABELS_REF.format(key=key), p["labels"])
        set_ref(HOLDOUT_REF.format(key=key), p["holdout"])
        set_ref(PROBES_REF.format(key=key), p["probes"])
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in suppressed:
        set_ref(SUPPRESSION_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    metrics = {"design": design(), "corpus": {key: p["stats"] for key, p in preps.items()}}
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.21-metrics.json"))
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.21-trajectories.json"))
    body = {"design": design(), "runs": [slim(r) for r in evaled]}
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.21-eval.json"))
    body = {"design": design(), "runs": [slim(r) for r in suppressed]}
    set_ref(SUPPRESSION_REF, put(json.dumps(body).encode(), name="ex-2.2.21-suppression.json"))
    body = {"runs": [slim(r) for r in confused]}
    set_ref(CONFUSION_REF, put(json.dumps(body).encode(), name="ex-2.2.21-confusion.json"))
    return {
        "n_runs": len(trained),
        "n_suppressed": len(suppressed),
        "eem": {r["label"]: r["task"]["eem"]["all"] for r in evaled},
    }


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx,
    arms: tuple[Arm, ...],
    n_lines: int,
    holdout_n: int,
    probe_n: int,
    n_traj_points: int,
    n_traj_eem: int,
    seeds: int | None = None,
    epochs: int = EPOCHS,
) -> dict:
    """The whole DAG: prepare both corpus conditions, train *arms*, evaluate every run, suppress on the runs of
    `SUPPRESSION_ARMS`, and publish. Sizes, seeds, and length are arguments so a smoke run takes the same path.
    """
    preps = {
        key: ctx.run(
            prepare_condition, OP_NAMES, K, RHO, rate, CUBE_RATE, n_lines, seed, holdout_n, probe_n, key, role="prep"
        )
        for key, rate, seed in ((MAIN_KEY, 0.0, CORPUS_SEED), (VERIFY_KEY, VERIFY_RATE, VERIFY_SEED))
    }
    if (n_lines, holdout_n, probe_n) == (N_LINES, HOLDOUT_CONTEXTS, N_TRAJ_PROBE):
        ctx.run(check_corpus, preps[MAIN_KEY]["stats"], role="prep")
    main_prep = preps[MAIN_KEY]
    rows = cells(arms, preps, seeds, epochs)
    n = len(rows)
    stride = [
        ex2217.traj_stride_for(
            epochs, ex2217.epoch_length_of(preps[r["corpus_key"]]["meta"].total_tokens, r["config"]), n_traj_points
        )
        for r in rows
    ]
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["pull"] for r in rows],
        [OP_NAMES] * n,
        [preps[r["corpus_key"]]["corpus"] for r in rows],
        [preps[r["corpus_key"]]["labels"] for r in rows],
        [preps[r["corpus_key"]]["probes"] for r in rows],
        [main_prep["holdout"]] * n,
        [K] * n,
        stride,
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    ckpt = [t["checkpoint"] for t in trained]
    evaled = ctx.map(
        eval_one,
        ckpt,
        [main_prep["holdout"]] * n,
        [preps[VERIFY_KEY]["holdout"] if r["corpus_key"] == VERIFY_KEY else None for r in rows],
        [OP_NAMES] * n,
        [K] * n,
        [r["label"] for r in rows],
        role="eval",
    )
    sup = [i for i, r in enumerate(rows) if r["arm"] in SUPPRESSION_ARMS + POST_HOC_SUPPRESSION_ARMS]
    suppressed = ctx.map(
        suppress_one,
        [ckpt[i] for i in sup],
        [main_prep["holdout"]] * len(sup),
        [OP_NAMES] * len(sup),
        [K] * len(sup),
        [rows[i]["label"] for i in sup],
        role="suppress",
    )
    confused = ctx.map(
        confuse_one,
        [ckpt[i] for i in sup],
        [main_prep["holdout"]] * len(sup),
        [OP_NAMES] * len(sup),
        [K] * len(sup),
        [rows[i]["label"] for i in sup],
        role="suppress",
    )
    slim_rows = [{k: v for k, v in r.items() if k not in ("config", "anchor", "anti")} for r in rows]
    return ctx.run(publish, slim_rows, preps, trained, evaled, suppressed, confused, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx, ARMS, N_LINES, HOLDOUT_CONTEXTS, N_TRAJ_PROBE, N_TRAJ_POINTS, N_TRAJ_EEM_PER_OP)


COMPUTE = {
    # Two corpus builds of 300,000 contexts with the posterior over the anchored contexts and held-out sets; the
    # corpus check; and the fan-in that writes every ref.
    "prep": dict(cpu=2, timeout=1800),
    # About 52,800 steps (200 epochs, about 20 minutes on an L4 in ex-2.2.19), with the trajectory's alignment and
    # scoring passes at about fifty points.
    "train": dict(gpu="L4", timeout=2 * 3600, watchdog=900, watchdog_grace=900),
    # Forward passes over 14,000 held-out contexts: the eval makes k + 3 of them, the suppression pass 29 (seven
    # edits at four sites, and the clean pass).
    "eval": dict(gpu="L4", timeout=900),
    "suppress": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="ex-2.2.21", main=main, roles=COMPUTE)
