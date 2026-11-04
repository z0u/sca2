# ruff: noqa: B018
# title: Directions found after training, beside the anchor

# A preregistered comparison: the anchored edit of the recipe of record against edits along directions fitted after
# training on the controls at the same seeds, each method tuned on a split it is never scored on. The plan is
# docs/m2/d2.2/post-hoc-comparison.md. The results were filled in place once the run had finished.
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesGrid, AxesRow, figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """An experiment module by path, registered while it loads (its dataclasses need it), as the reports do."""
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


ex = _load_sibling("post-hoc-compare", "post_hoc_compare_ex")
fc = ex.fc
fl = _load_sibling("fixed-labels", "post_hoc_compare_fl")
sd = _load_sibling("filtered-corpus", "post_hoc_compare_sd")

# --- The stored results ---------------------------------------------------------------------------------------


SEEDS: list[int] = list(ex.SEEDS)


def fetch_all() -> tuple[dict, dict, dict, dict, dict]:
    """The results and restricted selections of this experiment, the axis scoring of the anchored runs and of the
    fixed-labels runs, and per run the arrays of `finish_one`, keyed by label, with the held-out contexts (their
    ops, posteriors, query operands, and target-null scores) from the dose arrays of one anchored run, and the clean
    answers of the sans-diff model at each seed.
    """
    store = project_store()
    arrays = {r["label"]: ex.ARRAYS_REF.format(label=r["label"]) for r in ex.runs()}
    dose = fc.DOSE_ARRAYS_REF.format(label=fc.label_of(SEEDS[0], None))
    sans = {s: sd.DOSE_ARRAYS_REF.format(label=sd.label_of(s)) for s in SEEDS}
    refs = [ex.RESULTS_REF, ex.RESTRICTED_REF, fc.AXES_REF, fl.AXES_REF, dose, *arrays.values(), *sans.values()]
    got = store.get_refs(refs)
    have = {r: a for r, a in got.items() if a is not None}
    assert len(have) == len(refs), f"not published: {sorted(set(refs) - set(have))}"
    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        paths = dict(zip(refs, store.get_many([(have[r], into / f"{i}") for i, r in enumerate(refs)]), strict=True))
        body = json.loads(paths[ex.RESULTS_REF].read_text())
        restricted = json.loads(paths[ex.RESTRICTED_REF].read_text())
        axes = {
            "fixed": json.loads(paths[fc.AXES_REF].read_text()),
            "labels": json.loads(paths[fl.AXES_REF].read_text()),
        }
        with np.load(paths[dose]) as z:
            held = {k: z[k] for k in ("op_ids", "posterior", "query_pair", "null_eem", "clean_eem")}
        arr = {"held-out": held}
        for lab, ref in arrays.items():
            with np.load(paths[ref]) as z:
                arr[lab] = {k: z[k] for k in z.files}
        for s, ref in sans.items():
            with np.load(paths[ref]) as z:
                arr[("sans-diff", s)] = {k: z[k] for k in ("op_ids", "query_pair", "p_clean", "clean_eem")}
    return body, restricted, axes["fixed"], axes["labels"], arr


BODY, RESTRICTED_BODY, ANCHOR_AXES, LABELS_AXES, ARRAYS = fetch_all()


def fetch_pooled() -> tuple[dict, dict]:
    """What E2 to E5 read at the selections over the pooled fits alone, per control run: the results, and the
    arrays of `finish_one`, keyed by label.
    """
    store = project_store()
    labels = [r["label"] for r in ex.runs() if r["kind"] == "control"]
    arrays = {lab: ex.POOLED_ARRAYS_REF.format(label=lab) for lab in labels}
    refs = [ex.POOLED_REF, *arrays.values()]
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    assert len(have) == len(refs), f"not published: {sorted(set(refs) - set(have))}"
    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        paths = dict(zip(refs, store.get_many([(have[r], into / f"{i}") for i, r in enumerate(refs)]), strict=True))
        body = json.loads(paths[ex.POOLED_REF].read_text())
        arr = {}
        for lab, ref in arrays.items():
            with np.load(paths[ref]) as z:
                arr[lab] = {k: z[k] for k in z.files}
    return {r["label"]: r for r in body["runs"]}, arr


POOLED_FINISH, POOLED_ARRAYS = fetch_pooled()


def fetch_sans_probe() -> dict[str, dict]:
    """The fresh probe on the clean states of each sans-diff model, keyed by label."""
    store = project_store()
    got = store.get_refs([ex.SANS_PROBE_REF])[ex.SANS_PROBE_REF]
    assert got is not None, f"not published: {ex.SANS_PROBE_REF}"
    with tempfile.TemporaryDirectory() as tmp:
        path = store.get_many([(got, Path(tmp) / "p")])[0]
        return {r["label"]: r["probe"] for r in json.loads(path.read_text())["runs"]}


SANS_PROBE = fetch_sans_probe()
OPS: tuple[str, ...] = tuple(BODY["design"]["ops"])
DIFF = ex.ANCHORED_OP
D = OPS.index(DIFF)
CRITERION: float = BODY["design"]["spill_criterion"]  # the selectivity criterion of the D2.2 design
METHODS: tuple[str, ...] = ex.METHODS
FITTERS = ("dim", "ridge", "leace", "inlp")  # the standard fitters; the optimized direction is the ceiling
RANK1 = ("dim", "ridge", "leace")
SHOWN = {"dim": "difference in means", "ridge": "probe", "leace": "LEACE", "inlp": "INLP",
         "optimized": "optimized direction", "anchor": "SCA (anchor)",
         "sans": "sans-diff"}  # fmt: skip
RUNS = {(r["kind"], r["seed"]): r for r in BODY["runs"]}
REST = {r["label"]: r for r in RESTRICTED_BODY["runs"]}
ANCHORED_REMOVAL: float = BODY["anchored_removal"]  # the seed mean a fitter is matched to (spill at matched removal)
H4_MARGIN = 0.1  # H4: a fitter removes no more than this much less than the edit along e₁ on the anchored runs
MATCH_BAND = 0.05  # H2 and S1: an interval within this of zero on both sides means the two remove about as much
FISHER_P = 0.05
REJECT_COUNT = 5  # H1: a fitter over the criterion on this many runs or more, and told apart by the Fisher test


def anchor_runs(body: dict, label_of) -> dict[int, dict[str, Any]]:
    """Removal and spill of the edit along e₁ at every dose, per seed, net of the median over the other 63 axes of
    the run, as fixed-effects publishes them.
    """
    out = {}
    by_label = {r["label"]: r for r in body["runs"]}
    for s in SEEDS:
        r = by_label[label_of(s)]
        drop = np.asarray(r["drop"])  # (axes, doses, ops), the anchored axis first
        net = drop[0] - np.median(drop[1:], axis=0)
        raw = drop[0]
        den = r["clean"][D] - r["null"][D]
        out[s] = {"removal": net[:, D] / den, "spill": np.delete(net, D, axis=1).max(axis=1),
                  "raw_removal": raw[:, D] / den, "raw_spill": np.delete(raw, D, axis=1).max(axis=1),
                  "clean": np.asarray(r["clean"])}  # fmt: skip
    return out


ANCHOR = anchor_runs(ANCHOR_AXES, lambda s: fc.label_of(s, None))
FIXED_LABELS = anchor_runs(LABELS_AXES, lambda s: fl.label_of(s))
assert abs(np.mean([ANCHOR[s]["removal"][-1] for s in SEEDS]) - ANCHORED_REMOVAL) < 1e-9


def baselines(run: dict, method: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The two test-set baselines of configuration *i* (drops per op): the median over the permuted-label fits, and
    the median along the other 63 axes under the same operator form.
    """
    key = str(i)
    got = run["finish"]["methods"][method]["configs"].get(key) or REST[run["label"]]["baselines"][method][key]
    return np.median(np.asarray(got["permuted_drops"]), axis=0), np.asarray(got["axis_median_drop"])


def point(run: dict, method: str, sel: dict, which: str, rule: str) -> tuple[float, float] | None:
    """A point of the selection rule on the test set, (removal, spill): the test drops of its bracketing
    configurations, mixed with the weights found on the tuning set. *rule* is the baseline: "perm", "axis",
    "favor" (whichever of the two favors the method, measure by measure), or "raw". None where the frontier never
    reached the anchored removal.
    """
    p = sel[which]
    if not p["matched"]:
        return None
    g = run["test"][METHODS.index(method)]
    clean, null = np.asarray(g["clean"]), np.asarray(g["null"])
    den = clean[D] - null[D]
    rules = {"perm": (0,), "axis": (1,), "favor": (0, 1), "raw": (2,)}[rule]
    removal, spill = np.zeros(len(rules)), np.zeros(len(rules))
    for i, w in zip(p["configs"], p["weights"], strict=True):
        raw = clean - np.asarray(g["scores"][i])
        perm, axis = baselines(run, method, i) if rule != "raw" else (None, None)
        for j, r in enumerate(rules):
            net = raw - (perm, axis, 0.0)[r]
            removal[j] += w * net[D] / den
            spill[j] += w * np.delete(net, D).max()
    return float(removal.max()), float(spill.min())


def scope_selection(run: dict, method: str, scope: str) -> dict:
    """The selection of *method* on *run*: over the whole grid ("all", the frozen rule), over the sites H4 allows
    ("h4"), or over the pooled fits alone ("pooled").
    """
    return run["selection"][method] if scope == "all" else REST[run["label"]]["selections"][scope][method]


def measures(kind: str, method: str, scope: str = "all", rule: str | None = None) -> dict[str, np.ndarray]:
    """Per seed: removal at the spill criterion, its spill, and spill at matched removal (inf where not matched),
    net and raw. The controls take the baseline that favors the method; the anchored runs take the permuted one.
    """
    rule = rule or ("favor" if kind == "control" else "perm")
    out: dict[str, list] = {k: [] for k in ("removal", "spill_at", "spill", "raw_removal", "raw_spill")}
    for s in SEEDS:
        run = RUNS[(kind, s)]
        sel = scope_selection(run, method, scope)
        at = point(run, method, sel, "at_criterion", rule)
        raw = point(run, method, sel, "at_criterion", "raw")
        matched = point(run, method, sel, "at_matched", rule)
        raw_matched = point(run, method, sel, "at_matched", "raw")
        assert at is not None and raw is not None
        out["removal"].append(at[0])
        out["spill_at"].append(at[1])
        out["raw_removal"].append(raw[0])
        out["spill"].append(np.inf if matched is None else matched[1])
        out["raw_spill"].append(np.inf if raw_matched is None else raw_matched[1])
    return {k: np.asarray(v, float) for k, v in out.items()}


M_CONTROL = {m: measures("control", m) for m in METHODS}
M_POOLED = {m: measures("control", m, "pooled") for m in METHODS}
M_H4 = {m: measures("anchored", m, "h4") for m in FITTERS}
M_H4_AXIS = {m: measures("anchored", m, "h4", rule="axis") for m in FITTERS}
A_REMOVAL = np.array([ANCHOR[s]["removal"][-1] for s in SEEDS])
A_SPILL = np.array([ANCHOR[s]["spill"][-1] for s in SEEDS])
L_REMOVAL = np.array([FIXED_LABELS[s]["removal"][-1] for s in SEEDS])
L_SPILL = np.array([FIXED_LABELS[s]["spill"][-1] for s in SEEDS])
# The anchor spills past the criterion at no dose on any run, so its removal at the criterion is the full dose.
assert all((ANCHOR[s]["spill"] <= CRITERION).all() for s in SEEDS)


def t_interval(a, b) -> tuple[float, float]:
    """A 95% Welch t interval on the difference of the means of two samples of runs (a − b)."""
    ci = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False).confidence_interval(0.95)
    return float(ci.low), float(ci.high)


def over_count(spill: np.ndarray) -> int:
    return int((spill > CRITERION).sum())


def fisher_greater(k: int, k_ref: int, n: int = len(SEEDS)) -> float:
    """One-sided Fisher exact test: is a count of *k* runs out of *n* higher than *k_ref* by more than chance?"""
    return float(stats.fisher_exact(np.array([[k, n - k], [k_ref, n - k_ref]]), alternative="greater").pvalue)


# --- The verdicts ---------------------------------------------------------------------------------------------

H4_ACCEPTED = {m: bool(M_H4[m]["removal"].mean() >= A_REMOVAL.mean() - H4_MARGIN) for m in FITTERS}
A_OVER = over_count(A_SPILL)


def h1_verdict(m: str, meas: dict | None = None) -> str:
    k = over_count((meas or M_CONTROL[m])["spill"])
    if k < REJECT_COUNT:
        return "accepted"
    return "rejected" if fisher_greater(k, A_OVER) < FISHER_P else "unresolved"


def h2_verdict(m: str, meas: dict | None = None) -> str:
    lo, hi = t_interval(A_REMOVAL, (meas or M_CONTROL[m])["removal"])
    if lo > 0:
        return "rejected"
    if hi < 0 or (lo >= -MATCH_BAND and hi <= MATCH_BAND):
        return "accepted"
    return "unresolved"


def overall(verdicts: dict[str, str]) -> str:
    """Passes if every fitter with a resolved verdict was rejected, misses if every one was accepted."""
    resolved = [v for v in verdicts.values() if v != "unresolved"]
    if not resolved:
        return "Unresolved"
    if all(v == "rejected" for v in resolved):
        return "Passes"
    return "Misses" if all(v == "accepted" for v in resolved) else "Partial"


JUDGED = [m for m in FITTERS if H4_ACCEPTED[m]]
H1 = {m: h1_verdict(m) for m in FITTERS}
H2 = {m: h2_verdict(m) for m in FITTERS}
H1_VERDICT = overall({m: H1[m] for m in JUDGED}) if JUDGED else "Unresolved"
H2_VERDICT = overall({m: H2[m] for m in JUDGED}) if JUDGED else "Unresolved"
H4_VERDICT = "Passes" if all(H4_ACCEPTED.values()) else "Misses" if not any(H4_ACCEPTED.values()) else "Partial"

H3_PEERS = [m for m in RANK1 if H4_ACCEPTED[m]]
OPT = M_CONTROL["optimized"]
H3_REMOVAL = bool(H3_PEERS) and OPT["removal"].mean() >= max(M_CONTROL[m]["removal"].mean() for m in H3_PEERS)
H3_SPILL = bool(H3_PEERS) and over_count(OPT["spill"]) <= min(over_count(M_CONTROL[m]["spill"]) for m in H3_PEERS)
H3_VERDICT = ("Unresolved" if not H3_PEERS else "Passes" if H3_REMOVAL and H3_SPILL
              else "Misses" if not (H3_REMOVAL or H3_SPILL) else "Unresolved")  # fmt: skip

S1_CI = t_interval(A_REMOVAL, OPT["removal"])
S1_P = fisher_greater(over_count(OPT["spill"]), A_OVER)
S1_VERDICT = ("Matches" if S1_CI[0] >= -MATCH_BAND and S1_CI[1] <= MATCH_BAND and S1_CI[0] <= 0 <= S1_CI[1] and S1_P >= FISHER_P
              else "Falls short" if S1_CI[0] > 0 else "Unresolved")  # fmt: skip

# The Findings and the lede state these.
assert (H1_VERDICT, H2_VERDICT, H3_VERDICT, S1_VERDICT, H4_VERDICT) == (
    "Misses",
    "Misses",
    "Misses",
    "Matches",
    "Partial",
)

# --- Display helpers ------------------------------------------------------------------------------------------


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus, and no sign on a value that prints as zero."""
    if not np.isfinite(v):
        return "–"
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def span(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(v.mean(), spec)} ({num(v.min(), spec)} to {num(v.max(), spec)})"


def interval(ci: tuple[float, float]) -> str:
    return f"{num(ci[0])} to {num(ci[1])}"


def table_html(head: list[str], rows: list[list[str]], caption: str, *, grey: frozenset[int] = frozenset()) -> str:
    """A results table; the rows in *grey* (a fitter rejected under H4) are greyed out."""

    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=muted style="opacity:0.45"' if k in grey else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r))
        + "</tr>"
        for k, r in enumerate(rows)
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


NUMBER_WORD = dict(enumerate(("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")))


def h4_grey(names) -> frozenset[int]:
    """The rows, by position in *names*, of fitters rejected under H4, which the frozen rule of H4 greys out."""
    return frozenset(i for i, m in enumerate(names) if m in FITTERS and not H4_ACCEPTED[m])


# Theme inks are functions: `light_dark` picks by the theme active when it is called.
INK = {
    "anchor": ("#2a9d8f", "#7fd8c8"),
    "dim": ("#8d6a9f", "#c7a8d8"),
    "ridge": ("#d08c2a", "#f2b866"),
    "leace": ("#3a6ea5", "#8fb8e8"),
    "inlp": ("#b5485d", "#f08aa0"),
    "optimized": ("#555555", "#bbbbbb"),
    "sans": ("#8c6d46", "#d1b48c"),
}
MARKER = {"anchor": "D", "dim": "o", "ridge": "s", "leace": "^", "inlp": "v", "optimized": "*", "sans": "P"}


def ink(m: str) -> str:
    return light_dark(*INK[m])


def faint() -> str:
    return light_dark("#bbb", "#555")


def rule() -> str:
    return light_dark("#333", "#ddd")


def dots(ax, x: float, v: np.ndarray, m: str, *, rng, ms: float = 6.0, width: float = 0.08) -> None:
    """One column of per-seed dots with the seed mean on top, in the method's ink and marker shape."""
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.nan)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [np.nanmin(v), np.nanmax(v)], "-", color=ink(m), lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.6, color=ink(m), alpha=0.5, zorder=3, mew=0)
    ax.plot(x, np.nanmean(v), MARKER[m], ms=ms, color=ink(m), zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def hatch_side(ax, y: float, side: str, *, vertical: bool = False) -> None:
    """Hatch the side of a gate that misses it, after the marks, keeping the limits."""
    lo, hi = ax.get_xlim() if vertical else ax.get_ylim()
    rng_ = (lo, y) if side == "below" else (y, hi)
    kw = dict(
        facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="/", lw=0, zorder=0, alpha=light_dark(0.05, 0.16)
    )
    (ax.axvspan if vertical else ax.axhspan)(*rng_, **kw)
    (ax.set_xlim if vertical else ax.set_ylim)(lo, hi)


def describe(c: dict) -> str:
    """A selected configuration in a few words."""
    sl = c["slices"]
    where = "every slice" if len(sl) == 5 else f"slice {sl[0]}" if len(sl) == 1 else f"slices {sl[0]}–{sl[-1]}"
    site = {
        "pooled": "pooled",
        "answers": "example answers",
        "query =": "query `=`",
        "query operands": "query operands",
    }[c["site"]]
    rep = "mean" if c["replacement"] == "mean" else "zero"
    extra = f", rank {c['rank']}" if c["method"] == "inlp" else ""
    return f"{site}, {where}, dose {num(c['dose'])}, {rep}{extra}"


def chosen_sites(kind: str, m: str, scope: str = "all") -> str:
    """How often each site appears in the configuration E2 to E5 read, over the runs."""
    from collections import Counter

    got = Counter()
    for s in SEEDS:
        run = RUNS[(kind, s)]
        sel = scope_selection(run, m, scope)
        got[run["test"][METHODS.index(m)]["configs"][sel["at_criterion"]["configs"][0]]["site"]] += 1
    names = {"pooled": "pooled", "answers": "answers", "query =": "query =", "query operands": "operands"}
    return ", ".join(f"{names[k]} {v}" for k, v in got.most_common())


r"""
# Directions found after training, beside the anchor

/// tip |
<!-- lede -->
We compare the anchored edit with edits along directions found after training: four standard fitters, and one direction optimized against the scoring itself. Each is fitted on a control model with every label, tuned per run on data it is never scored on, and scored the same way as the anchor. Given role labels and a per-run grid, the fitters remove more of the op than the anchored edit within the same spill limit, mostly by editing the example answers, a role that a real corpus does not label. Restricted to fits that need no role labels, the best of them removes about as much as the anchor and spills past the criterion on a few more runs.
///

Placing a concept during training needs an extra training term, and it lowers the task score somewhat. This report asks whether that placement does better than searching for the concept afterwards, on the same model and with the same scoring.
"""

# %%

r"""
## Findings

- [The trade-off (E1)](#the-trade-off-e1) — On every run but one, each standard fitter climbs past the anchored removal before it reaches the spill criterion, mostly through edits at the example answers. With a configuration chosen on the other seven runs instead, the fitters remove a little less, and on one run in eight a choice made elsewhere misses the slice where that run holds the op.
- [Spill at matched removal (H1)](#spill-at-matched-removal-h1) — **Misses**. Turned up to the anchored removal, no standard fitter spills past the criterion on more than one run, so each is accepted.
- [Removal at the spill criterion (H2)](#removal-at-the-spill-criterion-h2) — **Misses**. LEACE and INLP remove more than the anchor within the criterion, the probe is Unresolved, and difference in means is left out under H4; with the pooled fits alone, LEACE is close to the anchor and the others remove less.
- [The ceiling against the fitters (H3)](#the-ceiling-against-the-fitters-h3) — **Misses**. The optimized direction, which acts at every position, removes less than the rank-1 fitters that edit the example answers.
- [The ceiling against the anchor (S1)](#the-ceiling-against-the-anchor-s1) — **Matches**. Net of the axis-median baseline that favors it, the optimized direction removes about as much as the anchor, and its two runs over the spill criterion are within chance of the anchored none.
- [The fitters find a placed concept (H4)](#the-fitters-find-a-placed-concept-h4) — **Partial**. The probe, LEACE, and INLP find the placed concept; difference in means misses the gate by less than 0.001.
- [Grading, landing, and where the answers go (E2)](#grading-landing-and-where-the-answers-go-e2) — The edits grade alike and send the answers to the same places; the fitters land a little further toward the target null than the anchor.
- [How many labels a fitter needs (E3)](#how-many-labels-a-fitter-needs-e3) — Difference in means, LEACE, and INLP keep their removal down to a fiftieth of the labels, and the probe drops to 0.84.
- [The write against each bound (E4)](#the-write-against-each-bound-e4) — The writes stay within their bounds apart from three sites on two runs, though most post-hoc edits act at one slice, where the write equals the bound.
- [What a fresh probe can still find (E5)](#what-a-fresh-probe-can-still-find-e5) — No edit brings a fresh probe near chance, LEACE included.
- [The two baseline rules (E6)](#the-two-baseline-rules-e6) — The two baselines disagree at configurations far past the spill criterion; at the points the verdicts read, no verdict of H1 or H2 changes under either one alone, and S1 would fall short under the permuted-label rule alone.
- [Without role labels](#without-role-labels) — Restricted to fits that need no role labels, LEACE and INLP remove about as much as the anchor and spill past the criterion at matched removal on two runs in eight against none; difference in means removes less and spills on six. Drawn again with these fits, E1 to E5 show the same picture, with the fitters a little lower.

/// admonition | How to read this report
This report was preregistered: the predictions, the grid, and the rule that picks a configuration from it were frozen at commit `bf73293`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place. A few analyses were added after the run, and each says so where it appears: the configuration chosen on other runs in E1, the reading of H4 under the axis baseline, the section [Without role labels](#without-role-labels), and the marks for the sans-diff model. They give no verdicts.

The hypotheses are judged one fitter at a time, and then as a whole. A fitter is *accepted* when it does what the test asks of a working method (in H1 and H2, matching or beating the anchor; in H4, finding a placed concept) and *rejected* when it does not. A hypothesis as a whole *passes*, *misses*, or is *partial*; a fitter or a hypothesis the rule cannot settle is *Unresolved*.
///
"""

# %%

r"""
## Why this experiment

The anchored pipeline trains the model with a pull toward the first basis vector e₁ on labelled `difference` contexts, and removes the op afterwards by projecting e₁ out of the residual stream[^residual] at every slice and position. On the recipe of record, the fixed-effects experiment found that this edit removes most of the op and costs the other ops less than the selectivity criterion allows. It also drops each context by about as much as that context needed `difference`, and the blocks after the edit keep what it writes within a bound set before the edit runs.

[^residual]: Residual stream: the running vector at each position, which every block reads from and adds its output to.

A reader who knows the steering and erasure literature will ask whether a direction found after training does as well. A conventional pipeline trains with no pull and fits a direction for the concept post hoc, from the states and the labels. It then projects that direction out, replaces the component along it with a mean, or erases it with LEACE. If one of those matched the performance of the anchor on this model, we would have to say that the placement added nothing a search could not find, in a model of this class.

Ex-2.2.1 asked this once, for *red* on the D2.1 checkpoints. Every fitted direction there that removed at least half of *red* cost the lines with no red far more than the anchored edit did. But that run gave the fitters little: three control seeds, labels from the token alone, one fit per slice, and no tuning. So this report gives the fitters every advantage we can, scores them the same way as the anchor, and asks where they fall on the trade-off between removal and spill.
"""

# %%

r"""
## Runs and methods

No new run is trained. The anchored runs are from the recipe of record: the eight `τ_ℓ fixed` runs of the fixed confirmation at seeds 712 to 719. The controls are the eight runs of the anneal confirmation at the same seeds, trained with no pull, with the same initialization, batches, and corpus. Every post-hoc method is fitted on a control and scored on it, so each method has eight runs, and each comparison with the anchor is between two samples of eight runs.

Anchored
:   The full edit along e₁, as published in fixed-effects. Nothing is refitted or retuned for this experiment.

Anchored, fixed labels
:   The same recipe trained on one 1/50 label draw held for the whole run, at the same seeds (the fixed-labels experiment), under the same edit. It is a robustness row beside the main table, from its stored scores; the hypotheses judge the anchored arm above, whose labels are drawn again at each visit, as one pass over a large corpus would draw them.

Difference in means
:   The unit vector from the mean state of the other contexts to the mean state of `difference` contexts. The direction most steering work uses. The states are unit vectors, so their mean lies inside the sphere, but only the direction of the difference is used, and the edit puts each state back on the sphere.

Ridge probe
:   The normalized weight vector of a linear readout of the target, fitted by least squares with an ℓ₂ penalty. It finds what can be read off the state linearly.

LEACE
:   Least-squares concept erasure ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)): the smallest affine edit, measured in the covariance-whitened norm, after which no linear probe can read the target. For one target it works with two directions: c, the covariance of the states with the target (for a binary target, the difference in means), and Σ⁻¹c, where Σ is the covariance of the states, which is the direction a least-squares probe reads. It reads how much of the concept a state holds along Σ⁻¹c and moves the state back along c until that reading is the mean, so afterwards the probe reads the same value for every state. The two directions differ, so the edit is oblique (not a perpendicular projection).

Iterated probe (INLP)
:   The ridge probe fitted, projected out, and fitted again on what is left, to a rank of 2, 4, or 8; the edit removes the whole span. It covers the case of a concept held across several directions, which an edit along one could not remove.

Optimized direction
:   A unit vector per edited slice, fitted by gradient descent on the control so that the edited answers on `difference` contexts move toward the target null while the answers on the other contexts stay where they were. It uses the target null, which no real pipeline has, so it is a diagnostic ceiling: about the best a rank-1 edit (an edit along one direction) of this control could do with the scoring in hand. INLP edits a span of several directions, so this ceiling does not bound it. [COAST](https://arxiv.org/abs/2605.01167) is its nearest published relative.

The first four are the standard fitters that H1 and H2 judge. Each is also fitted on the anchored runs, where training placed the concept on e₁, to check that it can find a concept that is there: it must remove about as much as the edit along e₁ does there, within the same spill limit. That check is H4, and a fitter that does not get through it is left out of the verdicts of H1, H2, H3, and S1. The sans-diff model, trained with no loss on `difference`, stays the reference for how close to the target null a model of this size comes.
"""

# %%

r"""
## What each method gets

We use three sets of contexts, and no context is in more than one: a fit set the directions are fitted on, a tuning set the configuration is chosen on, and the test set every published D2.2 score is taken on.

Fit set
:   280,000 fresh contexts, 40,000 per op, drawn from the training generator of the run with its replacement op noise: about the size of the training corpus, and drawn the same way. Each comes with its generating op and its posterior over ops. The states are taken from the clean pass of the run at every slice and position. The fits are made from means and covariances accumulated over the set, so the states need not be held in memory at once.

Tuning set
:   Another 14,000 fresh contexts from the same generator, drawn the same way.

Test set
:   The 14,000 held-out contexts of ex-2.2.21, 2,000 per op, so the anchored numbers need no recomputing.

Every method is fitted with a label on every `difference` context. The anchor had a label on a fiftieth of them at each visit, but the labels were drawn again at every visit. So over 400 epochs about three quarters of the `difference` contexts were labelled at some point, about one and a half times each.

So over the whole run the anchor met about as many distinct labelled contexts as the fit set holds, but parity in that count is no tie. The anchor pull saw one `difference` context in fifty on each pass over the corpus, a few hundred at a time, through a stochastic term, while the fitters see every label at once, in closed form. What the fitters lack is a term that shaped the whole representation around the concept, and that is what this experiment measures. The label ladder of E3 measures how much the label count itself matters to them.
<!-- REVIEW: the plan said the fitters get "fifty times the labels the anchor saw", and the first draft said "about level" for a fit set of 2,000 `difference` contexts. Under the context labeller (`sca.anchoring._context_mask`) the 1/50 draw is fresh at every visit; at 300k contexts of 24 tokens, a window of 96, an epoch of a quarter of the corpus, and 400 epochs, a context is wholly in view about 70 times, so about 0.75 of the 43k `difference` contexts draw a label at least once (34k distinct, about 650 per pass over the corpus). The fit set was enlarged to 40k `difference` contexts so the claim holds and the bottom rung of E3 (800 contexts) is not a sample-size test. Verify: `(1 - 0.02) ** 70`, and the fit set size above. -->

The fitters also get a graded target: the posterior on `difference` given the examples. A binary label marks the generating op even where the examples point to another op, and the graded target takes away that noise.

Each method then gets a grid over how it is fitted, where it acts, and how much it removes:

| Factor | Levels |
| --- | --- |
| Target | binary (the generating op); graded (the posterior on `difference`), for the probe, LEACE, and the optimized direction |
| Sites fitted | every position of a context, pooled; the example answers; the query `=`; the query operands |
| Slices edited | each slice alone; slices 1 to 3; every slice |
| Sequential refit | off (every fit on the clean states); or, when several slices are edited, on (the fit at each slice made on the states as they arrive under the edits before it) |
| Positions edited | every position; or the sites fitted |
| Dose | 0.25, 0.5, 0.75, and 1 of the component removed |
| Replacement | zero (the projection); or the mean coefficient over the contexts of the other ops (mean ablation) |
| Rank | 2, 4, or 8, for INLP, on the binary target |

The fits at the example answers, the query `=`, and the query operands use role labels a real corpus does not have. So they are generous diagnostics rather than candidate recipes, and the tables mark which sites each selected configuration used.

The anchor gets no grid. Its one configuration is the published edit, which was reached through some thirty experiments at other seeds. The fitters arrive with their own literatures behind them, and get a per-run grid on top.

Selection rule (frozen)
:   Per method and per run, every configuration is scored on the tuning set, net of the median drop under the same operator form along fifteen fixed coordinate axes other than e₁, and the frontier is taken: the configurations that no other beats on both removal and spill. Two points are read off the frontier. *Removal at the spill criterion* is where the frontier crosses a spill of 0.02, interpolated linearly between the two frontier configurations that bracket it; if no configuration is within the criterion it is the one with the least spill, and if every one is within it it is the one with the most removal. *Spill at matched removal* is where the frontier crosses the seed mean of the anchored removal at full dose, interpolated the same way; a frontier that never reaches that removal has no such point, and the run is recorded as not matched. The bracketing configurations are then scored on the test set, and each point is their test scores mixed with the weights found on the tuning set, so the test set chooses nothing, and the point can sit a little off the line on the test set; the table shows where it sits.
<!-- REVIEW: the first draft took the tuning frontier on raw drops. A typical coordinate axis of a control costs about 0.03 per op at full dose (reference-baseline), so raw 0.02 on the tuning set would have meant about −0.01 net, and the selection would have sat inside the criterion it is judged against, biased against the fitters. The axis median is fitter-independent, so fifteen axes per operator form cost about 2,500 passes per run. Verify: the axis subsample is fixed before the run, and the test-set baselines are the full ones of the Measurements. -->

Removal at the spill criterion mirrors what a practitioner with one of these methods would do: sweep the layer and the strength on a validation set, keep the side effects within a tolerance, and take the strongest edit that does.
"""

# %%

r"""
## Measurements

Every measurement is defined as in the D2.2 design and fixed-effects, and taken through `sca.intervention.apply`. So any method that produces a subspace and an operator is scored by the code that scores the anchor.

Removal
:   The drop in expected exact match on `difference` under the edit, as a share of the way from the clean score to the target null. Higher is better.

Spill
:   The largest drop on any other op under the edit. Lower is better, and the selectivity criterion is 0.02.

Net and raw
:   A net drop has a baseline drop subtracted; a raw drop does not. Published anchored scores are net of the median drop under the same edit along each of the other 63 coordinate axes of the run. A fitted direction is not an axis, so each post-hoc configuration gets a baseline in its own terms: the median drop over fifteen fits of the same fitter with the labels permuted across contexts. A permuted fit gives a direction with no concept in it, and the same kind of operator. On the controls the coordinate-axis median is computed too, and each hypothesis is judged on whichever of the two baselines favors the post-hoc method for that measure (the smaller for removal, the larger for spill). The anchor keeps its published baseline. Raw drops are shown beside net ones throughout.

Task change
:   The change in task score from the mean of the controls. It is zero for every post-hoc method, since the model is the control. For the anchor it is the published change, and the table of E1 shows it as the price of anchoring.
"""

# %%

r"""
## The trade-off (E1)

Each hypothesis takes a point from this trade-off, so it comes first. The figure draws the frontier of each run on the test set, with the point the selection rule read on it, beside the anchored dose curve of each run. The table after it gives, per method, the two points of the selection rule, the task change, and how many runs chose each site.

Each mark is net of the median drop along the fifteen coordinate axes of the tuning set under the same operator form, the baseline the selection rule uses on the tuning set. We use it here because the two test-set baselines were computed only at the configurations the rule chose.

<!-- REVIEW: the figure uses the tuning-set baseline on test-set drops, a mix the preregistration did not name. The points the verdicts read use the test-set baselines. Verify: is the mix fine for a figure where nothing is chosen? -->
"""

# %%


SCOPE_SITES = {"all": None, "pooled": ("pooled",)}


def run_cloud(method: str, seed: int, scope: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill of every configuration of *method* on the test set of one control run, net of the
    tuning-set axis median under its operator form; NaN for a configuration outside *scope*.
    """
    run = RUNS[("control", seed)]
    g = run["test"][METHODS.index(method)]
    clean, null = np.asarray(g["clean"]), np.asarray(g["null"])
    med = run["axes"]["median_drop"]
    base = np.array([med[str(ex.config_of(c).form)] for c in g["configs"]])
    net = clean[None] - np.asarray(g["scores"]) - base
    removal, spill = net[:, D] / (clean[D] - null[D]), np.delete(net, D, axis=1).max(axis=1)
    if (sites := SCOPE_SITES[scope]) is not None:
        out = np.array([c["site"] not in sites for c in g["configs"]])
        removal[out], spill[out] = np.nan, np.nan
    return removal, spill


def finite_frontier(removal: np.ndarray, spill: np.ndarray) -> np.ndarray:
    """`frontier` over the configurations with a score, with the indices of the whole grid."""
    ok = np.flatnonzero(np.isfinite(removal))
    return ok[ex.frontier(removal[ok], spill[ok])]


def cloud(method: str) -> tuple[np.ndarray, np.ndarray]:
    """Seed-mean removal and spill of every configuration of *method*, as `run_cloud` scores them."""
    per = [run_cloud(method, s) for s in SEEDS]
    return np.mean([r for r, _ in per], axis=0), np.mean([sp for _, sp in per], axis=0)


def run_fronts(method: str, scope: str = "all") -> list[tuple[np.ndarray, np.ndarray]]:
    """The frontier of each control run of *method*, as (spill, removal) lines."""
    out = []
    for s in SEEDS:
        r, sp = run_cloud(method, s, scope)
        f = finite_frontier(r, sp)
        out.append((sp[f], r[f]))
    return out


def shared_choice(method: str, scope: str = "all") -> list[dict]:
    """Per run left out, the point of the selection rule at the spill criterion on the frontier of the seed means of
    the other seven runs, on the tuning set: the bracketing configurations and their weights.
    """

    def tuning(s: int) -> tuple[np.ndarray, np.ndarray]:
        run = RUNS[("control", s)]
        sel = scope_selection(run, method, scope)
        r, sp = (np.array([np.nan if v is None else v for v in sel[k]], float) for k in ("removal", "spill"))
        return r, sp

    out = []
    for s in SEEDS:
        others = [tuning(o) for o in SEEDS if o != s]
        tr, ts = np.mean([a for a, _ in others], axis=0), np.mean([b for _, b in others], axis=0)
        out.append(ex.read_point(tr, ts, finite_frontier(tr, ts), CRITERION, "spill"))
    return out


def _mix(p: dict, v: np.ndarray) -> float:
    return float(sum(w * v[i] for i, w in zip(p["configs"], p["weights"], strict=True)))


def shared_point(method: str, scope: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill per run at a configuration chosen on the other seven runs (`shared_choice`), scored on the
    test set of the run left out, as `run_cloud` scores it. The fit is still made on the run left out; only the
    choice of target, site, slices, dose, refit, replacement, and rank comes from the others.
    """
    rem, spl = [], []
    for s, p in zip(SEEDS, shared_choice(method, scope), strict=True):
        r, sp = run_cloud(method, s, scope)
        rem.append(_mix(p, r))
        spl.append(_mix(p, sp))
    return np.asarray(rem), np.asarray(spl)


def shared_lines(method: str, scope: str = "all") -> tuple[list, tuple[np.ndarray, np.ndarray]]:
    """Per run left out, the frontier of the seed means of the other seven runs on the test set, as a (spill,
    removal) line, and where the choice of `shared_choice` sits on those means.
    """
    lines, er, es = [], [], []
    for s, p in zip(SEEDS, shared_choice(method, scope), strict=True):
        per = [run_cloud(method, o, scope) for o in SEEDS if o != s]
        r, sp = np.mean([a for a, _ in per], axis=0), np.mean([b for _, b in per], axis=0)
        f = finite_frontier(r, sp)
        lines.append((sp[f], r[f]))
        er.append(_mix(p, r))
        es.append(_mix(p, sp))
    return lines, (np.asarray(er), np.asarray(es))


CLOUD = {m: cloud(m) for m in METHODS}
FRONT = {m: ex.frontier(*CLOUD[m]) for m in METHODS}
RUN_FRONTS = {m: run_fronts(m) for m in METHODS}
SHARED = {m: shared_point(m) for m in METHODS}
H1["optimized"], H2["optimized"] = h1_verdict("optimized"), h2_verdict("optimized")
ANCHOR_CURVE = (
    np.mean([ANCHOR[s]["removal"] for s in SEEDS], axis=0),
    np.mean([ANCHOR[s]["spill"] for s in SEEDS], axis=0),
)
SPILL_LIM = (-0.03, 0.12)
ANCHOR_RUNS = [(ANCHOR[s]["spill"], ANCHOR[s]["removal"]) for s in SEEDS]


def sans_point() -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill per seed of the sans-diff model against the control at the same seed, raw: the drop in
    expected exact match per op from the control to sans-diff on the held-out contexts, with no edit and so no
    baseline.
    """
    ops, null = ARRAYS["held-out"]["op_ids"], ARRAYS["held-out"]["null_eem"]
    per_op = lambda v: np.array([v[ops == o].mean() for o in range(len(OPS))])  # noqa: E731
    rem, spl = [], []
    for s in SEEDS:
        c, sd_ = (
            per_op(ARRAYS[RUNS[("control", s)]["label"]]["clean/eem"]),
            per_op(ARRAYS[("sans-diff", s)]["clean_eem"]),
        )
        drop = c - sd_
        rem.append(drop[D] / (c[D] - per_op(null)[D]))
        spl.append(np.delete(drop, D).max())
    return np.asarray(rem), np.asarray(spl)


SANS_POINT = sans_point()


def _border(ax, color: str) -> None:
    for sp_ in ax.spines.values():
        sp_.set_linewidth(1.8)
        sp_.set_color(color)


def _e1_backdrop(ax, m: str, fronts: dict, anchor_curve: tuple) -> None:
    """The per-run frontiers of the other methods as hairlines, and the anchored dose curve of seed means."""
    for other in METHODS:
        if other != m:
            for sp, r in fronts[other]:
                ax.plot(sp, r, "-", color=faint(), lw=0.3, alpha=0.7, zorder=1)
    ax.plot(anchor_curve[1], anchor_curve[0], "--", color=ink("anchor"), lw=0.8, zorder=3)


def _e1_method(ax, m: str, fronts: list, chosen: tuple, accepted: bool, rejected: bool) -> None:
    """One method: a frontier per run, and on each the point the selection rule read at the spill criterion."""
    for sp, r in fronts:
        ax.plot(sp, r, "-", color=ink(m), lw=0.8, alpha=0.6, zorder=3)
    cr, cs = chosen
    ax.plot(cs, cr, MARKER[m], ms=4, color=ink(m), zorder=4, mec=light_dark("white", "#111"), mew=0.5)
    ax.set_title(SHOWN[m], fontsize=8)
    if accepted:
        _border(ax, light_dark("#c0392b", "#ff8a76"))
    if rejected:
        ax.axvspan(*SPILL_LIM, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="/", lw=0, alpha=light_dark(0.05, 0.16),
                   zorder=0)  # fmt: skip


def _e1_expected(ax, m: str, expected: tuple, chosen: tuple) -> None:
    """Where each choice sat on the line it was made on (hollow), joined to where it landed on its own run."""
    er, es = expected
    cr, cs = chosen
    for x0, y0, x1, y1 in zip(es, er, cs, cr, strict=True):
        ax.plot([x0, x1], [y0, y1], "-", color=ink(m), lw=0.6, alpha=0.7, zorder=4)
    ax.plot(es, er, MARKER[m], ms=4, mfc="none", mec=ink(m), mew=0.8, zorder=4)


def _e1_sans(ax, sans: tuple) -> None:
    """The sans-diff model against the control at the same seed: seed mean with bars over the range of runs."""
    r, sp = sans
    ax.errorbar(sp.mean(), r.mean(), xerr=[[sp.mean() - sp.min()], [sp.max() - sp.mean()]],
                yerr=[[r.mean() - r.min()], [r.max() - r.mean()]], fmt=MARKER["sans"], ms=5, color=ink("sans"),
                mec=light_dark("white", "#111"), mew=0.5, elinewidth=0.7, capsize=0, zorder=6)  # fmt: skip


def _e1_anchor(ax, anchor_runs: list, a_rem: np.ndarray, a_spill: np.ndarray) -> None:
    """The anchored dose curve of each run, ending at its full-dose diamond."""
    for sp, r in anchor_runs:
        ax.plot(sp, r, "-", color=ink("anchor"), lw=0.8, alpha=0.6, zorder=3)
    ax.plot(a_spill, a_rem, MARKER["anchor"], ms=4, color=ink("anchor"), zorder=4, mec=light_dark("white", "#111"),
            mew=0.5)  # fmt: skip
    ax.set_title("SCA (anchor)", fontsize=8, fontweight="bold")
    _border(ax, ink("anchor"))


@memo
def e1_draw(fronts: dict, anchor_runs: list, anchor_curve: tuple, chosen: dict, a_rem: np.ndarray,
            a_spill: np.ndarray, accepted: dict, rejected: dict, alt_text: str, name: str = "post-hoc-compare-tradeoff",
            caption: str | None = None, sans: tuple | None = None, expected: dict | None = None) -> str:  # fmt: skip
    @themed(
        name=name,
        alt_text=alt_text,
        caption=caption
        or f"""
            **Removal against spill, one line per run**, on the test set of the eight controls. Each panel is one
            method. A line is the frontier of one run: the configurations that no other configuration of that run
            beats on both removal and spill. The marker on each line is the point the selection rule read for that run
            at the spill criterion (H2), net of the axis median as the lines are, so every run has its own
            configuration; the tables take the baseline that favors the method, which can put a point a little
            to the left. In the anchor panel a line is the
            anchored edit of one run as its dose rises, ending at a diamond (full dose); the dashed curve in the other
            panels is their mean. The other methods are drawn as grey hairlines behind. The dotted vertical line is
            the spill criterion, and the dotted horizontal one the anchored removal. A red border marks a method
            accepted under both H1 and H2 (the optimized direction scored by the same rules, for the figure only), and
            a hatched panel a fitter rejected under H4. The plus is the sans-diff model, trained with no loss on
            `{DIFF}`, against the control at the same seed, raw since nothing is edited: seed mean, with bars over the
            runs. The spill axis is cut at {SPILL_LIM[1]}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True, sharey=True, layout="constrained")
        grid = cast(AxesGrid, axes)
        flat = [ax for row in grid for ax in row]
        for ax, m in zip(flat, ["anchor", *METHODS], strict=True):
            if m == "anchor":
                _e1_backdrop(ax, m, fronts, (np.array([]), np.array([])))
                _e1_anchor(ax, anchor_runs, a_rem, a_spill)
            else:
                _e1_backdrop(ax, m, fronts, anchor_curve)
                _e1_method(ax, m, fronts[m], chosen[m], accepted[m], rejected[m])
                if expected is not None:
                    _e1_expected(ax, m, expected[m], chosen[m])
            if sans is not None:
                _e1_sans(ax, sans)
            ax.axvline(CRITERION, color=rule(), lw=0.8, ls=":", zorder=1)
            ax.axhline(a_rem.mean(), color=rule(), lw=0.6, ls=":", zorder=1)
            ax.set_xlim(*SPILL_LIM)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)
        for ax in grid[1]:
            ax.set_xlabel("spill (net)", fontsize=8)
        for row in grid:
            row[0].set_ylabel(f"removal of {DIFF} (net)", fontsize=8)
        return fig

    return _plot()


E1_ALT = (
    "Six panels of removal against spill, one line per run. The anchor panel shows the anchored edit of each run "
    "rising with dose to a diamond near removal 0.88 and spill zero. Each method panel shows eight frontiers with a "
    "marker on each; the fitter lines rise steeply near zero spill and level off near removal 0.95, so their markers "
    "sit above the anchor at spill below 0.02, apart from one probe run near 0.77. The optimized direction lines "
    "stop near the anchor."
)


# The markers take the axis median of the test set, close to the tuning-set median the lines take; the favoring
# rule of the tables can put a point a little further left.
_AXIS = {m: measures("control", m, rule="axis") for m in METHODS}
CHOSEN = {m: (_AXIS[m]["removal"], _AXIS[m]["spill_at"]) for m in METHODS}
E1_ACCEPTED = {m: H1[m] == "accepted" and H2[m] == "accepted" for m in METHODS}
E1_REJECTED = {m: m in FITTERS and not H4_ACCEPTED[m] for m in METHODS}
assert all(chosen_sites("control", m) == "answers 8" for m in FITTERS)
assert max(over_count(M_CONTROL[m]["spill"]) for m in FITTERS) <= 1
e1_draw(
    RUN_FRONTS,
    ANCHOR_RUNS,
    ANCHOR_CURVE,
    CHOSEN,
    A_REMOVAL,
    A_SPILL,
    E1_ACCEPTED,
    E1_REJECTED,
    alt_text=E1_ALT,
    sans=SANS_POINT,
)

# %%

f"""
On every run but one, each standard fitter has a frontier that climbs past the anchored removal before it reaches the spill criterion. The configuration it picks at the criterion removes nearly all of `{DIFF}`.

On every run, the configuration each fitter chose at the criterion was fitted at the example answers and edits there alone; the table below counts the sites. The optimized direction, which acts at every position by design, stops about where the anchor does.
"""

# %%


def task_change() -> float:
    """The change in task score (expected exact match, mean over the ops) of the anchored runs from the controls."""
    a = np.mean([np.mean(RUNS[("anchored", s)]["finish"]["clean"]) for s in SEEDS])
    c = np.mean([np.mean(RUNS[("control", s)]["finish"]["clean"]) for s in SEEDS])
    return float(a - c)


TASK_CHANGE = task_change()


def e1_table() -> str:
    rows = [["anchor", span(A_REMOVAL), span(ANCHOR[s]["raw_removal"][-1] for s in SEEDS),
             span(A_SPILL), "0 of 8", num(TASK_CHANGE, ".3f"), "every position, every slice"]]  # fmt: skip
    rows.append(["anchor, fixed labels", span(L_REMOVAL), span(FIXED_LABELS[s]["raw_removal"][-1] for s in SEEDS),
                 span(L_SPILL), f"{over_count(L_SPILL)} of 8", "", "every position, every slice"])  # fmt: skip
    for m in METHODS:
        v = M_CONTROL[m]
        fin = v["spill"][np.isfinite(v["spill"])]
        unmatched = int((~np.isfinite(v["spill"])).sum())
        spill = (span(fin) if len(fin) else "–") + (f", {unmatched} not matched" if unmatched else "")
        rows.append([SHOWN[m], span(v["removal"]), span(v["raw_removal"]), spill,
                     f"{over_count(v['spill'])} of 8", num(0.0, ".3f"), chosen_sites("control", m)])  # fmt: skip
    grey = frozenset(i + 2 for i, m in enumerate(METHODS) if m in FITTERS and not H4_ACCEPTED[m])
    return table_html(
        [
            "method",
            "removal at criterion ↑",
            "raw",
            "spill at matched removal ↓",
            "over criterion",
            "task change",
            "sites (runs)",
        ],  # fmt: skip
        rows,
        """**The two points of the selection rule**, seed mean and range over eight runs on the test set. Removal at
        the spill criterion is net; the raw column has no baseline subtracted. The anchor is at full dose, where its
        spill is that of the edit. A method whose frontier never reaches the anchored removal on a run is not matched
        there and counts as over the criterion. The sites column counts the runs whose configuration at the
        criterion was fitted at each site. A greyed row is a fitter rejected under H4.""",
        grey=grey,
    )


e1_table()

f"""
So on this model a direction fitted after training, with every label and a per-run grid, removes more of `{DIFF}` within the criterion than the anchored edit does, and spills no more at matched removal. It costs the task nothing, since the model is the control, while the anchored runs score {num(-TASK_CHANGE, ".3f")} lower than the controls. The fixed-labels row removes less than the main anchored arm, as that experiment found.

The edge of the fitters comes mostly from fitting and editing at the example answers, a role a real corpus does not label. Restricted to the pooled fits, which need no role labels, the fitters lose most of that edge ([Without role labels](#without-role-labels)).
"""

# %%

r"""
### One configuration chosen on other runs

Each run above picks its own configuration, among about a thousand, on the model it edits. The anchor had no such choice on these runs: its recipe was settled on earlier seeds and run on these eight as it was. The next figure puts the fitters on the same terms. For each run in turn, the selection rule picks a configuration on the other seven, off the frontier of their seed means on the tuning set, and the fitter is then fitted on the run left out with that configuration. So the choice is made eight times per method: seed 712 takes the configuration chosen on 713 to 719, seed 713 the one chosen on 712 and 714 to 719, and so on. This analysis was not preregistered.
"""

# %%

SHARED_712 = {m: float(SHARED[m][0][SEEDS.index(712)]) for m in METHODS}
SHARED_REST = {m: float(np.delete(SHARED[m][0], SEEDS.index(712)).mean()) for m in METHODS}
OWN_REST = {m: float(np.delete(M_CONTROL[m]["removal"], SEEDS.index(712)).mean()) for m in METHODS}
SHARED_OVER = {m: over_count(SHARED[m][1]) for m in METHODS}
# Seed 712 is the run where a configuration chosen on the others fails: difference in means, LEACE, and INLP keep
# almost nothing there, and its own choices edit slice 1 where the others edit slice 2.
assert all(SHARED_712[m] < 0.1 for m in ("dim", "leace", "inlp")), SHARED_712
assert all(RUNS[("control", 712)]["test"][METHODS.index(m)]["configs"][RUNS[("control", 712)]["selection"][m]["at_criterion"]["configs"][0]]["slices"][0] == 1 for m in ("dim", "leace", "inlp"))  # fmt: skip
assert all(SHARED_REST[m] < OWN_REST[m] for m in FITTERS), (SHARED_REST, OWN_REST)
assert all(3 <= SHARED_OVER[m] <= 4 for m in FITTERS) and SHARED_OVER["optimized"] == 0, SHARED_OVER
E1_SHARED_ALT = (
    "Six panels of removal against spill. In each method panel, eight nearly identical lines (the frontiers of seven "
    "runs averaged) with hollow markers bunched at the spill criterion near removal 0.9, each joined to a filled "
    "marker for the run left out: most land close by, about half just past the criterion, and for difference in "
    "means, LEACE, and INLP one drops to near zero removal. The optimized direction stays near the anchor."
)
SHARED_LINES = {m: shared_lines(m) for m in METHODS}
e1_draw({m: SHARED_LINES[m][0] for m in METHODS}, ANCHOR_RUNS, ANCHOR_CURVE, {m: SHARED[m] for m in METHODS},
        A_REMOVAL, A_SPILL, dict.fromkeys(METHODS, False), dict.fromkeys(METHODS, False), alt_text=E1_SHARED_ALT,
        name="post-hoc-compare-shared", sans=SANS_POINT, expected={m: SHARED_LINES[m][1] for m in METHODS},
        caption=f"""
            **Removal against spill with a configuration chosen on the other runs**, on the test set of the eight
            controls. Each line is the frontier of the seed means of seven runs, one line for each run left out. The
            hollow marker is where the selection rule chose on those seven, at the spill criterion; the filled marker
            joined to it is what that configuration, fitted on the run left out, scores there. The anchor panel is as
            before, and the plus is the sans-diff model against the control at the same seed (seed mean, with bars
            over the runs). Spill here is net of the median along the coordinate axes of the tuning set, as in the
            figure before. The spill axis is cut at {SPILL_LIM[1]}.
        """)  # fmt: skip

# %%

f"""
With the configuration chosen on other runs, the fitters remove about {num(min(SHARED_REST[m] for m in FITTERS))} to {num(max(SHARED_REST[m] for m in FITTERS))} on seven of the eight runs, a little less than with a choice of their own ({num(min(OWN_REST[m] for m in FITTERS))} to {num(max(OWN_REST[m] for m in FITTERS))}), and land past the spill criterion on {NUMBER_WORD[min(SHARED_OVER[m] for m in FITTERS)]} or {NUMBER_WORD[max(SHARED_OVER[m] for m in FITTERS)]} of the eight. A point picked right at the criterion on other runs lands on either side of it about as often, while the anchor sits well inside it on every run.

The eighth run, seed 712, is where the choice fails to carry over. Difference in means, LEACE, and INLP keep almost nothing of their removal there ({num(min(SHARED_712[m] for m in ("dim", "leace", "inlp")))} to {num(max(SHARED_712[m] for m in ("dim", "leace", "inlp")))}): their choices on the other runs edit slice 2, and on this run their own choices edit slice 1. The probe keeps {num(SHARED_712["ridge"])} there, with an edit over slices 1 to 3. So where a post-hoc edit works moves from run to run, by a slice on one run in eight here, and a fixed recipe for a fitter has to cover that, while the anchor puts the concept on e₁ on every run. One run is a small sample; a larger set of seeds would say how often this happens.

"""

# %%

r"""
## Spill at matched removal (H1)

**What we expect.** If we turn each fitter up until it removes as much of `difference` as the anchor does, then on most runs it will cost another op more than the criterion allows.

**The rule.** For each fitter we count the runs, out of eight, whose spill at matched removal is over 0.02 net. A run whose frontier never reaches the anchored removal counts as over, since that fitter cannot remove as much at any spill. For the anchor we count the runs whose spill at full dose is over 0.02 net. A one-sided Fisher exact test[^fisher] then says whether the fitter count is higher than the anchored count by more than chance would give. A fitter is rejected if its count is five or more and the test gives p below 0.05, and accepted if its count is under five. A count of five or more that the test cannot tell apart from the anchored count (which can happen only if the anchor itself spills on one or more runs; five against one gives p of 0.06) would be outside the plan, and that fitter would be Unresolved. Among the fitters accepted under H4, H1 passes if every fitter with a resolved verdict was rejected, misses if every one was accepted, and is partial otherwise; it is Unresolved if no fitter is accepted under H4 or none has a resolved verdict. The Findings line names any fitter left out or Unresolved. A fitter accepted here would mean that, on this model, a searched direction serves as well as a placed one.

[^fisher]: Fisher exact test: a test of whether two counts out of the same number of trials differ by more than chance. With the anchor at none of eight, a fitter at four or more of eight gives p below 0.05.

The figure shows spill at matched removal for each run, with the anchor at full dose beside the fitters.
"""

# %%


@memo
def per_run_draw(name: str, values: dict, gate: float, side: str, ylabel: str, ylim: tuple, caption: str,
                 alt_text: str, top_label: str | None = None) -> str:  # fmt: skip
    """One column of runs per method, with the seed mean on top, a gate line, and the side that misses it hatched.
    An infinite value (a run not matched) is drawn at the top of the axis.
    """

    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 2.6), layout="constrained")
        rng = np.random.default_rng(0)
        names = list(values)
        for x, m in enumerate(names):
            v = np.asarray(values[m], float)
            shown = np.where(np.isfinite(v), v, ylim[1] - 0.04 * (ylim[1] - ylim[0]))
            dots(ax, x, shown, m.split(":")[0], rng=rng)
            if (~np.isfinite(v)).any():
                ax.text(x, ylim[1], f"{int((~np.isfinite(v)).sum())} {top_label}", ha="center", va="bottom", fontsize=6)
        ax.axhline(gate, color=rule(), lw=0.9, ls="--", zorder=2)
        ax.set_xticks(
            range(len(names)),
            [SHOWN[m.split(":")[0]] + (" (H4 out)" if m.endswith(":out") else "") for m in names],
            fontsize=7,
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(*ylim)
        hatch_side(ax, gate, side)
        return fig

    return _plot()


def key(m: str) -> str:
    return m + (":out" if m in FITTERS and not H4_ACCEPTED[m] else "")


H1_VALUES = {"anchor": A_SPILL} | {key(m): M_CONTROL[m]["spill"] for m in METHODS}
per_run_draw(
    "post-hoc-compare-h1",
    H1_VALUES,
    CRITERION,
    "above",
    "spill at matched removal (net)",
    (-0.08, 0.3),
    """**Spill at matched removal per run**, on the test set, net. A small dot per run and the seed mean as the
    method's marker; the anchor at full dose. The dashed line is the criterion and the side over it is hatched. A run
    whose frontier never reaches the anchored removal is drawn at the top and counted there. The axis is cut at 0.3.""",
    "Spill at matched removal per run for the anchor and five methods. Nearly every run of every method sits under "
    "the criterion line; a few probe and optimized runs sit over it or are not matched.",
    top_label="not matched",
)

# %%


def h1_table() -> str:
    rows = [["anchor (full dose)", f"{A_OVER} of 8", "", ""]]
    for m in METHODS:
        k = over_count(M_CONTROL[m]["spill"])
        out = m in FITTERS and not H4_ACCEPTED[m]
        verdict = H1[m] + (" (left out under H4)" if out else "") + (" (not judged)" if m == "optimized" else "")
        rows.append([SHOWN[m], f"{k} of 8", num(fisher_greater(k, A_OVER)), verdict])
    grey = frozenset(i + 1 for i, m in enumerate(METHODS) if m in FITTERS and not H4_ACCEPTED[m])
    return table_html(
        ["method", "runs over the criterion", "Fisher p", "verdict"],
        rows,
        """**Runs over the criterion at matched removal**, out of eight, with the one-sided Fisher p against the
        anchored count. A fitter is rejected at five or more runs with p below 0.05, and accepted under five.""",
        grey=grey,
    )


h1_table()

# %%

assert H1_VERDICT == "Misses" and all(H1[m] == "accepted" for m in FITTERS), (
    "the prose of H1 says every fitter was accepted"
)

f"""
**{H1_VERDICT}.** No standard fitter spills past the criterion on more than one run once it is turned up to the anchored removal, so each is accepted, and H1 misses. Difference in means is left out of the verdict under H4, and it would have been accepted too. Turned up that far, a fitted direction costs the other ops about as little as the anchored edit does.
"""

# %%

r"""
## Removal at the spill criterion (H2)

**What we expect.** If we hold each fitter to the same spill limit as the anchor, then it will remove less of `difference` than the anchor does.

**The rule.** For each fitter we take the difference of seed means between the anchored removal (at full dose, or where its own dose curve crosses the criterion if it spills past it) and the removal at the spill criterion of the fitter, with a 95% Welch t interval[^welch]. A fitter is rejected if the interval lies wholly above zero (the anchor removes more). It is accepted if the interval lies wholly below zero (the fitter removes more), or contains zero and lies within 0.05 of it on both sides (the two remove about as much). An interval that contains zero and reaches further than 0.05 on either side would mean the seeds could not tell, and that fitter would be Unresolved. Among the fitters accepted under H4, H2 passes if every fitter with a resolved verdict was rejected, misses if every one was accepted, and is partial otherwise; it is Unresolved if no fitter is accepted under H4 or none has a resolved verdict. The Findings line names any fitter left out or Unresolved.

A small edge for a fitter would get it accepted under this rule, and it would not surprise us. Every run writes the evidence for the op partly off e₁ (the low-runs experiment). So the edit along e₁ leaves a little behind, while a fitted direction can take in both parts.

[^welch]: Welch t interval: a 95% confidence interval for the gap between two means that lets the two groups scatter by different amounts. An interval wholly above zero means the eight anchored runs remove more than the eight fitted ones by more than seed-to-seed scatter accounts for.

The figure shows removal at the spill criterion for each run.
"""

# %%

per_run_draw(
    "post-hoc-compare-h2",
    {"anchor": A_REMOVAL} | {key(m): M_CONTROL[m]["removal"] for m in METHODS},
    float(A_REMOVAL.mean()),
    "below",
    f"removal of {DIFF} at the criterion (net)",
    (0.5, 1.02),
    """**Removal at the spill criterion per run**, on the test set, net. A small dot per run and the seed mean as
    the method's marker; the anchor at full dose. The dashed line is the seed mean of the anchor, and the side
    under it is hatched.""",
    "Removal at the spill criterion per run for the anchor and five methods. The four standard fitters sit above the "
    "anchor line, near 0.95, with one probe run near 0.77; the optimized direction sits on the anchor line.",
)

# %%


def h2_table(meas: dict, verdicts: dict | None, caption: str) -> str:
    rows = [["anchor", span(A_REMOVAL), num(A_REMOVAL.std(ddof=1)), "", ""]]
    for m in METHODS:
        v = meas[m]["removal"]
        ci = t_interval(A_REMOVAL, v)
        row = [SHOWN[m], span(v), num(v.std(ddof=1)), interval(ci)]
        if verdicts is not None:
            out = m in FITTERS and not H4_ACCEPTED[m]
            row.append(
                verdicts[m] + (" (left out under H4)" if out else "") + (" (not judged)" if m == "optimized" else "")
            )
        else:
            row.append("")
        rows.append(row)
    grey = frozenset(i + 1 for i, m in enumerate(METHODS) if m in FITTERS and not H4_ACCEPTED[m])
    return table_html(["method", "removal at criterion ↑", "seed sd", "anchor minus method, 95% interval",
                       "verdict" if verdicts else ""], rows, caption, grey=grey)  # fmt: skip


h2_table(
    M_CONTROL,
    H2,
    """**Removal at the spill criterion**, seed mean and range over eight runs, net, with the seed standard
    deviation and the 95% Welch interval on the anchored seed mean minus the method seed mean. A fitter is rejected
    if the interval lies above zero, and accepted if it lies below zero or within 0.05 of it on both sides.""",
)

# %%

assert H2_VERDICT == "Misses" and H2["ridge"] == "unresolved" and H2["leace"] == H2["inlp"] == "accepted"

f"""
**{H2_VERDICT}.** LEACE and INLP remove more than the anchor within the criterion, by more than the seeds scatter, so both are accepted. The probe removes more on seven runs but stops far short on one, which widens its interval to span {num(t_interval(A_REMOVAL, M_CONTROL["ridge"]["removal"])[0])} to just above zero. That interval contains zero but reaches further than 0.05 from it, so the verdict of the probe is Unresolved.

Every resolved verdict is an acceptance, so H2 misses. Difference in means would have been accepted too, but it is left out under H4.

We expected a small edge for a fitter, since part of the evidence lies off e₁. The edge here is larger, and it comes mostly from where the fitters act: the rule picks the configurations fitted at the example answers, which edit those positions alone and use role labels a real corpus does not have.

[Without role labels](#without-role-labels) repeats the comparison with every fitter restricted to the pooled fits, which is the like-for-like reading for the anchor.
"""

# %%

r"""
## The ceiling against the fitters (H3)

**What we expect.** If we search for a direction with the scoring in hand, then it does better than every standard rank-1 fitter.

**The rule.** The comparison takes the rank-1 fitters accepted under H4: difference in means, the probe, and LEACE. Removal by the optimized direction at the spill criterion is at least the highest seed mean among them, and its count over the criterion at matched removal (as in H1) is at most the lowest. It passes if both hold, and misses if neither does; one without the other would be outside the plan, and the verdict would be Unresolved, as it would be if no rank-1 fitter is accepted under H4. The comparison is between seed means and counts, with no interval, which is enough for a check that the ceiling is a ceiling.

INLP is left out of H3 because it edits a span of two to eight directions, which a rank-1 ceiling does not bound; its two measures are shown beside the others. If INLP does better than the ceiling, the control holds the op across several directions, which E5 looks at from another side.

LEACE is rank-1 but oblique, and the ceiling is a perpendicular edit, so LEACE could in principle beat it too. We keep it in, since LEACE beating the ceiling would say the oblique form matters.

"""

# %%


def h3_table() -> str:
    rows = []
    for m in (*METHODS[-1:], *FITTERS):
        v = M_CONTROL[m]
        note = (
            "ceiling"
            if m == "optimized"
            else "outside the rule"
            if m == "inlp"
            else "left out under H4"
            if not H4_ACCEPTED[m]
            else "compared"
        )
        rows.append([SHOWN[m], span(v["removal"]), f"{over_count(v['spill'])} of 8", note])
    grey = frozenset(i + 1 for i, m in enumerate(FITTERS) if not H4_ACCEPTED[m])
    return table_html(
        ["method", "removal at criterion ↑", "over the criterion at matched removal ↓", "role in H3"],
        rows,
        """**The optimized direction beside the standard fitters**, on the controls: removal at the spill criterion
        (seed mean and range, net) and the runs over the criterion at matched removal, out of eight.""",
        grey=grey,
    )


h3_table()

# %%

assert H3_VERDICT == "Misses" and not H3_REMOVAL and not H3_SPILL

f"""
**{H3_VERDICT}.** The optimized direction removes less than any of the rank-1 fitters it is compared with, and is over the criterion at matched removal on more runs. Both shortfalls come from two runs on which its frontier never reaches the anchored removal. INLP, outside the rule, does better than it too.

So it is no ceiling for the fitters here, for the reason seen in H2: the search acts at every position, as it was planned to, while the fitters picked edits at the example answers alone. Against the pooled fits ([Without role labels](#without-role-labels)) it does about as well as the best of them.
"""

# %%

r"""
## The ceiling against the anchor (S1)

**What we expect.** This section predicts nothing: it is a frozen rule that decides between two outcomes, and its badge is Decided, or Unresolved. Either the best rank-1 search on the control does as well as the anchor, or it still falls short.

**The rule.** The ceiling *matches* the anchor if the 95% Welch interval[^welch] on the difference of seed means in removal at the spill criterion contains zero and lies within 0.05 of it on both sides (as for an accepted fitter in H2), and the Fisher test[^fisher] cannot tell its count over the criterion at matched removal apart from the anchored count; then the advantage of placing the concept is that no search is needed. The ceiling *falls short* if the interval lies wholly above zero; then the control holds the op in a way no rank-1 edit removes cleanly, which is the stronger claim for the anchor. Any other result (the interval wholly below zero, an interval that contains zero but reaches further than 0.05, or a count the Fisher test does tell apart) would be outside the plan, and the verdict would be Unresolved.
<!-- REVIEW: H3 was split in two at the paper round before the freeze. As one hypothesis, its verdict sat on the clause that changes the post least (the ceiling beats the fitters), and the comparison with the anchor, which decides the framing, carried none. Verify: S1 names both outcomes and the Findings show which held. A second review made the comparison with the anchor an S section, since it predicts nothing, and gave "matches" the same 0.05 band as an accepted fitter in H2, so an interval too wide to tell is Unresolved rather than a match. -->

"""

# %%

table_html(
    ["", "anchor", "optimized direction"],
    [
        ["removal at criterion ↑", span(A_REMOVAL), span(OPT["removal"])],
        ["over the criterion at matched removal ↓", f"{A_OVER} of 8", f"{over_count(OPT['spill'])} of 8"],
        ["anchor minus optimized, 95% interval", interval(S1_CI), ""],
        ["Fisher p (optimized count higher)", num(S1_P), ""],
    ],
    """**The optimized direction beside the anchor**, on the test set, net. The two runs counted over the criterion
    are runs whose frontier never reaches the anchored removal.""",
)

# %%

assert S1_VERDICT == "Matches"

f"""
**{S1_VERDICT}.** The interval on the gap in removal straddles zero and lies within 0.05 of it, and the Fisher test cannot tell the counts apart, so the best rank-1 search acting at every position does about as well as the anchor. By the frozen rule, the advantage of placing the concept is then that no search is needed.

That conclusion takes the search as the reference. H2 shows that fitters acting at the example answers do better still, and we come back to this in the discussion.

The removal here is net of the axis median, the baseline that favors the method. Under the permuted-label baseline alone the optimized direction would remove far less, and the ceiling would fall short (E6).
"""

# %%

r"""
## The fitters find a placed concept (H4)

**What we expect.** If we fit on the anchored runs, where training placed the op on e₁, then each standard fitter removes about as much as the edit along e₁ does, within the same spill limit. A fitter that cannot do that where the concept is known to be is no fair stand-in for its method on the controls.

**The rule.** Restricted to fits on the pooled positions or at the example answers, a fitter is accepted if its removal at the spill criterion on the anchored runs is no more than 0.1 below what the edit along e₁ removes at full dose on the same runs (seed means), and rejected otherwise. A fitter that finds e₁ but has to stop short of it to stay within the criterion, as INLP may when its extra directions cost spill, is rejected. H4 passes if every standard fitter is accepted, misses if every one is rejected, and is partial otherwise. On the anchored runs the fitter takes its permuted-label baseline, and the edit along e₁ keeps its published one. The query `=` is left out: the clean alignment there is near zero in the anchored runs (the mid-block-edit experiment), so a fit at that site is not expected to find e₁.

Beside the verdict, and not part of it, we report how close each fitted direction is to e₁: the cosine with e₁[^cosine] at each of slices 1 to 3, for the target and site of the selected configuration, given as the seed mean of its absolute value. For INLP it is the largest cosine between e₁ and the span.

[^cosine]: Cosine with e₁: how closely a direction lines up with e₁, 1 when it lies along e₁ and 0 when it is perpendicular to it.
<!-- REVIEW: the first draft also gated H4 on a cosine of 0.9 with e₁. The paper round asked whether that assumes every `difference` context sits fully on e₁; it does not, and the low-runs experiment found part of the evidence off e₁ in every run, so a fitted direction can remove the op while pointing partly elsewhere. H4 now gates on removal within the spill criterion, which is what the fitter is used for, and reports the cosine. Verify: the cosines below. -->

The rule for a rejected fitter, frozen here: a fitter rejected under H4 is left out of the verdicts of H1, H2, H3, and S1, its rows stay in the tables of every section, greyed out, and the report names it as a method the comparison could not fairly score. A fitter that cannot remove a placed concept would likely do badly on the controls too, so leaving it out leans toward the fitters, which is the cautious direction for a claim about the anchor.

The figure shows the removal at the spill criterion of each fitter on the anchored runs, restricted to the two sites, beside the edit along e₁.
"""

# %%

per_run_draw(
    "post-hoc-compare-h4",
    {"anchor": A_REMOVAL} | {key(m): M_H4[m]["removal"] for m in FITTERS},
    float(A_REMOVAL.mean() - H4_MARGIN),
    "below",
    f"removal of {DIFF} at the criterion (net)",
    (0.3, 1.02),
    f"""**Removal at the spill criterion on the anchored runs**, fits on the pooled positions or at the example
    answers, net of the permuted-label baseline; the edit along e₁ at full dose beside them, net of its published
    baseline. The dashed line is {H4_MARGIN} under the seed mean of the edit along e₁, and the side under it is
    hatched.""",
    "Removal on the anchored runs for the edit along e₁ and the four fitters. LEACE sits above the anchor; the "
    "probe and INLP sit just under it; difference in means sits on the gate line, with one run near 0.42.",
)

# %%


def cosines(m: str) -> np.ndarray:
    """|cos| with e₁ at slices 1 to 3 per anchored run, at the configuration the rule chose over the whole grid."""
    return np.abs([RUNS[("anchored", s)]["finish"]["methods"][m]["cosine_e1"] for s in SEEDS])


def h4_table() -> str:
    rows = []
    for m in FITTERS:
        v = M_H4[m]
        cos = cosines(m).mean(axis=0)
        rows.append([SHOWN[m], span(v["removal"]), span(v["raw_removal"]), *[num(c) for c in cos],
                     "accepted" if H4_ACCEPTED[m] else "rejected"])  # fmt: skip
    return table_html(
        ["fitter", "removal at criterion ↑", "raw", "cos, slice 1", "slice 2", "slice 3", "verdict"],
        rows,
        f"""**The fitters on the anchored runs**, restricted to the pooled positions and the example answers: removal
        at the spill criterion (seed mean and range, net of the permuted-label baseline, and raw), and the seed mean
        of the absolute cosine with e₁ at slices 1 to 3. The edit along e₁ removes {span(A_REMOVAL)}; a fitter is
        accepted at {num(A_REMOVAL.mean() - H4_MARGIN, ".3f")} or more. The cosines are for the configuration the rule chose
        over the whole grid, which on most runs is the same site. For LEACE the cosine is that of its covariance
        vector, which is the difference in means, so its row matches that of difference in means where both chose
        the same fit.""",
    )


h4_table()

# %%

assert H4_VERDICT == "Partial" and not H4_ACCEPTED["dim"] and all(H4_ACCEPTED[m] for m in ("ridge", "leace", "inlp"))
H4_DIM_GAP = float(A_REMOVAL.mean() - H4_MARGIN - M_H4["dim"]["removal"].mean())
assert 0 < H4_DIM_GAP < 0.001, H4_DIM_GAP  # the prose says "by less than 0.001"

f"""
**{H4_VERDICT}.** The probe, LEACE, and INLP each remove within {H4_MARGIN} of what the edit along e₁ removes, and LEACE more. Difference in means falls short of the gate by less than 0.001 ({num(M_H4["dim"]["removal"].mean(), ".3f")} against {num(A_REMOVAL.mean() - H4_MARGIN, ".3f")}; both print as 0.78 in the table), so it is rejected and left out of the other verdicts.

That rejection is fragile, and its cause is the baseline more than the fitter. Raw, difference in means removes as much as the others; it is the permuted-label baseline that takes it under the gate. Net of the coordinate-axis median instead, the baseline the anchored edit itself is scored against, difference in means removes {num(M_H4_AXIS["dim"]["removal"].mean())}, well above the gate, and every fitter would be accepted (removals {num(min(M_H4_AXIS[m]["removal"].mean() for m in FITTERS))} to {num(max(M_H4_AXIS[m]["removal"].mean() for m in FITTERS))}). Had it been kept in, H1 and H2 would have accepted it too, so leaving it out changes no verdict. This reading under the axis baseline was added after the run.

On the anchored runs a permuted fit is not neutral. The difference of the means of two random halves of the states points along the directions where the states vary most, and training made e₁ one of those. So a permuted fit removes part of the op too, which puts the net removal of every fitter on these runs below its raw one. E6 compares the baselines on the controls.

The directions of difference in means, LEACE, and INLP on these runs point well toward e₁ at the slice where the fit was made, without lying on it, as the low-runs experiment led us to expect. The probe points much less toward e₁ and still removes the op, likely because a probe turns its direction away from where the states vary most, and training made e₁ one of those directions.

<!-- REVIEW: the run's selection code did not restrict H4 to the pooled fits and the example answers, as the rule above says. A second pass after the run (about $0.64) ran the selection again over that part of the grid, on the same tuning scores, and computed the test-set baselines of any configuration it chose that the main pass had not scored; the table of the pooled fits in H2 comes from the same pass. The cosines in the table still come from the configuration chosen over the whole grid, which is at the example answers or pooled on nearly every run. Verify: is the second pass described well enough in the method, and is the cosine source fine? -->
"""

# %%

r"""
## Grading, landing, and where the answers go (E2)

At the configuration each method selects for removal at the spill criterion, we run the scoring suite of the fixed-effects experiment. The anchored edit is scored on the same held-out contexts as the other methods.

These measures read each context on its own, where H1 to H4 average the drop over all the contexts of an op. The baselines were taken per op, so they cannot be subtracted context by context, and the drops here are raw.

Grading is the share of the way to the target null removed in each bin of the posterior on `difference`. Landing is how far the edit moves the answers toward the target null, in total variation. A matrix of where the edited model puts its answers on `difference` contexts shows where they go.
"""

# %%

HELD = ARRAYS["held-out"]
OP_IDS = HELD["op_ids"].astype(int)
IS_D = OP_IDS == D
POST_D = HELD["posterior"][:, D].astype(np.float64)
NULL_EEM = HELD["null_eem"].astype(np.float64)
POST_BINS = (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)
BIN_LABEL = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–0.95", "≥ 0.95"]
BIN = np.clip(np.digitize(POST_D, POST_BINS[1:-1]), 0, len(BIN_LABEL) - 1)
for _s in SEEDS:  # the arrays of this experiment are on the same contexts, in the same order
    _a = ARRAYS[fc.label_of(_s, None)]["clean/eem"]
    assert _a.shape == NULL_EEM.shape
assert np.allclose(ARRAYS[fc.label_of(SEEDS[0], None)]["clean/eem"], HELD["clean_eem"], atol=2e-3)
P = ex.ex2216._get_posterior()
TABLE = P.build_table(ex.ex2218.table_of(OPS))


def label_of(kind: str, s: int) -> str:
    return RUNS[(kind, s)]["label"]


def method_finish(m: str, s: int, pooled: bool = False) -> dict:
    """What `finish_one` read for method *m* on the control at seed *s*: at the selection over the whole grid, or
    with *pooled*, over the pooled fits alone (the optimized direction is always pooled).
    """
    if pooled and m in FITTERS:
        return POOLED_FINISH[label_of("control", s)]["methods"][m]
    return RUNS[("control", s)]["finish"]["methods"][m]


def edited(m: str, s: int, pooled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean and edited expected exact match per context, and the clean and edited answer distributions on the
    `difference` contexts, for a method on the control at seed *s* (with *pooled*, at its pooled selection), or for
    the anchor on the anchored run.
    """
    if pooled and m in FITTERS:
        a = POOLED_ARRAYS[label_of("control", s)]
    else:
        a = ARRAYS[label_of("anchored" if m == "anchor" else "control", s)]
    return a["clean/eem"], a[f"{m}/eem"], a["clean/p_difference"], a[f"{m}/p_difference"]


def share_by_bin(m: str, pooled: bool = False) -> np.ndarray:
    """The share of the way to the target null removed, per bin of the posterior on `difference`, per seed."""
    out = []
    for s in SEEDS:
        clean, ed, _, _ = edited(m, s, pooled)
        drop, dep = clean - ed, clean - NULL_EEM
        out.append([drop[(BIN == b) & IS_D].mean() / dep[(BIN == b) & IS_D].mean() for b in range(len(BIN_LABEL))])
    return np.asarray(out)


E2_METHODS = ("anchor", *METHODS)
SHARE = {m: share_by_bin(m) for m in E2_METHODS}


def sans_share_by_bin() -> np.ndarray:
    """`share_by_bin` for the sans-diff model in place of an edit: the drop from the control at the same seed."""
    out = []
    for s in SEEDS:
        clean, ed = edited("dim", s)[0], ARRAYS[("sans-diff", s)]["clean_eem"]
        drop, dep = clean - ed, clean - NULL_EEM
        out.append([drop[(BIN == b) & IS_D].mean() / dep[(BIN == b) & IS_D].mean() for b in range(len(BIN_LABEL))])
    return np.asarray(out)


SANS_SHARE = sans_share_by_bin()


@memo
def grading_draw(share: dict, alt_text: str, name: str = "post-hoc-compare-grading", caption: str | None = None) -> str:
    @themed(
        name=name,
        alt_text=alt_text,
        caption=caption
        or """
            **The share of the way to the target null removed, by how sure the context is of `difference`**, on the
            held-out `difference` contexts, raw, seed mean over eight runs; the anchor on the anchored runs, the
            methods on the controls at the configuration each selected for removal at the spill criterion, and the
            sans-diff model (no edit) against the control at the same seed.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 2.6), layout="constrained")
        x = np.arange(len(BIN_LABEL))
        for m, v in share.items():
            ax.plot(x, np.mean(v, axis=0), "-", marker=MARKER[m], ms=4, color=ink(m), lw=2.0 if m == "anchor" else 1.0,
                    label=SHOWN[m], zorder=4 if m == "anchor" else 3)  # fmt: skip
        ax.axhline(1.0, color=rule(), lw=0.6, ls=":")
        ax.set_xticks(x, BIN_LABEL, fontsize=7)
        ax.set_xlabel(f"posterior on {DIFF}", fontsize=8)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel("share removed", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.legend(loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


grading_draw(
    {m: SHARE[m] for m in E2_METHODS} | {"sans": SANS_SHARE},
    alt_text="Share removed by posterior bin for the anchor and five methods, on an axis from 0 to 1. Every line is "
    "nearly flat, between about 0.82 and 0.98, rising a little from the lowest bin to the middle one; the anchor and "
    "INLP sit at the top and the optimized direction at the bottom.",
)

# %%


def contexts_of(op_ids: np.ndarray, query_pair: np.ndarray) -> Any:
    empty = np.zeros((len(op_ids), 0), np.int64)
    return P.Contexts(op_ids.astype(np.int64), empty, empty, query_pair.astype(np.int64))


@memo
def target_null_d(posterior: np.ndarray, op_ids: np.ndarray, query_pair: np.ndarray) -> np.ndarray:
    """The answer distribution of the target null on the `difference` contexts."""
    ctx = contexts_of(op_ids, query_pair)
    null = ex.ex2216._target_null(P, TABLE, ctx, posterior.astype(np.float64), D)
    return P.predictive(TABLE, ctx, null)[op_ids == D]


Q_NULL_D = target_null_d(HELD["posterior"], OP_IDS, HELD["query_pair"])


def tv(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Total variation between a palette distribution and a model one, the mass off the palette counted as distance."""
    return 0.5 * (np.abs(q - p).sum(-1) + np.clip(1.0 - p.sum(-1), 0, None))


def landing(m: str, pooled: bool = False) -> np.ndarray:
    """The share of the clean distance to the target null that the edit covers, in total variation, per seed."""
    out = []
    for s in SEEDS:
        _, _, pc, pe = edited(m, s, pooled)
        out.append(1 - tv(Q_NULL_D, pe.astype(np.float64)).mean() / tv(Q_NULL_D, pc.astype(np.float64)).mean())
    return np.asarray(out)


LANDING = {m: landing(m) for m in E2_METHODS}


def share_all(m: str, pooled: bool = False) -> np.ndarray:
    out = []
    for s in SEEDS:
        clean, ed, _, _ = edited(m, s, pooled)
        out.append((clean - ed)[IS_D].mean() / (clean - NULL_EEM)[IS_D].mean())
    return np.asarray(out)


SHARE_ALL = {m: share_all(m) for m in E2_METHODS}
table_html(
    ["", "share removed ↑", "landing ↑"],
    [[SHOWN[m], span(SHARE_ALL[m]), span(LANDING[m])] for m in E2_METHODS],
    """**How much and how far**, on the held-out `difference` contexts, seed mean and range, raw. The share removed
    is the drop as a share of the way to the target null, over all of them. The landing is the share of the clean
    distance to the target null, in total variation, that the edit covers. A greyed row is a fitter rejected under
    H4.""",
    grey=h4_grey(E2_METHODS),
)

# %%


@memo
def gives_table(pair: np.ndarray) -> np.ndarray:
    """For each op and context, which colors that op can give on the query pair: `(ops, n, colors)`."""
    n = len(pair)
    out = np.zeros((TABLE.n_ops, n, P.N_COLORS), dtype=bool)
    rows = np.arange(n)
    for o in range(TABLE.n_ops):
        idx, ok = TABLE.idx[o, pair], TABLE.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            out[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    return out


GIVES_D = gives_table(HELD["query_pair"].astype(np.int64))[:, IS_D]  # (ops, difference contexts, colors)
TRUE_D = GIVES_D[D]


def where_mass(p: np.ndarray) -> np.ndarray:
    """On the `difference` contexts: the mass on the answers `difference` can give (at D), and on the answers each
    other op can give and `difference` cannot.
    """
    p = p.astype(np.float64)
    return np.array([(p * (TRUE_D if o == D else GIVES_D[o] & ~TRUE_D)).sum(1).mean() for o in range(len(OPS))])


SANS_D = {}
for _s in SEEDS:
    _z = ARRAYS[("sans-diff", _s)]
    assert (_z["op_ids"] == HELD["op_ids"]).all() and (_z["query_pair"] == HELD["query_pair"]).all()
    SANS_D[_s] = _z["p_clean"][IS_D]
WHERE = {"clean (control)": np.mean([where_mass(edited("dim", s)[2]) for s in SEEDS], axis=0),
         "target null": where_mass(Q_NULL_D),
         "sans-diff": np.mean([where_mass(SANS_D[s]) for s in SEEDS], axis=0),
         "clean (anchored)": np.mean([where_mass(edited("anchor", s)[2]) for s in SEEDS], axis=0)}  # fmt: skip
WHERE |= {SHOWN[m]: np.mean([where_mass(edited(m, s)[3]) for s in SEEDS], axis=0) for m in E2_METHODS}


@memo
def where_draw(rows: dict, alt_text: str, name: str = "post-hoc-compare-where", caption: str | None = None) -> str:
    @themed(
        name=name,
        alt_text=alt_text,
        caption=caption
        or f"""
            **Where the answers go on `{DIFF}` contexts**, seed mean over eight runs. Each row is a model or an edit;
            each column the mass on the colors the column op can give for the query operands and `{DIFF}` cannot.
            The `{DIFF}` column (the mass on its own answers) has a bold label, is printed in grey italics, and is
            left out of the color scale; the anchored edit has a border in its own color. Values of at least 0.01 are printed. Rules separate the unedited references (above), the
            anchored edit, and the post-hoc edits (below).
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.patches import Rectangle

        names = list(rows)
        m = np.array([rows[n] for n in names])
        fig, ax = plt.subplots(figsize=(5.0, 0.5 * len(names) + 1.0), layout="constrained")
        show = m.copy()
        show[:, D] = np.nan
        cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
        cmap.set_bad(light_dark("#fff", "#111"))
        vmax = float(np.nanmax(show))
        ax.imshow(show, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        for i, j in np.ndindex(*m.shape):
            if j == D:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6, color=light_dark("#777", "#999"),
                        fontstyle="italic")  # fmt: skip
            elif m[i, j] >= 0.01:
                dark_cell = (m[i, j] / vmax > 0.55) == (light_dark(0, 1) == 0)
                ax.text(
                    j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6, color="#fff" if dark_cell else "#000"
                )
        ax.set_xticks(range(len(OPS)), OPS, fontsize=7, rotation=30, ha="right")
        ax.get_xticklabels()[D].set_fontweight("bold")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        a = names.index(SHOWN["anchor"])
        ax.get_yticklabels()[a].set_fontweight("bold")
        for i in (3, 4):
            ax.axhline(i + 0.5, color=light_dark("#000", "#fff"), lw=0.8)
        ax.add_patch(
            Rectangle((-0.5, a - 0.5), len(OPS), 1, fill=False, ec=ink("anchor"), lw=2.0, zorder=5, clip_on=False)
        )
        return fig

    return _plot()


where_draw(
    WHERE,
    alt_text="A matrix of where the answers go on difference contexts, one row per model or edit and one column per "
    "op. The clean rows put most mass on difference answers; the target null and every edit move it to the answers of "
    "the other ops, mostly the same ones.",
)

# %%

_BIN_MEAN = {m: SHARE[m].mean(axis=0) for m in E2_METHODS}
_LOW = [v[0] for v in _BIN_MEAN.values()]
_MID = [v[2] for v in _BIN_MEAN.values()]
_FIT_LAND = [LANDING[m].mean() for m in FITTERS]
_STAYS = [WHERE[SHOWN[m]][D] for m in E2_METHODS]

f"""
Every edit grades its drop by how sure the context is of `{DIFF}`, and they all grade about alike. The share of the way to the target null removed stays between {num(min(_LOW))} and {num(max(_MID))} in every bin, so each edit drops a context by about as much as that context needed `{DIFF}`. The share is lowest where the context is least sure, {num(min(_LOW))} to {num(max(_LOW))} below a posterior of 0.2, and highest between 0.5 and 0.8, at {num(min(_MID))} to {num(max(_MID))}.

It dips a little for the surest contexts, most for the probe and the optimized direction. The anchor and INLP sit at the top in every bin, and the optimized direction at the bottom.

The fitters land a little further toward the target null than the anchor: {num(min(_FIT_LAND))} to {num(max(_FIT_LAND))} of the clean distance, against {num(LANDING["anchor"].mean())}. The optimized direction, which is fitted toward the target null, lands furthest at {num(LANDING["optimized"].mean())}, about where the sans-diff model lands. The ranges over runs overlap.

None of the edits gets all the way, and they fall short in the same way. The answers they move off `{DIFF}` go to the answers of the other ops in about the proportions the target null gives, a fifth to a quarter each, as do those of the sans-diff model, which was trained with no loss on `{DIFF}`; and {num(min(_STAYS))} to {num(max(_STAYS))} stays on the answers of `{DIFF}` itself.

The anchor puts a little less than the methods or the target null on the answers of `lighten`, and a little more on those of `sat-hsv`. On grading and on where the answers go, the anchored edit and the post-hoc edits are hard to tell apart.

<!-- REVIEW: the plan also named the gain on the contexts of other ops by bin, and the landing on the scale from the control to the sans-diff model. Neither is drawn here: the first because the spill of H1 and H2 already reads the other ops, the second because the sans-diff arrays are on another set of runs. Verify: is leaving them out fine, or should they be added? -->
"""

# %%

r"""
## How many labels a fitter needs (E3)

At the configuration each method selects, we repeat the fit with labels on every `difference` context of the fit set, then on a fixed half, a tenth, a twenty-fifth, and a fiftieth of them. The last two pair with the anchored runs of the fixed-labels experiment, which held one draw at those rates. The unlabelled `difference` contexts are counted among the other contexts, as an unlabelled context is for the anchor.

A fixed fiftieth is 800 contexts: about what the anchor met in one pass over the corpus, and far fewer than it met over training. So in that respect the lowest rate is harsher than what the anchor had. In another it is kinder: the fitters still see every label of the fixed set at once.

The removal and spill here are those of the configuration E2 to E5 read, with its own baselines: for each method, the lower-spill configuration of the pair that brackets removal at the spill criterion, which for every fitter is a fit at the example answers. So at every label count the numbers differ slightly from the points of H2, which interpolate between two configurations.
"""

# %%


def ladder(m: str, pooled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill of *m* at each rung of the label ladder, per seed: `(seeds, rungs)` each, at the
    configuration E2 to E5 read, net of its baselines by the rule that favors the method.
    """
    rem, spl = [], []
    for s in SEEDS:
        run = RUNS[("control", s)]
        i = scope_selection(run, m, "pooled" if pooled else "all")["at_criterion"]["configs"][0]
        g = run["test"][METHODS.index(m)]
        clean, null = np.asarray(g["clean"]), np.asarray(g["null"])
        den = clean[D] - null[D]
        bases = baselines(run, m, i)
        drops = np.asarray(method_finish(m, s, pooled)["ladder_drops"])
        rem.append([max((d[D] - b[D]) / den for b in bases) for d in drops])
        spl.append([min(np.delete(d - b, D).max() for b in bases) for d in drops])
    return np.asarray(rem), np.asarray(spl)


RATES = ex.LADDER
LADDER = {m: ladder(m) for m in FITTERS}
RATE_NAME = {1.0: "all", 0.5: "1/2", 0.1: "1/10", 0.04: "1/25", 0.02: "1/50"}
# The anchored runs of fixed-labels at each label rate, held fixed for the run.
FIXED_AT = {
    rate: np.array([r["removal"][-1] for r in anchor_runs(LABELS_AXES, lambda s, k=k: fl.label_of(s, rate=k)).values()])
    for k, rate in fl.RATES.items()
}
assert set(FIXED_AT) <= set(RATES)


@memo
def ladder_draw(ladder: dict, anchor: float, fixed: dict, alt_text: str, name: str = "post-hoc-compare-ladder",
                caption: str | None = None) -> str:  # fmt: skip
    @themed(
        name=name,
        alt_text=alt_text,
        caption=caption
        or """
            **Removal against the share of `difference` contexts labelled**, on the test set, net of the baselines,
            seed mean (marker) and range (bar) over the eight controls, at the configuration each fitter selected
            with every label. The dotted line is the anchored removal at the recipe of record, which draws a
            fiftieth of the labels afresh on every visit. The diamonds are the anchored runs of fixed-labels, which
            hold one draw for the whole run.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 2.8), layout="constrained")
        x = np.log10(RATES)
        step = 0.035
        for k, (m, (rem, _)) in enumerate(ladder.items()):
            dx = (k - 1.5) * step
            ax.vlines(x + dx, rem.min(0), rem.max(0), color=ink(m), lw=0.8, alpha=0.5)
            ax.plot(x + dx, rem.mean(0), "-", marker=MARKER[m], ms=4, color=ink(m), lw=1.0, label=SHOWN[m])
        ax.axhline(anchor, color=ink("anchor"), lw=1.0, ls=":", label="anchor, labels redrawn")
        fx = np.log10(sorted(fixed))
        fv = [fixed[r] for r in sorted(fixed)]
        dx = 2.5 * step
        ax.vlines(fx + dx, [v.min() for v in fv], [v.max() for v in fv], color=ink("anchor"), lw=0.8, alpha=0.5)
        ax.plot(fx + dx, [v.mean() for v in fv], MARKER["anchor"], ms=5, color=ink("anchor"), ls="none",
                mec=light_dark("white", "#111"), mew=0.5, label="anchor, fixed labels")  # fmt: skip
        ax.set_xticks(x, [RATE_NAME[r] for r in RATES], fontsize=7)
        ax.invert_xaxis()
        ax.set_xlabel(f"share of {DIFF} contexts labelled", fontsize=8)
        ax.set_ylabel("removal", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.legend(loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


ladder_draw(
    LADDER,
    float(A_REMOVAL.mean()),
    FIXED_AT,
    alt_text="Removal against the share of difference contexts labelled, from all down to one in fifty. Difference in "
    "means, LEACE, and INLP stay flat near 0.96; the probe falls to about 0.84 at one in fifty, with one run near 0.5. "
    "The anchored runs with fixed labels sit lower, near 0.81.",
)

# %%

table_html(
    ["", *(RATE_NAME[r] for r in RATES)],
    [
        [SHOWN[m], *(f"{num(rem[:, j].mean())} ({over_count(spl[:, j])})" for j in range(len(RATES)))]
        for m, (rem, spl) in LADDER.items()
    ]
    + [["anchor, fixed labels", *(num(FIXED_AT[r].mean()) if r in FIXED_AT else "–" for r in RATES)]],
    """**Removal by the share of labels**, seed mean over eight runs, with the number of runs over the spill
    criterion in brackets. The anchored row is the fixed-labels experiment, at the full dose, where it spills on no
    run. A greyed row is a fitter rejected under H4.""",
    grey=h4_grey(LADDER),
)

# %%

_LAST = {m: LADDER[m][0][:, -1] for m in FITTERS}
_FLAT = [m for m in FITTERS if LADDER[m][0][:, 0].mean() - _LAST[m].mean() < 0.02]
assert set(_FLAT) == {"dim", "leace", "inlp"}, _FLAT
_RIDGE_LOW = _LAST["ridge"].min()

f"""
Difference in means, LEACE, and INLP need few labels: from every label down to a fiftieth, their removal moves by less than 0.02, so a mean or a covariance over 800 labelled contexts already points the right way. The probe drops to {num(_LAST["ridge"].mean())} at a fiftieth, with one run at {num(_RIDGE_LOW)} and one over the spill criterion.

The anchor with fixed labels removes {num(FIXED_AT[0.02].mean())} at a fiftieth, below the three flat fitters at the same rate. So the fitters do not need many labels, and the number of labels is not where the anchor has an edge. The fitters at a fiftieth are fitted at the example answers, so the rate is matched with the anchor and the site is not.
"""

# %%

r"""
## The write against each bound (E4)

The write bound of the anchor does not apply as it stands to a direction that is not an axis, or to an oblique eraser, so each method gets the bound in its own terms: the angle its operator turns the clean states of the other ops through (the 99% quantile, per slice and role).

The write is the angle the operator turns those states through as they arrive, already changed by the edits at the earlier slices. Where the blocks after an edit put the concept back, the method shows a write over its own bound. A site counts as over when the write passes the bound by more than both tolerances of the D2.2 design: a quarter over the bound, and about a degree.
"""

# %%

ABS_TOL = np.radians(1.13)  # the absolute tolerance of the write-floor experiment, about a degree
REL_TOL = 1.25
ROLE = ("a", "?", "b", "=", "y", ",")


def write_over(m: str, pooled: bool = False) -> tuple[int, list[tuple[int, int, str, float, float]]]:
    """How many runs edit more than one slice at the configuration E2 to E5 read, and the sites (seed, slice, role,
    write, bound in degrees) past the embedding where the write goes over both tolerances.
    """
    kind = "anchored" if m == "anchor" else "control"
    several, over = 0, []
    for s in SEEDS:
        fin = RUNS[(kind, s)]["finish"]
        got = None if m == "anchor" else method_finish(m, s, pooled)
        w = fin["anchor"]["write"] if got is None else got["write"]
        several += got is None or len(got["main_config"]["slices"]) > 1
        W, B = np.asarray(w["write"]), np.asarray(w["bound"])
        for sl, r in zip(*np.nonzero((W > REL_TOL * B) & (W - B > ABS_TOL)), strict=True):
            if sl > 0:
                over.append((s, int(sl), ROLE[r], float(np.degrees(W[sl, r])), float(np.degrees(B[sl, r]))))
    return several, over


WRITE_OVER = {m: write_over(m) for m in E2_METHODS}
assert [(m, s) for m in E2_METHODS for s in sorted({o[0] for o in WRITE_OVER[m][1]})] == [
    ("ridge", 716),
    ("optimized", 717),
]


def _sites(over: list) -> str:
    if not over:
        return "none"
    return "; ".join(f"{s}, slice {sl} at `{r}`: {num(w, '.1f')}° against {num(b, '.1f')}°" for s, sl, r, w, b in over)


table_html(
    ["", "runs editing several slices", "sites over"],
    [[SHOWN[m], str(WRITE_OVER[m][0]), _sites(WRITE_OVER[m][1])] for m in E2_METHODS],
    """**The write against each bound**, on 2,000 test contexts of the other ops, at the configuration E2 to E5
    read. A site is over when the write passes the bound by more than a quarter and by more than about a degree;
    each is given as seed, slice, and role, with the write and the bound at the 99% quantile. A greyed row is a
    fitter rejected under H4.""",
    grey=h4_grey(E2_METHODS),
)

# %%

f"""
The write stays within its bound almost everywhere, though for most of the post-hoc edits there is little to measure. An edit at one slice has no edits before it, so its write equals its bound, and most of the configurations read here edit slice 2 alone.

Of the {NUMBER_WORD[sum(WRITE_OVER[m][0] for m in METHODS)]} runs where a method edits several slices, two show a site over: the probe at seed 716 and the optimized direction at seed 717. The anchored edit acts at every slice and shows no site over, as in the fixed-effects experiment.
"""

# %%

r"""
## What a fresh probe can still find (E5)

LEACE promises that no linear probe can read the concept after its edit, and the anchor promises nothing about that. Here we fit a fresh ridge probe for the binary target on the edited states of the test set, per slice and per method, with cross-validation over contexts.

The evidence for the op can be recomputed from the operand colors (the block1-edit experiment), so a method may leave the op readable even where the answer no longer uses it. A probe that still finds the op does not mean the edit failed to remove it from the answer, and a probe that finds nothing does not bound what the edit writes.

We report it because it is the measure the erasure literature uses, and the one on which the anchor may come off worse. We expect LEACE to bring the probe near chance at the slices it edits, since that is what it is built to do.

We expect the anchored edit and the projections to leave the op readable well above chance, since the evidence can be recomputed and part of it lies off any one direction (the low-runs experiment). If the projections also bring the probe near chance, the op is held along fewer directions than that suggests.

The query `=` is the position whose output predicts the answer the task scores, so a model that uses the op must have it there in some form; the example answers are where block 1 writes the evidence for it (the block1-edit experiment). We score the probe by the area under the ROC curve[^auc] at the query `=` and at the example answers, pooled. It is fitted on half of 7,000 test contexts and scored on the other half.

[^auc]: Area under the ROC curve (AUC): the chance that the probe scores a random `difference` context above a random context of another op. 0.5 is chance, 1 a perfect split.
"""

# %%

PROBE_SITES = ("query =", "answers")


def probe_auc(kind: str, which: str, pooled: bool = False) -> np.ndarray:
    """Seed-mean AUC per site and slice, `(sites, slices)`: *which* is "clean", "anchor", or a method."""
    out = []
    for s in SEEDS:
        fin = RUNS[(kind, s)]["finish"]
        p = fin["probe_clean"] if which == "clean" else fin["anchor"]["probe"] if which == "anchor" else method_finish(which, s, pooled)["probe"]  # fmt: skip
        out.append([[p[f"{site}, slice {sl}"] for sl in range(ex.N_SLICES)] for site in PROBE_SITES])
    return np.mean(out, axis=0)


AUC = {"clean control": probe_auc("control", "clean"), "clean anchored": probe_auc("anchored", "clean")}
AUC |= {SHOWN[m]: probe_auc("control", m) for m in METHODS} | {SHOWN["anchor"]: probe_auc("anchored", "anchor")}
SANS_AUC = np.mean(
    [
        [[SANS_PROBE[f"sans-diff-s{s}"][f"{site}, slice {sl}"] for sl in range(ex.N_SLICES)] for site in PROBE_SITES]
        for s in SEEDS
    ],
    axis=0,
)
AUC["sans-diff"] = SANS_AUC


@memo
def probe_draw(auc: dict, alt_text: str, name: str = "post-hoc-compare-probe", caption: str | None = None) -> str:
    @themed(
        name=name,
        alt_text=alt_text,
        caption=caption
        or f"""
            **What a fresh probe finds after each edit**, the area under the ROC curve of a ridge probe for
            `{DIFF}` on the edited states, by slice, seed mean over eight runs; at the query `=` (left) and the
            example answers (right). The dashed grey line is the clean control, the dashed teal one the clean
            anchored model, the dashed brown one the sans-diff model (trained with no loss on `{DIFF}`, clean), and
            the dotted line chance.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.arange(ex.N_SLICES)
        key = {SHOWN[m]: m for m in E2_METHODS}
        for j, (ax, site) in enumerate(zip(axes, PROBE_SITES, strict=True)):
            for name, v in auc.items():
                if name.startswith("clean") or name == "sans-diff":
                    c = (
                        ink("anchor")
                        if "anchored" in name
                        else ink("sans")
                        if name == "sans-diff"
                        else light_dark("#888", "#999")
                    )
                    ax.plot(x, v[j], "--", color=c, lw=0.9, label=name)
                else:
                    m = key[name]
                    ax.plot(x, v[j], "-", marker=MARKER[m], ms=3.5, color=ink(m), lw=2.0 if m == "anchor" else 1.0,
                            label=name, zorder=4 if m == "anchor" else 3)  # fmt: skip
            ax.axhline(0.5, color=rule(), lw=0.6, ls=":")
            ax.set_title("query =" if site == "query =" else "example answers", fontsize=8)
            ax.set_xticks(x, [str(i) for i in x], fontsize=7)
            ax.set_xlabel("slice", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("probe AUC", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=7)
        return fig

    return _plot()


probe_draw(
    AUC,
    alt_text="Probe AUC by slice at the query equals sign and the example answers. Clean, the probe reaches about 0.9 at "
    "the query equals sign at slice 3 and 0.83 at the answers. After every edit it falls to about 0.65 to 0.7 at the "
    "query equals sign and 0.65 to 0.8 at the answers; none reaches chance.",
)

# %%

_Q3 = {n: v[0, 3] for n, v in AUC.items()}
_A2 = {n: v[1, 2] for n, v in AUC.items()}
_EDITS = [SHOWN[m] for m in E2_METHODS]
assert min(_Q3[n] for n in _EDITS) > 0.6 and min(_A2[n] for n in _EDITS) > 0.6
assert min(_Q3[n] for n in _EDITS) - 0.05 < _Q3["sans-diff"] < max(_Q3[n] for n in _EDITS) + 0.02, _Q3

f"""
No edit brings the probe near chance. Before any edit, a probe finds the op at the query `=` of slice 3 with an AUC of {num(_Q3["clean control"])} on the control and {num(_Q3["clean anchored"])} on the anchored model. After the edits it scores {num(min(_Q3[n] for n in _EDITS))} to {num(max(_Q3[n] for n in _EDITS))} there, lowest after the anchored edit.

The sans-diff model, trained with no loss on `{DIFF}`, leaves {num(_Q3["sans-diff"])} for a fresh probe at the same site, about what the edits leave. So most of what the probe finds after an edit is there in a model that never learned the op, likely because the operand colors carry it.

At the example answers the order turns around: the anchored edit leaves the op more readable ({num(_A2[SHOWN["anchor"]])} at slice 2) than difference in means, LEACE, or INLP ({num(min(_A2[SHOWN[m]] for m in ("dim", "leace", "inlp")))} to {num(max(_A2[SHOWN[m]] for m in ("dim", "leace", "inlp")))}), which edit there.

LEACE does not do what we expected of it, and we do not know why. At the slice and site it edits, mostly at full dose, a fresh probe still reads {num(_A2[SHOWN["leace"]])}, about what difference in means leaves.

Three candidates fit what we see:
**(a)** The model keeps its states at unit length, so every edit ends by scaling the state back to length one. Rescaling by a length that depends on what was taken out can bring some of the op back along every other direction, while the guarantee of LEACE is for the linear edit alone.
**(b)** Our LEACE takes a small ridge on a covariance that is close to rank-deficient (some directions barely vary), so it may leave low-variance directions that a fresh probe with a weak penalty can read.
**(c)** On two runs it was fitted to the graded target, where the guarantee is for the posterior rather than the binary label the probe reads.

<!-- REVIEW: the first draft offered the return to the sphere as the one candidate; the claims review added the other two and asked that the paragraph not lean on any of them unchecked. A one-seed check (probe before the return to the sphere, and LEACE without the ridge) is in the backlog. Verify: fine to leave it unchecked for this report? -->
"""

# %%

r"""
## The two baseline rules (E6)

On the controls, we compare the permuted-label baseline with the coordinate-axis median, configuration by configuration, for the configurations scored on the test set. The Adam optimizer concentrates energy on some axes of a control, so the two can differ.

The baseline decides how much of a drop counts as the edit, and each hypothesis takes whichever rule favors the fitters. So if the two rules differ by about the size of a gap that a verdict turns on, that verdict rests on the choice of baseline.
"""

# %%


def rule_pairs() -> dict[str, np.ndarray]:
    """Per method, net removal and spill under each test-set baseline at every configuration scored on the test
    set: `(configurations, 4)`, columns removal (permuted, axis) and spill (permuted, axis).
    """
    out = {}
    for m in METHODS:
        rows = []
        for s in SEEDS:
            run = RUNS[("control", s)]
            g = run["test"][METHODS.index(m)]
            clean, null = np.asarray(g["clean"]), np.asarray(g["null"])
            for key in run["finish"]["methods"][m]["configs"]:
                raw = clean - np.asarray(g["scores"][int(key)])
                perm, axis = baselines(run, m, int(key))
                rows.append([(raw[D] - b[D]) / (clean[D] - null[D]) for b in (perm, axis)]
                            + [np.delete(raw - b, D).max() for b in (perm, axis)])  # fmt: skip
        out[m] = np.asarray(rows)
    return out


PAIRS = rule_pairs()


@memo
def rules_draw(pairs: dict, alt_text: str) -> str:
    @themed(
        name="post-hoc-compare-baselines",
        alt_text=alt_text,
        caption="""
            **The two test-set baselines, configuration by configuration**, on the controls: net removal (left) and
            net spill (right) under the permuted-label baseline against the same under the coordinate-axis median,
            one mark per run and configuration. On the diagonal the two rules agree; below it, the permuted baseline
            takes more of the drop as background. The grey lines in the right panel mark the spill criterion on each
            axis.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, (j, title) in zip(axes, ((0, "removal"), (2, "spill")), strict=True):
            for m, v in pairs.items():
                ax.plot(v[:, j + 1], v[:, j], MARKER[m], ms=3.5, color=ink(m), alpha=0.7, mew=0, ls="none",
                        label=SHOWN[m])  # fmt: skip
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], color=rule(), lw=0.6, ls=":")
            if j == 2:
                ax.axvline(CRITERION, color=faint(), lw=0.6)
                ax.axhline(CRITERION, color=faint(), lw=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"{title}, axis median", fontsize=8)
            ax.set_ylabel(f"{title}, permuted labels", fontsize=8)
            ax.tick_params(labelsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=5, frameon=False, fontsize=7)
        return fig

    return _plot()


rules_draw(
    PAIRS,
    alt_text="Two scatter plots of the permuted-label baseline against the axis median, per configuration. For removal, "
    "most marks lie on or below the diagonal; for spill, many lie well below it, where the permuted edit spills about "
    "as much as the fitted one.",
)

# %%

M_RULE = {(m, r): measures("control", m, rule=r) for m in METHODS for r in ("perm", "axis")}
RULE_VERDICTS = {
    (m, r): (h1_verdict(m, M_RULE[(m, r)]), h2_verdict(m, M_RULE[(m, r)])) for m in FITTERS for r in ("perm", "axis")
}
table_html(
    ["", "removal, permuted", "removal, axis", "over, permuted", "over, axis", "H1, H2: permuted", "H1, H2: axis"],
    [
        [
            SHOWN[m],
            num(M_RULE[(m, "perm")]["removal"].mean()),
            num(M_RULE[(m, "axis")]["removal"].mean()),
            str(over_count(M_RULE[(m, "perm")]["spill"])),
            str(over_count(M_RULE[(m, "axis")]["spill"])),
            *(", ".join(RULE_VERDICTS[(m, r)]) if m in FITTERS else "–" for r in ("perm", "axis")),
        ]
        for m in METHODS
    ],  # fmt: skip
    """**The points of H1 and H2 under each baseline alone**, seed means over eight controls: removal at the spill
    criterion, and the number of runs over the criterion at the anchored removal, with the verdicts each would give.
    The hypotheses take the rule that favors the fitter, measure by measure. A greyed row is a fitter rejected
    under H4.""",
    grey=h4_grey(METHODS),
)

# %%

_PERM_COST = [M_RULE[(m, "axis")]["removal"].mean() - M_RULE[(m, "perm")]["removal"].mean() for m in FITTERS]
_SAME = all(RULE_VERDICTS[(m, r)] == (H1[m], H2[m]) for m in FITTERS for r in ("perm", "axis"))
_CHANGED = [(m, r) for m in FITTERS for r in ("perm", "axis") if RULE_VERDICTS[(m, r)] != (H1[m], H2[m])]
# REVIEW: the first draft said no verdict changes under either rule alone, from a check over H1 and H2 only. S1
# reads the axis rule for removal (the favoring rule), and under the permuted rule alone the optimized direction's
# removal falls to 0.58 and the interval on the gap lies wholly above zero, which the frozen rule reads as "falls
# short". Verify: `_S1_PERM_CI[0] > 0`.
_S1_PERM = M_RULE[("optimized", "perm")]["removal"]
_S1_PERM_CI = t_interval(A_REMOVAL, _S1_PERM)
assert _S1_PERM_CI[0] > 0, _S1_PERM_CI

f"""
The two rules disagree a lot on single configurations and little at the points the verdicts read. At some configurations a permuted-label fit costs every op about as much as the fitted edit does, while an edit along a coordinate axis costs almost nothing. Such a configuration shows little spill under the permuted rule and a lot under the axis rule, well past the criterion: the marks far below the diagonal in the right panel.

At the points the verdicts read, the axis rule gives the higher removal and the permuted rule the lower spill, so the favoring rule combines the two. Taken alone, the permuted rule lowers removal by {num(min(_PERM_COST))} to {num(max(_PERM_COST))} for the fitters and by {num(M_RULE[("optimized", "axis")]["removal"].mean() - M_RULE[("optimized", "perm")]["removal"].mean())} for the optimized direction, whose permuted searches remove much of the op themselves. {"No verdict of H1 or H2 changes under either rule." if _SAME else "The verdicts of H1 and H2 that change: " + ", ".join(f"{SHOWN[m]} under the {r} rule" for m, r in _CHANGED) + "."}

S1 is the exception. It uses the axis rule for removal, and under the permuted rule alone the optimized direction removes {num(_S1_PERM.mean())}. The interval on the gap then lies wholly above zero ({interval(_S1_PERM_CI)}), and the ceiling would fall short of the anchor.

A permuted search is not neutral either, since each context keeps its own target null under the permutation. So the search still pulls the `{DIFF}` contexts that land in the permuted labelled set toward the null, against the keep term on the rest, and it finds a direction that removes much of the op.
"""

# %%

r"""
## Without role labels

The anchor uses no role labels. Its pull pools over the positions of a labelled context, and its edit acts at every position and slice. A fit at the example answers, the query `=`, or the query operands uses a role, which a real corpus does not label. So the like-for-like comparison restricts every fitter to the pooled fits: the selection rule runs again over the configurations fitted at every position, pooled, on the same runs and with the same baselines. The fitters keep a label on every `difference` context and a grid tuned per run over the target, slices, dose, refit, replacement, and rank, which the anchor does not have.

This reading was not preregistered, so it gives no verdict. The table applies the rules of H1 and H2 as a guide to how far apart the methods are. After it, the figures of E1 to E5 are drawn again with the pooled fits alone.
"""

# %%

per_run_draw(
    "post-hoc-compare-pooled",
    {"anchor": A_REMOVAL} | {key(m): M_POOLED[m]["removal"] for m in METHODS},
    float(A_REMOVAL.mean()),
    "below",
    f"removal of {DIFF} at the criterion (net)",
    (0.1, 1.02),
    """**Removal at the spill criterion per run, with the pooled fits alone**, on the test set, net. A small dot per
    run and the seed mean as the method's marker; the anchor at full dose. The dashed line is the seed mean of the
    anchor, and the side under it is hatched.""",
    "Removal at the spill criterion per run with the pooled fits alone. LEACE and INLP sit near the anchor line, "
    "the probe a little below, and difference in means well below with one run near 0.2; the optimized direction "
    "sits on the anchor line.",
)

# %%

POOLED_CI = {m: t_interval(A_REMOVAL, M_POOLED[m]["removal"]) for m in METHODS}
POOLED_RULES = {m: (h1_verdict(m, M_POOLED[m]), h2_verdict(m, M_POOLED[m])) for m in METHODS}
POOLED_ROWS = ["anchor", *METHODS]
table_html(
    ["method", "removal at criterion ↑", "anchor minus method, 95% interval", "over at matched removal",
     "H1, H2 rules would read"],
    [["anchor", span(A_REMOVAL), "", f"{A_OVER} of 8", ""]]
    + [[SHOWN[m], span(M_POOLED[m]["removal"]), interval(POOLED_CI[m]), f"{over_count(M_POOLED[m]['spill'])} of 8" + (f" ({u} not matched)" if (u := int((~np.isfinite(M_POOLED[m]["spill"])).sum())) else ""),
        ", ".join(POOLED_RULES[m])] for m in METHODS],
    """**The comparison with the pooled fits alone**: removal at the spill criterion (seed mean and range, net),
    the 95% Welch interval on the anchored seed mean minus the method seed mean, the runs over the criterion at
    matched removal out of eight, and what the rules of H1 and H2 would read; no verdict. The optimized direction is
    always pooled, so its row is the same as in H2. A greyed row is a fitter rejected under H4.""",
    grey=h4_grey(POOLED_ROWS),
)  # fmt: skip

# %%

POOLED_UNMATCHED = {m: int((~np.isfinite(M_POOLED[m]["spill"])).sum()) for m in METHODS}
POOLED_OVER = {m: over_count(M_POOLED[m]["spill"]) for m in METHODS}
assert POOLED_OVER["leace"] == POOLED_OVER["inlp"] == 2
assert A_OVER == 0 and POOLED_OVER["dim"] == 6 and POOLED_RULES["dim"][0] == "rejected"
assert all(POOLED_OVER[m] in (2, 3) for m in ("ridge", "leace", "inlp", "optimized")), POOLED_OVER
assert all(POOLED_RULES[m][1] == "unresolved" for m in FITTERS)
_P2 = fisher_greater(2, A_OVER)
assert not any(POOLED_UNMATCHED[m] for m in FITTERS)

f"""
Without role labels the fitters lose most of their edge in removal. LEACE removes {num(M_POOLED["leace"]["removal"].mean())} against the anchored {num(A_REMOVAL.mean())}, with an interval on the gap of {interval(POOLED_CI["leace"])}. INLP ({num(M_POOLED["inlp"]["removal"].mean())}) and the probe ({num(M_POOLED["ridge"]["removal"].mean())}) come close on average with a wider spread, and difference in means removes {num(M_POOLED["dim"]["removal"].mean())}, with one run at {num(M_POOLED["dim"]["removal"].min())}. Read by the rule of H2, every fitter's interval reaches further than 0.05 from zero on one side, so none is a match and none falls short.

At matched removal the order turns around. The anchored edit spills past the criterion on no run; LEACE and INLP do on {NUMBER_WORD[POOLED_OVER["leace"]]} runs each, the probe on {NUMBER_WORD[POOLED_OVER["ridge"]]}, and difference in means on {NUMBER_WORD[POOLED_OVER["dim"]]}, which the rule of H1 would reject. Every fitter reaches the anchored removal on every run, so these are runs that spill, not runs that fall short. The Fisher test cannot tell two runs in eight from none (p {num(_P2)}), so for LEACE and INLP this is a lean, not a separation.

So in the setting the anchor is built for, the best post-hoc eraser removes about as much as the anchored edit and, even with a grid tuned per run, spills past the criterion on a few more runs. Difference in means, the direction most steering work uses, removes less and spills more.
"""

# %%

r"""
### The trade-off with the pooled fits

As in E1, one frontier per run, now over the pooled fits alone, with the point the selection rule read on it. The optimized direction panel is the same as in E1.
"""

# %%

RUN_FRONTS_POOLED = {m: run_fronts(m, "pooled") for m in METHODS}
e1_draw(RUN_FRONTS_POOLED, ANCHOR_RUNS, ANCHOR_CURVE, {m: (v["removal"], v["spill_at"]) for m in METHODS if (v := measures("control", m, "pooled", rule="axis"))},
        A_REMOVAL, A_SPILL, dict.fromkeys(METHODS, False), {m: E1_REJECTED[m] for m in METHODS},
        alt_text="Six panels of removal against spill, one line per run, with the pooled fits alone. The fitter lines "
        "rise more slowly than in E1 and level off lower; most markers sit near the anchor line at the spill "
        "criterion, with one difference in means marker near 0.2 and a few probe markers lower.",
        name="post-hoc-compare-pooled-tradeoff", sans=SANS_POINT, caption=f"""
            **Removal against spill with the pooled fits alone, one line per run**, on the test set of the eight
            controls, drawn as in E1: a line is the frontier of one run over the configurations fitted at every
            position, pooled, and its marker the point the selection rule read on it at the spill criterion, net of
            the axis median as in E1. The anchor panel and the plus for the sans-diff model are as in E1, and a hatched
            panel is a fitter rejected under H4. The spill axis is cut at {SPILL_LIM[1]}.
        """)  # fmt: skip

# %%

SHARED_POOLED = {m: shared_point(m, "pooled") for m in METHODS}
SP_712 = {m: float(SHARED_POOLED[m][0][SEEDS.index(712)]) for m in FITTERS}
SP_REST = {m: float(np.delete(SHARED_POOLED[m][0], SEEDS.index(712)).mean()) for m in FITTERS}
SP_OVER = {m: over_count(SHARED_POOLED[m][1]) for m in FITTERS}
assert all(v < 0.5 for m, v in SP_712.items() if m != "dim") and SP_712["dim"] > 0.7, SP_712
assert all(len(method_finish(m, 712, True)["main_config"]["slices"]) == 1 and method_finish(m, 712, True)["main_config"]["slices"][0] == 1 for m in ("ridge", "leace", "inlp"))  # fmt: skip

f"""
The frontiers rise more slowly than at the example answers and level off lower, so each run settles for less removal at the criterion. A configuration chosen on the other seven runs, as in E1, removes {num(min(SP_REST.values()))} to {num(max(SP_REST.values()))} on seven runs and lands past the criterion on {NUMBER_WORD[min(SP_OVER.values())]} to {NUMBER_WORD[max(SP_OVER.values())]} of the eight. Seed 712 again takes a slice-1 edit for LEACE, INLP, and the probe, which keep {num(min(SP_712[m] for m in ("ridge", "leace", "inlp")))} to {num(max(SP_712[m] for m in ("ridge", "leace", "inlp")))} there with a choice made on the other runs; difference in means keeps {num(SP_712["dim"])}, its choice on the others being a slice-2 edit that still works there.
"""

# %%

r"""
### Grading and where the answers go with the pooled fits

As in E2, on the held-out `difference` contexts, raw, at the configuration each fitter selected over the pooled fits alone.
"""

# %%

SHARE_POOLED = {m: share_by_bin(m, True) for m in E2_METHODS}
grading_draw(
    SHARE_POOLED | {"sans": SANS_SHARE},
    alt_text="Share removed by posterior bin with the pooled fits alone, on an axis from 0 to 1. The anchor and LEACE "
    "sit near the top, INLP and the probe a little lower, and difference in means lowest, near 0.6 to 0.8.",
    name="post-hoc-compare-pooled-grading",
    caption="""
        **The share of the way to the target null removed, by how sure the context is of `difference`, with the
        pooled fits alone**, on the held-out `difference` contexts, raw, seed mean over eight runs, drawn as in E2.
    """,
)

# %%

LANDING_POOLED = {m: landing(m, True) for m in E2_METHODS}
SHARE_ALL_POOLED = {m: share_all(m, True) for m in E2_METHODS}
table_html(
    ["", "share removed ↑", "landing ↑"],
    [[SHOWN[m], span(SHARE_ALL_POOLED[m]), span(LANDING_POOLED[m])] for m in E2_METHODS],
    """**How much and how far with the pooled fits alone**, on the held-out `difference` contexts, seed mean and
    range, raw, as in E2. A greyed row is a fitter rejected under H4.""",
    grey=h4_grey(E2_METHODS),
)

# %%

WHERE_POOLED = {k: WHERE[k] for k in ("clean (control)", "target null", "sans-diff", "clean (anchored)")}
WHERE_POOLED |= {SHOWN[m]: np.mean([where_mass(edited(m, s, True)[3]) for s in SEEDS], axis=0) for m in E2_METHODS}
where_draw(
    WHERE_POOLED,
    alt_text="A matrix of where the answers go on difference contexts with the pooled fits alone, laid out as in E2. "
    "Every edit moves mass to the answers of the other ops in about the proportions of the target null; difference "
    "in means leaves about a fifth on the difference answers.",
    name="post-hoc-compare-pooled-where",
    caption=f"""
        **Where the answers go on `{DIFF}` contexts with the pooled fits alone**, seed mean over eight runs, drawn as
        in E2; the four reference rows above the first rule are the same.
    """,
)

# %%

_STAY_P = {m: WHERE_POOLED[SHOWN[m]][D] for m in E2_METHODS}
assert SHARE_ALL_POOLED["anchor"].mean() > max(SHARE_ALL_POOLED[m].mean() for m in FITTERS)
assert LANDING_POOLED["leace"].mean() > LANDING_POOLED["anchor"].mean() > LANDING_POOLED["dim"].mean()
assert _STAY_P["dim"] > 0.15 and _STAY_P["anchor"] < _STAY_P["leace"] < 0.08

f"""
With the pooled fits the edits still grade alike, and the anchored edit removes the largest share ({num(SHARE_ALL_POOLED["anchor"].mean())}), with LEACE next at {num(SHARE_ALL_POOLED["leace"].mean())} and difference in means lowest at {num(SHARE_ALL_POOLED["dim"].mean())}. LEACE lands a little further toward the target null than the anchor ({num(LANDING_POOLED["leace"].mean())} against {num(LANDING_POOLED["anchor"].mean())}), and difference in means less far ({num(LANDING_POOLED["dim"].mean())}). The answers go where they went in E2, except that difference in means leaves {num(_STAY_P["dim"])} on the answers of `{DIFF}`, against {num(_STAY_P["anchor"])} for the anchor and {num(_STAY_P["leace"])} for LEACE.
"""

# %%

r"""
### Labels, write, and probe with the pooled fits
"""

# %%

LADDER_POOLED = {m: ladder(m, True) for m in FITTERS}
ladder_draw(
    LADDER_POOLED,
    float(A_REMOVAL.mean()),
    FIXED_AT,
    alt_text="Removal against the share of difference contexts labelled with the pooled fits alone. Every fitter is "
    "flat from all labels down to one in fifty: LEACE near 0.91, INLP near 0.87, the probe near 0.84, and difference "
    "in means near 0.70. The anchored runs with fixed labels sit near 0.81 to 0.88.",
    name="post-hoc-compare-pooled-ladder",
    caption="""
        **Removal against the share of `difference` contexts labelled, with the pooled fits alone**, on the test set,
        net of the baselines, seed mean (marker) and range (bar) over the eight controls, drawn as in E3. The dotted
        line is the anchored removal with labels redrawn, and the diamonds the anchored runs of fixed-labels.
    """,
)

# %%

_FLAT_P = {m: float(np.ptp(LADDER_POOLED[m][0].mean(0))) for m in FITTERS}
assert max(_FLAT_P.values()) < 0.02, _FLAT_P
WRITE_POOLED = {m: write_over(m, True) for m in FITTERS}
assert all(WRITE_POOLED[m] == (0, []) for m in FITTERS), WRITE_POOLED
AUC_POOLED = {k: AUC[k] for k in ("clean control", "clean anchored", "sans-diff")}
AUC_POOLED |= {SHOWN[m]: probe_auc("control", m, True) for m in METHODS} | {SHOWN["anchor"]: AUC[SHOWN["anchor"]]}
probe_draw(
    AUC_POOLED,
    alt_text="Probe AUC by slice with the pooled fits alone, laid out as in E5. After every edit the probe falls to "
    "about 0.64 to 0.74 at the query equals sign at slice 3 and 0.70 to 0.79 at the answers; none reaches chance.",
    name="post-hoc-compare-pooled-probe",
    caption=f"""
        **What a fresh probe finds after each edit, with the pooled fits alone**, the area under the ROC curve of a
        ridge probe for `{DIFF}` on the edited states, by slice, seed mean over eight runs, drawn as in E5.
    """,
)

# %%

_Q3_P = {n: v[0, 3] for n, v in AUC_POOLED.items()}
_A2_P = {n: v[1, 2] for n, v in AUC_POOLED.items()}
_EDITS_P = [SHOWN[m] for m in E2_METHODS]
assert min(_Q3_P[n] for n in _EDITS_P) > 0.6 and min(_A2_P[n] for n in _EDITS_P) > 0.6

f"""
The label ladder stays flat for every pooled fit: from every label down to a fiftieth, the seed-mean removal moves by less than 0.02. At a fiftieth, LEACE, INLP, and the probe remove {num(min(LADDER_POOLED[m][0][:, -1].mean() for m in ("ridge", "leace", "inlp")))} to {num(max(LADDER_POOLED[m][0][:, -1].mean() for m in ("ridge", "leace", "inlp")))}, above the {num(FIXED_AT[0.02].mean())} of the anchor with fixed labels at the same rate, and difference in means {num(LADDER_POOLED["dim"][0][:, -1].mean())}. So the number of labels is still not where the anchor has an edge.

Every pooled selection edits one slice, so its write equals its bound and E4 has nothing to measure. Slice 2 on most runs, and slice 1 on seed 712 for LEACE, INLP, and the probe.

A fresh probe still finds the op after every edit, as in E5: {num(min(_Q3_P[n] for n in _EDITS_P))} to {num(max(_Q3_P[n] for n in _EDITS_P))} at the query `=` of slice 3, and {num(min(_A2_P[n] for n in _EDITS_P))} to {num(max(_A2_P[n] for n in _EDITS_P))} at the example answers of slice 2. LEACE leaves the least at the answers, and the anchored edit the least at the query `=`.
"""

# %%

r"""
## Discussion

On this model, a direction found after training does about as well as the anchored edit. With role labels and a grid tuned on each run, the standard fitters remove more of `difference` within the spill criterion, mostly by editing the example answers alone. Restricted to fits that need no role labels, which is the comparison the anchor was built for, LEACE and INLP remove about as much as the anchor and spill past the criterion on a few more runs, and difference in means falls behind on both. So the anchored edit matches the best post-hoc eraser we tried, and does better than the direction most steering work uses. We have not shown that it does better than LEACE.

The fitters had help the anchor did not. They saw every label at once, and each run chose among about a thousand configurations scored on that run. The anchor had one configuration, but it came from some thirty experiments at other seeds, so its search happened once, earlier. The configuration chosen on other runs in E1 is the closest this report comes to comparing the two kinds of search: for the fitters it works on most runs, and on one run it misses because that run holds the op a slice earlier. One run in eight is too few to say how often a fixed recipe for a fitter would miss, though it fits the idea that placing the concept saves finding where it sits in each new model.

Placing the concept had costs here too: an extra training term, and a small drop in task score. What it offers that a post-hoc method lacks is a location chosen before training, and this experiment uses that only in the plainest way, one concept on one axis. Whether a chosen location helps more when several concepts are placed at once, when they are arranged in a designed subspace, or when a concept is harder to find after training, is open, and nothing here bears on it either way.

None of the edits, LEACE included, left a fresh probe at chance. For the anchor that was expected, since part of the evidence lies off e₁. For LEACE it was not, and we don't know the cause. The sans-diff model, which never learned the op, leaves a probe about as much, so what remains may be what any model of these contexts holds. Either way, on this model taking the op out of the answer and making it unreadable are different properties, and a claim of erasure needs to say which one it means.

The comparison covers one concept in one small model on a synthetic task, with eight runs per arm, and several readings depend on the choice of baseline (E6). The pooled comparison and the configuration chosen on other runs were added after the run, so they describe what we saw rather than test a prediction.
"""

# %%

r"""
## Method

**The fits.** Difference in means takes the binary target only. The ridge penalty is fixed at 0.01, the value ex-2.2.1 used; it is added to the unnormalized Gram matrix of the fit states, so with this many states it barely regularizes. That is fine here: with 280,000 states in 64 dimensions a least-squares probe has little room to overfit, and the penalty is there to keep the solve stable where the covariance is nearly flat. A strong penalty would turn the probe toward the difference in means, which the grid has already. LEACE takes a small ridge on the covariance, since the stream is unit-norm and its covariance is close to rank-deficient. INLP fits the ridge probe, projects its direction out of the fit states, and repeats. When several slices are edited, each gets its own fit. The anchor uses e₁ at every slice, but a fitted direction of the control can differ from slice to slice. With sequential refit, the fit at slice ℓ is made on the fit set as it arrives under the edits at the slices before ℓ, which keeps an oblique eraser on the states it will see (ex-2.2.1 found it extrapolates poorly off them). Mean ablation replaces the coefficient with its mean over the fit contexts of the other ops.

**The optimized direction.** A unit vector per edited slice, initialized at the difference in means and fitted by Adam for a few hundred steps on the fit set. The loss is the KL divergence[^kl] from the target null to the edited answers on `difference` contexts, plus a weight times the KL divergence from the clean answers to the edited ones on the other contexts, at full dose. The weight takes four values, and the dose and slice sets are crossed as for the others. Its configurations join the frontier like any other.

[^kl]: KL divergence: how far one probability distribution is from another, zero when they match.

**Grid size.** About 500 configurations for difference in means, 1,000 for the probe and for LEACE, 1,500 for INLP, and 450 for the optimized direction, counting doses; each is one scoring pass, reading the logits at the query `=` (which predicts the answer) only. The whole grid is scored on the tuning set, and again on the test set for the figure of E1, where nothing is chosen. A fit is made once per (target, site, slice, refit) and reused across doses and positions.

**The baseline.** On the tuning set, the median over fifteen coordinate axes other than e₁, drawn once with a fixed seed before the run and shared by every run. On the test set, fifteen permuted fits per configuration, at the bracketing configurations of each point. LEACE is homogeneous in its covariance vector, so a permuted fit is a full-strength oblique erasure along a reference direction. The coordinate-axis median applies the same operator form (slices, positions, dose, replacement) along each of the 63 axes other than e₁; it does not depend on the fitter, so it is computed once per form.

**Comparisons.** Each method against the anchor as two samples of eight runs: the difference of seed means with a 95% Welch t interval[^welch], and Fisher's exact test[^fisher] for counts of runs over the criterion.

**Smoke test.** Before the full run, the whole pipeline runs end to end on two stored runs outside the comparison: the control of ex-2.2.23 at seed 704 and the `falloff` run at the same seed. Every stage runs: the fits for every method and factor, both baselines, the selection rule, the test scoring, the label ladder, the optimized direction, the measures of E2 to E6, and the tables, figures, and print of this report, with the one pair standing in for the eight (an interval over one run a side means nothing, but the code that computes it runs). The grid is thinned so that each level of each factor is visited at least once, and each stage is timed per pass, which gives the cost of the full run. So a fault anywhere shows up before the costly part. The smoke test also gives two early readings: H4 on the anchored run, so that a fitter that can't find a placed concept shows up before the full run, and where a probe for the op scores best on the control.

**The query operands join the sites.** Before the freeze, whether the control holds the evidence for the op at the example answers, as the anchored model does, was not known: the block1-edit experiment found that the control gathers the operand colors at each example answer in block 1, but it fitted no probe for the op on a control. The smoke test fitted one at every role and slice of the control at seed 704. It scored best at the two operands of the query (an area under the ROC curve of 0.95 at slice 3), above the query `=` (0.92) and the example answers (0.83). The query answer scored higher still (0.99), but it comes after the `=` that predicts the scored answer, so an edit there cannot change it, and it was left out. So the query operands joined the sites fitted, as the plan said they would.

**Details fixed with the code.** A sequential refit makes the fit at each slice under the full-dose edits at the slices before it, at every position, with zero replacement, and is reused across the doses, positions, and replacements scored with it; mean ablation under a refit uses the mean state of the other ops as it arrives under those edits. A permuted fit takes no sequential refit: it reuses the clean permuted fit at each slice, since a direction with no concept in it barely changes the states the next fit sees. The optimized direction acts at every position; its keep weight takes the values 0.3, 1, 3, and 10, and each search runs 300 Adam steps of 512 contexts at a learning rate of 0.01. With the graded target, the term toward the target null is weighted by the posterior on `difference` and the keep term by the rest. Its mean ablation is scored along the searched direction without searching again, and its permuted baseline uses five searches rather than fifteen, since each is a gradient search. A permuted search is no blank: each context keeps its own target null under the permutation, so the search still pulls the `difference` contexts among its labels toward the null, and the favoring rule takes the axis median for its removal. E2, E4, and E5 read, for each method, the bracketing configuration of removal at the spill criterion that lies within the criterion. The label ladder of E3 leaves out the optimized direction, which reads the target null rather than labels; on the graded target an unlabelled `difference` context has a target of zero. The probe of E5 reads the query `=` and the example answers at each slice, on 7,000 test contexts split in half, and reports the area under the ROC curve. The write of E4 is measured on 2,000 test contexts of the other ops, with a role per position within an equation. The selection of H4 over the pooled fits and the example answers, and the pooled-only selection of H2, were run as a second pass after the main one, on the same tuning scores; it scored the test-set baselines of the configurations it chose that the main pass had not.

**Left out.** The same comparison for *red* on the D2.1 checkpoints, where ex-2.2.1 gave the fitters far less, is not run here: this report is about the inferred op.

**Cost.** No training. Each run takes about 6,000 scoring passes: the grid on the tuning and test sets, the axis baseline on the tuning set, the permuted fits, and the ladder. The anchored runs take the same for H4, so sixteen runs come to about 100,000 passes, plus the state passes over the fit set and for sequential refit, and the optimized direction. The plan estimated eight to ten L4-hours, about \$15, with \$25 as the ceiling. The run took L40S cards: the full run cost \$15.80 and the second pass for the restricted selections \$0.64, and with the smoke tests and the speed-ups between them the experiment came to \$22.14. The passes added after the paper rounds on the results, E2 to E5 at the pooled selections and the probe on the sans-diff models, brought the total to \$22.72.
"""

# %%

r"""
## References

[ex-2.2.1](/docs/m2/ex-2.2.1/report.py)
:   On the D2.1 checkpoints, every fitted direction that removed at least half of *red* from the control cost the lines with no red 0.37 or more, an order of magnitude above the anchored edit.

[fixed-effects](/docs/m2/fixed-effects/report.py)
:   The full scoring suite on the recipe of record at seeds 712 to 719: removal 0.88 at the full dose, spill below the criterion at every dose, graded drops, a landing of 0.50, and a write within the bound.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at fresh seeds 712 to 719 keeps every anchor; it is the recipe of record, and its checkpoints and dose scoring are stored.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   Trained the controls at seeds 712 to 719 that this comparison fits on, sharing initialization, batches, and corpus with the anchored runs.

[block1-edit](/docs/m2/block1-edit/report.py)
:   Block 1 writes the evidence for the op onto e₁ at the example answers, and a later block reads it at the query `=`.

[mid-block-edit](/docs/m2/mid-block-edit/report.py)
:   Editing the mid-block states as well gains nothing and spills; in passing it shows the query `=` near zero alignment throughout.

[reference-baseline](/docs/m2/reference-baseline/report.py)
:   The median drop along the other coordinate axes of the run replaces the paired control as the baseline, and the published numbers barely move.

[sans-diff model](/docs/m2/filtered-corpus/report.py)
:   A model trained with no loss on `difference` comes 0.58 of the way from the control to the target null, against 0.50 for the edited recipe of record, and misses in the same kind of way.

[fixed-labels](/docs/m2/fixed-labels/report.py)
:   Holding one label draw fixed for the run, as a real labelled set would be, keeps the anchor at every rate tried, but a few runs end with the edit removing much less of the op.

[low-runs](/docs/m2/low-runs/report.py)
:   In every run the evidence that an example fits `difference` lies partly on e₁ and partly off it, and the edit along e₁ never touches the off-axis part.

[ex-2.2.21](/docs/m2/ex-2.2.21/report.py)
:   The 14,000 held-out contexts, 2,000 per op, that every D2.2 score is taken on.
"""
