# ruff: noqa: B018
# title: Soft pool scout

# An exploratory report in two stages: sixteen schedule conditions of `pool 1–3, soft τ` at its eight soft-pool seeds,
# screened at epoch 64 for lost anchors and early spill (stage 1), then the leaders carried on to epoch 400 (stage 2).
# `experiment.py` beside this script trains and scores the runs.
import json
import re
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


def by_label(body: dict) -> dict[str, dict]:
    return {r["label"]: r for r in body["runs"]}


SCREEN_BODY = fetch_json(ex.SCREEN_REF)
SCREEN = by_label(SCREEN_BODY)
SCREEN_TRAJ = fetch_json(ex.SCREEN_TRAJ_REF)
# The base: spill-timing's reruns of soft-pool's `pool 1–3, soft τ`, scored at every record.
BASE_TRAJ = fetch_json(ex.spt.TRAJ_REF)
# Spill-timing's scoring of the loss-rate screen's epoch-64 checkpoints of `falloff` and `step, soft τ`.
QUIET = by_label(fetch_json(ex.spt.STORED_REF))
DESIGN = SCREEN_BODY["design"]

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
STOP = DESIGN["stop_epoch"]
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
UNIT = 6  # tokens per equation: a ? b = y and a separator
ANSWERS = [UNIT * e + 4 for e in range(ex.K)]
QUERY_ANSWER = UNIT * ex.K + 4
SLICE_TICK = ["emb", "1", "2", "3", "4"]
LOST_MAX, KEPT_MIN = ex.lr.LOST_MAX, ex.lr.KEPT_MIN
CONDITIONS = [ex.Condition(**{k: v for k, v in c.items() if k != "anchor"}) for c in DESIGN["conditions"]]
for c in CONDITIONS:  # JSON turns the warm-up tuple into a list
    object.__setattr__(c, "warmup", tuple(c.warmup))
NAMES = [c.name for c in CONDITIONS]
SHOWN = {c.name: c.shown_as for c in CONDITIONS}
BY_NAME = {c.name: c for c in CONDITIONS}


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- Each run at the stop -------------------------------------------------------------------------------------


def reference_spill(clean, edited, ref) -> float:
    """The largest drop on another op under the full edit, less the median drop under the reference edits
    (spill-timing): the median reference score less the edit score, at its largest over the other ops.
    """
    del clean  # the clean score cancels
    net = np.median(np.asarray(ref), axis=0) - np.asarray(edited)
    return float(net[OTHER].max())


def worst_op(edited, ref) -> str:
    net = np.median(np.asarray(ref), axis=0) - np.asarray(edited)
    return OPS[OTHER[int(net[OTHER].argmax())]]


def at_stop(traj: dict) -> int:
    """The index of the first record at the stop epoch (a stopped run records it twice)."""
    return int(np.flatnonzero(np.isclose(np.asarray(traj["epoch"]), STOP))[0])


def run_at_stop(name: str, seed: int) -> dict[str, Any]:
    """Everything stage 1 measures on one run, from the screen for a new condition and from spill-timing's rerun for
    the base, so the two are scored the same way.
    """
    if name == ex.STORED:
        traj = BASE_TRAJ[ex.spt.label_of(seed)]["traj"]
        i = at_stop(traj)
        scores = {k: traj[k][i] for k in ("es_clean", "es_edit", "es_sham")}
    else:
        label = ex.label_of(name, seed)
        traj, scores = SCREEN_TRAJ[label]["traj"], SCREEN[label]
        i = at_stop(traj)
    alpha = np.asarray(traj["alpha_anchored"])[i]  # (slices, positions), on the `difference` probes
    return {
        "window": ex.lr.window_mean(traj),
        "spill": reference_spill(scores["es_clean"], scores["es_edit"], scores["es_sham"]),
        "op": worst_op(scores["es_edit"], scores["es_sham"]),
        "alpha_ans": alpha[:, ANSWERS].mean(axis=1),
        "alpha_q": alpha[:, QUERY_ANSWER],
    }


RUNS = {(n, s): run_at_stop(n, s) for n in NAMES for s in SEEDS}


def per_seed(name: str, key: str) -> np.ndarray:
    return np.array([RUNS[(name, s)][key] for s in SEEDS])


WINDOW = {n: per_seed(n, "window") for n in NAMES}
SPILL = {n: per_seed(n, "spill") for n in NAMES}
CALLS = {n: [ex.lr.call_of(v) for v in WINDOW[n]] for n in NAMES}
N_LOST = {n: sum(c != "kept" for c in CALLS[n]) for n in NAMES}
PEAK = {n: SLICE_TICK[int(per_seed(n, "alpha_ans").mean(axis=0).argmax())] for n in NAMES}

# The quiet recipes at the stop: the loss-rate screen's checkpoints, pooled over its seeds.
QUIET_SPILL = {
    arm: np.array(
        [
            reference_spill(r["es_clean"], r["es_edit"], r["es_sham"])
            for lab, r in QUIET.items()
            if lab.startswith(f"{arm}-s") and lab.endswith(f"-e{STOP}")
        ]
    )
    for arm in ("falloff", "step-soft")
}
QUIET_SHOWN = {"falloff": "falloff", "step-soft": "step, soft τ"}
# α at the stop on the same checkpoints, from the loss-rate screen's trajectories.
QUIET_TRAJ = fetch_json(ex.lr.TRAJ_REF)


class QuietAlpha(TypedDict):
    ans: np.ndarray
    q: np.ndarray
    n: int


def quiet_alpha(arm: str) -> QuietAlpha:
    """The seed-mean α by slice at the stop, at the example answers and at the query answer, over the screen runs."""
    maps = []
    for lab, r in QUIET_TRAJ.items():
        if lab.startswith(f"{arm}-s") and lab.endswith(f"-e{STOP}"):
            maps.append(np.asarray(r["traj"]["alpha_anchored"])[at_stop(r["traj"])])
    m = np.mean(maps, axis=0)
    return {"ans": m[:, ANSWERS].mean(axis=1), "q": m[:, QUERY_ANSWER], "n": len(maps)}


QUIET_ALPHA = {arm: quiet_alpha(arm) for arm in QUIET_SHOWN}


def weight_at_stop(name: str) -> float:
    """λ_a at the stop as a share of its peak."""
    from sca.anchoring import AnchorSpec

    spec = BY_NAME[name].anchor(ex.st.schedules()[0])
    return float(AnchorSpec(**spec)(STOP) / spec["peak"])


STRETCH_AT_STOP = weight_at_stop("stretch")

# Ranking: no lost or uncertain call, then the least mean spill at the stop.
RANKED = sorted(NAMES, key=lambda n: (N_LOST[n] > 0, float(SPILL[n].mean())))
N_RUNS = len(RUNS)
N_KEPT = sum(c == "kept" for n in NAMES for c in CALLS[n])
LOWEST_CALL = min(float(WINDOW[n].min()) for n in NAMES)

# The crossing, pooled over its four timing levels at each slice temperature.
CROSSING = [c.name for c in ex.CROSSING]
LEVELS = sorted({c.slice_tau for c in ex.CROSSING})
LEVEL_RUNS = {lt: [n for n in CROSSING if BY_NAME[n].slice_tau == lt] for lt in LEVELS}
LEVEL_SPILL = {lt: np.concatenate([SPILL[n] for n in ns]) for lt, ns in LEVEL_RUNS.items()}
LEVEL_OVER = {lt: int((v > CRITERION).sum()) for lt, v in LEVEL_SPILL.items()}
LEVEL_ALPHA1 = {lt: float(np.concatenate([per_seed(n, "alpha_ans")[:, 1] for n in ns]).mean()) for lt, ns in LEVEL_RUNS.items()}  # fmt: skip
OVER_OPS: dict[str, int] = {}
for r in RUNS.values():
    if r["spill"] > CRITERION:
        OVER_OPS[r["op"]] = OVER_OPS.get(r["op"], 0) + 1
N_OVER = sum(OVER_OPS.values())
TOP_OP = max(OVER_OPS, key=OVER_OPS.__getitem__)

# --- Stage 2: the finished runs ---------------------------------------------------------------------------------
# The two leaders at epoch 400, beside the three recipes recipe-compare scored at the same seeds, each measured the
# same way: the dose edit along every coordinate axis (reference-baseline), the edits restricted to slice sets
# (split-anti-seeds), the dose arrays and the write arrays (edit-effects).

rc = ex._load_sibling("recipe-compare", "soft_pool_scout_rc")
LEADERS: list[str] = list(ex.LEADERS)
FALLOFF, STEP_SOFT = "falloff", "step-soft"
# The conditions of stage 2, by their names here: the two quiet recipes, the base, then the leaders.
FINISHED = [FALLOFF, STEP_SOFT, ex.STORED, *LEADERS]
# The labels of the stored recipes in the runs of split-anti-seeds and soft-pool.
STORED_AS = {FALLOFF: "falloff", STEP_SOFT: "step-soft", ex.STORED: "pool-123-soft"}
CONTROL = "control"
FIN_SHOWN = {FALLOFF: "falloff", STEP_SOFT: "step, soft τ"} | {n: SHOWN[n] for n in (ex.STORED, *LEADERS)}
GAMMAS: list[float] = list(ex.rb.DOSE_GAMMAS)
FULL = len(GAMMAS) - 1
E1 = 0  # the anchored axis, in code
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
Q = 0.99  # the quantile the bound and the write are taken at, as in edit-effects
TOLERANCE = 1.25  # ex-2.2.1 allowed the write a quarter over the bound


def fin_label(name: str, seed: int) -> str:
    """The label of a finished run in the store of the experiment that trained it."""
    return f"{STORED_AS.get(name, name)}-s{seed}"


def fetch_arrays(refs: dict[str, str], keep: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """The `.npz` behind each ref, by label, with only the *keep* arrays."""
    store = project_store()
    got = store.get_refs(list(refs.values()))
    missing = [r for r, a in got.items() if a is None]
    assert not missing, f"not published: {missing}"
    arts = {r: a for r, a in got.items() if a is not None}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(arts[r], Path(tmp) / f"{lbl}.npz") for lbl, r in refs.items()])
        for lbl, p in zip(refs, paths, strict=True):
            with np.load(p) as z:
                out[lbl] = {k: z[k] for k in keep}
    return out


_RB = {r["label"]: r for r in fetch_json(ex.rb.AXES_REF)["runs"]}
AX = {(n, s): _RB[fin_label(n, s) if n == FALLOFF else ex.rb.SOFT_POOL + fin_label(n, s)] for n in STORED_AS for s in SEEDS}  # fmt: skip
AX |= {(CONTROL, s): _RB[f"{CONTROL}-s{s}"] for s in SEEDS}
AX |= {(r["condition"], r["model_seed"]): r for r in fetch_json(ex.AXES_REF)["runs"]}

LEAN = {
    r["label"]: r
    for ref in (rc.sas.RESULTS_REF, rc.sp.RESULTS_REF, ex.RESULTS_REF)
    for r in fetch_json(ref)["runs"]
    if r["model_seed"] in SEEDS
}
LEAN_CONTROL = {
    s: r
    for r in fetch_json(rc.sas.lean.RESULTS_REF)["runs"]
    for s in SEEDS
    if r["label"] == rc.sas.ex2223.label_of(CONTROL, rc.sas.EPOCHS, s)
}
assert set(LEAN_CONTROL) == set(SEEDS)
DOSE_SUMMARY = {r["label"]: r for ref in (rc.sas.DOSE_REF, rc.sp.DOSE_REF, ex.DOSE_REF) for r in fetch_json(ref)["runs"]}  # fmt: skip


def dose_ref(name: str, seed: int) -> str:
    lbl = fin_label(name, seed)
    if name in LEADERS:
        return ex.DOSE_ARRAYS_REF.format(label=lbl)
    return (ex.sp.DOSE_ARRAYS_REF if name in (STEP_SOFT, ex.STORED) else rc.sas.DOSE_ARRAYS_REF).format(label=lbl)


def write_ref(name: str, seed: int) -> str:
    lbl = fin_label(name, seed)
    if name in LEADERS:
        return ex.WRITE_ARRAYS_REF.format(label=lbl)
    return (rc.WRITE_ARRAYS_REF if name in (STEP_SOFT, ex.STORED) else rc.ee.ARRAYS_REF).format(label=lbl)


DOSE = fetch_arrays(
    {fin_label(n, s): dose_ref(n, s) for n in (*FINISHED, CONTROL) for s in SEEDS},
    ("op_ids", "posterior", "clean_eem", "null_eem", "eem"),
)
ALPHA = fetch_arrays({fin_label(n, s): write_ref(n, s) for n in FINISHED for s in SEEDS}, ("clean", "arriving", "op_ids"))  # fmt: skip
FITS_D = fetch_arrays({"fits": rc.ee.FITS_REF}, ("fits",))["fits"]["fits"][:, :, D]
OP_IDS = DOSE[fin_label(CONTROL, SEEDS[0])]["op_ids"].astype(int)
IS_D = OP_IDS == D
for _lbl in DOSE:
    assert (DOSE[_lbl]["op_ids"] == OP_IDS).all()
for _lbl in ALPHA:
    assert (ALPHA[_lbl]["op_ids"] == OP_IDS).all()


def axis_scores(r: dict) -> dict[str, Any]:
    """Removal at the full dose and the worst spill over doses for the e₁ edit on one run, net of the median drop
    over every other coordinate axis of the same run (the D2.2 rule), and removal with nothing subtracted.
    """
    drop = np.asarray(r["drop"])  # (axes, doses, ops)
    net = drop[E1] - np.median(drop[1:], axis=0)
    span = r["clean"][D] - r["null"][D]
    spill = net[:, OTHER].max(axis=1)
    worst = int(spill.argmax())
    return {
        "removal": float(net[FULL, D] / span),
        "raw": float(drop[E1][FULL, D] / span),
        "spill": float(spill.max()),
        "op": OPS[OTHER[int(net[worst, OTHER].argmax())]],
    }


def control_median_spill(name: str, s: int) -> float:
    """The worst spill with the median taken over the other axes of the control in place of those of the run."""
    net = np.asarray(AX[(name, s)]["drop"])[E1] - np.median(np.asarray(AX[(CONTROL, s)]["drop"])[1:], axis=0)
    return float(net[:, OTHER].max())


FIN = {(n, s): axis_scores(AX[(n, s)]) for n in FINISHED for s in SEEDS}


def fin_seed(name: str, key: str) -> np.ndarray:
    return np.array([FIN[(name, s)][key] for s in SEEDS])


REMOVAL = {n: fin_seed(n, "removal") for n in FINISHED}
RAW = {n: fin_seed(n, "raw") for n in FINISHED}
FIN_SPILL = {n: fin_seed(n, "spill") for n in FINISHED}
FIN_OVER = {n: int((FIN_SPILL[n] > CRITERION).sum()) for n in FINISHED}
CTRL_OVER = {n: sum(control_median_spill(n, s) > CRITERION for s in SEEDS) for n in FINISHED}
TASK = {n: np.array([np.asarray(AX[(n, s)]["clean"]) - np.asarray(AX[(CONTROL, s)]["clean"]) for s in SEEDS]) for n in FINISHED}  # fmt: skip
N_FIN_LOST = sum(int((REMOVAL[n] < LOST_BELOW).sum()) for n in LEADERS)
FIN_OPS = {n: [FIN[(n, s)]["op"] for s in SEEDS] for n in FINISHED}


def paired(name: str, key: str, against: str) -> np.ndarray:
    """The difference from *against* at each seed, which shares its initialization, batches and label draws."""
    v = {"removal": REMOVAL, "spill": FIN_SPILL}[key]
    return v[name] - v[against]


# Spill along training, from the checkpoints each leader kept and the one it finished on (eight reference axes, as
# spill-timing scored it), with the base from spill-timing's reruns, scored at every record.
ALONG_EPOCHS = (*ex.KEEP_EPOCHS, ex.EPOCHS)
_ALONG = fetch_json(ex.ALONG_REF)["runs"]
ALONG = {
    n: np.array(
        [
            [
                next(
                    reference_spill(r["es_clean"], r["es_edit"], r["es_sham"])
                    for r in _ALONG
                    if r["condition"] == n and r["model_seed"] == s and r["epoch"] == e
                )
                for e in ALONG_EPOCHS
            ]
            for s in SEEDS
        ]
    )
    for n in LEADERS
}


def base_along(seed: int) -> list[float]:
    traj = BASE_TRAJ[ex.spt.label_of(seed)]["traj"]
    ep = np.asarray(traj["epoch"])
    out = []
    for e in ALONG_EPOCHS:
        i = int(np.flatnonzero(np.isclose(ep, e))[0])
        out.append(reference_spill(traj["es_clean"][i], traj["es_edit"][i], traj["es_sham"][i]))
    return out


ALONG[ex.STORED] = np.array([base_along(s) for s in SEEDS])  # (seeds, epochs)
AT_STOP_COL = ALONG_EPOCHS.index(STOP)
ALONG_NAMES = [ex.STORED, *LEADERS]
# The runs that are quiet at the stop and go over the criterion or near it (within a quarter) later on.
NEAR = 0.75 * CRITERION
LATE = {n: [s for i, s in enumerate(SEEDS) if ALONG[n][i, AT_STOP_COL] <= NEAR < ALONG[n][i, AT_STOP_COL + 1 :].max()] for n in ALONG_NAMES}  # fmt: skip
EARLY = {n: [s for i, s in enumerate(SEEDS) if ALONG[n][i, : AT_STOP_COL + 1].max() > CRITERION] for n in ALONG_NAMES}  # fmt: skip


# The edits restricted to slice sets, net of the same edit on the control at the seed (split-anti-seeds).
SLICE_SETS = ("embedding", "block 1", "blocks 2 to 4", "block 4", "blocks 1 to 4", "every slice")


def slice_net(name: str, s: int, edit: str) -> np.ndarray:
    return np.asarray(LEAN[fin_label(name, s)]["edits"][edit]) - np.asarray(LEAN_CONTROL[s]["edits"][edit])


def slice_removal(name: str, s: int, edit: str) -> float:
    r = LEAN[fin_label(name, s)]
    return float(slice_net(name, s, edit)[D] / (r["clean"][D] - r["null"][D]))


SLICE_REMOVAL = {n: {e: np.array([slice_removal(n, s, e) for s in SEEDS]) for e in SLICE_SETS} for n in FINISHED}
SLICE_SPILL = {n: {e: np.array([slice_net(n, s, e)[OTHER].max() for s in SEEDS]) for e in SLICE_SETS} for n in FINISHED}  # fmt: skip
LEAN_R = {n: np.array([LEAN[fin_label(n, s)]["r_light_emb"] for s in SEEDS]) for n in FINISHED}
LEAN_R[CONTROL] = np.array([LEAN_CONTROL[s]["r_light_emb"] for s in SEEDS])


@memo
def turn_maps(clean: np.ndarray, arriving: np.ndarray, op_ids: np.ndarray, fits_d: np.ndarray) -> dict:
    """The bound and the write of one run as alignments, `(slices, positions)`: the 99th percentile of |α| over the
    contexts of the other ops, clean and as it arrives under the full edit, with the answers that fit `difference`
    left out at the answers, as in edit-effects. Beside it, the same at the example answers that do not fit, by slice,
    and the anchor by slice at the example answers and the query answer of `difference` contexts.
    """
    other = op_ids != D
    c, a = np.abs(clean.astype(np.float32)), np.abs(arriving.astype(np.float32))
    bound, write = np.quantile(c[:, other], Q, axis=1), np.quantile(a[:, other], Q, axis=1)
    for e, p in enumerate(ANSWERS + [QUERY_ANSWER]):
        keep = other & ~fits_d[:, e]
        bound[:, p] = np.quantile(c[:, keep, p], Q, axis=1)
        write[:, p] = np.quantile(a[:, keep, p], Q, axis=1)
    cc = np.concatenate([c[:, other & ~fits_d[:, e], p] for e, p in enumerate(ANSWERS)], axis=1)
    return {
        "bound": bound,
        "write": write,
        "ans_bound": np.quantile(cc, Q, axis=1),
        "alpha_d": c[:, op_ids == D][:, :, ANSWERS].mean(axis=(1, 2)),
        "alpha_q": c[:, op_ids == D, QUERY_ANSWER].mean(axis=1),
    }


TURN = {(n, s): turn_maps(*(ALPHA[fin_label(n, s)][k] for k in ("clean", "arriving")), OP_IDS, FITS_D) for n in FINISHED for s in SEEDS}  # fmt: skip


def turn_mean(name: str, key: str) -> np.ndarray:
    return np.mean([TURN[(name, s)][key] for s in SEEDS], axis=0)


def angle(alpha) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(alpha, 0, 1)))


RATIO = {n: np.arcsin(turn_mean(n, "write")) / np.arcsin(turn_mean(n, "bound")) for n in FINISHED}
for _n in FINISHED:
    assert np.allclose(RATIO[_n][0], 1.0, atol=1e-3), "the embedding sees the clean stream"


def over_tolerance(name: str) -> list[tuple[int, int, int, float, float]]:
    """Per run, the sites past the embedding where the write goes over the tolerance: seed, slice, position, the
    ratio, and the bound there in degrees.
    """
    out = []
    for s in SEEDS:
        t = TURN[(name, s)]
        r = np.arcsin(t["write"]) / np.arcsin(t["bound"])
        for sl, p in zip(*np.nonzero(r[1:] > TOLERANCE), strict=True):
            out.append((s, int(sl) + 1, int(p), float(r[sl + 1, p]), float(angle(t["bound"][sl + 1, p]))))
    return out


OVER_TOL = {n: over_tolerance(n) for n in FINISHED}
FIN_ALPHA = {n: np.array([TURN[(n, s)]["alpha_d"] for s in SEEDS]) for n in FINISHED}
FIN_PEAK = {n: FIN_ALPHA[n][:, 1:].argmax(axis=1) + 1 for n in FINISHED}


def drop_of(lbl: str) -> np.ndarray:
    r = DOSE[lbl]
    return r["clean_eem"] - r["eem"][-1]


def share(name: str, s: int) -> float:
    """The share of the way to the target null the full edit takes on `difference` contexts, net of the control at
    the seed, as a ratio of means.
    """
    lbl = fin_label(name, s)
    drop = drop_of(lbl) - drop_of(fin_label(CONTROL, s))
    dep = DOSE[lbl]["clean_eem"] - DOSE[lbl]["null_eem"]
    return float(drop[IS_D].mean() / dep[IS_D].mean())


SHARE = {n: np.array([share(n, s) for s in SEEDS]) for n in FINISHED}
LANDING = {n: np.array([DOSE_SUMMARY[fin_label(n, s)]["landing"] for s in SEEDS]) for n in FINISHED}

# --- Encoding ---------------------------------------------------------------------------------------------------

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
QUIET_INK = light_dark("#777", "#aaa")
TAU_INK = {0.1: light_dark("#08306b", "#c6dbef"), 0.2: light_dark("#2171b5", "#6baed6"), 0.4: light_dark("#6baed6", "#2171b5")}  # fmt: skip
WARMUP_INK = {True: light_dark("#d95f02", "#fdae6b"), False: light_dark("#7570b3", "#bcbddc")}  # soft τ, fixed τ


def tex(shown: str) -> str:
    """A condition name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("λ_a", "$\\lambda_a$")


def ink(c: ex.Condition):
    if c.name in [w.name for w in ex.WARMUPS]:
        return WARMUP_INK[c.sharp_by is not None]
    return TAU_INK[c.slice_tau]


def marker(c: ex.Condition) -> str:
    if c.warmup == ex.DELAY:
        return ">"
    if c.warmup == ex.STRETCH:
        return "<"
    if not c.slice_soft:
        return "s"
    return {10.0: "v", 20.0: "D", 40.0: "o", None: "P"}[c.sharp_by]


# --- The schedules ----------------------------------------------------------------------------------------------


def schedules_figure() -> str:
    from sca.anchoring import AnchorSpec

    base = ex.st.schedules()[0]
    epoch = np.linspace(0, 100, 401)
    data = {"epoch": epoch.tolist(), "conditions": []}
    for c in CONDITIONS:
        spec = AnchorSpec(**c.anchor(base))
        taus = np.array([spec.taus(e) for e in epoch])
        data["conditions"].append(
            {"name": c.name, "weight": spec(epoch).tolist(), "tau_p": taus[:, 0].tolist(), "tau_l": taus[:, 1].tolist()}
        )
    alt = """
        Three panels over epochs 0 to 100. Top: λ_a rises from zero to 0.1 by epoch 40 for most conditions, by 60
        from a start at 20 for the delay, and by 80 for the stretch. Middle: τ_p falls from 1 to 0.1 by epoch 10,
        20 or 40, or sits at 0.1 throughout for fixed τ. Bottom: τ_ℓ falls from 4 to 0.1, 0.2 or 0.4 on the same
        windows, or sits at its end value throughout.
    """
    return schedules_draw(data, alt)


@memo
def schedules_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-schedules",
        alt_text=alt_text,
        caption=f"""
            **The schedules the conditions change.** The anchor weight $\\lambda_a$, the position pool temperature
            $\\tau_p$ and the slice pool temperature $\\tau_\\ell$ (both on a log scale) over the first hundred
            epochs, one line per condition, in its ink and marker; where conditions share a schedule, the later line is
            drawn over the earlier, with the markers staggered. The dotted line is the stop at epoch {STOP}. After
            epoch 100 every condition is the same.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6), layout="constrained", sharex=True)
        ep = np.asarray(data["epoch"])
        handles = []
        for i, c in enumerate(data["conditions"]):
            cond = BY_NAME[c["name"]]
            # Markers staggered by condition, so the ones on a shared schedule stay apart.
            every = (2 * i, 80)
            for a_, key in zip(axes, ("weight", "tau_p", "tau_l"), strict=True):
                (h,) = a_.plot(ep, c[key], color=ink(cond), lw=0.9, marker=marker(cond), ms=3, markevery=every,
                               alpha=0.85, label=tex(SHOWN[c["name"]]))  # fmt: skip
            handles.append(h)
        for a_ in axes:
            a_.axvline(STOP, color=RULE, lw=0.6, ls=":", zorder=0)
        fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=6)
        axes[0].set_ylabel("$\\lambda_a$", fontsize=8)
        axes[0].set_ylim(0, 0.11)
        for a_, label in zip(axes[1:], ("$\\tau_p$", "$\\tau_\\ell$"), strict=True):
            a_.set_yscale("log")
            a_.set_ylabel(label, fontsize=8)
            a_.minorticks_off()
        axes[1].set_yticks([0.1, 1], ["0.1", "1"])
        axes[2].set_yticks([0.1, 0.2, 0.4, 1, 4], ["0.1", "0.2", "0.4", "1", "4"])
        axes[-1].set_xlim(0, 100)
        axes[-1].set_xlabel("epoch", fontsize=8)
        for a_ in axes:
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


# --- E1 and E2: the call and the spill at the stop ---------------------------------------------------------------


def screen_figure() -> str:
    data = {
        "names": NAMES,
        "window": {n: WINDOW[n].tolist() for n in NAMES},
        "spill": {n: SPILL[n].tolist() for n in NAMES},
        "quiet": {QUIET_SHOWN[a]: v.tolist() for a, v in QUIET_SPILL.items()},
        "lost_max": LOST_MAX,
        "kept_min": KEPT_MIN,
        "criterion": CRITERION,
    }
    alt = """
        Two panels with one column per condition. Top: the call sits between about 0.5 and 0.6 for every condition,
        far above the kept line at 0.28; the lowest single run, on the stretched warm-up, is at 0.36. Bottom: the
        four τ_ℓ 0.1 conditions sit just below zero with every seed under 0.01, like the falloff and step, soft τ
        columns at the right. The τ_ℓ 0.2 conditions sit near zero, with one seed at 0.13 (τ_ℓ fixed) and two near 0.03. The τ_ℓ 0.4
        conditions and the λ_a conditions have means between about 0 and 0.02 and seeds up to 0.05 to 0.18.
    """
    return screen_draw(data, alt)


def hatch(ax, lo: float, hi: float, pattern: str) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig): denser is worse."""
    ax.axhspan(
        lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch=pattern, lw=0, alpha=0.12, zorder=0
    )


def _dots(ax, x: float, v, color, m: str, rng) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


@memo
def screen_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-screen",
        alt_text=alt_text,
        caption=f"""
            **Each condition at epoch {STOP}.** One column per condition, one small dot per seed, the seed mean as
            the large mark and the seed range as a thin bar. Top: α at the example answers at slice
            {ex.lr.TRACK_SLICE}, averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]} (the loss-rate call);
            below the dotted line a run is called lost (cross-hatched), above the dashed line kept, and between
            the two uncertain (lightly hatched). Bottom: the spill at epoch
            {STOP}, the largest drop on another op under the full edit less the median drop under eight reference
            edits; the dashed line is the criterion, with the side over it hatched. The gray columns at the right are the loss-rate screen's
            epoch-{STOP} checkpoints of the two quiet recipes, at {len(QUIET_SPILL["falloff"])} seeds each.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.6, 4.6), layout="constrained", sharex=True)
        rng = np.random.default_rng(0)
        for i, n in enumerate(names):
            c = BY_NAME[n]
            _dots(ax, i, data["window"][n], ink(c), marker(c), rng)
            _dots(bx, i, data["spill"][n], ink(c), marker(c), rng)
        for j, v in enumerate(data["quiet"].values()):
            _dots(bx, len(names) + 0.5 + j, v, QUIET_INK, "o", rng)
        ax.axhline(data["lost_max"], color=RULE, lw=0.6, ls=":", zorder=1)
        ax.axhline(data["kept_min"], color=RULE, lw=0.6, ls="--", zorder=1)
        ax.set_ylim(0, 1)
        hatch(ax, 0, data["lost_max"], "xx")
        hatch(ax, data["lost_max"], data["kept_min"], "/")
        ax.set_ylabel("α, epochs 44–64", fontsize=8)
        bx.axhline(0, color=ZERO, lw=0.5, zorder=0)
        bx.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
        bx.set_ylabel(f"spill at epoch {STOP}", fontsize=8)
        ticks = list(range(len(names))) + [len(names) + 0.5 + j for j in range(len(data["quiet"]))]
        bx.set_xticks(
            ticks,
            [tex(SHOWN[n]) + (" (base)" if n == ex.STORED else "") for n in names] + list(data["quiet"]),
            rotation=60,
            ha="right",
            fontsize=6,
        )
        bx.set_xlim(-0.6, ticks[-1] + 0.6)
        lo, hi = bx.get_ylim()
        hatch(bx, data["criterion"], hi, "//")
        bx.set_ylim(lo, hi)
        for a_ in (ax, bx):
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


def screen_table() -> str:
    rows = []
    for n in RANKED:
        q = per_seed(n, "alpha_q").mean(axis=0)[ex.lr.TRACK_SLICE]
        a = per_seed(n, "alpha_ans").mean(axis=0)
        rows.append(
            [
                f"`{SHOWN[n]}`" + (" (base)" if n == ex.STORED else ""),
                f"{8 - N_LOST[n]}/8",
                num(float(WINDOW[n].min()), ".2f"),
                num(float(SPILL[n].mean())),
                f"{num(float(SPILL[n].min()))} to {num(float(SPILL[n].max()))}",
                str(int((SPILL[n] > CRITERION).sum())),
                PEAK[n],
                num(float(a.max()), ".2f"),
                num(float(q), ".2f"),
            ]
        )
    return table_html(
        ["condition", "kept ↑", "lowest call", "spill ↓", "seed range", "over", "peak slice", "α at peak", "α, query"],
        rows,
        f"""
        **The conditions at epoch {STOP}, ranked.** Ranked by whether any run was not called kept, then by the mean
        spill. Kept: the runs the loss-rate call keeps (α at the example answers at slice {ex.lr.TRACK_SLICE},
        averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]}, at least {KEPT_MIN:g}). Lowest call: the
        lowest of those averages. Spill: the largest drop on another op under the full edit at epoch {STOP}, less
        the median drop under the reference edits, as a seed mean, with its seed range and the number of runs over
        the criterion ({CRITERION:g}). Peak slice and α at peak: where the seed-mean α at the example answers is
        highest at epoch {STOP}, and its value. α, query: the seed-mean α at the query answer at slice
        {ex.lr.TRACK_SLICE}.
        """,
    )


# --- E3: where the op is heading --------------------------------------------------------------------------------


def depth_figure(names: list[str]) -> str:
    data = {
        "names": names,
        "all": [per_seed(n, "alpha_ans").mean(axis=0).tolist() for n in NAMES],
        "seeds": {n: per_seed(n, "alpha_ans").tolist() for n in names},
        "query": {n: per_seed(n, "alpha_q").tolist() for n in names},
    }
    alt = """
        Four panels of α against slice, from the embedding to slice 4. In every panel the example answers rise from
        about zero at the embedding to a peak of about 0.55 at slice 3 and fall to about 0.37 at slice 4, and the
        query answer peaks near 0.85 at slice 3. At slice 1 the example answers sit near 0.22 for τ_ℓ 0.1, against
        about 0.3 to 0.37 in the other three panels.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-depth",
        alt_text=alt_text,
        caption=f"""
            **Where the op sits at epoch {STOP}, by slice.** α at the example answers (solid, one line per seed) and
            at the query answer (dashed), at each slice, for the three slice temperatures at the base timing and for the condition with the highest mean spill.
            The hairlines behind every panel are the seed means of every condition.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, len(names), figsize=(6.6, 2.4), layout="constrained", sharey=True, squeeze=False)
        x = np.arange(len(SLICE_TICK))
        for ax, n in zip(axes[0], names, strict=True):
            c = BY_NAME[n]
            for g in data["all"]:
                ax.plot(x, g, color=GHOST, lw=0.4, zorder=0)
            for a, q in zip(data["seeds"][n], data["query"][n], strict=True):
                ax.plot(x, a, color=ink(c), lw=0.9, marker=marker(c), ms=2.5, zorder=2)
                ax.plot(x, q, color=ink(c), lw=0.6, ls="--", zorder=1)
            ax.set_title(tex(SHOWN[n]), fontsize=7)
            ax.set_xticks(x, SLICE_TICK, fontsize=7)
            ax.set_ylim(-0.05, 1)
            ax.tick_params(labelsize=7)
        axes[0][0].set_ylabel("α", fontsize=8)
        return fig

    return _plot()


def quiet_figure(names: list[str]) -> str:
    data = {
        "names": names,
        "ghosts": [per_seed(n, "alpha_ans").mean(axis=0).tolist() for n in NAMES],
        "ghosts_q": [per_seed(n, "alpha_q").mean(axis=0).tolist() for n in NAMES],
        "ans": {n: per_seed(n, "alpha_ans").mean(axis=0).tolist() for n in names},
        "q": {n: per_seed(n, "alpha_q").mean(axis=0).tolist() for n in names},
        "quiet": {QUIET_SHOWN[a]: {"ans": v["ans"].tolist(), "q": v["q"].tolist()} for a, v in QUIET_ALPHA.items()},
    }
    alt = """
        Two panels of seed-mean α against slice. Left, at the example answers: every line starts near zero at the
        embedding. At slice 1 the two τ_ℓ 0.1 conditions and the two quiet recipes are together near 0.23, and the
        base is at about 0.37. The pool conditions peak near 0.56 at slice 3 and fall to about 0.38 at slice 4;
        falloff and step, soft τ level off near 0.5 from slice 3 to slice 4. Right, at the query answer: the same
        ordering at slice 1 (about 0.27, 0.33 and 0.43), all near 0.85 at slice 3, then the pool falls to about 0.62
        at slice 4 while the quiet recipes stay near 0.85.
    """
    return quiet_draw(data, alt)


@memo
def quiet_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-quiet",
        alt_text=alt_text,
        caption=f"""
            **The pool beside the quiet recipes at epoch {STOP}.** Seed-mean α by slice at the example answers
            (left) and at the query answer (right): the two stage-2 conditions and the base in their ink and
            marker, and the loss-rate screen's `falloff` and `step, soft τ` at the same epoch
            ({QUIET_ALPHA["falloff"]["n"]} seeds each) in gray. The hairlines are the other conditions.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained", sharey=True)
        x = np.arange(len(SLICE_TICK))
        for ax, key, ghosts in zip(axes, ("ans", "q"), (data["ghosts"], data["ghosts_q"]), strict=True):
            for g in ghosts:
                ax.plot(x, g, color=GHOST, lw=0.4, zorder=0)
            for n in data["names"]:
                c = BY_NAME[n]
                ax.plot(x, data[key][n], color=ink(c), marker=marker(c), lw=1.1, ms=4, label=tex(SHOWN[n]), zorder=2)
            for (name, v), m, ls in zip(data["quiet"].items(), ("D", "v"), ("-", "--"), strict=True):
                ax.plot(x, v[key], color=QUIET_INK, marker=m, ls=ls, lw=1.1, ms=4, label=name, zorder=1)
            ax.set_xticks(x, SLICE_TICK, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-0.05, 1)
        axes[0].set_title("example answers", fontsize=8)
        axes[1].set_title("query answer", fontsize=8)
        axes[0].set_ylabel("α", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=6)
        return fig

    return _plot()


def slice1_figure() -> str:
    data = {
        "runs": [
            {"name": n, "alpha1": float(RUNS[(n, s)]["alpha_ans"][1]), "spill": RUNS[(n, s)]["spill"]}
            for n in NAMES
            for s in SEEDS
        ],  # fmt: skip
        "criterion": CRITERION,
    }
    alt = """
        A scatter of 136 runs: α at the example answers at slice 1 on the horizontal axis, from about 0.15 to 0.45,
        and spill at epoch 64 on the vertical, from about −0.02 to 0.18. The τ_ℓ 0.1 runs (dark blue) sit at the
        left, between about 0.16 and 0.26, and all below the criterion. The runs over the criterion all sit right of
        about 0.26, most from the τ_ℓ 0.4 and λ_a conditions.
    """
    return slice1_draw(data, alt)


@memo
def slice1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-slice1",
        alt_text=alt_text,
        caption=f"""
            **Spill beside the anchor at slice 1.** One mark per run at epoch {STOP}: α at the example answers at
            slice 1 (horizontal), and the spill (vertical), in the ink and marker of its condition. The dashed line
            is the criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
        for r in data["runs"]:
            c = BY_NAME[r["name"]]
            ax.plot(r["alpha1"], r["spill"], marker(c), color=ink(c), ms=3.5, alpha=0.75, mew=0)
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=0)
        ax.set_xlabel("α at the example answers, slice 1", fontsize=8)
        ax.set_ylabel(f"spill at epoch {STOP}", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


# --- E4 to E6: the finished runs ---------------------------------------------------------------------------------

FIN_MARKER = {FALLOFF: "D", STEP_SOFT: "v"}


def fin_ink(name: str):
    return QUIET_INK if name in FIN_MARKER else ink(BY_NAME[name])


def fin_marker(name: str) -> str:
    return FIN_MARKER.get(name) or marker(BY_NAME[name])


def fin_tick(name: str) -> str:
    return tex(FIN_SHOWN[name]) + (" (base)" if name == ex.STORED else "")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


def finished_figure() -> str:
    data = {
        "names": FINISHED,
        "removal": {n: REMOVAL[n].tolist() for n in FINISHED},
        "spill": {n: FIN_SPILL[n].tolist() for n in FINISHED},
        "task": {n: TASK[n].mean(axis=1).tolist() for n in FINISHED},
        "criterion": CRITERION,
        "task_gate": TASK_MEAN_GATE,
    }
    alt = """
        Three panels with one column per condition. Left: removal sits between about 0.85 and 0.9 for every
        condition, with the two τ_ℓ 0.1 conditions highest and τ_ℓ 0.1, τ_ℓ fixed the tightest. Middle: spill sits
        near zero for every condition but the base, whose seeds reach 0.055; the two τ_ℓ 0.1 conditions have two
        seeds and one seed at the criterion line, none over it. Right: the task change sits near zero, the pool
        conditions a little higher than falloff and step, soft τ, all above the gate.
    """
    return finished_draw(data, alt)


@memo
def finished_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-finished",
        alt_text=alt_text,
        caption=f"""
            **Removal, spill and task at epoch {ex.EPOCHS}, by condition.** One column per condition, the
            {len(SEEDS)} seeds as small dots over a thin bar for their range, and the seed mean as the large mark.
            **Left:** removal at the full dose, net of the median over the other axes of the run. **Middle:** spill,
            the largest net drop on another op at any dose; the dashed line is the criterion, with the side over it
            hatched. **Right:** the task change, the mean clean score over ops less that of the control at the seed;
            the dashed line is the task gate on that mean, with the failing side hatched.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), layout="constrained")
        rng = np.random.default_rng(0)
        for ax, key in zip(axes, ("removal", "spill", "task"), strict=True):
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], fin_ink(n), fin_marker(n), rng)
            ax.set_xticks(range(len(names)), [fin_tick(n) for n in names], rotation=40, ha="right", fontsize=6)
            ax.set_xlim(-0.6, len(names) - 0.4)
            ax.tick_params(labelsize=7)
        axes[0].set_title("removal ↑", fontsize=8)
        axes[1].set_title("spill ↓", fontsize=8)
        axes[2].set_title("task change ↑", fontsize=8)
        for ax, y, below in ((axes[1], data["criterion"], False), (axes[2], data["task_gate"], True)):
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=1)
            lo, hi = ax.get_ylim()
            if below:
                lo = min(lo, y - 0.005)
                hatch(ax, lo, y, "//")
            else:
                hatch(ax, y, hi, "//")
            ax.set_ylim(lo, hi)
        return fig

    return _plot()


def finished_table() -> str:
    rows = []
    for n in FINISHED:
        pair = []
        for key in ("removal", "spill"):
            if n in LEADERS:
                g = paired(n, key, ex.STORED)
                spec = ".2f" if key == "removal" else ".3f"
                pair.append(f"{fmt(g, spec)}; {int((g > 0).sum())} up, {int((g < 0).sum())} down")
            else:
                pair.append("—")
        t = TASK[n]
        worst = np.unravel_index(int(t.argmin()), t.shape)
        rows.append(
            [
                f"`{FIN_SHOWN[n]}`" + (" (base)" if n == ex.STORED else ""),
                fmt(REMOVAL[n]),
                pair[0],
                fmt(FIN_SPILL[n], ".3f"),
                pair[1],
                f"{FIN_OVER[n]} of {len(SEEDS)}",
                f"{CTRL_OVER[n]} of {len(SEEDS)}",
                fmt(t.mean(axis=1), ".3f"),
                f"{num(float(t.min()), '.3f')} (`{OPS[worst[1]]}`, {SEEDS[worst[0]]})",
            ]
        )
    return table_html(
        [
            "condition",
            "removal ↑",
            "removal less base, paired",
            "spill ↓",
            "spill less base, paired",
            "over criterion ↓",
            "over, control median ↓",
            "task change ↑",
            "worst op change ↑",
        ],  # fmt: skip
        rows,
        f"""
        **Removal, spill and task at epoch {ex.EPOCHS}**, seed mean and range over {len(SEEDS)} runs. Removal and
        spill are net of the median drop over the other 63 coordinate axes of the run, and "over, control median"
        counts the runs over the criterion with the median taken in the control at the seed instead. The paired
        columns are the condition less the base at the same seed, with the count of seeds on each side of zero. The
        task change is the mean over ops of the clean score less that of the control at the seed; the worst op change
        is the largest loss on one op at any seed. The rows for `falloff`, `step, soft τ` and the base are those of
        recipe-compare.
        """,
    )


def finished_seed_table() -> str:
    rows = []
    for n in (ex.STORED, *LEADERS):
        rows.append([f"`{FIN_SHOWN[n]}`", "removal"] + [num(float(v), ".2f") for v in REMOVAL[n]])
        rows.append(
            ["", "spill"]
            + [f"**{num(float(v))}**" if v > CRITERION else num(float(v)) for v in FIN_SPILL[n]]
        )  # fmt: skip
        rows.append(["", "op"] + [f"`{o}`" for o in FIN_OPS[n]])
    return table_html(
        ["condition", "measure", *[str(s) for s in SEEDS]],
        rows,
        """
        **Removal and spill by seed**, as in the table above, with the op the spill is largest on. Bold marks a run
        over the criterion.
        """,
    )


def along_figure() -> str:
    data = {
        "names": ALONG_NAMES,
        "epochs": list(ALONG_EPOCHS),
        "spill": {n: ALONG[n].tolist() for n in ALONG_NAMES},
        "seeds": SEEDS,
        "criterion": CRITERION,
        "stop": STOP,
    }
    alt = """
        Three panels of spill against epoch, from 16 to 400 on a log scale, one line per seed. Base: three seeds
        rise over the criterion by epoch 64 (711 already at 32), and 707 and 711 stay over it to the end; two more
        seeds climb to near the criterion after epoch 128. τ_ℓ 0.1, by 40, and τ_ℓ 0.1, τ_ℓ fixed: every seed stays
        under 0.016 up to epoch 128; after that two seeds (705 and 707) and one seed (707) rise to about the
        criterion by epoch 400.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-along",
        alt_text=alt_text,
        caption=f"""
            **Spill along training.** The spill of each run (one line per seed) at the checkpoints kept at epochs
            {", ".join(str(e) for e in data["epochs"][:-1])} and at the end, scored as at the stop, against the
            median of eight reference edits. The base is spill-timing's rerun of soft-pool, scored at every record.
            The dashed line is the criterion and the dotted line the stop of stage 1. The seeds named are the ones
            that go over the criterion or come within a quarter of it.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, len(names), figsize=(6.6, 2.6), layout="constrained", sharey=True)
        ep = np.asarray(data["epochs"])
        for ax, n in zip(axes, names, strict=True):
            c = BY_NAME[n]
            for v in data["spill"][n]:
                v = np.asarray(v)
                loud = v.max() > NEAR
                ax.plot(ep, v, color=ink(c) if loud else GHOST, lw=1.0 if loud else 0.6, marker=marker(c),
                        ms=3 if loud else 2, zorder=2 if loud else 1)  # fmt: skip
            # Seed labels at the right end, nudged apart so they do not overlap.
            ends = sorted((v[-1], s) for s, v in zip(data["seeds"], data["spill"][n], strict=True) if max(v) > NEAR)
            ys: list[float] = []
            for y, _ in ends:
                ys.append(max(y, ys[-1] + 0.0045) if ys else y)
            for (y, s), ty in zip(ends, ys, strict=True):
                ax.annotate(str(s), (ep[-1], y), xytext=(ep[-1] * 1.07, ty), fontsize=6, va="center", color=ink(c))
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
            ax.axvline(data["stop"], color=RULE, lw=0.6, ls=":", zorder=0)
            ax.set_xscale("log")
            ax.set_xticks(ep, [str(e) for e in ep], fontsize=6)
            ax.minorticks_off()
            ax.set_xlim(ep[0] * 0.85, ep[-1] * 1.6)
            ax.set_title(tex(SHOWN[n]) + (" (base)" if n == ex.STORED else ""), fontsize=7)
            ax.set_xlabel("epoch", fontsize=7)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("spill", fontsize=8)
        return fig

    return _plot()


def restricted_table() -> str:
    cols = ("block 1", "blocks 2 to 4")
    rows = []
    for n in FINISHED:
        rows.append(
            [
                f"`{FIN_SHOWN[n]}`" + (" (base)" if n == ex.STORED else ""),
                f"slice {int(np.median(FIN_PEAK[n]))}: {int((FIN_PEAK[n] == np.median(FIN_PEAK[n])).sum())}",
            ]  # fmt: skip
            + [fmt(SLICE_REMOVAL[n][e]) for e in cols]
            + [
                f"{fmt(SLICE_SPILL[n][e], '.3f')}; {int((SLICE_SPILL[n][e] > CRITERION).sum())} over"
                for e in ("embedding", "block 1", "blocks 1 to 4", "every slice")
            ]  # fmt: skip
            + [fmt(LEAN_R[n])]
        )
    rows.append(["control", ""] + [""] * 6 + [fmt(LEAN_R[CONTROL])])
    return table_html(
        [
            "condition",
            "peak α, seeds",
            "removal, block 1 alone ↑",
            "removal, blocks 2 to 4 ↑",
            "spill, embedding alone ↓",
            "spill, block 1 alone ↓",
            "spill, blocks 1 to 4 ↓",
            "spill, every slice ↓",
            "lightness on e₁, embedding table",
        ],  # fmt: skip
        rows,
        """
        **Where the evidence sits, and where the edit finds it.** The slice at which α at the example answers of
        `difference` contexts peaks, with the count of seeds. Removal and spill under the full edit restricted to a
        set of slices, at every position, net of the same edit on the control at the seed (the median baseline exists
        only for the edit at every slice), with the count of runs over the criterion; seed mean and range. The last
        column is the correlation, over the color embeddings, of the e₁ component with lightness (the lean of
        embedding-lean). The first three rows are those of recipe-compare.
        """,
    )


def edit_table() -> str:
    rows = []
    for n in FINISHED:
        r = RATIO[n][1:]
        sl, p = np.unravel_index(int(np.nanargmax(r)), r.shape)
        over = OVER_TOL[n]
        ab = angle(turn_mean(n, "ans_bound"))
        rows.append(
            [
                f"`{FIN_SHOWN[n]}`" + (" (base)" if n == ex.STORED else ""),
                f"{float(r[sl, p]):.2f} (slice {int(sl) + 1}, {float(angle(turn_mean(n, 'bound')[sl + 1, p])):.1f}°)",
                f"{len(over)} (≤ {max(o[4] for o in over):.0f}°)" if over else "0",
                f"{ab[0]:.0f}°, {ab[1]:.0f}°",
                fmt(SHARE[n]),
                fmt(LANDING[n]),
            ]
        )
    return table_html(
        [
            "condition",
            "largest write over bound ↓",
            "sites over 1.25×, all runs ↓",
            "bound at example answers, emb and slice 1 ↓",
            "share removed ↑",
            "landing ↑",
        ],  # fmt: skip
        rows,
        f"""
        **The write, the grading and the landing.** The largest ratio of write to bound in angle over the slices past
        the embedding and every position, from the seed-mean quantiles as in edit-effects, with where it falls and
        the bound there. The sites over {TOLERANCE}× count slice and position pairs over all {len(SEEDS)} runs, each
        run taken on its own, with the largest bound among them. The bound at the example answers is over those that
        do not fit `difference`. The share removed is over every `difference` context, net of the control at the
        seed, and the landing is the share of the way to the target null in total variation; seed mean and range.
        The first three rows are those of recipe-compare.
        """,
    )


# %%
rf"""
# Soft pool scout

/// tip |
<!-- lede -->
A sharper pool over slices stopped the early spill of `pool 1–3, soft τ`. Of sixteen schedules of the pooled pull screened at epoch {STOP}, those with the slice temperature ending at 0.1 had no run spill early, however soon or late the temperatures sharpened.

We carried two of these conditions on to epoch {ex.EPOCHS}. Both stay within the spill criterion on every seed, and both remove a little more than the recipe of record at no cost to the task. Holding the slice temperature at 0.1 from the start is the cleaner of the two, and a candidate for the recipe of record.
///

[Recipe-compare](/docs/m2/recipe-compare/report.py) left one thing between `pool 1–3, soft τ` and the recipe of record: two of its eight runs go over the spill criterion, and the spill enters at block 1, the one slice the pool pulls and the step leaves alone. [Spill-timing](/docs/m2/spill-timing/report.py) found that both of those runs spill early, while the pool sharpens, and that this early kind already shows at epoch {STOP}. So a schedule that keeps block 1 closed should show it at epoch {STOP}, at a sixth of the cost of a full run.

This report trains sixteen conditions of the pool at its eight seeds ({SEEDS[0]} to {SEEDS[-1]}), each changing when the two pool temperatures sharpen, how sharp the pool over slices ends, or when the anchor weight warms up. Every run stops at epoch {STOP}, where we score whether it is keeping the anchor and how much the edit spills. This is stage 1. In stage 2, the two conditions with $\tau_\ell$ ending at 0.1 that change least (sharpening by epoch 40 as the base does, and held at 0.1 from the start) are carried on to epoch {ex.EPOCHS} from where they stopped, and measured as recipe-compare measured the three recipes, with the spill along training from checkpoints kept on the way.
"""

# %%

rf"""
## Observations

- [Lost anchors (E1)](#lost-anchors-e1): no schedule lost the anchor. All {N_RUNS} runs, the base included, are called kept, the lowest well above the line. The fixed temperatures and the stretched warm-up keep it a little less firmly.
- [Spill at epoch {STOP} (E2)](#spill-at-epoch-{STOP}-e2): the final slice temperature decides the spill; when the temperatures sharpen hardly matters. At τ_ℓ 0.1 no run is over the criterion, and the four conditions sit at or below the quiet recipes. At 0.4 (the base) about a quarter of the runs are over. Moving the $\lambda_a$ warm-up shifts the spill by less than the seeds vary, and fixing the temperatures from the start spills the most.
- [Where the evidence is heading (E3)](#where-the-evidence-is-heading-e3): every condition peaks at slice 3, and none leans on the query answer. What changes is slice 1: a sharper pool over slices puts less of the anchor there, and the runs that spill are the ones with more of it at slice 1.
- [Removal, spill and task at epoch {ex.EPOCHS} (E4)](#removal-spill-and-task-at-epoch-{ex.EPOCHS}-e4): both finished conditions keep the anchor, remove a little more than `falloff` and the base, and stay within the spill criterion on every seed, with one or two seeds at the line. Neither costs the task. Holding $\tau_\ell$ fixed is the tighter of the two on removal and spill.
- [Spill along training (E5)](#spill-along-training-e5): no finished run spills early. The spill that remains comes late, after epoch 128, on one or two seeds, and reaches about the criterion by the end. So the screen at epoch {STOP} called the early kind correctly, and the late kind is what stage 2 adds.
- [Where the spill enters (E6)](#where-the-spill-enters-e6): the evidence still peaks at slice 3 on every seed. The edit at block 1 alone costs the other ops about what it costs them under the step, and less than under the base. With $\tau_\ell$ fixed that holds seed by seed; when it sharpens by 40 it holds on the mean only. The other ops sit as far from e₁ at the first two slices as under the step, and the lean of the color embeddings along lightness falls back to the level of the step.

## Scope and method

This is an exploratory study, with no preregistration and no gate. Every condition starts from `pool 1–3, soft τ` (the base, which is not the recipe of record): the pull is pooled over the slice and position pairs of slices 1 to 3, the position pool temperature $\tau_p$ sharpens from 1 to 0.1 and the slice pool temperature $\tau_\ell$ from 4 to 0.4 over the first forty epochs, and the anchor weight $\lambda_a$ warms up over the same forty epochs to its peak of 0.1. The anti weight is that of `falloff`, the recipe of record, and the rest of training is as in [soft-pool](/docs/m2/soft-pool/report.py).

Twelve conditions cross two factors. One is $\tau_\ell$ at the end, with levels 0.1, 0.2 and 0.4 (the base), each starting from 4. The other is when the temperatures are sharp: by epoch 10, 20 or 40 (the base), both on the same window, or $\tau_\ell$ at its end value from epoch 0 while $\tau_p$ still sharpens by 40, which tells the two temperatures apart. Five more conditions move the $\lambda_a$ warm-up at the base temperatures: a delay (zero until epoch 20, then up by 60) and a stretch (up by 80), each with the soft temperatures and with both temperatures fixed at their end values from the start, and the fixed temperatures with the base warm-up. The fixed-temperature conditions might have lost the anchor, which is what they were there to show: if a delayed pull keeps it at a fixed τ, the soft start was a delay by other means. The schedules are drawn below.

{schedules_figure()}

Each run trains on the full 400-epoch schedule and stops at epoch {STOP}, keeping what it needs to carry on (a resumed run matches a straight run bit for bit). The base is not trained again: [spill-timing](/docs/m2/spill-timing/report.py) trained it at these seeds with the same scoring at every record, so its epoch-{STOP} record is the base here. The epoch-{STOP} checkpoints of `falloff` and `step, soft τ` left by [the loss-rate screen](/docs/m2/loss-rate/report.py) are reference points on the same scale.

A screen at epoch {STOP} can rank the conditions by the early kind of spill and by lost anchors. It cannot clear a condition: the late kind of spill, which grows on `value-hsv` after epoch 100 on seeds that learn it late, is invisible here. Spill-timing found the base about 0.02 above the quiet recipes at epoch {STOP}, pooled over seeds, so the seed mean of eight runs can tell an early-spilling condition from a quiet one at about that resolution, and a single run says little. We compare many conditions, so a mean a little apart from its neighbors is a lead only. The pattern across a factor, which pools 32 runs at each level, is firmer.

### The measurements

- The call: α (the cosine of the state with e₁) at the example answers at slice {ex.lr.TRACK_SLICE}, averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]}. As in the loss-rate screen, a run is called lost below {LOST_MAX:g}, kept from {KEPT_MIN:g}, and uncertain between.
- Spill at epoch {STOP}: the largest drop on another op when e₁ is projected out at every position and slice, less the median drop under eight reference edits (each projecting out one other basis direction), over the whole held-out set, as in spill-timing. A drop is in expected exact match. Lower is better. The criterion is {CRITERION:g}, though at epoch {STOP} it is a guide only, since it is set for a finished run.
- Where the evidence for the op sits: α at the example answers and at the query answer, by slice, at epoch {STOP}.

### Stage 2

Each run of the two finished conditions ($\tau_\ell$ ending at 0.1 and sharpening by epoch 40, and $\tau_\ell$ held at 0.1 from the start while $\tau_p$ sharpens by 40) carries on from epoch {STOP} to epoch {ex.EPOCHS}. The checkpoints kept at epochs {", ".join(str(e) for e in ex.KEEP_EPOCHS)} along the way are scored for the spill.

We measure the finished runs as [recipe-compare](/docs/m2/recipe-compare/report.py) measured `falloff`, `step, soft τ`, and the base at the same seeds, and set them beside its numbers for those three. Each run at a seed shares its initialization, batches, and label draws with the base and with the control of ex-2.2.23 at that seed. So each comparison is paired by seed, and the paired difference from the base says whether a gap is larger than the variation between seeds.

Eight seeds are too few to add to a count of lost anchors. The loss-rate screen took that count at forty seeds for the step recipes, and nobody has taken it for the pool.

- Removal: the drop in the score of `difference` under the full edit, as a share of the way from the clean score to the target null. It is net of the median drop under the same edit along each of the other 63 coordinate axes of the run (the rule of the D2.2 design). A run that removes less than {LOST_BELOW:g} has lost the anchor.
- Spill: the largest drop on another op at any dose, net of the same median. The criterion is {CRITERION:g}.
- Task change: the clean score per op, less that of the control at the seed. The task gate fails below {TASK_MEAN_GATE:g} on the mean over ops, or below {TASK_OP_GATE:g} on any one op.
- Spill along training: the spill at each kept checkpoint and at the end, against eight reference edits, as at the stop.
- Restricted edits, the write, the grading, and the landing, as in recipe-compare: the full edit made at a set of slices only, net of the same edit on the control; how much e₁ the blocks after the edit put back on the other ops, against the bound the clean pass sets; the share removed on `difference` contexts; and how far the edited answers move toward the target null.
"""

# %%

rf"""
## Lost anchors (E1)

The fixed-temperature conditions were the ones most likely to lose the anchor, since a sharp pool from the first step is what let the pull latch onto the query answer on one seed in slice-terms. Did any schedule lose it by epoch {STOP}?

None did. Every run is called kept, and the lowest call ({LOWEST_CALL:.2f}, on a stretched warm-up) is still above the line by a wide margin. The figure (top) shows the calls close together across all sixteen conditions. The fixed temperatures and the stretch sit a little lower than the rest, which fits a pull that has had less time at full strength by epoch {STOP}, but none comes near the uncertain band.

{screen_figure()}

Two soft starts are in play: $\tau_p$ sharpening from 1, and $\tau_\ell$ sharpening from 4. The conditions with $\tau_\ell$ fixed drop the second, the fixed-τ conditions drop both, and all of them kept the anchor on every seed. So in this pool neither soft start is what keeps the anchor, at least to epoch {STOP} and at these seeds. The step may differ: in the loss-rate screen, the step at a fixed $\tau_p$ (`falloff`) lost the anchor on one seed of forty and the step with the soft start on none, which fits a soft $\tau_p$ that helps the step, though one loss is too few to show it. Eight seeds would miss a loss rate of one in ten or so, and a run can still lose the anchor late, so the fixed temperatures are not cleared for this pool either.
"""

# %%

rf"""
## Spill at epoch {STOP} (E2)

If spill enters through block 1 while the pool is soft, then sharpening sooner, sharpening the slice pool further, or holding the pull back until e₁ is cleared might each close it. Which of them lowers the spill at epoch {STOP}?

The slice temperature at the end does, and the others barely move it. The figure (bottom) groups the crossing by $\tau_\ell$: at 0.1 the four timing levels all sit at or below the quiet recipes, with no run of 32 over the criterion; at 0.2 the early spill is a little higher, with {LEVEL_OVER[0.2]} runs over; at 0.4, the base level, {LEVEL_OVER[0.4]} of 32 are over. Within each level, sharpening by epoch 10, 20 or 40, or holding $\tau_\ell$ fixed from the start, makes no difference we can see at eight seeds. So for the early spill, what matters is how sharp the slice pool ends, and how soon it gets there barely does. That also covers where the slice pool starts: at 0.1, sharpening from 4 and holding at 0.1 from epoch 0 spill about the same, so a start in between would be unlikely to differ.

This is a stronger effect of $\tau_\ell$ than [slice-terms](/docs/m2/slice-terms/report.py) saw. Its E4 trained the pool over slices with $\tau_\ell$ from 0.1 to 1.6 and found no trend in removal, with the spill a little higher at the two softest levels. But that was one seed, chosen because its pull had settled on block 4, at a fixed $\tau_p$ and with the pool over blocks 1 to 4. Here the pool leaves block 4 out, and the spill is a per-seed event: at $\tau_\ell$ 0.4 most seeds stay quiet, and one seed would most likely have been one of those.

{screen_table()}

The $\lambda_a$ conditions all stay well above the $\tau_\ell$ 0.1 level. One of them is not on an equal footing: under the stretched warm-up, $\lambda_a$ is still at {STRETCH_AT_STOP:.0%} of its peak at the stop, so its spill at epoch {STOP} is of a pull not yet at full strength. A twenty-epoch delay, which gives e₁ time to clear under the full learning rate, lowered the mean spill somewhat, at the soft temperatures and at the fixed ones. The stretched warm-up raised it, and so did fixing both temperatures with the base warm-up. Each of these means rests on one or two seeds that spill far more than the rest, so at eight seeds the differences among them could be chance. None of them comes near what sharpening the slice pool does.

Of the {N_OVER} runs over the criterion, {OVER_OPS[TOP_OP]} spill most on `{TOP_OP}`, the op on which the base spills early in spill-timing.

These are early numbers. A sharper slice pool could still spill late, on `value-hsv` as the base does on its slow seeds, and only a finished run can show that. What we take from the screen is that $\tau_\ell$ is the lever on the early kind, and that the level at 0.1 brings the pool to the quiet recipes on it.
"""

# %%

rf"""
## Where the evidence is heading (E3)

A sharper slice pool could lower the spill in a way we would not want: by settling the pull on one slice and one role, as the joint pool did in [slice-terms](/docs/m2/slice-terms/report.py), where one seed latched onto the query answer and the op could not be removed. Where does the evidence for the op sit at epoch {STOP}?

In the same place in every condition. The seed-mean α at the example answers peaks at slice 3 in all seventeen conditions, and α at the query answer is about the same everywhere (the table above). The figure shows the three levels of $\tau_\ell$ at the base timing, beside the condition that spilled most: the shapes are nearly the same, with one difference at slice 1.

{depth_figure(["lt0.1-by40", "lt0.2-by40", ex.STORED, "fixed-stretch"])}

Beside the quiet recipes (below), the two conditions at $\tau_\ell$ 0.1 have about as much of the evidence at slice 1 as the step does, though the step pulls nothing there: some alignment reaches slice 1 from the pull on the slices after it. The base has half as much again. Past slice 1 the shapes differ as recipe-compare found at the end of training: the step holds the evidence level from slice 3 to slice 4, where the pool lets it fall back, at the example answers and at the query answer alike.

{quiet_figure(["lt0.1-by40", "lt0.1-fixed-lt", ex.STORED])}

That difference is the one that tracks the spill. At $\tau_\ell$ 0.1, α at the example answers at slice 1 averages {LEVEL_ALPHA1[0.1]:.2f}, against {LEVEL_ALPHA1[0.4]:.2f} at 0.4. Across all {N_RUNS} runs, the runs over the criterion are the ones with more of the anchor at slice 1, though at the same α some spill and some do not.

{slice1_figure()}

This fits the account recipe-compare gave: the pool spills when it puts part of the pull on block 1, and the edit then takes something from the other ops there. A sharper slice pool takes more of the pull off slice 1 and puts it on slice 3, so block 1 holds less of the anchor and the edit has less to take. It fits without proving it; the screen shows the two moving together, and a restricted edit on the finished runs (block 1 alone) would test it.
"""

# %%

rf"""
## Removal, spill and task at epoch {ex.EPOCHS} (E4)

Do the two finished conditions keep the anchor, stay within the spill criterion and leave the task alone at the end of training, seed by seed, against the base and the two quiet recipes?

They do. No finished run lost the anchor: the lowest removal is {min(REMOVAL[n].min() for n in LEADERS):.2f}. Both conditions remove a little more than the base on most seeds (the paired column of the table), and a little more than `falloff` on the mean. Holding $\tau_\ell$ fixed gives the tightest removal of the five, with all eight seeds between {REMOVAL[LEADERS[1]].min():.2f} and {REMOVAL[LEADERS[1]].max():.2f}.

{finished_figure()}

{finished_table()}

On spill, neither condition has a run over the criterion, where the base has {FIN_OVER[ex.STORED]}. The change shows best on the two seeds the base spills on most: at 711 both conditions sit below zero, and at 707 both come down to the line.

That leaves one or two seeds per condition at the criterion, each within a thousandth of it: 705 and 707 under `{SHOWN[LEADERS[0]]}`, and 707 under `{SHOWN[LEADERS[1]]}`. Taking the median over the other axes of the control at the seed, in place of those of the run, puts one run of `{SHOWN[LEADERS[0]]}` over the line (705), and none of `{SHOWN[LEADERS[1]]}`.

On the task, both pass the gate on the seed mean, and no op at any seed goes past the per-op line, so the pool keeps the task change near zero that recipe-compare found for it. One run of `{SHOWN[LEADERS[0]]}` (708) is a little under the line for the mean over ops, at the seed where `falloff` is lowest too.

{finished_seed_table()}

## Spill along training (E5)

[Spill-timing](/docs/m2/spill-timing/report.py) found two kinds of spill under the base: an early kind, which rises while the pool sharpens and already shows at epoch {STOP}, and a late kind, which grows after epoch 100 on seeds that learn `value-hsv` late. Which kind is left on the seeds that come near the criterion at epoch {ex.EPOCHS}?

The late kind. Under the base, three seeds go over the criterion by epoch {STOP}, and 711 does so already at epoch 32. Under the two finished conditions no seed goes over it, and up to epoch 128 the largest spill is {ALONG[LEADERS[0]][:, :4].max():.3f} under `{SHOWN[LEADERS[0]]}` and {ALONG[LEADERS[1]][:, :4].max():.3f} under `{SHOWN[LEADERS[1]]}`. After that, two seeds under `{SHOWN[LEADERS[0]]}` ({" and ".join(str(s) for s in LATE[LEADERS[0]])}) and one under `{SHOWN[LEADERS[1]]}` ({LATE[LEADERS[1]][0]}) climb to about the criterion by the end.

{along_figure()}

So the screen at epoch {STOP} called the early kind correctly: the conditions it found quiet stayed quiet through the part of training it covers.

The late kind here is about as large as on the late seeds of the base (705 and 710 come to about 0.02 there too). It lands on different ops from those spill-timing saw, though: on `hue-hsv` and `darken` at 705 and 707 under `{SHOWN[LEADERS[0]]}`, and on `mix` at 707 under `{SHOWN[LEADERS[1]]}`, where the late kind in the base was on `value-hsv`. Two or three runs are too few to say whether that is a different kind of late spill or the same one landing elsewhere.

## Where the spill enters (E6)

Recipe-compare traced the spill of the base to the edit at block 1, and E3 found that a sharper slice pool puts less of the anchor at slice 1. Does the edit at block 1 alone now cost the other ops less, and does the rest of what the edit does stay as it was?

{restricted_table()}

The evidence settles where it did. Under both conditions α at the example answers peaks at slice 3 on every seed, as under the base. The edit at blocks 2 to 4 removes most of the op on every seed but 710, which falls short under every recipe. The edit at block 4 alone removes nothing. The edit at the embedding alone spills nothing, and neither does the edit at blocks 2 to 4.

The edit at block 1 alone costs the other ops less than under the base on the mean, and about what it costs them under `step, soft τ`: the mean is between the step and the base under `{SHOWN[LEADERS[0]]}` and below the step under `{SHOWN[LEADERS[1]]}`.

Seed by seed the two conditions differ. `{SHOWN[LEADERS[1]]}` is below the base on six seeds of eight, where `{SHOWN[LEADERS[0]]}` is below it on three, and the lower mean under `{SHOWN[LEADERS[0]]}` comes from 711, the seed the base spills on most at block 1. So holding $\tau_\ell$ fixed closes block 1 about as far as leaving it out of the pull does, and sharpening to 0.1 by epoch 40 does so on the mean, less surely seed by seed.

As recipe-compare noted, the block-1 edit on its own overstates what the full edit costs: under `{SHOWN[LEADERS[1]]}` it goes over the criterion on two seeds whose full edit stays within it.
<!-- REVIEW: added the paired-by-seed reading of the block-1 edit. On the mean both conditions sit below the base, but `τ_ℓ 0.1, by 40` is above the base on five seeds of eight (SLICE_SPILL[n]["block 1"] against the base's, seed by seed), and without 711 its mean is above the base's, so the mean alone overstated it. The same holds for the control-netted edit at every slice, below (above the base on five seeds). Verify: the per-seed block-1 and every-slice spill of the base and the two leaders. -->

With the edit at every slice, net of the control rather than of the median, `{SHOWN[LEADERS[0]]}` has three runs over the criterion and `{SHOWN[LEADERS[1]]}` none. Under this baseline `step, soft τ` has two over, so these counts are best compared with one another, and on them holding $\tau_\ell$ fixed is the quieter of the two.

The embedding table moves with the slice pool as well. Under the base the e₁ components of the color embeddings follow lightness closely on every seed (the lean of [embedding-lean](/docs/m2/embedding-lean/report.py)). Under both finished conditions the lean is back at about the level of the step, and varies from seed to seed as it does there, which fits a pull that reaches slice 1 less.

{edit_table()}

What the edit does downstream barely changes, and where it changes it moves toward the step. The write stays close to the bound past the embedding, with fewer sites over the tolerance than under the base, and the share removed and the landing are about the same as under every recipe.

Recipe-compare found the bound at the example answers of the other ops about 3× as wide at the embedding under the base as under `falloff`. Here it is back at the width of `falloff` at the embedding, and a little narrower at slice 1. So the other ops sit as far from e₁ at the first two slices as under the step.

## Discussion

With $\tau_\ell$ at 0.1, the pool over slices 1 to 3 keeps the anchor, removes a little more than the recipe of record, costs the task nothing, and stays within the spill criterion on every seed. The early spill that put two runs of the base over the criterion is gone.

So are most of the signs that block 1 is open to the edit. The wider bound at the first two slices and the lean in the embedding table are gone under both conditions, and the larger cost of the block-1 edit is gone under the condition with $\tau_\ell$ fixed.

Of the two, holding $\tau_\ell$ at 0.1 from the start is the cleaner. It removes the most and the most evenly, has no run over the criterion under either median, and none under the control-netted edit at every slice. It also has one schedule fewer than the base, since only $\tau_p$ sharpens.

The other condition is close behind on the median baseline of the D2.2 design, where the gap rests on two or three runs near the line. On the edits netted against the control the gap is larger: there the other condition sits at about the level of the base on most seeds, and only its mean is lower.

What is left is the late kind of spill, which brings one or two seeds per condition to about the criterion by the end. It is about as large as under the base on its late seeds, so it does not seem to come from the sharper slice pool, and nothing in this study acts on it. When it appears, the term still changing is the anti weight, partway through its anneal, so if anything in the recipe acts on the late kind, that anneal is the likelier place.

Before the pool with $\tau_\ell$ fixed at 0.1 could replace `falloff` as the recipe of record, it would need two things the step has: a count of lost anchors at forty seeds, which a loss-rate screen would give, and a set of fresh seeds, which a confirmation at seeds 712 to 719 with their controls would give. Since the pool chooses the slice within a range set by the architecture, it would be the easier recipe to carry to another model.
<!-- REVIEW: was "replace `step, soft τ`"; the report's own Scope and method and the D2.2 design doc name `falloff` as the recipe of record, and the two things named (forty seeds, fresh seeds) are what `falloff` has. Verify: docs/m2/d2.2/design.md, the recipe block. -->
"""
