"""Other-op gain under the fitted edits: do the directions fitted after training share the gain of filtered-corpus E4?

Section E4 of the filtered-corpus report found that the edit along e₁ improves the anchored model on contexts of the
other ops whose examples point partly at `difference`: the gain over the clean model grows with the posterior on
`difference`, and in the top bin it is about three-quarters of the gain the target null predicts over the Bayes
predictor. This probe takes the same measurement for the edits of the post-hoc comparison, each applied to the control
at the same seed (712 to 719), at the configuration its E sections score: removal at the spill criterion, once with the
selection over the whole grid (where every fitter picked the example answers, a site that needs role labels) and once
over the pooled fits alone. No new passes: `finish_one` of the post-hoc comparison stored the expected exact match of
each context on the held-out set, clean and edited, and the dose arrays of the fixed confirmation hold the anchored
edit. The bins and the contexts are those of E4.

Results (2026-11-05, production arrays). Gain over the clean model on the contexts of the other ops, seed mean, by bin
of the posterior on `difference` (10180, 1042, 664, and 114 contexts), and the share of the target-null gain in the top
bin (seed mean, then range):

    edit                      < 0.05   0.05–0.2  0.2–0.5   ≥ 0.5    top-bin share
    target null                0.003    0.054     0.100    0.328    1
    anchored (e₁, all slices) −0.012    0.044     0.076    0.246    0.75 (0.67 to 0.83)
    LEACE, whole grid          0.000    0.061     0.083    0.273    0.83 (0.60 to 0.96)
    INLP, whole grid          −0.001    0.058     0.081    0.270    0.82 (0.55 to 0.95)
    diff. in means, whole     −0.001    0.061     0.082    0.273    0.83 (0.59 to 0.96)
    probe, whole grid          0.000    0.063     0.085    0.274    0.84 (0.53 to 0.94)
    optimized (pooled)        −0.016    0.044     0.073    0.260    0.80 (0.59 to 0.86)
    LEACE, pooled             −0.006    0.054     0.076    0.264    0.81 (0.58 to 0.91)
    INLP, pooled              −0.008    0.050     0.070    0.249    0.76 (0.52 to 0.88)
    diff. in means, pooled    −0.008    0.034     0.057    0.193    0.59 (0.08 to 0.98)
    probe, pooled             −0.004    0.055     0.073    0.244    0.75 (0.42 to 0.83)

So the gain is shared. Every fitted edit climbs toward the target null on these contexts as the anchored edit does,
rising with the posterior. In the top bin the fitters selected over the whole grid get a little further than the
anchored edit, and the pooled ones about as far. Difference in means at its pooled selection is the exception: one of
its runs gains almost nothing there, and another a quarter of the target-null gain. The fitted edits vary more from
seed to seed than the anchored edit; at seed 717 every standard fitter gets about half of the target-null gain, where
the anchored edit gets 0.78. The anchored edit also costs a little on the contexts that point nowhere near `difference`
(the first bin), where the fitters selected over the whole grid cost nothing; they were selected on the tuning set to
spill little on the other ops, and the anchored edit was not tuned. The two clean models start from about the same
place: on average both sit within 0.06 of the Bayes predictor in every bin, and a little above it in the top bin.

The anchored numbers are those of E4 and agree to four places with `finish_one`'s own edit along e₁ on the anchored
runs (`anchor (finish_one)` in the output). The top bin holds 114 contexts, so a share moves by a few points with a
handful of them.

    bin/mini run docs/m2/other-op-gain-fitted/experiment.py --app modal --max-containers 8 --budget 30m
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment


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


ph = _load_sibling("post-hoc-compare", "other_op_gain_ph")
fc, ex2216, ex2218 = ph.fc, ph.ex2216, ph.ex2218
NAME = "other-op-gain-fitted"
SEEDS: tuple[int, ...] = ph.SEEDS
D = ph.D
OTHER_BINS = (0.0, 0.05, 0.2, 0.5, 1.0)
"""The bins of the posterior on `difference` that filtered-corpus E4 takes, over the contexts of the other ops."""
BIN_LABEL = ("< 0.05", "0.05–0.2", "0.2–0.5", "≥ 0.5")


def _fetch(refs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    import tempfile
    from pathlib import Path

    from mini.store import project_store

    store = project_store()
    got = store.get_refs(refs)
    missing = [r for r, a in got.items() if a is None]
    assert not missing, f"not published: {missing}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(got[r], Path(tmp) / f"{i}.npz") for i, r in enumerate(refs)])
        out = {}
        for r, path in zip(refs, paths, strict=True):
            with np.load(path) as z:
                out[r] = {k: z[k] for k in z.files}
    return out


def _results_body() -> dict:
    import json
    import tempfile
    from pathlib import Path

    from mini.store import project_store

    store = project_store()
    art = store.get_refs([ph.RESULTS_REF, ph.POOLED_REF])
    with tempfile.TemporaryDirectory() as tmp:
        p = store.get_many([(art[r], Path(tmp) / f"{i}.json") for i, r in enumerate(art)])
        return dict(zip(art, (json.loads(Path(x).read_text()) for x in p), strict=True))


def measure() -> dict[str, Any]:
    """The gain over the clean model on the contexts of the other ops, by bin of the posterior on `difference`,
    for each fitter of the post-hoc comparison on the controls (at the selection over the whole grid, which may use
    role labels, and at the selection over the pooled fits alone), for the edit along e₁ on the anchored runs (the
    full edit of the fixed confirmation, as E4 takes it, and `finish_one`'s own edit along e₁ as a check), and for
    the target null over the Bayes predictor. Per seed, a row per bin.
    """
    controls = {r["seed"]: r["label"] for r in ph.runs() if r["kind"] == "control"}
    anchored = {r["seed"]: r["label"] for r in ph.runs() if r["kind"] == "anchored"}
    dose_ref = {s: fc.DOSE_ARRAYS_REF.format(label=fc.label_of(s, None)) for s in SEEDS}
    full_ref = {lab: ph.ARRAYS_REF.format(label=lab) for lab in [*controls.values(), *anchored.values()]}
    pooled_ref = {lab: ph.POOLED_ARRAYS_REF.format(label=lab) for lab in controls.values()}
    arr = _fetch([*dose_ref.values(), *full_ref.values(), *pooled_ref.values()])

    held = arr[dose_ref[SEEDS[0]]]
    op_ids = held["op_ids"].astype(np.int64)
    post = held["posterior"].astype(np.float64)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ph.OP_NAMES))
    empty = np.zeros((len(op_ids), 0), np.int64)
    ctx = P.Contexts(op_ids, empty, empty, held["query_pair"].astype(np.int64))
    ideal = P.expected_match(table, ctx, post)
    null = held["null_eem"].astype(np.float64)
    other = op_ids != D
    obin = np.clip(np.digitize(post[:, D], OTHER_BINS[1:-1]), 0, len(BIN_LABEL) - 1)

    def by_bin(g: np.ndarray) -> list[float]:
        return [float(g[(obin == b) & other].mean()) for b in range(len(BIN_LABEL))]

    for s in SEEDS:  # every array is on the held-out contexts, in the order of the dose arrays
        assert (arr[dose_ref[s]]["op_ids"] == held["op_ids"]).all()
        a = arr[full_ref[anchored[s]]]
        assert np.allclose(a["clean/eem"], arr[dose_ref[s]]["clean_eem"], atol=2e-3)

    bodies = _results_body()
    main_cfg = {r["label"]: {m: v["main_config"] for m, v in r["finish"]["methods"].items()} for r in bodies[ph.RESULTS_REF]["runs"]}  # fmt: skip
    pooled_cfg = {r["label"]: {m: v["main_config"] for m, v in r["methods"].items()} for r in bodies[ph.POOLED_REF]["runs"]}  # fmt: skip

    gain: dict[str, dict[int, list[float]]] = {"target null": {s: by_bin(null - ideal) for s in SEEDS}}
    # The room each clean model leaves below the Bayes predictor, which bounds how far an edit can climb toward it.
    headroom = {
        "control": {s: by_bin(ideal - arr[full_ref[controls[s]]]["clean/eem"]) for s in SEEDS},
        "anchored": {s: by_bin(ideal - arr[dose_ref[s]]["clean_eem"]) for s in SEEDS},
    }
    gain["anchor (dose arrays)"] = {s: by_bin(arr[dose_ref[s]]["eem"][-1] - arr[dose_ref[s]]["clean_eem"]) for s in SEEDS}  # fmt: skip
    gain["anchor (finish_one)"] = {s: by_bin(arr[full_ref[anchored[s]]]["anchor/eem"] - arr[full_ref[anchored[s]]]["clean/eem"]) for s in SEEDS}  # fmt: skip
    for m in ph.METHODS:
        a = {s: arr[full_ref[controls[s]]] for s in SEEDS}
        gain[f"{m} (whole grid)"] = {s: by_bin(a[s][f"{m}/eem"] - a[s]["clean/eem"]) for s in SEEDS}
        if m in ph.STANDARD:
            p = {s: arr[pooled_ref[controls[s]]] for s in SEEDS}
            gain[f"{m} (pooled)"] = {s: by_bin(p[s][f"{m}/eem"] - p[s]["clean/eem"]) for s in SEEDS}
    configs = {s: {"whole grid": main_cfg[controls[s]], "pooled": pooled_cfg.get(controls[s], {})} for s in SEEDS}
    n_bin = [int(((obin == b) & other).sum()) for b in range(len(BIN_LABEL))]
    return {"bins": list(BIN_LABEL), "n_bin": n_bin, "gain": gain, "headroom": headroom, "configs": configs}


def summary(out: dict[str, Any]) -> dict[str, Any]:
    """Seed mean and range of the gain per bin, and the share of the target-null gain in the top bin."""
    top_null = float(np.mean([v[-1] for v in out["gain"]["target null"].values()]))
    rows = {}
    for k, per_seed in out["gain"].items():
        g = np.array(list(per_seed.values()))
        rows[k] = {
            "mean": g.mean(0).round(4).tolist(),
            "min": g.min(0).round(4).tolist(),
            "max": g.max(0).round(4).tolist(),
            "top_share": round(float(g[:, -1].mean() / top_null), 3),
            "top_share_by_seed": (g[:, -1] / top_null).round(3).tolist(),
        }
    headroom = {k: np.array(list(v.values())).mean(0).round(4).tolist() for k, v in out["headroom"].items()}
    return {"bins": out["bins"], "n_bin": out["n_bin"], "rows": rows, "headroom": headroom}


def main(ctx: Ctx) -> dict:
    out = ctx.run(measure, role="read")
    return {"summary": summary(out)}


experiment = Experiment(name=NAME, main=main, roles={"read": dict(cpu=2, memory=8192, timeout=1800)})
