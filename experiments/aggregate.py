"""Aggregate completed runs: per-fold metrics, five-fold summaries and paired contrasts.

Contrasts are ``baseline - other`` in percentage points, with a nominal 95% paired
t interval, a two-sided paired t test and an exact Wilcoxon signed-rank test
(zero differences omitted).  With five folds the smallest attainable two-sided
Wilcoxon p-value is 0.0625; all tests are unadjusted and exploratory.
Outputs are written to ``runs/summary/``.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = RUNS / "summary"
variants = json.loads((REPO / "experiments/configs/variants.json").read_text(encoding="utf8"))


def fold_metrics(root):
    paths = list((root / "ablation/runs").glob("*.json"))
    assert len(paths) == 1, (root, paths)
    detail = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    cm = np.asarray(detail["test"]["confusion_matrix"])
    assert cm.shape == (15, 15) and (cm.sum(1) > 0).all()
    recall = cm.diagonal() / cm.sum(1)
    s = detail["summary"]
    return dict(acc=100 * np.trace(cm) / cm.sum(), ba=100 * recall.mean(), mr=100 * recall.min(),
                recall_class10=100 * recall[10], test_rows=int(cm.sum()), loss=s.get("loss"),
                pairwise=s.get("pairwise_ala_strength"), traj=s.get("trajectory_consistency_weight"),
                physics=s.get("physics_teacher"), gate=s.get("physics_min_distance"), sigma=s.get("physics_sigma"),
                decay=s.get("physics_decay_end"), scale=s.get("head_scale"), hybrid_alpha=s.get("hybrid_alpha"),
                epochs=s.get("epochs_ran"), best_epoch=s.get("best_epoch"))


rows = []
for name in variants:
    for fold in range(5):
        root = RUNS / name / f"fold{fold}"
        if (root / "COMPLETE.json").exists():
            rows.append(dict(model=name, fold=fold, **fold_metrics(root)))
teacher = RUNS / "teacher/teacher_per_fold.csv"
if teacher.exists():
    t = pd.read_csv(teacher)
    for _, r in t[t.part == "test"].iterrows():
        rows.append(dict(model=r.model, fold=int(r.fold), acc=r.acc, ba=r.ba, mr=r.mr, test_rows=int(r.rows)))
if not rows:
    sys.exit("No completed runs under runs/.")
OUT.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "per_fold.csv", index=False)


def wilcoxon_exact(values):
    try:
        return stats.wilcoxon(values, alternative="two-sided", method="exact")
    except TypeError:  # older SciPy
        return stats.wilcoxon(values, alternative="two-sided", mode="exact")


def paired(reference, other, metric):
    a = frame[frame.model == reference].set_index("fold")[metric]
    b = frame[frame.model == other].set_index("fold")[metric]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return None
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    nonzero = d[d != 0]
    return dict(n=n, mean=mean, sd=sd, ci_low=mean - half, ci_high=mean + half,
                p_t=float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else float("nan"),
                p_wilcoxon=float(wilcoxon_exact(nonzero).pvalue) if len(nonzero) else float("nan"),
                n_positive=int((d > 0).sum()), per_fold=" ".join(f"{v:+.3f}" for v in d))


summary = []
for model in frame.model.unique():
    d = frame[frame.model == model]
    row = dict(model=model, n_folds=len(d))
    for k in ("acc", "ba", "mr"):
        row[k + "_mean"], row[k + "_sd"] = d[k].mean(), d[k].std(ddof=1)
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "summary.csv", index=False)

contrasts = []
if (frame.model == "baseline").any():
    for model in frame.model.unique():
        if model == "baseline":
            continue
        for k in ("acc", "ba", "mr"):
            p = paired("baseline", model, k)
            if p:
                contrasts.append(dict(reference="baseline", compared=model, metric=k, **p))
contrasts = pd.DataFrame(contrasts)
contrasts.to_csv(OUT / "paired_contrasts.csv", index=False)

pd.set_option("display.width", 200)
print("== Five-fold means (sample SD, ddof=1) ==")
print(summary.round(3).to_string(index=False))
if len(contrasts):
    print("\n== Paired contrasts: baseline - compared (percentage points) ==")
    print(contrasts[["compared", "metric", "n", "mean", "sd", "ci_low", "ci_high", "p_t", "p_wilcoxon", "n_positive"]].round(4).to_string(index=False))
print(f"\nWrote {OUT}/per_fold.csv, summary.csv and paired_contrasts.csv")
