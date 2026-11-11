"""Physical baselines on the archived grouped folds (CPU, about one to two minutes).

For each fold, using training rows only, this fits
  * ``lr``     - fractional-linear surrogate + multinomial logistic regression (the teacher of the paper);
  * ``argmin`` - the surrogate response of minimum magnitude among indices 0-14, without label fitting.
Both are scored on the same test rows as the student.  The script also writes an
analytic estimate of the teacher's stored constants and per-sample operations;
these are estimates, not on-board measurements.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from iatopt import core  # noqa: E402

OUT = REPO / "runs/teacher"
OUT.mkdir(parents=True, exist_ok=True)
bundle = core.load_dataset(REPO / "data/x_train.csv", REPO / "data/y_train.csv")
raw, labels = bundle.raw, bundle.labels
rows, cost = [], {}
for fold in range(5):
    z = np.load(REPO / "data/splits/original" / f"fold{fold}.npz")
    for kind in ("lr", "argmin"):
        teacher = core.PhysicsTeacher.fit(raw, labels, z["train"], kind=kind)
        for part in ("valid", "test"):
            m = core.metrics(labels[z[part]], teacher.predict_raw(raw[z[part]]))
            rows.append(dict(model="teacher_" + kind, fold=fold, seed=42 + fold, part=part,
                             acc=100 * m["accuracy"], ba=100 * m["balanced_accuracy"],
                             mr=100 * m["minimum_class_recall"], rows=int(len(z[part]))))
        np.savez_compressed(OUT / f"teacher_{kind}_fold{fold}_test_predictions.npz", row=z["test"],
                            truth=labels[z["test"]], prediction=teacher.predict_raw(raw[z["test"]]))
        if fold == 0:
            nfreq = int(len(teacher.physics.frequencies))
            coef_reals = 15 * nfreq * 3 * 2  # complex (a, b, c) for indices 0-14 at every frequency
            lr_reals = int(teacher.model.coef_.size + teacher.model.intercept_.size + 2 * 15) if kind == "lr" else 0
            reals = coef_reals + lr_reals
            cost["teacher_" + kind] = dict(
                frequencies=nfreq, surrogate_real_parameters=coef_reals, classifier_real_parameters=lr_reals,
                total_real_parameters=reals, bytes_float32=4 * reals, bytes_float32_plus_freq_table=4 * reals + 2 * nfreq,
                # per index: two complex products (4 real each), one complex division (about 8 real
                # multiplications + 2 real divisions) and one magnitude (2 multiplications + 1 square root)
                real_multiplications=15 * (4 + 4 + 8 + 2) + (15 + 15 * 15 if kind == "lr" else 0),
                real_divisions=15 * 2, square_roots=15, inputs_used="f, g2Re, g2Im",
                note="Analytic operation counts, not on-board measurements.")
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "teacher_per_fold.csv", index=False)
(OUT / "teacher_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf8")
test = frame[frame.part == "test"]
print(test.groupby("model")[["acc", "ba", "mr"]].agg(["mean", lambda s: s.std(ddof=1)]).round(3).to_string())
print(json.dumps(cost, indent=2))
print(f"Wrote {OUT / 'teacher_per_fold.csv'} and teacher_cost.json. Next: python experiments/aggregate.py")
