"""Summarise a QAT screening sweep: one table, one dataset load.

    python summarize_qat.py model/qat_v2_*            (glob expanded by the shell)
    python summarize_qat.py --prefix model/qat_v2_    (or let this script glob)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infer_int8 import Int8Model                       # noqa: E402
from iatopt.core import build_features, load_dataset, metrics   # noqa: E402

DATA = ROOT.parent / "esp32p4_iat_demo/input/data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument("--prefix", default=None, help="glob <prefix>* instead of listing dirs")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    directories = list(args.dirs)
    if args.prefix:
        directories += sorted(Path().glob(args.prefix + "*"))
    directories = [d for d in directories if (d / "model_int8.npz").exists()]
    if not directories:
        raise SystemExit("no directory with model_int8.npz found")

    bundle = load_dataset(DATA / "x_train.csv", DATA / "y_train.csv")
    rows = []
    for directory in directories:
        model = Int8Model(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        x = model.standardise(build_features(bundle.raw, model.meta["feature_spec"]))
        result = metrics(bundle.labels, model.predict(x))
        size = sum(v.size for k, v in model.d.items() if k.endswith("_w_int"))
        size += 4 * sum(v.size for k, v in model.d.items()
                        if k.endswith("_b") or k.endswith("_w_scale"))
        rows.append({
            "run": directory.name,
            "scheme": meta.get("qat_scheme", "minmax"),
            "epochs": meta.get("training_epochs", 0),
            "init": "fp32" if meta.get("qat_init_from") else "scratch",
            "kd": meta.get("qat_kd_weight", 0.0),
            "freeze": meta.get("qat_freeze_at", 0.0),
            "int8_acc": result["accuracy"] * 100,
            "int8_bal": result["balanced_accuracy"] * 100,
            "int8_f1": result["macro_f1"] * 100,
            "int8_min_recall": result["minimum_class_recall"] * 100,
            "kib": size / 1024,
        })

    rows.sort(key=lambda r: -r["int8_acc"])
    header = (f"{'run':34s} {'scheme':7s} {'ep':>4s} {'init':>7s} {'kd':>5s} {'frz':>5s} "
              f"{'INT8 acc':>9s} {'bal':>8s} {'macroF1':>8s} {'min-recall':>11s} {'KiB':>7s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run'][:34]:34s} {r['scheme']:7s} {r['epochs']:4d} {r['init']:>7s} "
              f"{r['kd']:5.2f} {r['freeze']:5.2f} {r['int8_acc']:8.4f}% {r['int8_bal']:7.4f}% "
              f"{r['int8_f1']:7.4f}% {r['int8_min_recall']:10.4f}% {r['kib']:7.3f}")
    print("\n参考  FP32 v5 全量拟合 = 99.1702%   旧 minmax QAT 347ep = 98.6080%   目标 >= 99%")

    if args.csv:
        import csv
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已写入 {args.csv}")


if __name__ == "__main__":
    main()
