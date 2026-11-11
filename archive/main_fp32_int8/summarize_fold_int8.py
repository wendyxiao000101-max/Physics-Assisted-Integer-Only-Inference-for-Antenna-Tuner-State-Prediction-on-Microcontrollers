# -*- coding: utf-8 -*-
"""汇总各折的 FP32 / INT8 留出集成绩，输出论文可以直接用的表格。

    python summarize_fold_int8.py model/foldexp
    python summarize_fold_int8.py model/foldexp --subdir int8_bal --compare int8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "minimum_class_recall")


def collect(root: Path, subdir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(f"*/{subdir}/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": path.parent.parent.name, "protocol": m["protocol"], "seed": m["seed"],
               "rows_test": m["rows_test"], "alpha": m["hybrid_alpha"],
               "selection": m.get("selection_criterion", "accuracy"),
               "clipped": m["input_rows_clipped_by_train_range"],
               "acc_max": m["int32_accumulator_max"],
               "agreement": m["agreement_with_fp32_on_holdout"],
               "int8_backbone_acc": m["int8_backbone_holdout_accuracy"] * 100}
        for k in KEYS:
            row[f"fp32_{k}"] = m["fp32_holdout"][k] * 100
            row[f"int8_{k}"] = m["int8_holdout"][k] * 100
        row["cost_pp"] = row["int8_accuracy"] - row["fp32_accuracy"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"{root} 下没有找到 {subdir}/metrics.json")
    return pd.DataFrame(rows).sort_values(["protocol", "seed"]).reset_index(drop=True)


def block(frame: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)
    print(f"{'run':22s} {'FP32 acc':>10s} {'INT8 acc':>10s} {'代价':>9s} "
          f"{'INT8 bal':>9s} {'INT8 最差类':>12s} {'一致率':>8s}")
    for _, r in frame.iterrows():
        print(f"{r['run']:22s} {r['fp32_accuracy']:9.4f}% {r['int8_accuracy']:9.4f}% "
              f"{r['cost_pp']:+8.4f} {r['int8_balanced_accuracy']:8.4f}% "
              f"{r['int8_minimum_class_recall']:11.4f}% {r['agreement']*100:7.4f}%")
    for protocol, group in frame.groupby("protocol"):
        label = "随机五种子" if protocol == "random" else "扫描五折"
        print(f"\n【{label}】n={len(group)}   选择准则={group['selection'].iloc[0]}")
        for tag, prefix in (("FP32", "fp32_"), ("INT8", "int8_")):
            cells = "  ".join(f"{k}={group[prefix+k].mean():.4f}±{group[prefix+k].std(ddof=1):.4f}"
                              for k in KEYS)
            print(f"  {tag}: {cells}")
        cost = group["cost_pp"]
        print(f"  量化代价: {cost.mean():+.4f} ± {cost.std(ddof=1):.4f} pp   "
              f"逐折 {'  '.join(f'{v:+.3f}' for v in cost)}")
        try:
            from scipy.stats import ttest_rel, wilcoxon
            print(f"  配对 t p={ttest_rel(group['int8_accuracy'], group['fp32_accuracy'])[1]:.4f}   "
                  f"Wilcoxon p={wilcoxon(group['int8_accuracy'], group['fp32_accuracy']).pvalue:.4f}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  （检验跳过：{exc}）")


def compare(new: pd.DataFrame, old: pd.DataFrame, new_tag: str, old_tag: str) -> None:
    merged = new.merge(old, on=["run", "protocol", "seed"], suffixes=("_new", "_old"))
    print("\n" + "=" * 100)
    print(f"选择准则对比：{old_tag} → {new_tag}   （骨干模型相同，唯一变量是检查点选择准则）")
    print("-" * 100)
    print(f"{'run':22s} {'旧 acc':>9s} {'新 acc':>9s} {'Δacc':>8s} | "
          f"{'旧最差类':>9s} {'新最差类':>9s} {'Δ最差类':>9s}")
    for _, r in merged.iterrows():
        print(f"{r['run']:22s} {r['int8_accuracy_old']:8.4f}% {r['int8_accuracy_new']:8.4f}% "
              f"{r['int8_accuracy_new']-r['int8_accuracy_old']:+7.4f} | "
              f"{r['int8_minimum_class_recall_old']:8.4f}% {r['int8_minimum_class_recall_new']:8.4f}% "
              f"{r['int8_minimum_class_recall_new']-r['int8_minimum_class_recall_old']:+8.4f}")
    for protocol, g in merged.groupby("protocol"):
        label = "随机五种子" if protocol == "random" else "扫描五折"
        d_acc = g["int8_accuracy_new"] - g["int8_accuracy_old"]
        d_bal = g["int8_balanced_accuracy_new"] - g["int8_balanced_accuracy_old"]
        d_min = g["int8_minimum_class_recall_new"] - g["int8_minimum_class_recall_old"]
        print(f"\n【{label}】Top-1 {d_acc.mean():+.4f} pp   平衡 {d_bal.mean():+.4f} pp   "
              f"最差类 {d_min.mean():+.4f} pp")
        print(f"  最差类标准差 {g['int8_minimum_class_recall_old'].std(ddof=1):.4f} → "
              f"{g['int8_minimum_class_recall_new'].std(ddof=1):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=Path("model/foldexp"))
    ap.add_argument("--subdir", default="int8")
    ap.add_argument("--compare", default=None, help="另一个子目录名，做并排对比")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    frame = collect(args.root, args.subdir)
    block(frame, f"结果：{args.subdir}")
    if args.compare:
        compare(frame, collect(args.root, args.compare), args.subdir, args.compare)

    csv = args.csv or Path(f"reports/int8_generalisation_{args.subdir}.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\n已写入 {csv}")
    print("注：FP32 与 INT8 走同一份划分；INT8 的训练、校准、选点只用 train+valid，test 仅评测时使用一次。")


if __name__ == "__main__":
    main()
