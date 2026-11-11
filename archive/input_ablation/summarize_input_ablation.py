# -*- coding: utf-8 -*-
"""汇总 train.py --mode ablate 产出的逐折结果，输出论文可直接用的输入消融表。

    python summarize_input_ablation.py <results_input_ablation_A\\ablation\\raw_runs.csv> ^
        [<...B\\ablation\\raw_runs.csv>] --out tables

对每个输入空间给出：逐折成绩、均值 ± 标准差、与 raw5 的配对差、配对 t 与 Wilcoxon、
参数量与 MACs。所有检验都是**逐折配对**的——同一折里 raw5 与另一个输入空间用的是
完全相同的划分，所以配对检验才是正确的口径。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "raw5"
PROTOCOL_LABEL = {"random": "随机 5 种子", "sweep_group": "扫描五折"}


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["block"] = path.parent.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired(frame: pd.DataFrame, column: str, by_block: bool = False) -> pd.DataFrame:
    """按协议配对。

    默认把所有传入的 raw_runs.csv 当成一个池子（分批跑但配方相同时就该这样），
    这样第二批里没有 raw5 也能和第一批的 raw5 逐折配对。加 --split-by-block
    才会按来源目录分开统计。
    """
    keys = ["block", "protocol"] if by_block else ["protocol"]
    rows = []
    for key, group in frame.groupby(keys):
        block, protocol = key if by_block else ("pooled", key)
        wide = group.pivot_table(index="seed", columns="feature", values=column)
        if REFERENCE not in wide.columns:
            continue
        base = wide[REFERENCE]
        for feature in wide.columns:
            delta = wide[feature] - base
            row = {"block": block, "protocol": protocol, "feature": feature,
                   "n": int(wide[feature].notna().sum()),
                   "mean": wide[feature].mean() * 100,
                   "std": wide[feature].std(ddof=1) * 100,
                   "delta_vs_raw5_pp": delta.mean() * 100,
                   "delta_std_pp": delta.std(ddof=1) * 100,
                   "wins_vs_raw5": int((delta > 0).sum()),
                   "per_fold": " ".join(f"{v*100:.3f}" for v in wide[feature])}
            if feature != REFERENCE and row["n"] > 1 and delta.std(ddof=1) > 0:
                try:
                    from scipy.stats import ttest_rel, wilcoxon
                    row["paired_t_p"] = float(ttest_rel(wide[feature], base, nan_policy="omit")[1])
                    row["wilcoxon_p"] = float(wilcoxon(wide[feature], base).pvalue)
                except Exception:                                      # noqa: BLE001
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("tables"))
    ap.add_argument("--metric", default="test_accuracy")
    ap.add_argument("--split-by-block", action="store_true",
                    help="按来源目录分开统计（默认合并，适用于分批跑同一配方）")
    args = ap.parse_args()

    frame = load(args.raw_runs)
    frame = frame.drop_duplicates(subset=["feature", "protocol", "seed"], keep="last")
    args.out.mkdir(parents=True, exist_ok=True)
    print("每个输入空间的完成折数：")
    print(frame.groupby(["feature", "protocol"]).size().to_string())

    size = (frame.groupby("feature", as_index=False)
            .agg(input_dim=("input_dim", "first"), params=("params", "first"),
                 macs=("macs", "first")))

    for metric, tag in ((args.metric, "top1"),
                        ("test_balanced_accuracy", "balanced"),
                        ("test_minimum_class_recall", "min_recall")):
        if metric not in frame.columns:
            continue
        table = paired(frame, metric, args.split_by_block).merge(size, on="feature", how="left")
        path = args.out / f"T15_input_ablation_{tag}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        if tag != "top1":
            continue
        for (block, protocol), group in table.groupby(["block", "protocol"]):
            print(f"\n===== {block}   {PROTOCOL_LABEL.get(protocol, protocol)} "
                  f"(n={int(group['n'].iloc[0])}) =====")
            print(f"{'feature':14s} {'dim':>4s} {'params':>7s} {'MACs':>6s} "
                  f"{'Top-1':>17s} {'Δ vs raw5':>12s} {'胜/负':>7s} {'配对 t':>9s}")
            for _, r in group.sort_values("input_dim", ascending=False).iterrows():
                p = f"{r['paired_t_p']:.4f}" if not pd.isna(r.get("paired_t_p", np.nan)) else "—"
                print(f"{r['feature']:14s} {int(r['input_dim']):4d} {int(r['params']):7d} "
                      f"{int(r['macs']):6d} {r['mean']:10.4f}±{r['std']:5.4f} "
                      f"{r['delta_vs_raw5_pp']:+11.4f} {r['wins_vs_raw5']:4d}/{int(r['n'])-r['wins_vs_raw5']:<3d} {p:>9s}")
    print(f"\n已写入 {args.out}/T15_input_ablation_*.csv")
    print("提示：Δ 与配对检验都是逐折配对的（同折同划分），不要用独立样本 t 检验。")


if __name__ == "__main__":
    main()
