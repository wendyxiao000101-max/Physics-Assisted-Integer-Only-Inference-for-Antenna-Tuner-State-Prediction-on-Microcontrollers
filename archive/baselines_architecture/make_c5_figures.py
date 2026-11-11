# -*- coding: utf-8 -*-
"""C5 的 Pareto 图 F14（v2：真实编码体积、按验证集选前沿）。

    python C5\\scripts\\make_c5_figures.py --root C5

读：
  * tables/T22v2_baseline_pareto.csv   baseline_pareto_v2.py 的输出（必需）
  * tables/T22c_gbm_ext.csv            baseline_gbm_ext.py 的输出（可选，XGBoost/LightGBM）
  * tables/T22v2_budget_table.csv / T22c_budget_with_gbm.csv（打印预算表）

与 v1 的区别（对应核查意见 §三.6 / §四.6）：
  * 横轴是**真实编码字节**：树 / GBDT 为 tree_codec 编码后的长度（含文件头与偏移表），
    精度也是从同一份字节推理得到的；kNN 计入训练样本与标签，逻辑回归为 float32 参数。
  * 前沿按**验证集**精度构造（字节递增、valid 精度递增者入选），画出的是这些配置的 test 精度。
  * 不再画 CE 消融块的网络点（那是另一套训练配方，不能与主实验同图比较）。
  * 只画两个网络点，均来自主实验（tables/T1_main_results.csv，sweep_group 五折）：
      交付 INT8     4,656 B（实测 blob 大小）            94.422 ± 0.881 %
      FP32 折叠后   11,136 B（2,784 参数 × 4 B，仅权重）  95.084 ± 1.299 %
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INT8 = {"bytes": 4656, "top1": 94.422072, "std": 0.880965}
# 逐折 INT8 test Top-1（tables/T3_quantisation_cost_per_fold.csv，sweep_group seeds 42–46）
INT8_PER_FOLD = {42: 95.55851882244332, 43: 95.037372247928, 44: 93.45499663961124,
                 45: 93.72054048392164, 46: 94.33893352812272}
FP32 = {"bytes": 2784 * 4, "top1": 95.083936, "std": 1.298742}
FAMILY = {"extra_trees": ("Extra Trees", "o"), "random_forest": ("Random Forest", "s"),
          "hist_gb": ("Hist. GBDT", "^"), "decision_tree": ("Decision tree", "v"),
          "xgboost": ("XGBoost", "<"), "lightgbm": ("LightGBM", ">"),
          "knn": ("k-NN", "D"), "logistic_regression": ("Logistic regression", "P")}


def valid_frontier(table: pd.DataFrame) -> pd.DataFrame:
    """按字节递增扫描，valid 精度创新高者入选；返回这些配置（画 test 精度）。"""
    points = table.sort_values(["bytes_max", "valid_acc"], ascending=[True, False]).reset_index(drop=True)
    best, keep = -np.inf, []
    for index, row in points.iterrows():
        if row["valid_acc"] > best:
            best = row["valid_acc"]
            keep.append(index)
    return points.loc[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("C5"))
    args = ap.parse_args()
    tables = args.root / "tables"
    parts = [pd.read_csv(tables / "T22v2_baseline_pareto.csv")]
    if (tables / "T22c_gbm_ext.csv").exists():
        parts.append(pd.read_csv(tables / "T22c_gbm_ext.csv"))
    table = pd.concat(parts, ignore_index=True)
    if "bytes_max" not in table:
        table["bytes_max"] = table["bytes"]
    table["bytes_max"] = table["bytes_max"].fillna(table["bytes"])
    front = valid_frontier(table)
    front.to_csv(tables / "T22v2_valid_frontier.csv", index=False, encoding="utf-8-sig")

    print("=== 按验证集构造的前沿（报告 test）===")
    for _, r in front.iterrows():
        print(f"  {r['config'] + ' [' + r['representation'] + ']':62s} {r['bytes_max']:11,.0f} B  "
              f"valid {r['valid_acc']:.3f}  test {r['test_acc']:.3f}±{r['test_acc_std']:.3f}")
    for name in ("T22c_budget_with_gbm.csv", "T22v2_budget_table.csv"):
        if (tables / name).exists():
            print(f"\n=== 预算表（{name}）===")
            print(pd.read_csv(tables / name).to_string(index=False))
            break
    paired_vs_int8(args.root, tables)
    print(f"\n网络（主实验）：INT8 {INT8['bytes']:,} B {INT8['top1']:.3f}±{INT8['std']:.3f}；"
          f"FP32 折叠 {FP32['bytes']:,} B {FP32['top1']:.3f}±{FP32['std']:.3f}")
    draw(table, front, args.root / "figures")


def paired_vs_int8(root: Path, tables: Path) -> None:
    """交付 INT8 与预算表选中的基线逐折配对（t(4) CI；五折训练集重叠，仅作描述）。"""
    try:
        from scipy import stats
        t_crit = lambda df: float(stats.t.ppf(0.975, df))              # noqa: E731
    except ImportError:                                                 # 没装 scipy 时只支持 n = 5
        t_crit = lambda df: {4: 2.7764451051977987}[df]                # noqa: E731
    raws = [pd.read_csv(root / "raw" / "baseline_pareto_v2_raw.csv")]
    if (root / "raw" / "gbm_ext_raw.csv").exists():
        raws.append(pd.read_csv(root / "raw" / "gbm_ext_raw.csv"))
    raw = pd.concat(raws, ignore_index=True)
    budget_file = tables / ("T22c_budget_with_gbm.csv" if (tables / "T22c_budget_with_gbm.csv").exists()
                            else "T22v2_budget_table.csv")
    rows = []
    for _, b in pd.read_csv(budget_file).iterrows():
        text = b.get("selected_by_valid")
        if not isinstance(text, str) or "[" not in text:
            continue
        config, representation = text.rsplit(" [", 1)
        runs = raw[(raw.config == config) & (raw.representation == representation.rstrip("]"))]
        runs = runs.set_index("seed").test_acc
        delta = np.array([INT8_PER_FOLD[k] - runs[k] for k in sorted(INT8_PER_FOLD)])
        half = t_crit(len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append({"budget": b["budget"], "scope": b.get("scope", "全部基线"), "baseline": text,
                     "int8_minus_baseline": delta.mean(), "ci95_lo": delta.mean() - half,
                     "ci95_hi": delta.mean() + half, "int8_wins": int((delta > 0).sum()),
                     "per_fold": " ".join(f"{v:+.3f}" for v in delta)})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "T22v2_paired_vs_int8.csv", index=False, encoding="utf-8-sig")
    print("\n=== 交付 INT8 − 预算内最优基线（逐折配对，t(4) CI，仅作描述）===")
    for _, r in out.iterrows():
        print(f"  {r['budget']:30s} {r['baseline']:58s} Δ {r['int8_minus_baseline']:+.3f} "
              f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]  INT8 胜 {r['int8_wins']}/5")


def draw(table: pd.DataFrame, front: pd.DataFrame, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 9, "axes.linewidth": 0.8})
    S1, S2 = "#2a78d6", "#eb6834"
    folder.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6.6, 4.1))

    for family, group in table.groupby("model"):
        _, marker = FAMILY.get(family, (family, "o"))
        ax.scatter(group["bytes"], group["test_acc"], s=9, alpha=0.30, marker=marker,
                   color="#9aa5b1", linewidths=0, zorder=2)
    ax.plot(front["bytes_max"], front["test_acc"], "-", lw=1.4, color="#5b6670", zorder=3,
            label="Non-neural frontier (selected on validation; test shown; x = largest fold)")

    ax.annotate("", xy=(INT8["bytes"], INT8["top1"]), xytext=(FP32["bytes"], FP32["top1"]),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=S2, shrinkA=6, shrinkB=8), zorder=6)
    ax.scatter([FP32["bytes"]], [FP32["top1"]], s=46, marker="o", facecolor="white", edgecolor=S1,
               linewidths=1.5, zorder=7, label="FP32 folded, 2,784 params x 4 B (weights only)")
    ax.scatter([INT8["bytes"]], [INT8["top1"]], s=190, marker="*", color=S2, edgecolor="white",
               linewidths=0.7, zorder=8, label="Deployed INT8 blob, 4,656 B")
    ratio = FP32["bytes"] / INT8["bytes"]
    ax.annotate(f"{ratio:.1f}$\\times$ smaller\n{INT8['top1'] - FP32['top1']:+.2f} pp",
                ((FP32["bytes"] * INT8["bytes"]) ** 0.5, FP32["top1"]), textcoords="offset points",
                xytext=(-10, 8), fontsize=7.4, color=S2, ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=9)
    for cap in (INT8["bytes"], FP32["bytes"]):
        ax.axvline(cap, lw=0.8, ls=":", color=S2, alpha=0.6, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Real encoded model size (bytes, log scale)")
    ax.set_ylabel("Grouped 5-fold test Top-1 (%)")
    low = max(40.0, float(np.floor(min(table["test_acc"].quantile(0.05), front["test_acc"].min()) / 5) * 5))
    ax.set_ylim(low, 99)
    ax.grid(lw=0.4, color="#e3e6e9", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(folder / f"F14_accuracy_size_pareto.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(figure)
    print(f"\n已写入 {folder}/F14_accuracy_size_pareto.png/.pdf")


if __name__ == "__main__":
    main()
