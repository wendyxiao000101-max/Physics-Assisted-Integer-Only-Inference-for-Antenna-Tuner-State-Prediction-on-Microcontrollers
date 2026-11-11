# -*- coding: utf-8 -*-
"""从原始逐折 CSV 重建 C5 的全部配对统计表（T23 / T23b / T24 / T24b）。

    python C5\\scripts\\c5_stats.py --root C5

回应核查意见 §三.2 与 §四.8：
  * 95 % 置信区间改用 **t 分布（df = n − 1 = 4）** 的临界值 2.776，
    与表中的配对 t 检验一致；此前误用 1.96，区间偏窄约 30 %，
    还出现过"区间不跨零、p > 0.05"的自相矛盾。
  * **删除"两骨干合并 n = 10"**：两组差值共享同一组五折划分，不是 10 个独立样本。
  * 逐折差值、配对 t、Wilcoxon **全部保留**，不只挑显著的那个。
  * 表头标明：五折交叉验证的训练集相互重叠，折间并不独立，
    这些 p 值只能作为描述性证据，不能当作强显著性。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {"top1": "test_accuracy", "ba": "test_balanced_accuracy",
           "min_recall": "test_minimum_class_recall"}

# (对比名, a, b, 口径说明)——口径说明直接写进表里，防止被读成"只差一个变量"
CONTRASTS_T23 = [
    ("residual vs plain MLP (near-matched parameter budget)",
     "residual32_gelu_relu_hybrid", "mlp64_32_gelu_hybrid",
     "整体架构比较：宽度 32 vs 64/32、线性层 4 vs 3、跳连有无同时不同；不能归因于跳连本身"),
    ("classifier head: hybrid vs linear (same MLP backbone)",
     "mlp64_32_gelu_hybrid", "mlp64_32_gelu",
     "同一 MLP 骨干，仅分类头不同"),
    ("residual hybrid vs MLP linear head",
     "residual32_gelu_relu_hybrid", "mlp64_32_gelu",
     "骨干与分类头同时不同"),
    ("MLP width 128-64 vs 64-32 (same head)",
     "mlp128_64_gelu_hybrid", "mlp64_32_gelu_hybrid",
     "同为混合头 MLP，仅宽度不同"),
    ("residual vs wider MLP 128-64",
     "residual32_gelu_relu_hybrid", "mlp128_64_gelu_hybrid",
     "整体架构比较，MLP 参数约为 3.6 倍"),
    ("residual vs narrower MLP 32-16",
     "residual32_gelu_relu_hybrid", "mlp32_16_gelu_hybrid",
     "整体架构比较，MLP 参数约为 0.35 倍"),
]
CONTRASTS_T24 = [
    ("residual: cosine vs hybrid head",
     "residual32_gelu_relu_cosine", "residual32_gelu_relu_hybrid", "同一残差骨干，仅分类头不同"),
    ("MLP 64-32: cosine vs hybrid head",
     "mlp64_32_gelu_cosine", "mlp64_32_gelu_hybrid", "同一 MLP 骨干，仅分类头不同"),
    ("cosine head: residual vs MLP 64-32",
     "residual32_gelu_relu_cosine", "mlp64_32_gelu_cosine",
     "整体架构比较（同上，不能归因于跳连本身）"),
]


CONTRASTS_T27 = [
    ("skip connection only: residual vs no-skip",
     "residual32_gelu_relu_hybrid", "residual32_noskip_gelu_relu_hybrid",
     "唯一差别为跳连：同宽度、同层数、同参数 2,800、同种子初始权重逐元素相同"),
]


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    table = (frame.groupby("architecture", as_index=False)
             .agg(n=("seed", "count"), top1=("test_accuracy", "mean"), top1_std=("test_accuracy", "std"),
                  ba=("test_balanced_accuracy", "mean"), min_recall=("test_minimum_class_recall", "mean"),
                  params=("params", "first"), macs=("macs", "first"), best_epoch=("best_epoch", "mean")))
    for column in ("top1", "top1_std", "ba", "min_recall"):
        table[column] *= 100
    table["per_fold_top1"] = [" ".join(f"{v * 100:.3f}" for v in
                                       frame[frame.architecture == a].sort_values("seed").test_accuracy)
                              for a in table.architecture]
    if "hybrid_alpha" in frame:
        table["alpha_per_fold"] = [" ".join(f"{v:.6f}" for v in
                                            frame[frame.architecture == a].sort_values("seed").hybrid_alpha)
                                   if a.endswith("_hybrid") else "" for a in table.architecture]
    return table.sort_values("top1", ascending=False)


def contrasts(frame: pd.DataFrame, spec) -> pd.DataFrame:
    panels = {k: frame.pivot_table(index="seed", columns="architecture", values=v) * 100
              for k, v in METRICS.items()}
    rows = []
    for label, a, b, note in spec:
        if a not in panels["top1"] or b not in panels["top1"]:
            continue
        row = {"contrast": label, "arm_a": a, "arm_b": b, "note": note}
        for key, panel in panels.items():
            paired = panel[[a, b]].dropna()
            delta = (paired[a] - paired[b]).to_numpy()
            n = len(delta)
            se = delta.std(ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, n - 1)
            row[f"{key}_n"] = n
            row[f"{key}_delta"] = delta.mean()
            row[f"{key}_ci95_lo"] = delta.mean() - t_crit * se
            row[f"{key}_ci95_hi"] = delta.mean() + t_crit * se
            row[f"{key}_wins"] = int((delta > 0).sum())
            row[f"{key}_losses"] = int((delta < 0).sum())
            row[f"{key}_t_p"] = float(stats.ttest_rel(paired[a], paired[b])[1])
            row[f"{key}_wilcoxon_p"] = float(stats.wilcoxon(paired[a], paired[b]).pvalue)
            row[f"{key}_per_fold"] = " ".join(f"{v:+.3f}" for v in delta)
        row["t_crit_df4"] = stats.t.ppf(0.975, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===  （95 % CI 用 t(df=n−1)；逐折配对；五折训练集重叠，p 值仅作描述）")
    for _, r in table.iterrows():
        flag = ""
        crosses = r["top1_ci95_lo"] <= 0 <= r["top1_ci95_hi"]
        if (r["top1_t_p"] < 0.05) == crosses:
            flag = "  ← 区间与 p 值不一致，请检查"
        print(f"  {r['contrast']:52s} Δ {r['top1_delta']:+7.3f} "
              f"[{r['top1_ci95_lo']:+.3f}, {r['top1_ci95_hi']:+.3f}]  "
              f"t p={r['top1_t_p']:.4f}  W p={r['top1_wilcoxon_p']:.4f}  "
              f"{r['top1_wins']}/{r['top1_losses']}{flag}")
        print(f"  {'':52s} 逐折 {r['top1_per_fold']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("C5"))
    args = ap.parse_args()
    raw, out = args.root / "raw", args.root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    head = pd.read_csv(raw / "c5_head_lr15_raw_runs.csv")
    cosine = pd.read_csv(raw / "c5_cosine_raw_runs.csv")

    summary(head).to_csv(out / "T23_c5_architecture_head.csv", index=False, encoding="utf-8-sig")
    t23b = contrasts(head, CONTRASTS_T23)
    t23b.to_csv(out / "T23b_c5_paired_contrasts.csv", index=False, encoding="utf-8-sig")
    summary(cosine).to_csv(out / "T24_cosine_vs_hybrid_head.csv", index=False, encoding="utf-8-sig")
    t24b = contrasts(cosine, CONTRASTS_T24)
    t24b.to_csv(out / "T24b_cosine_paired_contrasts.csv", index=False, encoding="utf-8-sig")

    show("T23b 架构 / 分类头", t23b)
    show("T24b 纯余弦 vs 混合头", t24b)
    print(f"\n已写入 {out}/T23*.csv 与 T24*.csv（不含两骨干合并的 n=10 检验）")

    # ---- 同骨干去跳连（run_c5_noskip.bat 跑完、把 raw_runs.csv 拷进 raw/ 后才有）
    noskip = raw / "c5_noskip_raw_runs.csv"
    if noskip.exists():
        frame = pd.read_csv(noskip)
        anchor = frame[frame.architecture == "residual32_gelu_relu_hybrid"].sort_values("seed")
        ref = head[head.architecture == "residual32_gelu_relu_hybrid"].sort_values("seed")
        if len(anchor) == len(ref) == 5:
            drift = np.abs(anchor.test_accuracy.to_numpy() - ref.test_accuracy.to_numpy()).max() * 100
            print(f"\n锚点复现：本块 residual 与 results_c5_head_lr15 逐折最大差 {drift:.4f} pp"
                  + ("" if drift < 1e-9 else "  ← 不为 0：GPU 非确定性，配对只用本块内部的 residual"))
        summary(frame).to_csv(out / "T27_noskip.csv", index=False, encoding="utf-8-sig")
        t27b = contrasts(frame, CONTRASTS_T27)
        t27b.to_csv(out / "T27b_noskip_paired.csv", index=False, encoding="utf-8-sig")
        show("T27b 同骨干只去跳连（CE 配方）", t27b)
    else:
        print(f"\n（未找到 {noskip}：跑完 run_c5_noskip.bat 后把 results_c5_noskip\\ablation\\raw_runs.csv 拷成它）")
    noskip_v5, b0 = raw / "c5_noskip_v5_raw_runs.csv", raw / "teacher_B0_clean_raw_runs.csv"
    if noskip_v5.exists() and b0.exists():
        frame = pd.concat([pd.read_csv(noskip_v5), pd.read_csv(b0)], ignore_index=True)
        t27c = contrasts(frame, CONTRASTS_T27)
        t27c.to_csv(out / "T27c_noskip_v5_paired.csv", index=False, encoding="utf-8-sig")
        show("T27c 同骨干只去跳连（ALA 主配方，对照 B0_clean）", t27c)


if __name__ == "__main__":
    main()
