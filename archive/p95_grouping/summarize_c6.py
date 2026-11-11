# -*- coding: utf-8 -*-
"""汇总 C6 所需的全部配对统计（不训练，只读已有结果）。

    python summarize_c6.py

输出（本目录）：
  C6_physics_gain.csv   物理增益：原分组 5 折、P95 5 折、两者合并 10 折
  C6_quant_cost.csv     量化代价 INT8-A − FP32：原分组、P95、合并 10 折、随机 5 次
  C6_per_fold.csv       全部逐折数字，便于核对

统计口径：
  * 单组 5 折：配对 t（df = 4）、95 % CI 用 t(0.975, 4) = 2.776、Wilcoxon 符号秩（n = 5 时最小 0.0625）。
  * 合并 10 折（两种分组各 5 折）：两组划分的训练集大量重叠，折间不独立，
    不能直接用 t(df = 9)。这里报 Nadeau–Bengio 修正的重复交叉验证 t：
    方差乘以 (1/k + n_test/n_train)，df = k − 1，k = 10；n_test/n_train 取各折实际行数的均值。
  * 全部标注 exploratory；原分组是开发折（方法在其上选定），P95 是未参与方法选择的另一种划分，
    但仍是同一份数据，不构成独立测试集。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
C7 = HERE.parent / "C7补充验证_20260925"
ROOT = HERE.parents[1]                                      # <PROJECT_ROOT>
T5 = ROOT / "论文实验数据" / "核对补充" / "T5_recovered_per_run.csv"
A_RANDOM = HERE.parent / "C8修订" / "A_per_run.csv"
SEEDS = [42, 43, 44, 45, 46]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ablation(path: Path) -> dict[int, dict]:
    rows = {int(r["seed"]): r for r in read_csv(path)}
    missing = [s for s in SEEDS if s not in rows]
    if missing:
        raise SystemExit(f"{path} 缺少 seed {missing}")
    return {s: {"acc": float(rows[s]["test_accuracy"]) * 100,
                "ba": float(rows[s]["test_balanced_accuracy"]) * 100,
                "mr": float(rows[s]["test_minimum_class_recall"]) * 100} for s in SEEDS}


def paired(label: str, diff, note: str = "", ratio: float | None = None) -> dict:
    d = np.asarray(diff, dtype=float)
    n = len(d)
    sd = d.std(ddof=1)
    row = {"contrast": label, "n": n, "mean_pp": d.mean(), "sd_pp": sd,
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
           "per_fold": " ".join(f"{v:+.3f}" for v in d), "note": note}
    if ratio is None:                                        # 单组：普通配对 t
        se = sd / np.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        row.update(method="paired t, df=n-1", ci_lo=d.mean() - crit * se, ci_hi=d.mean() + crit * se,
                   t_p=float(stats.ttest_1samp(d, 0.0).pvalue))
    else:                                                    # 合并：Nadeau–Bengio 修正
        se = np.sqrt((1.0 / n + ratio) * sd ** 2)
        crit = stats.t.ppf(0.975, n - 1)
        t = d.mean() / se
        row.update(method=f"Nadeau-Bengio corrected t, df={n - 1}, n_test/n_train={ratio:.3f}",
                   ci_lo=d.mean() - crit * se, ci_hi=d.mean() + crit * se,
                   t_p=float(2 * stats.t.sf(abs(t), n - 1)))
    nonzero = d[d != 0]
    # 合并 10 折时折间不独立，Wilcoxon 没有对应的修正版本，不报，避免被误读成 p = 0.002 之类
    row["wilcoxon_p"] = (float(stats.wilcoxon(nonzero).pvalue)
                         if ratio is None and len(nonzero) >= 1 else float("nan"))
    return row


def main() -> int:
    per_fold, physics, quant = [], [], []

    # ---------------- 物理增益
    p95_v5 = ablation(C7 / "training" / "ablation" / "raw_runs.csv")
    p95_off = ablation(HERE / "training" / "ablation" / "raw_runs.csv")
    t5 = read_csv(T5)
    orig = {}
    for config in ("baseline_champion", "v5_final"):
        rows = {int(r["seed"]): float(r["test_accuracy"]) * 100 for r in t5
                if r["config"] == config and r["protocol"] == "sweep_group"}
        orig[config] = [rows[s] for s in SEEDS]
    d_orig = np.array(orig["v5_final"]) - np.array(orig["baseline_champion"])
    for metric in ("acc", "ba", "mr"):
        d = np.array([p95_v5[s][metric] - p95_off[s][metric] for s in SEEDS])
        physics.append(paired(f"P95: v5 − 同配方无物理 [{metric}]", d,
                              "未参与方法选择的划分；对照与 v5 仅差物理项"))
        for s, v in zip(SEEDS, d):
            per_fold.append({"table": "physics", "grouping": "P95", "seed": s, "metric": metric,
                             "a": p95_v5[s][metric], "b": p95_off[s][metric], "diff": v})
    physics.insert(0, paired("原分组: v5 − 无物理基线（T5 冠军配方）[acc]", d_orig,
                             "开发折；基线为早期冠军配方，不只差物理项"))
    for s, a, b in zip(SEEDS, orig["v5_final"], orig["baseline_champion"]):
        per_fold.append({"table": "physics", "grouping": "Original", "seed": s, "metric": "acc",
                         "a": a, "b": b, "diff": a - b})

    # ---------------- 量化代价（INT8-A − FP32）
    cls = read_csv(C7 / "classification_folds.csv")
    by = {}
    for r in cls:
        by[(r["grouping"], int(r["fold"]), r["model"])] = r
    d_group, sizes = {}, []
    for grouping in ("Original", "P95"):
        diffs = []
        for fold in range(5):
            fp, q = by[(grouping, fold, "FP32")], by[(grouping, fold, "INT8-A")]
            diff = float(q["Acc"]) - float(fp["Acc"])
            diffs.append(diff)
            sizes.append(int(fp["n"]))
            per_fold.append({"table": "quant", "grouping": grouping, "seed": 42 + fold, "metric": "acc",
                             "a": float(q["Acc"]), "b": float(fp["Acc"]), "diff": diff})
        d_group[grouping] = np.array(diffs)
        quant.append(paired(f"{grouping}: INT8-A − FP32 [acc]", diffs,
                            "P95 第 2 折（seed 43）折叠门限放宽到 0.006，已在正文披露" if grouping == "P95" else "开发折"))
    total = 96768
    ratio = float(np.mean(sizes) / (0.8125 * (total - np.mean(sizes))))   # 训练 = 非测试部分的 81.25 %
    quant.append(paired("合并 10 折（原分组 + P95）: INT8-A − FP32 [acc]",
                        np.r_[d_group["Original"], d_group["P95"]], "两种划分，训练集重叠", ratio=ratio))
    if A_RANDOM.exists():
        rnd = [float(r["delta"]) for r in read_csv(A_RANDOM) if r["protocol"] == "random"]
        quant.append(paired("随机 5 次: INT8-A − FP32 [acc]", rnd, "随机划分，保持 n = 5"))

    physics.append(paired("合并 10 折（原分组 + P95）: v5 − 无物理 [acc]",
                          np.r_[d_orig, [p95_v5[s]["acc"] - p95_off[s]["acc"] for s in SEEDS]],
                          "两组基线定义不同（原：冠军配方；P95：同配方关物理），仅作描述", ratio=ratio))

    # ---------------- 输出
    for name, rows in (("C6_physics_gain.csv", physics), ("C6_quant_cost.csv", quant),
                       ("C6_per_fold.csv", per_fold)):
        with (HERE / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def show(title, rows):
        print(f"\n=== {title} ===")
        for r in rows:
            print(f"  {r['contrast']:46s} n={r['n']:2d}  Δ {r['mean_pp']:+.3f} ± {r['sd_pp']:.3f}  "
                  f"CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  t p={r['t_p']:.4f}  W p={r['wilcoxon_p']:.4f}  "
                  f"{r['wins']}/{r['losses']}")
            print(f"  {'':46s} 逐折 {r['per_fold']}")
    show("物理增益（pp）", physics)
    show("量化代价 INT8-A − FP32（pp）", quant)
    print(f"\n合并 10 折的修正系数 n_test/n_train = {ratio:.3f}；全部结果为 exploratory。")
    print(f"已写入 {HERE / 'C6_physics_gain.csv'}、C6_quant_cost.csv、C6_per_fold.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
