# -*- coding: utf-8 -*-
"""把"教师质量 vs 学生收益"各档位的逐折结果汇总成论文可直接引用的表与图。

    python summarize_teacher_quality.py results_teacher_*\\ablation\\raw_runs.csv ^
        [--teacher-test tables\\T20_teacher_same_protocol_*.csv] --out tables --figures figures

对每个档位给出：
  * 学生逐折测试 Top-1 / BA / 最小类召回，均值 ± 标准差；
  * 相对"无物理教师"基线的**逐折配对**收益 Δ，配对 t 与 Wilcoxon；
  * 教师自身的留出准确率（train.py 逐折日志里的 physics_teacher_validation_accuracy），
    以及（若提供 --teacher-test）教师在**与学生完全相同的测试行**上的准确率；
  * 教师质量与学生收益的相关：Pearson r、Spearman ρ、加权最小二乘斜率。

所有检验都是逐折配对的——同一折里各档位用的是完全相同的划分。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "A0_none"


def arm_of(path: Path) -> str:
    """results_teacher_B2_eta015\\ablation\\raw_runs.csv -> B2_eta015"""
    name = path.parent.parent.name
    return re.sub(r"^results_teacher_", "", name)


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["arm"] = arm_of(path)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["protocol"] == "sweep_group"].copy()
    return frame.drop_duplicates(subset=["arm", "protocol", "seed"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_runs", nargs="+", type=Path)
    ap.add_argument("--teacher-test", nargs="*", type=Path, default=[],
                    help="teacher_same_protocol.py 产出的 T20_*.csv，用来给出教师在测试行上的准确率")
    ap.add_argument("--out", type=Path, default=Path("tables"))
    ap.add_argument("--figures", type=Path, default=None)
    args = ap.parse_args()

    frame = load(args.raw_runs)
    print("各档位完成折数：")
    print(frame.groupby("arm").size().to_string())

    wide = frame.pivot_table(index="seed", columns="arm", values="test_accuracy")
    if BASELINE not in wide.columns:
        raise SystemExit(f"缺少基线档位 {BASELINE}（results_teacher_{BASELINE}），无法计算收益。")
    base = wide[BASELINE]

    # 教师在同折测试行上的准确率（可选）
    teacher_test: dict[str, pd.Series] = {}
    for path in args.teacher_test:
        table = pd.read_csv(path)
        table = table[table["protocol"] == "sweep_group"]
        variant = str(table["teacher_variant"].iloc[0])
        teacher_test[variant] = table.set_index("seed")["test_acc"]

    def match(arm: str) -> pd.Series | None:
        kind = "argmin" if "argmin" in arm else "lr"
        if arm == BASELINE:
            return None
        if "shuffle" in arm:
            key = [k for k in teacher_test if k.startswith(kind) and "shuffle" in k]
        elif (m := re.search(r"eta(\d+)", arm)):
            eta = f"{int(m.group(1)) / 10 ** (len(m.group(1)) - 1):g}"
            key = [k for k in teacher_test if k.startswith(kind) and f"eta{eta}" in k]
        else:
            key = [k for k in teacher_test if k.startswith(kind) and "clean" in k]
        return teacher_test[key[0]] if key else None

    rows = []
    for arm in sorted(wide.columns):
        group = frame[frame["arm"] == arm].set_index("seed")
        delta = wide[arm] - base
        row = {
            "arm": arm,
            "n": int(wide[arm].notna().sum()),
            "teacher_valid_acc": group["physics_teacher_validation_accuracy"].mean() * 100
            if "physics_teacher_validation_accuracy" in group else np.nan,
            "teacher_valid_acc_std": group["physics_teacher_validation_accuracy"].std(ddof=1) * 100
            if "physics_teacher_validation_accuracy" in group else np.nan,
            "student_top1": wide[arm].mean() * 100,
            "student_top1_std": wide[arm].std(ddof=1) * 100,
            "student_ba": group["test_balanced_accuracy"].mean() * 100,
            "student_min_recall": group["test_minimum_class_recall"].mean() * 100,
            "gain_pp": delta.mean() * 100,
            "gain_std_pp": delta.std(ddof=1) * 100,
            "wins": int((delta > 0).sum()),
            "per_fold": " ".join(f"{v * 100:.3f}" for v in wide[arm]),
        }
        series = match(arm)
        if series is not None:
            row["teacher_test_acc"] = float(series.reindex(wide.index).mean())
        if arm != BASELINE and row["n"] > 1 and delta.std(ddof=1) > 0:
            from scipy.stats import ttest_rel, wilcoxon
            row["paired_t_p"] = float(ttest_rel(wide[arm], base, nan_policy="omit")[1])
            row["wilcoxon_p"] = float(wilcoxon(wide[arm], base).pvalue)
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "T21_teacher_quality_vs_student_gain.csv",
                 index=False, encoding="utf-8-sig")

    print("\n===== 教师质量 vs 学生收益（扫描五折，学生配方逐档位完全相同） =====")
    header = (f"{'arm':14s} {'教师留出':>9s} {'教师测试':>9s} {'学生 Top-1':>17s} "
              f"{'Δ vs 无物理':>13s} {'胜/负':>7s} {'配对 t':>9s}")
    print(header)
    for _, r in table.sort_values("teacher_valid_acc", na_position="first").iterrows():
        tv = f"{r['teacher_valid_acc']:8.3f}%" if not pd.isna(r["teacher_valid_acc"]) else "       —"
        tt = f"{r['teacher_test_acc']:8.3f}%" if not pd.isna(r.get("teacher_test_acc", np.nan)) else "       —"
        p = f"{r['paired_t_p']:.4f}" if not pd.isna(r.get("paired_t_p", np.nan)) else "—"
        print(f"{r['arm']:14s} {tv:>9s} {tt:>9s} {r['student_top1']:10.4f}±{r['student_top1_std']:5.4f} "
              f"{r['gain_pp']:+12.4f} {r['wins']:4d}/{int(r['n']) - r['wins']:<3d} {p:>9s}")

    # ---- 档位两两配对对比（这才是主统计量；n=5 折的配对检验远比 n=3 个档位的回归可靠）
    from scipy.stats import ttest_rel, wilcoxon
    metrics = {"top1": "test_accuracy", "ba": "test_balanced_accuracy",
               "min_recall": "test_minimum_class_recall"}
    panels = {tag: frame.pivot_table(index="seed", columns="arm", values=column) * 100
              for tag, column in metrics.items() if column in frame.columns}
    arms = list(wide.columns)
    contrasts = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            row = {"arm_a": a, "arm_b": b}
            for tag, panel in panels.items():
                if a not in panel or b not in panel:
                    continue
                delta = (panel[a] - panel[b]).dropna()
                row[f"{tag}_delta"] = delta.mean()
                row[f"{tag}_std"] = delta.std(ddof=1)
                row[f"{tag}_wins"] = int((delta > 0).sum())
                if len(delta) > 1 and delta.std(ddof=1) > 0:
                    row[f"{tag}_t_p"] = float(ttest_rel(panel[a].dropna(), panel[b].dropna())[1])
                    row[f"{tag}_w_p"] = float(wilcoxon(panel[a].dropna(), panel[b].dropna()).pvalue)
                if tag == "top1":
                    row["top1_per_fold"] = " ".join(f"{v:+.2f}" for v in delta)
            contrasts.append(row)
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(args.out / "T21c_teacher_arm_contrasts.csv", index=False, encoding="utf-8-sig")

    print("\n===== 档位两两配对对比（逐折配对，n = 折数） =====")
    print(f"{'对比':28s} {'ΔTop-1':>9s} {'胜/负':>7s} {'t p':>8s} "
          f"{'ΔBA':>9s} {'t p':>8s} {'Δmin-rec':>10s} {'t p':>8s}")
    for _, r in contrast.iterrows():
        def cell(tag, width=9, fmt=".4f"):
            v, p = r.get(f"{tag}_delta", np.nan), r.get(f"{tag}_t_p", np.nan)
            return (f"{v:+{width}{fmt}}" if not pd.isna(v) else " " * width,
                    f"{p:8.4f}" if not pd.isna(p) else "       —")
        d1, p1 = cell("top1")
        d2, p2 = cell("ba")
        d3, p3 = cell("min_recall", 10)
        wins = int(r.get("top1_wins", 0))
        n = int(table.loc[table["arm"] == r["arm_a"], "n"].iloc[0])
        print(f"{r['arm_a'] + ' − ' + r['arm_b']:28s} {d1} {wins:4d}/{n - wins:<3d} {p1} "
              f"{d2} {p2} {d3} {p3}")

    # ---- 相关性：次要统计量，只在趋势确实单调时才给拟合斜率
    fit = table[table["arm"] != BASELINE].dropna(subset=["teacher_valid_acc"])
    monotone = False
    if len(fit) >= 3:
        from scipy.stats import pearsonr, spearmanr
        x, y = fit["teacher_valid_acc"].to_numpy(), fit["gain_pp"].to_numpy()
        r, pr = pearsonr(x, y)
        rho, ps = spearmanr(x, y)
        monotone = abs(rho) >= 0.9
        print(f"\n教师留出准确率 ↔ 学生收益（只有 n={len(fit)} 个档位，作次要参考）：")
        print(f"  Pearson r = {r:+.4f} (p = {pr:.4f})   Spearman ρ = {rho:+.4f} (p = {ps:.4f})")
        record = {"n_arms": len(fit), "pearson_r": r, "pearson_p": pr,
                  "spearman_rho": rho, "spearman_p": ps, "monotone": monotone}
        if monotone:
            slope, intercept = np.polyfit(x, y, 1)
            record |= {"slope_pp_per_pct": slope, "intercept_pp": intercept}
            print(f"  线性拟合：学生收益 (pp) = {slope:+.4f} × 教师准确率 (%) {intercept:+.4f}")
        else:
            print("  |ρ| < 0.9，关系不是单调的——**不要**报线性斜率，"
                  "按上面的两两对比陈述（阶跃而非斜坡）。")
        pd.DataFrame([record]).to_csv(args.out / "T21b_teacher_quality_correlation.csv",
                                      index=False, encoding="utf-8-sig")
    else:
        print(f"\n只有 {len(fit)} 个带教师的档位，至少要 3 个才能给相关性。先跑完 Tier 1。")

    if args.figures is not None and len(fit) >= 2:
        draw(table, fit, base, wide, args.figures, frame)
    print(f"\n已写入 {args.out}/T21_teacher_quality_vs_student_gain.csv")


def draw(table: pd.DataFrame, fit: pd.DataFrame, base: pd.Series,
         wide: pd.DataFrame, folder: Path, frame: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "font.size": 9, "axes.linewidth": 0.8})
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    S1_LIGHT = "#8fb8e8"

    def colour_of(arm: str) -> str:
        if "shuffle" in arm:
            return S3                      # 打乱频率列：物理形状对、内容错的零假设
        if "argmin" in arm:
            return S2                      # 纯物理规则教师
        if "eta" in arm:
            return S1_LIGHT                # 被加噪劣化的 lr 教师
        return S1                          # 干净的 lr 教师（交付版）

    # (a) 收益 vs 教师质量
    ax = axes[0]
    for _, r in fit.iterrows():
        ax.errorbar(r["teacher_valid_acc"], r["gain_pp"],
                    yerr=r["gain_std_pp"] / np.sqrt(max(r["n"], 1)),
                    fmt="o", ms=6, color=colour_of(r["arm"]), capsize=3, lw=1.0, zorder=3)
        ax.annotate(r["arm"].split("_", 1)[-1], (r["teacher_valid_acc"], r["gain_pp"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5, color="#444")
    if len(fit) >= 3:
        from scipy.stats import spearmanr
        if abs(spearmanr(fit["teacher_valid_acc"], fit["gain_pp"])[0]) >= 0.9:
            slope, intercept = np.polyfit(fit["teacher_valid_acc"], fit["gain_pp"], 1)
            span = np.linspace(fit["teacher_valid_acc"].min() - 3,
                               fit["teacher_valid_acc"].max() + 3, 50)
            ax.plot(span, slope * span + intercept, "--", lw=1.0, color="#888", zorder=1)
    ax.axhline(0, lw=0.9, color="#333", zorder=2)
    ax.set_xlim(36, 104)
    ax.set_xlabel("Teacher held-out accuracy (%)")
    ax.set_ylabel("Student Top-1 gain over\nno-physics baseline (pp)")
    ax.set_title("(a) Gain vs teacher quality", fontsize=9.5)
    ax.grid(axis="y", lw=0.4, color="#ddd", zorder=0)

    # (b) 三个指标相对基线的增益（零基准，避免截断坐标轴夸大差异）
    ax = axes[1]
    metrics = [("test_accuracy", "Top-1"), ("test_balanced_accuracy", "BA"),
               ("test_minimum_class_recall", "Min-class\nrecall")]
    arms = [a for a in table.sort_values("teacher_valid_acc")["arm"] if a != BASELINE]
    width = 0.8 / max(len(arms), 1)
    positions = np.arange(len(metrics))
    for index, arm in enumerate(arms):
        gains, errors = [], []
        for column, _ in metrics:
            if frame is None or column not in frame.columns:
                gains.append(np.nan); errors.append(0.0); continue
            panel = frame.pivot_table(index="seed", columns="arm", values=column) * 100
            delta = (panel[arm] - panel[BASELINE]).dropna()
            gains.append(delta.mean())
            errors.append(delta.std(ddof=1) / np.sqrt(max(len(delta), 1)))
        offset = (index - (len(arms) - 1) / 2) * width
        ax.bar(positions + offset, gains, width=width * 0.92, yerr=errors, capsize=2.5,
               color=colour_of(arm), zorder=2, error_kw={"lw": 0.8, "ecolor": "#555"},
               label=arm.split("_", 1)[-1])
    ax.axhline(0, lw=0.9, color="#333", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in metrics], fontsize=8)
    ax.set_ylabel("Gain over no-physics baseline (pp)")
    ax.set_title("(b) Only the correct physics helps", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=1,
              title="teacher", title_fontsize=7.5)
    ax.grid(axis="y", lw=0.4, color="#ddd", zorder=0)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(folder / "F13_teacher_quality_vs_student_gain.png", dpi=300)
    figure.savefig(folder / "F13_teacher_quality_vs_student_gain.pdf")
    print(f"已写入 {folder}/F13_teacher_quality_vs_student_gain.png")


if __name__ == "__main__":
    main()
