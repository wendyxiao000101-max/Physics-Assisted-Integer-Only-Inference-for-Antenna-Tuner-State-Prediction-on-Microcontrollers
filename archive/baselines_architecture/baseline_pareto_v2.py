# -*- coding: utf-8 -*-
"""C5 非神经基线 v2：精度与字节出自同一实现，配置按验证集选择。

    python C5\\scripts\\baseline_pareto_v2.py --data <数据目录> --foldexp model\\foldexp --root C5

相对 v1（baseline_pareto.py）的改动，逐条对应核查意见：

  §三.6  精度与体积同口径。每个森林配置报告两种**真实表示**：
           hard     叶存类别、多数投票          —— 字节 = tree_codec 编码后的真实长度
           soft_u8  叶存 15 个 uint8 概率、累加  —— 同上
         不再用 sklearn 的浮点软投票精度去配硬投票的字节数。
         GBDT 用 float32 叶值编码并**从字节推理**；单棵决策树 hard = soft。
         字节数含文件头与每棵树的偏移表（v1 只算了节点）。
  §四.6  公平性：一律**只在 train 上拟合**（v1 用了 train+valid）；
         Pareto 前沿与预算表**按 valid 均值选配置**，再报告 test。
         按 test 选出的"探索性前沿"另列一列，明确标注。
  预算判定按**五折中最大的字节数**（每一折的模型都要装得下），表中另列均值。
  §三.6  kNN / 逻辑回归：输入按 train 统计量标准化；kNN 字节计入每样本 1 B 标签。

森林 / 决策树的两种表示在扫描里用 `tree.apply` 精确模拟（与编码器逐样本等价），
代表性配置再**真正编码并从字节推理**，断言与模拟逐样本一致、字节数与公式一致。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_codec as tc                                             # noqa: E402

CLASSES = 15
INT8_BYTES, INT8_TOP1 = 4656, 94.422072          # tables/T1_main_results.csv, sweep_group INT8
FP32_FOLDED_BYTES, FP32_TOP1 = 2784 * 4, 95.083936


def build_raw5(raw: np.ndarray) -> np.ndarray:
    return np.column_stack((raw[:, 2], raw[:, 0], raw[:, 1], raw[:, 4], raw[:, 5]))


def configurations(quick: bool):
    trees = [1, 3, 10, 30, 100] if quick else [1, 2, 3, 5, 10, 20, 30, 60, 100, 200]
    depths = [4, 6, 8, 12, None] if quick else [3, 4, 5, 6, 8, 10, 12, 16, None]
    for n in trees:
        for d in depths:
            yield "extra_trees", {"n_estimators": n, "max_depth": d}
            yield "random_forest", {"n_estimators": n, "max_depth": d}
    for d in depths:
        yield "decision_tree", {"max_depth": d}
    for iters in ([10, 30, 100] if quick else [5, 10, 20, 50, 100, 200]):
        for leaves in ([8, 31] if quick else [4, 8, 15, 31]):
            yield "hist_gb", {"max_iter": iters, "max_leaf_nodes": leaves}
    for k in (1, 5, 15):
        yield "knn", {"n_neighbors": k}
    yield "logistic_regression", {}


def build(name, params, seed):
    if name == "extra_trees":
        return ExtraTreesClassifier(max_features=None, class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(l2_regularization=1e-3, random_state=seed, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **params)
    return LogisticRegression(max_iter=2000, C=10.0)


def forest_sizes(estimators, classes=CLASSES):
    internal = sum(int((e.tree_.children_left != -1).sum()) for e in estimators)
    leaves = sum(int((e.tree_.children_left == -1).sum()) for e in estimators)
    head = 8 + 4 * len(estimators)
    return {"hard": head + 5 * internal + 1 * leaves,
            "soft_u8": head + 5 * internal + (1 + classes) * leaves,
            "internal": internal, "leaves": leaves}


def forest_predictions(estimators, x, classes=CLASSES):
    """与 tree_codec 的 hard / soft_u8 解码逐样本等价的快速模拟。"""
    hard = np.zeros((len(x), classes), dtype=np.int64)
    soft = np.zeros((len(x), classes), dtype=np.int64)
    rows = np.arange(len(x))
    for e in estimators:
        leaf = e.tree_.apply(x.astype(np.float32))
        value = e.tree_.value[:, 0, :classes]
        total = value.sum(axis=1, keepdims=True)
        dist = np.divide(value, total, out=np.zeros_like(value), where=total > 0)
        hard[rows, dist[leaf].argmax(1)] += 1
        soft += np.round(dist[leaf] * 255).astype(np.int64)
    return hard.argmax(1), soft.argmax(1)


def score(y, pred):
    return float(np.mean(pred == y) * 100), float(balanced_accuracy_score(y, pred) * 100)


def evaluate(name, params, seed, x, y, train, valid, test):
    """返回若干行：每种表示一行（repr, bytes, valid/test 精度）。"""
    model = build(name, params, seed)
    rows = []
    if name in ("knn", "logistic_regression"):
        mean, std = x[train].mean(0), x[train].std(0)
        std[std == 0] = 1.0
        z = (x - mean) / std
        model.fit(z[train], y[train])
        pv, pt = model.predict(z[valid]), model.predict(z[test])
        if name == "knn":
            size = 8 + 40 + len(train) * (5 * 2 + 1)      # 头 + 标准化参数 + 每样本 int16×5 与 u8 标签
        else:
            size = 8 + 40 + (5 + 1) * CLASSES * 4
        rows.append(("float", size, pv, pt))
    elif name == "hist_gb":
        model.fit(x[train], y[train])
        blob = tc.encode(model, tc.GBDT_F32)
        rows.append(("gbdt_f32", len(blob), tc.predict(blob, x[valid]), tc.predict(blob, x[test])))
    else:
        model.fit(x[train], y[train])
        estimators = getattr(model, "estimators_", [model])
        sizes = forest_sizes(estimators)
        hv, sv = forest_predictions(estimators, x[valid])
        ht, st = forest_predictions(estimators, x[test])
        rows.append(("hard", sizes["hard"], hv, ht))
        if name != "decision_tree":                   # 单棵树 soft 与 hard 预测相同，只是更大
            rows.append(("soft_u8", sizes["soft_u8"], sv, st))
    out = []
    for representation, size, pv, pt in rows:
        va, vb = score(y[valid], pv)
        ta, tb = score(y[test], pt)
        out.append({"representation": representation, "bytes": size,
                    "valid_acc": va, "valid_ba": vb, "test_acc": ta, "test_ba": tb})
    return out


def verify_representatives(table, x, y, folds, n=6):
    """挑预算附近与最高精度的配置，真编码、从字节推理，断言与扫描模拟一致。"""
    picks = []
    for cap in (INT8_BYTES, FP32_FOLDED_BYTES, 10 ** 12):
        within = table[(table.bytes_max <= cap) & table.representation.isin(["hard", "soft_u8", "gbdt_f32"])]
        if len(within):
            picks.append(within.sort_values(["valid_acc", "bytes_max"], ascending=[False, True]).iloc[0])
    extra = table[table.representation == "gbdt_f32"].sort_values("valid_acc", ascending=False).head(1)
    picks += [r for _, r in extra.iterrows()]
    soft = table[table.representation == "soft_u8"].sort_values("valid_acc", ascending=False).head(1)
    picks += [r for _, r in soft.iterrows()]
    unique, seen = [], set()
    for r in picks:                                   # 去重：两个预算可能选中同一配置
        key = (r["config"], r["representation"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    picks = unique
    checks = []
    seed, train, valid, test = folds[0]
    for r in picks[:n]:
        model = build(r["model"], eval(r["params"]), seed)
        model.fit(x[train], y[train])
        mode = {"hard": tc.HARD, "soft_u8": tc.SOFT_U8, "gbdt_f32": tc.GBDT_F32}[r["representation"]]
        blob = tc.encode(model, mode)
        from_bytes = tc.predict(blob, x[test])
        if mode == tc.GBDT_F32:
            reference = model.predict(x[test])
            formula = len(blob)
        else:
            est = getattr(model, "estimators_", [model])
            h, s = forest_predictions(est, x[test])
            reference = h if mode == tc.HARD else s
            formula = forest_sizes(est)["hard" if mode == tc.HARD else "soft_u8"]
        agree = float(np.mean(from_bytes == reference) * 100)
        checks.append({"config": r["config"], "representation": r["representation"], "fold_seed": seed,
                       "encoded_bytes": len(blob), "formula_bytes": formula,
                       "bytes_match": len(blob) == formula,
                       "agree_with_scan_pct": agree, "test_acc_from_bytes": float(np.mean(from_bytes == y[test]) * 100)})
        if len(blob) != formula or agree < 100.0:
            raise AssertionError(f"编码核验失败：{checks[-1]}")
    return pd.DataFrame(checks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="含 x_train.csv / y_train.csv 的目录")
    ap.add_argument("--foldexp", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("C5"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.data / "x_train.csv", encoding="gbk").to_numpy(dtype=np.float64)
    y = pd.read_csv(args.data / "y_train.csv").to_numpy().ravel().astype(np.int64)
    x = build_raw5(raw)
    folds = []
    for d in sorted(args.foldexp.iterdir()):
        protocol, _, seed = d.name.rpartition("_seed")
        if protocol == "sweep_group" and (d / "split.npz").exists():
            z = np.load(d / "split.npz")
            folds.append((int(seed), z["train"], z["valid"], z["test"]))
    if len(folds) != 5:
        print(f"期望 5 个 sweep_group 折，找到 {len(folds)}", file=sys.stderr)
        return 2

    raw_path = args.root / "raw" / "baseline_pareto_v2_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    records = []
    if raw_path.exists():
        previous = pd.read_csv(raw_path)
        records = previous.to_dict("records")
        done = set(zip(previous.config, previous.seed))
    grid = list(configurations(args.quick))
    for index, (name, params) in enumerate(grid, 1):
        label = name + ("" if not params else "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")")
        started = time.perf_counter()
        for seed, train, valid, test in folds:
            if (label, seed) in done:
                continue
            for r in evaluate(name, params, seed, x, y, train, valid, test):
                records.append({"model": name, "config": label, "params": repr(params), "seed": seed, **r})
        pd.DataFrame(records).to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"[{index:3d}/{len(grid)}] {label:46s} {time.perf_counter() - started:6.1f}s", flush=True)

    frame = pd.DataFrame(records)
    table = (frame.groupby(["model", "config", "params", "representation"], as_index=False)
             .agg(n=("seed", "count"), bytes=("bytes", "mean"), bytes_max=("bytes", "max"),
                  valid_acc=("valid_acc", "mean"), test_acc=("test_acc", "mean"),
                  test_acc_std=("test_acc", "std"), test_ba=("test_ba", "mean")))
    table.to_csv(args.root / "tables" / "T22v2_baseline_pareto.csv", index=False, encoding="utf-8-sig")

    budgets = []
    for cap, label in ((INT8_BYTES, "≤ 4,656 B（交付 INT8 容器）"),
                       (FP32_FOLDED_BYTES, "≤ 11,136 B（FP32 折叠参数 2,784×4）"),
                       (10 ** 12, "不限")):
        within = table[table.bytes_max <= cap]          # 每一折的模型都要装得下，不按均值
        by_valid = within.sort_values(["valid_acc", "bytes_max"], ascending=[False, True]).iloc[0]  # 平票取更小者
        by_test = within.sort_values("test_acc", ascending=False).iloc[0]
        budgets.append({"budget": label, "cap_bytes": cap,
                        "selected_by_valid": f"{by_valid.config} [{by_valid.representation}]",
                        "bytes": by_valid.bytes, "bytes_max": by_valid.bytes_max, "valid_acc": by_valid.valid_acc,
                        "test_acc": by_valid.test_acc, "test_acc_std": by_valid.test_acc_std,
                        "exploratory_best_by_test": f"{by_test.config} [{by_test.representation}]",
                        "exploratory_test_acc": by_test.test_acc})
    budget = pd.DataFrame(budgets)
    budget.to_csv(args.root / "tables" / "T22v2_budget_table.csv", index=False, encoding="utf-8-sig")
    print("\n=== 预算表（按 valid 选配置，报告 test；探索性列按 test 选，仅供参考）===")
    for _, r in budget.iterrows():
        print(f"  {r['budget']:34s} {r['selected_by_valid']:52s} {r['bytes']:11,.0f} B  "
              f"test {r['test_acc']:.3f}±{r['test_acc_std']:.3f}   (探索性 {r['exploratory_test_acc']:.3f})")
    print(f"  参照：交付 INT8 {INT8_BYTES:,} B  test {INT8_TOP1:.3f}")

    checks = verify_representatives(table, x, y, folds)
    checks.to_csv(args.root / "tables" / "T22v2_encoding_checks.csv", index=False, encoding="utf-8-sig")
    print("\n=== 代表性配置：真编码 + 从字节推理 ===")
    print(checks.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
