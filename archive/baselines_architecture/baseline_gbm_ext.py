# -*- coding: utf-8 -*-
"""C5 补充基线：XGBoost / LightGBM，与 baseline_pareto_v2 同一协议、同一字节口径。

    python C5\\scripts\\baseline_gbm_ext.py --data <数据目录> --foldexp model\\foldexp --root C5

回应核查意见"补 XGBoost / LightGBM"：
  * 同样 5 个 sweep_group 折（model/foldexp/sweep_group_seed4x/split.npz），
    **只在 train 上拟合，按 valid 均值选配置，报告 test**；
  * 每个模型记两种体积：
      native_bytes  库自带序列化（XGBoost UBJSON / LightGBM 文本）——常见但臃肿，仅作参照；
      codec_bytes   转成 tree_codec 的 GBDT_F32 紧凑字节（与 hist_gb 同一格式），
                    **从字节推理**并断言与原生预测 argmax 一致（≥ 99.9 %，差异来自 float32 近平票）；
    预算表只用 codec_bytes，精度用从字节推理出的 test 精度。
  * 阈值换成整数是无损的：输入全为整数；XGBoost 规则 x < s ⟺ x ≤ ⌈s⌉−1，
    LightGBM 规则 x ≤ s ⟺ x ≤ ⌊s⌋。
  * 基线（初始 margin）不去猜库的内部约定，而是用原生 margin 减去逐类叶值和反推，
    并断言它在所有样本上是常数。

没装库会直接提示安装命令；只装了其中一个也能跑。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_codec as tc                                             # noqa: E402
from baseline_pareto_v2 import (CLASSES, FP32_FOLDED_BYTES, INT8_BYTES,  # noqa: E402
                                INT8_TOP1, build_raw5)

try:
    import xgboost as xgb
except ImportError:                                                 # noqa: BLE001
    xgb = None
try:
    import lightgbm as lgb
except ImportError:                                                 # noqa: BLE001
    lgb = None


def configurations(quick: bool):
    rounds = [5, 20, 100] if quick else [5, 10, 20, 50, 100, 200]
    rates = [0.3] if quick else [0.1, 0.3]
    if xgb is not None:
        for n in rounds:
            for depth in ([3, 6] if quick else [2, 3, 4, 6]):
                for lr in rates:
                    yield "xgboost", {"n_estimators": n, "max_depth": depth, "learning_rate": lr}
    if lgb is not None:
        for n in rounds:
            for leaves in ([8, 31] if quick else [4, 8, 15, 31]):
                for lr in rates:
                    yield "lightgbm", {"n_estimators": n, "num_leaves": leaves, "learning_rate": lr}


# ---------------------------------------------------------------- 树结构转换
def xgb_trees(booster):
    trees = []
    for text in booster.get_dump(dump_format="json"):
        def conv(node):
            if "leaf" in node:
                return ("leaf", float(node["leaf"]))
            feature = int(str(node["split"]).lstrip("f"))
            children = {c["nodeid"]: c for c in node["children"]}
            thr = math.ceil(float(node["split_condition"])) - 1     # x < s ⟺ x ≤ ⌈s⌉−1
            return ("split", feature, thr, conv(children[node["yes"]]), conv(children[node["no"]]))
        trees.append(conv(json.loads(text)))
    return trees


def lgb_trees(booster):
    trees = []
    for info in booster.dump_model()["tree_info"]:
        def conv(node):
            if "leaf_value" in node:
                return ("leaf", float(node["leaf_value"]))
            if node.get("decision_type", "<=") != "<=":
                raise ValueError(f"不支持的 decision_type {node.get('decision_type')}")
            thr = math.floor(float(node["threshold"]))              # x ≤ s ⟺ x ≤ ⌊s⌋
            return ("split", int(node["split_feature"]), thr,
                    conv(node["left_child"]), conv(node["right_child"]))
        trees.append(conv(info["tree_structure"]))
    return trees


def _max_abs_leaf(node) -> float:
    if node[0] == "leaf":
        return abs(float(node[1]))
    return max(_max_abs_leaf(node[3]), _max_abs_leaf(node[4]))


def fit(name, params, seed, x, y):
    if name == "xgboost":
        model = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=-1,
                                  random_state=seed, verbosity=0, **params)
        model.fit(x, y)
        booster = model.get_booster()
        native = len(booster.save_raw("ubj"))
        trees = xgb_trees(booster)

        def margin(z):
            return model.predict(z, output_margin=True)
    else:
        model = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, random_state=seed,
                                   verbose=-1, **params)
        model.fit(x, y)
        booster = model.booster_
        native = len(booster.model_to_string().encode("utf-8"))
        trees = lgb_trees(booster)

        def margin(z):
            return model.predict(z, raw_score=True)
    if len(trees) % CLASSES:
        raise AssertionError(f"树数 {len(trees)} 不是类数 {CLASSES} 的整数倍")
    probe = tc.encode_generic_gbdt(trees, CLASSES, np.zeros(CLASSES))
    reference = x[:: max(1, len(x) // 4000)]
    offset = margin(reference) - tc.leaf_sums(probe, reference)
    spread = float(np.abs(offset - np.median(offset, axis=0)).max())
    # 叶值按 float32 存储，每片叶子的舍入误差 ≤ |v|·2^-24。容差按"各树最大 |叶值| 之和"放宽，
    # 否则学习率大、叶值发散（LightGBM lr=0.3 时 |叶值| 可达 1e4–1e6）的模型会被误判为转换错误。
    # 转换本身的正确性由下面 evaluate() 里"与原生 argmax 一致 ≥ 99.9 %"把关。
    leaf_mass = sum(_max_abs_leaf(t) for t in trees)
    tolerance = 1e-3 + 2.0 ** -23 * leaf_mass
    if spread > tolerance:
        raise AssertionError(f"{name}: 原生 margin − 叶值和 不是常数（偏差 {spread:.3g} > 容差 {tolerance:.3g}），"
                             "树结构转换有误")
    blob = tc.encode_generic_gbdt(trees, CLASSES, np.median(offset, axis=0))
    model._c5_max_abs_leaf = max(_max_abs_leaf(t) for t in trees)
    return model, blob, native


def evaluate(name, params, seed, x, y, train, valid, test):
    model, blob, native = fit(name, params, seed, x[train], y[train])
    pv, pt = tc.predict(blob, x[valid]), tc.predict(blob, x[test])
    native_test = np.asarray(model.predict(x[test])).astype(np.int64).ravel()
    agree = float(np.mean(pt == native_test) * 100)
    if agree < 99.9:
        raise AssertionError(f"{name}{params} seed{seed}: 字节推理与原生预测只一致 {agree:.3f} %")
    return {"representation": "gbdt_f32", "bytes": len(blob), "native_bytes": native,
            "n_trees": int.from_bytes(blob[4:6], "little"),
            "valid_acc": float(np.mean(pv == y[valid]) * 100),
            "test_acc": float(np.mean(pt == y[test]) * 100),
            "native_test_acc": float(np.mean(native_test == y[test]) * 100),
            "agree_with_native_pct": agree,
            "max_abs_leaf": float(getattr(model, "_c5_max_abs_leaf", float("nan")))}


def dump_failure(name, params, seed, x, y, train, test, folder: Path) -> None:
    """出错时保存原生模型文本与少量样本的原生 raw score，便于在没装该库的机器上离线复现转换。"""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tag = f"{name}_" + "_".join(f"{k}{v}" for k, v in params.items()) + f"_seed{seed}"
        if name == "xgboost":
            model = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=-1,
                                      random_state=seed, verbosity=0, **params).fit(x[train], y[train])
            text = "\n".join(model.get_booster().get_dump(dump_format="json"))
            raw_score = model.predict(x[test[:2000]], output_margin=True)
        else:
            model = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, random_state=seed,
                                       verbose=-1, **params).fit(x[train], y[train])
            text = json.dumps(model.booster_.dump_model())
            raw_score = model.predict(x[test[:2000]], raw_score=True)
        (folder / f"{tag}.json").write_text(text, encoding="utf-8")
        np.savez_compressed(folder / f"{tag}.npz", x=x[test[:2000]], raw_score=raw_score,
                            label=np.asarray(model.predict(x[test[:2000]])))
    except Exception as error:                               # noqa: BLE001
        print(f"  （保存诊断文件失败：{error}）", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--foldexp", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("C5"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if xgb is None and lgb is None:
        print("xgboost 与 lightgbm 都没装。先运行：\n"
              "  .\\.venv-gpu\\Scripts\\python.exe -m pip install xgboost lightgbm", file=sys.stderr)
        return 2
    print(f"xgboost {getattr(xgb, '__version__', '未安装')}   lightgbm {getattr(lgb, '__version__', '未安装')}")

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

    raw_path = args.root / "raw" / "gbm_ext_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    (args.root / "tables").mkdir(parents=True, exist_ok=True)
    records, done = [], set()
    if raw_path.exists():
        previous = pd.read_csv(raw_path)
        records, done = previous.to_dict("records"), set(zip(previous.config, previous.seed))
    fail_path = args.root / "raw" / "gbm_ext_failures.csv"
    failures = []
    pd.DataFrame(columns=["model", "config", "seed", "error", "traceback"]).to_csv(
        fail_path, index=False, encoding="utf-8-sig")             # 清掉上一次的失败记录
    grid = list(configurations(args.quick))
    for index, (name, params) in enumerate(grid, 1):
        label = name + "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"
        started = time.perf_counter()
        for seed, train, valid, test in folds:
            if (label, seed) in done:
                continue
            try:
                result = evaluate(name, params, seed, x, y, train, valid, test)
            except Exception as error:                           # noqa: BLE001
                # 不中断整次扫描：记下出错的配置与完整报错，交给人工诊断；该配置不进汇总
                failures.append({"model": name, "config": label, "seed": seed,
                                 "error": f"{type(error).__name__}: {error}",
                                 "traceback": traceback.format_exc()})
                pd.DataFrame(failures).to_csv(fail_path, index=False, encoding="utf-8-sig")
                dump_failure(name, params, seed, x, y, train, test, args.root / "raw" / "gbm_ext_failures")
                print(f"  ✗ {label} seed{seed}: {type(error).__name__}: {error}", flush=True)
                break
            records.append({"model": name, "config": label, "params": repr(params), "seed": seed, **result})
        pd.DataFrame(records).to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"[{index:3d}/{len(grid)}] {label:60s} {time.perf_counter() - started:6.1f}s", flush=True)

    frame = pd.DataFrame(records)
    if failures:
        print(f"\n⚠ {len(failures)} 个配置出错并被跳过，详情见 {fail_path} 与 raw/gbm_ext_failures/，请发给我诊断。")
    complete = frame.groupby("config").seed.transform("count") == len(folds)
    frame = frame[complete]                                   # 只汇总五折都跑完的配置
    table = (frame.groupby(["model", "config", "params", "representation"], as_index=False)
             .agg(n=("seed", "count"), bytes=("bytes", "mean"), bytes_max=("bytes", "max"),
                  native_bytes=("native_bytes", "mean"), valid_acc=("valid_acc", "mean"),
                  test_acc=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                  native_test_acc=("native_test_acc", "mean"),
                  min_agree_with_native_pct=("agree_with_native_pct", "min")))
    table.to_csv(args.root / "tables" / "T22c_gbm_ext.csv", index=False, encoding="utf-8-sig")

    # 与 v2 表合并后重做预算表（v2 表若还没生成就只用本表）
    v2 = args.root / "tables" / "T22v2_baseline_pareto.csv"
    merged = pd.concat([pd.read_csv(v2), table], ignore_index=True) if v2.exists() else table
    if "bytes_max" not in merged or merged["bytes_max"].isna().any():
        print("提示：T22v2 表缺 bytes_max 列（旧版），请先重跑 baseline_pareto_v2.py", file=sys.stderr)
        merged["bytes_max"] = merged["bytes_max"].fillna(merged["bytes"])
    rows = []
    for cap, label in ((INT8_BYTES, "≤ 4,656 B（交付 INT8 容器）"),
                       (FP32_FOLDED_BYTES, "≤ 11,136 B（FP32 折叠参数 2,784×4）"),
                       (10 ** 12, "不限")):
        for scope, pool in (("全部基线", merged), ("仅 XGBoost/LightGBM", table)):
            within = pool[pool.bytes_max <= cap]
            if not len(within):
                rows.append({"budget": label, "scope": scope, "selected_by_valid": "（无配置装得下）"})
                continue
            r = within.sort_values(["valid_acc", "bytes_max"], ascending=[False, True]).iloc[0]
            rows.append({"budget": label, "scope": scope,
                         "selected_by_valid": f"{r.config} [{r.representation}]",
                         "bytes": r.bytes, "bytes_max": r.bytes_max,
                         "valid_acc": r.valid_acc, "test_acc": r.test_acc, "test_acc_std": r.test_acc_std})
    budget = pd.DataFrame(rows)
    budget.to_csv(args.root / "tables" / "T22c_budget_with_gbm.csv", index=False, encoding="utf-8-sig")

    print("\n=== 预算表（含 XGBoost/LightGBM；按 valid 选，报告 test；体积 = 紧凑字节，五折取最大）===")
    for _, r in budget.iterrows():
        if pd.isna(r.get("bytes")):
            print(f"  {r['budget']:30s} {r['scope']:18s} {r['selected_by_valid']}")
        else:
            print(f"  {r['budget']:30s} {r['scope']:18s} {r['selected_by_valid']:62s} "
                  f"{r['bytes_max']:9,.0f} B  test {r['test_acc']:.3f}±{r['test_acc_std']:.3f}")
    print(f"  参照：交付 INT8 {INT8_BYTES:,} B  test {INT8_TOP1:.3f}")
    print(f"\n原生序列化体积（仅参照）：{table.native_bytes.min():,.0f} – {table.native_bytes.max():,.0f} B；"
          f"字节推理与原生预测最低一致率 {table.min_agree_with_native_pct.min():.3f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
