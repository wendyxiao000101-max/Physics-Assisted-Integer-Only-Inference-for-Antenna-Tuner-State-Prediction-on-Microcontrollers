# -*- coding: utf-8 -*-
"""逐折验证：混合头 FP32 → 头部折叠 → 跨层均衡，三步之后逐样本 Top-1 是否一致。

    python C5\\scripts\\verify_fold_consistency.py --project <iat_fp32_optimized 绝对路径>

回应核查意见 §三.3：此前只核对了"折叠丢了哪些参数"，没有在同一个检查点上
比较折叠前后的逐样本预测。本脚本对 model/foldexp/ 下每一折：

  step 0  原始 residual32_gelu_relu_hybrid（FP32，混合头，含 (1-α)(Wx+b) 残余）
  step 1  仅折叠分类头（qat_fold_pipeline.fold_cosine_head 的做法：
          行归一化 W、去掉偏置与 α、去掉样本侧 1/‖x‖、保留 ×logit_scale）
  step 2  折叠 + 跨层均衡（iatopt.quantization.cross_layer_equalize_block）

分别报告 α、样本数、step0→1、step1→2 的 Top-1 不一致条数、准确率差、
由对变错、由错变对。**只比较 argmax**——折叠已经改变了 logit 的尺度，
直接比 logit 数值没有意义。

不依赖 torch：.pth 是 zip 里的 pickle + 原始存储，本脚本自带一个最小加载器。
前向用 float64 numpy 实现，与 torch float32 可能在极近平局的样本上有差别，
所以另外报告每折 step0 的 Top-1 与 metadata.json 里记录的 fp32_holdout 是否吻合，
作为实现正确性的自检。任何自检失败都以非零退出码结束。
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import erf

RAW5_COLUMNS = [0, 1, 2, 4, 5]          # g1_re, g1_im, freq, g2_re, g2_im
FOLD_GATE = 1e-3                          # qat_fold_pipeline.py --alpha-tolerance 默认值
CLE_MAX_SCALE = 16.0                      # iatopt.quantization.cross_layer_equalize_block 默认值


# ------------------------------------------------------------------ 无 torch 加载器

def load_state_dict(path: Path) -> "OrderedDict[str, np.ndarray]":
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split("/")[0]
    dtypes = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
              "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16}

    class Storage:
        def __init__(self, kind, key):
            self.array = np.frombuffer(archive.read(f"{prefix}/data/{key}"), dtype=dtypes[kind])

    def rebuild_tensor(storage, offset, size, stride, *_):
        if len(size) == 0:
            return storage.array[offset].copy()
        itemsize = storage.array.itemsize
        return np.lib.stride_tricks.as_strided(
            storage.array[offset:], shape=tuple(size),
            strides=tuple(s * itemsize for s in stride)).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if module == "torch" and name.endswith("Storage"):
                return name
            raise pickle.UnpicklingError(f"不允许的类型 {module}.{name}")

        def persistent_load(self, pid):
            _, kind, key, _location, _numel = pid
            kind = kind if isinstance(kind, str) else getattr(kind, "__name__", str(kind))
            return Storage(kind, key)

    return Unpickler(io.BytesIO(archive.read(f"{prefix}/data.pkl"))).load()


# ------------------------------------------------------------------ 前向（float64）

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def relu(x):
    return np.maximum(x, 0.0)


def normalize_rows(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def features(state, x, fc1_w=None, fc1_b=None, fc2_w=None):
    stem = gelu(x @ state["network.0.weight"].T + state["network.0.bias"])
    w1 = state["network.2.block.0.weight"] if fc1_w is None else fc1_w
    b1 = state["network.2.block.0.bias"] if fc1_b is None else fc1_b
    w2 = state["network.2.block.3.weight"] if fc2_w is None else fc2_w
    branch = relu(stem @ w1.T + b1) @ w2.T + state["network.2.block.3.bias"]
    return relu(stem + branch)


def hybrid_logits(state, x, scale):
    h = features(state, x)
    w, b = state["network.3.weight"], state["network.3.bias"]
    alpha = 1.0 / (1.0 + np.exp(-float(state["network.3.alpha_logit"])))
    linear = h @ w.T + b
    angular = scale * (normalize_rows(h) @ normalize_rows(w).T)
    return (1.0 - alpha) * linear + alpha * angular, alpha


def folded_logits(state, x, scale, cle=False):
    fc1_w = state["network.2.block.0.weight"].astype(np.float64)
    fc1_b = state["network.2.block.0.bias"].astype(np.float64)
    fc2_w = state["network.2.block.3.weight"].astype(np.float64)
    scales = None
    if cle:
        first = np.abs(fc1_w).max(axis=1)
        second = np.abs(fc2_w).max(axis=0)
        valid = (first > 1e-12) & (second > 1e-12)
        scales = np.ones_like(first)
        scales[valid] = np.sqrt(second[valid] / first[valid])
        scales = np.clip(scales, 1.0 / CLE_MAX_SCALE, CLE_MAX_SCALE)
        fc1_w = fc1_w * scales[:, None]
        fc1_b = fc1_b * scales
        fc2_w = fc2_w / scales[None, :]
    h = features(state, x, fc1_w, fc1_b, fc2_w)
    return (h @ normalize_rows(state["network.3.weight"]).T) * scale, scales


def compare(before, after, labels):
    a, b = before.argmax(1), after.argmax(1)
    ra, rb = a == labels, b == labels
    return {"disagree": int((a != b).sum()),
            "acc_before": float(ra.mean() * 100), "acc_after": float(rb.mean() * 100),
            "right_to_wrong": int((ra & ~rb).sum()), "wrong_to_right": int((~ra & rb).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="默认 <project>/../esp32p4_iat_demo/input/data")
    ap.add_argument("--out", type=Path, default=None, help="结果 CSV，默认 <project>/C5/tables/T25_fold_consistency.csv")
    ap.add_argument("--full-fit", type=Path, default=None,
                    help="交付的全量 FP32 模型目录，默认 <project>/model/residual32_physics_dual_champion；"
                         "在全部 96,768 行上评估（它没有留出集）")
    args = ap.parse_args()

    project = args.project.resolve()
    data = args.data or project.parent / "esp32p4_iat_demo" / "input" / "data"
    out = args.out or project / "C5" / "tables" / "T25_fold_consistency.csv"
    raw5 = pd.read_csv(data / "x_train.csv", encoding="gbk").values[:, RAW5_COLUMNS].astype(np.float64)
    labels = pd.read_csv(data / "y_train.csv").to_numpy().ravel()

    rows, failures = [], []
    full_fit = args.full_fit or project / "model" / "residual32_physics_dual_champion"
    targets = [f for f in sorted((project / "model" / "foldexp").iterdir()) if (f / "model_fp32.pth").exists()]
    if (full_fit / "model_fp32.pth").exists():
        targets.append(full_fit)
    for fold in targets:
        meta = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("architecture") != "residual32_gelu_relu_hybrid":
            continue
        state = {k: np.asarray(v, dtype=np.float64) for k, v in load_state_dict(fold / "model_fp32.pth").items()}
        scaler = joblib.load(fold / "scaler.pkl")
        is_full = not (fold / "split.npz").exists()
        test = np.arange(len(labels)) if is_full else np.load(fold / "split.npz")["test"]
        x = (raw5[test] - scaler["mean"]) / scaler["scale"]
        y = labels[test]
        scale = float(meta.get("head_scale", 20.0))

        step0, alpha = hybrid_logits(state, x, scale)
        step1, _ = folded_logits(state, x, scale, cle=False)
        step2, cle_scales = folded_logits(state, x, scale, cle=True)
        recorded = (95965 / 96768 * 100) if is_full else float(meta["fp32_holdout"]["accuracy"]) * 100
        s01, s12, s02 = compare(step0, step1, y), compare(step1, step2, y), compare(step0, step2, y)

        margins = np.sort(step0, axis=1)
        gap = margins[:, -1] - margins[:, -2]
        residual = (1.0 - alpha) * np.abs(features(state, x) @ state["network.3.weight"].T
                                            + state["network.3.bias"]).max(axis=1)
        row = {"fold": ("FULL-FIT (deployed) " + fold.name) if is_full else fold.name,
               "scope": "all 96,768 rows" if is_full else "fold test rows", "alpha": alpha, "one_minus_alpha": 1.0 - alpha,
               "passes_gate_1e-3": bool(1.0 - alpha <= FOLD_GATE),
               "n_test": len(test), "recorded_fp32_acc": recorded, "reproduced_fp32_acc": s01["acc_before"],
               "reproduce_diff_pp": s01["acc_before"] - recorded,
               "fold_disagree": s01["disagree"], "fold_acc_after": s01["acc_after"],
               "fold_delta_pp": s01["acc_after"] - s01["acc_before"],
               "fold_right_to_wrong": s01["right_to_wrong"], "fold_wrong_to_right": s01["wrong_to_right"],
               "cle_disagree": s12["disagree"], "cle_delta_pp": s12["acc_after"] - s12["acc_before"],
               "total_disagree": s02["disagree"],
               "min_top2_gap_before": float(gap.min()),
               "max_dropped_residual": float(residual.max()),
               "rows_residual_ge_gap": int((2 * residual >= gap).sum()),
               "cle_scale_min": float(cle_scales.min()), "cle_scale_max": float(cle_scales.max())}
        rows.append(row)
        if abs(row["reproduce_diff_pp"]) > 0.02:
            failures.append(f"{fold.name}: 复现 FP32 准确率 {s01['acc_before']:.4f} 与记录 {recorded:.4f} 相差超过 0.02 pp")

    if not rows:
        print("没有找到任何折的 model_fp32.pth", file=sys.stderr)
        return 2
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{'fold':20.20s} {'1-α':>9s} {'门限':>4s} {'n_test':>6s} {'复现差':>7s} "
          f"{'折叠不一致':>8s} {'对→错':>5s} {'错→对':>5s} {'CLE不一致':>8s}")
    for _, r in table.iterrows():
        print(f"{r['fold']:20.20s} {r['one_minus_alpha']:9.2e} {'过' if r['passes_gate_1e-3'] else '否':>4s} "
              f"{r['n_test']:6d} {r['reproduce_diff_pp']:+7.4f} {r['fold_disagree']:8d} "
              f"{r['fold_right_to_wrong']:5d} {r['fold_wrong_to_right']:5d} {r['cle_disagree']:8d}")
    folds = table[table["scope"] == "fold test rows"]
    print(f"\n十折合计 {folds['n_test'].sum():,} 个测试样本：头部折叠不一致 {folds['fold_disagree'].sum()} 条，"
          f"CLE 不一致 {folds['cle_disagree'].sum()} 条；逐样本充分条件（2×残余 ≥ 分差）不满足的行 "
          f"{table['rows_residual_ge_gap'].sum()} 条。")
    if (table["fold_disagree"] + table["cle_disagree"]).sum() > 0:
        print("注意：存在不一致样本，不能写'在所测数据上未改变预测'。")
    print(f"已写入 {out}")
    for message in failures:
        print("自检失败：" + message, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
