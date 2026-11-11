# -*- coding: utf-8 -*-
"""教师在**与学生完全相同的测试行**上的 Acc / BA / 最小类召回。

只用 numpy + sklearn，不需要 torch。对每个 foldexp/<protocol>_seed<k>/split.npz：
  1. 用该折的 train 行拟合 Mobius + 多项逻辑回归教师（与 train.py L401 同一路径）；
  2. 在同折的 train / valid / test 行上分别评估；
  3. 输出逐折表，供论文 VI-C 引用。

    python teacher_same_protocol.py --data <x_train.csv> <y_train.csv> \
        --foldexp <model/foldexp 目录> --out tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_quality import corrupt_coefficients, describe          # noqa: E402

GAMMA_SCALE = 32768.0
SWITCH_STATES = 16
CLASSES = 15


def fit_mobius(raw, rows, min_rows=4):
    sub = raw[rows]
    g1 = (sub[:, 0] + 1j * sub[:, 1]) / GAMMA_SCALE
    g2 = (sub[:, 4] + 1j * sub[:, 5]) / GAMMA_SCALE
    freq = sub[:, 2].astype(np.int64)
    state = sub[:, 3].astype(np.int64)
    freqs = np.sort(np.unique(freq))
    coef = np.zeros((SWITCH_STATES, len(freqs), 3), dtype=np.complex128)
    fitted = np.zeros((SWITCH_STATES, len(freqs)), dtype=bool)
    res = []
    for col, value in enumerate(freqs):
        at = freq == value
        for s in range(SWITCH_STATES):
            mask = at & (state == s)
            if int(mask.sum()) < min_rows:
                continue
            z, w = g2[mask], g1[mask]
            design = np.column_stack((z, np.ones_like(z), -z * w))
            sol, *_ = np.linalg.lstsq(design, w, rcond=None)
            coef[s, col] = sol
            fitted[s, col] = True
            a, b, c = sol
            res.append(np.abs((a * z + b) / (c * z + 1.0) - w))
    for s in range(SWITCH_STATES):
        avail = np.flatnonzero(fitted[s])
        if len(avail) == 0:
            raise ValueError(f"switch state {s} has no rows")
        for col in np.flatnonzero(~fitted[s]):
            coef[s, col] = coef[s, avail[np.argmin(np.abs(avail - col))]]
    median = float(np.median(np.concatenate(res))) if res else float("nan")
    return freqs, coef, fitted, median


def columns_of(freqs, frequency):
    frequency = np.asarray(frequency, dtype=np.float64)
    cols = np.clip(np.searchsorted(freqs, frequency), 0, len(freqs) - 1)
    left = np.clip(cols - 1, 0, len(freqs) - 1)
    use_left = np.abs(freqs[left] - frequency) < np.abs(freqs[cols] - frequency)
    return np.where(use_left, left, cols)


def magnitudes(freqs, coef, g2, frequency, classes=CLASSES):
    cols = columns_of(freqs, frequency)
    out = np.empty((len(g2), classes))
    for s in range(classes):
        a, b, c = coef[s, cols].T
        out[:, s] = np.abs((a * g2 + b) / (c * g2 + 1.0))
    return out


def features_of(freqs, coef, raw, rows):
    g2 = (raw[rows, 4] + 1j * raw[rows, 5]) / GAMMA_SCALE
    return magnitudes(freqs, coef, g2, raw[rows, 2])


def metrics(y_true, y_pred):
    return {
        "acc": float(np.mean(y_pred == y_true)) * 100,
        "ba": float(balanced_accuracy_score(y_true, y_pred)) * 100,
        "min_recall": float(np.min(recall_score(y_true, y_pred, average=None,
                                                labels=np.unique(y_true),
                                                zero_division=0))) * 100,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs=2, type=Path, required=True, metavar=("X_CSV", "Y_CSV"))
    ap.add_argument("--foldexp", type=Path, required=True)
    ap.add_argument("--kind", default="lr", choices=["lr", "argmin"])
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Möbius 系数乘性复高斯噪声强度 η（教师质量调节器）")
    ap.add_argument("--shuffle", action="store_true",
                    help="把同一开关态的频率列随机重排（物理形状正确、内容错误的对照）")
    ap.add_argument("--corrupt-seed", type=int, default=20260918)
    ap.add_argument("--protocols", nargs="+", default=None,
                    help="只评估这些协议（random / sweep_group），默认全部")
    ap.add_argument("--out", type=Path, default=Path("tables"))
    args = ap.parse_args()

    raw = pd.read_csv(args.data[0], encoding="gbk").to_numpy(dtype=np.float64)
    labels = pd.read_csv(args.data[1]).to_numpy().ravel().astype(np.int64)
    assert raw.shape == (96768, 6) and labels.shape == (96768,), (raw.shape, labels.shape)

    rows = []
    for folder in sorted(args.foldexp.iterdir()):
        split = folder / "split.npz"
        if not split.exists():
            continue
        protocol, _, seed = folder.name.rpartition("_seed")
        if args.protocols and protocol not in args.protocols:
            continue
        d = np.load(split)
        train, valid, test = d["train"], d["valid"], d["test"]
        freqs, coef, fitted, median = fit_mobius(raw, train)
        coef = corrupt_coefficients(coef, args.noise, args.shuffle, args.corrupt_seed)
        if args.kind == "lr":
            feat = features_of(freqs, coef, raw, train)
            mean, scale = feat.mean(0), feat.std(0)
            scale[scale < 1e-12] = 1.0
            model = LogisticRegression(max_iter=3000, C=10.0)
            model.fit(((feat - mean) / scale).astype(np.float32), labels[train])

            def predict(idx):
                f = features_of(freqs, coef, raw, idx)
                s = model.decision_function(((f - mean) / scale).astype(np.float32))
                full = np.full((len(s), CLASSES), -30.0)
                full[:, model.classes_] = s
                return full.argmax(1)
        else:
            def predict(idx):
                return (-features_of(freqs, coef, raw, idx)).argmax(1)

        row = {"protocol": protocol, "seed": int(seed),
               "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
               "mobius_residual_median": median,
               "mobius_cells_fitted": int(fitted.sum()),
               "mobius_cells_total": int(fitted.size),
               "teacher_kind": args.kind,
               "teacher_noise": args.noise,
               "teacher_shuffle": bool(args.shuffle),
               "teacher_variant": f"{args.kind}-{describe(args.noise, args.shuffle, args.corrupt_seed)}"}
        for tag, idx in (("train", train), ("valid", valid), ("test", test)):
            for k, v in metrics(labels[idx], predict(idx)).items():
                row[f"{tag}_{k}"] = v
        rows.append(row)
        print(f"{folder.name:22s} train {row['train_acc']:7.4f} "
              f"valid {row['valid_acc']:7.4f} test {row['test_acc']:7.4f} "
              f"(test BA {row['test_ba']:7.4f}, min-recall {row['test_min_recall']:7.4f})",
              flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.kind}_{describe(args.noise, args.shuffle, args.corrupt_seed)}".replace("(", "_").replace(")", "")
    path = args.out / f"T20_teacher_same_protocol_{tag}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n===== 同口径汇总（均值 ± 标准差，%） =====")
    for protocol, g in table.groupby("protocol"):
        print(f"\n-- {protocol} (n={len(g)}) --")
        for tag in ("train", "valid", "test"):
            print(f"  {tag:5s} Acc {g[f'{tag}_acc'].mean():7.4f}±{g[f'{tag}_acc'].std(ddof=1):6.4f}"
                  f"   BA {g[f'{tag}_ba'].mean():7.4f}±{g[f'{tag}_ba'].std(ddof=1):6.4f}"
                  f"   min-recall {g[f'{tag}_min_recall'].mean():7.4f}")
        print(f"  逐折 test Acc: {' '.join(f'{v:.4f}' for v in g['test_acc'])}")
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    sys.exit(main())
