# -*- coding: utf-8 -*-
"""C5 统一延迟基准：交付 INT8 网络与预算表里选出的树基线，同一台机器、同一口径计时。

    python C5\\scripts\\bench_c5_latency.py --data <数据目录> --foldexp model\\foldexp --root C5

**这是桌面 CPU 上的软件延迟，不是 MCU 实测**。它回答的是"同样用纯 Python 逐样本跑、
同样从紧凑字节推理时，两类模型的相对代价"，再配上与硬件无关的运算计数；
板端延迟以 ESP32-P4 实测为准。

每个模型报告三种口径：
  scalar_py_batch1    纯 Python、逐样本（batch = 1）、只读字节：INT8 用交付的 inference.py，
                      树用 tree_codec.ScalarRunner。报告中位数 / P95 / P99（µs）。
  numpy_batched       NumPy 向量化、整折一次推理，折算到每样本（µs/样本），吞吐口径。
  sklearn_batch1      仅树基线：库原生 predict 单样本（Python 调用开销主导，仅作参照）。
运算计数（与硬件无关）：
  INT8   MAC 2,688 / GELU 查表 32 / 重量化 111 次（每样本固定）
  树      实测每样本比较次数（均值 / 最大）+ 叶累加次数（hard: T，soft_u8: 15T，gbdt: T）

固定：单线程（BLAS/OMP = 1，在导入 numpy 前设置）、预热、多次重复；记录 CPU、库版本。
计时边界：只含 predict 调用本身，不含读 CSV、构造输入、写结果。
模型：树基线在 sweep_group_seed42 的 train 上拟合（与预算表同配置），推理行取该折 test。
      INT8 计时用交付 blob（sha256 154100ef…）；但交付模型在全量数据上训练，fold42 test
      对它不是留出集，所以精度列改用该折自己的 INT8 模型（model/foldexp/sweep_group_seed42/int8）。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                      # noqa: E402
import ast                                                           # noqa: E402
import hashlib                                                       # noqa: E402
import importlib.util                                                # noqa: E402
import json                                                          # noqa: E402
import platform                                                      # noqa: E402
import statistics                                                    # noqa: E402
import subprocess                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tree_codec as tc                                              # noqa: E402
from baseline_pareto_v2 import build, build_raw5                     # noqa: E402

SHIFT = 24
SHAPES = [(5, 32), (32, 32), (32, 32), (32, 15)]
INT8_MACS = sum(a * b for a, b in SHAPES)                            # 2,688
INT8_REQUANT = sum(b for _, b in SHAPES)                             # 111


class IntegerModel:
    """I8R32V02 的 NumPy 向量化整数核，与 inference.py 逐位等价（会断言）。"""

    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        if len(raw) != 4656 or raw[:8] != b"I8R32V02":
            raise ValueError(f"{path} 不是 4,656 字节的 I8R32V02 模型")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        pos = 8

        def get(count, dtype):
            nonlocal pos
            out = np.frombuffer(raw, dtype=dtype, count=count, offset=pos).copy()
            pos += out.nbytes
            return out
        self.offset, self.step = get(5, "<i4"), get(5, "<i4")
        self.hz = int(get(1, "<i4")[0])
        self.lut = get(1024, "i1")
        self.layers = [(get(i * o, "i1").reshape(o, i).astype(np.int64), get(o, "<i4").astype(np.int64),
                        get(o, "<i4").astype(np.int64)) for i, o in SHAPES]
        self.skip = int(get(1, "<i4")[0])
        assert pos == len(raw)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        x = np.clip((2 * (feats.astype(np.int64) - self.offset) + self.step) // (2 * self.step), 0, 255) - 128

        def linear(q, i, zp):
            w, b, m = self.layers[i]
            return ((q - zp) @ w.T + b) * m

        def rq(v):
            return (v + (1 << (SHIFT - 1))) >> SHIFT
        h = self.lut[np.clip(rq(linear(x, 0, -128)), -512, 511) + 512].astype(np.int64)
        a = np.clip(rq(linear(h, 1, self.hz - 128)), 0, 255) - 128
        z = np.clip(rq(linear(a, 2, -128) + (h - (self.hz - 128)) * self.skip), 0, 255) - 128
        return rq(linear(z, 3, -128)).argmax(1)


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20)
            if out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:                                            # noqa: BLE001
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def per_call_us(function, inputs, warmup: int, repeats: int):
    for item in inputs[:warmup]:
        function(item)
    samples = []
    for _ in range(repeats):
        for item in inputs:
            start = time.perf_counter_ns()
            function(item)
            samples.append((time.perf_counter_ns() - start) / 1e3)
    ordered = sorted(samples)
    return {"n_calls": len(samples), "median_us": statistics.median(samples),
            "p95_us": ordered[int(0.95 * (len(ordered) - 1))], "p99_us": ordered[int(0.99 * (len(ordered) - 1))],
            "mean_us": float(np.mean(samples))}


def batched_us(function, block, repeats: int):
    function(block)
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function(block)
        times.append((time.perf_counter_ns() - start) / 1e3 / len(block))
    return statistics.median(times)


def selected_baselines(root: Path):
    """从预算表取按 valid 选出的配置（去重）。"""
    picks = []
    for name in ("T22c_budget_with_gbm.csv", "T22v2_budget_table.csv"):
        path = root / "tables" / name
        if path.exists():
            for text in pd.read_csv(path).selected_by_valid.dropna():
                if "[" in text and text not in picks:
                    picks.append(text)
    catalog = []
    for name in ("T22v2_baseline_pareto.csv", "T22c_gbm_ext.csv"):
        path = root / "tables" / name
        if path.exists():
            catalog.append(pd.read_csv(path))
    if not picks or not catalog:
        raise SystemExit("找不到预算表：先跑 baseline_pareto_v2.py（以及可选的 baseline_gbm_ext.py）")
    catalog = pd.concat(catalog, ignore_index=True)
    out = []
    for text in picks:
        config, representation = text.rsplit(" [", 1)
        representation = representation.rstrip("]")
        row = catalog[(catalog.config == config) & (catalog.representation == representation)]
        if len(row):
            r = row.iloc[0]
            out.append((r.model, ast.literal_eval(r.params), representation, config))
    return out


def fit_baseline(model_name, params, representation, seed, x, y):
    if model_name in ("xgboost", "lightgbm"):
        from baseline_gbm_ext import fit
        model, blob, _ = fit(model_name, params, seed, x, y)
        return model, blob
    model = build(model_name, params, seed)
    model.fit(x, y)
    mode = {"hard": tc.HARD, "soft_u8": tc.SOFT_U8, "gbdt_f32": tc.GBDT_F32}[representation]
    return model, tc.encode(model, mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--foldexp", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=HERE.parent)
    ap.add_argument("--int8", type=Path, default=HERE.parent / "evidence" / "deployed_int8" / "model_int8.bin")
    ap.add_argument("--samples", type=int, default=2000, help="batch=1 计时的样本数")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--sklearn-samples", type=int, default=300)
    args = ap.parse_args()

    raw = pd.read_csv(args.data / "x_train.csv", encoding="gbk").to_numpy(dtype=np.float64)
    y = pd.read_csv(args.data / "y_train.csv").to_numpy().ravel().astype(np.int64)
    x_tree = build_raw5(raw)                                        # (freq, g1re, g1im, g2re, g2im)
    x_int8 = raw[:, [0, 1, 2, 4, 5]].astype(np.int64)               # (g1re, g1im, freq, g2re, g2im)
    split = np.load(args.foldexp / "sweep_group_seed42" / "split.npz")
    train, test = split["train"], split["test"]
    timed = test[:args.samples]
    rows = []

    # ---------------- 交付 INT8
    spec = importlib.util.spec_from_file_location("deployed_inference", args.int8.parent / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scalar = module.Predictor(str(args.int8))
    vector = IntegerModel(args.int8)
    as_lists = [[int(v) for v in r] for r in x_int8[timed]]
    scalar_pred = np.array([scalar.predict(r) for r in as_lists])
    if not np.array_equal(scalar_pred, vector.predict(x_int8[timed])):
        raise AssertionError("INT8 NumPy 核与 inference.py 不一致")
    # 交付模型是全量数据训练的，fold42 的 test 对它不是留出集；精度改用该折自己的 INT8 模型
    # （同一 I8R32V02 格式、同一计算量，计时不受权重影响）。
    fold_blob = args.foldexp / "sweep_group_seed42" / "int8" / "model_int8.bin"
    fold_acc = float(np.mean(IntegerModel(fold_blob).predict(x_int8[test]) == y[test]) * 100) \
        if fold_blob.exists() else float("nan")
    rows.append({"model": "residual32 INT8 (deployed blob; acc from fold-42 INT8)", "representation": "I8R32V02",
                 "bytes": args.int8.stat().st_size,
                 "test_acc_fold42": fold_acc,
                 **{f"scalar_{k}": v for k, v in per_call_us(scalar.predict, as_lists, args.warmup, args.repeats).items()},
                 "numpy_batched_us_per_sample": batched_us(vector.predict, x_int8[test], args.repeats),
                 "sklearn_batch1_median_us": None,
                 "macs": INT8_MACS, "lut_lookups": 32, "requant_ops": INT8_REQUANT,
                 "comparisons_mean": 0, "comparisons_max": 0, "leaf_accumulations": 0, "n_trees": 0})
    print(f"INT8  sha256 {vector.sha256[:12]}…  scalar median {rows[-1]['scalar_median_us']:.1f} µs")

    # ---------------- 预算表选出的树基线
    for model_name, params, representation, config in selected_baselines(args.root):
        model, blob = fit_baseline(model_name, params, representation, 42, x_tree[train], y[train])
        runner = tc.ScalarRunner(blob)
        inputs = [[int(v) for v in r] for r in x_tree[timed]]
        results = [runner.predict_one(r) for r in inputs]
        scalar_pred = np.array([r[0] for r in results])
        visited = np.array([r[1] for r in results])
        if not np.array_equal(scalar_pred, tc.predict(blob, x_tree[timed])):
            raise AssertionError(f"{config}: 纯 Python 解码与 NumPy 解码不一致")
        native = None
        if hasattr(model, "n_jobs"):
            model.set_params(n_jobs=1)                  # 单样本计时不要线程池开销
        if args.sklearn_samples:
            singles = [x_tree[i:i + 1] for i in timed[:args.sklearn_samples]]
            native = per_call_us(model.predict, singles, min(50, len(singles)), 1)["median_us"]
        accumulate = {"hard": 1, "soft_u8": tc_classes(blob), "gbdt_f32": 1}[representation] * runner.n_trees
        rows.append({"model": config, "representation": representation, "bytes": len(blob),
                     "test_acc_fold42": float(np.mean(tc.predict(blob, x_tree[test]) == y[test]) * 100),
                     **{f"scalar_{k}": v for k, v in
                        per_call_us(lambda r: runner.predict_one(r), inputs, args.warmup, args.repeats).items()},
                     "numpy_batched_us_per_sample": batched_us(lambda b: tc.predict(blob, b), x_tree[test],
                                                               max(1, args.repeats // 2)),
                     "sklearn_batch1_median_us": native,
                     "macs": 0, "lut_lookups": 0, "requant_ops": 0,
                     "comparisons_mean": float(visited.mean()), "comparisons_max": int(visited.max()),
                     "leaf_accumulations": accumulate, "n_trees": runner.n_trees})
        print(f"{config} [{representation}]  {len(blob):,} B  scalar median {rows[-1]['scalar_median_us']:.1f} µs")

    host = platform.node() or "host"
    table = pd.DataFrame(rows)
    out = args.root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"T26_latency_{host}.csv", index=False, encoding="utf-8-sig")
    environment = {
        "label": "桌面 CPU 软件延迟（非 MCU 实测）",
        "cpu": cpu_name(), "logical_cores": os.cpu_count(), "platform": platform.platform(),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "threads_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "clock": "time.perf_counter_ns", "warmup_calls": args.warmup, "repeats": args.repeats,
        "batch1_samples": len(timed), "fold": "sweep_group_seed42 (train 拟合, test 计时)",
        "timing_boundary": "只含 predict 调用；不含读 CSV / 构造输入 / 写结果",
        "int8_sha256": vector.sha256,
    }
    (out / f"T26_latency_{host}.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    view = table[["model", "representation", "bytes", "test_acc_fold42", "scalar_median_us", "scalar_p95_us",
                  "numpy_batched_us_per_sample", "sklearn_batch1_median_us", "macs",
                  "comparisons_mean", "leaf_accumulations"]]
    print("\n" + view.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n{environment['cpu']} | Python {environment['python']} | 单线程 | 桌面 CPU 软件延迟，非 MCU 实测")
    print(f"已写入 {out / f'T26_latency_{host}.csv'} 与 .json")
    return 0


def tc_classes(blob: bytes) -> int:
    return blob[6]


if __name__ == "__main__":
    sys.exit(main())
