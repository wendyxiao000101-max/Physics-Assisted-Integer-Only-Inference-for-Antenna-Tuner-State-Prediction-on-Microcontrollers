# -*- coding: utf-8 -*-
"""C6 补跑：P95 分组下的"同配方无物理"对照（FP32 × 5 折），跑完自动汇总 C6 统计。

    cd <PROJECT_ROOT>\\output\\C6补充_P95无物理
    <PROJECT_ROOT>\\iat_fp32_optimized\\.venv-gpu\\Scripts\\python.exe run_c6_p95_nophys.py

做法（不改任何已有代码）：
  * 复用 ..\\C7补充验证_20260925\\train_p95.py：它固定使用该目录下 splits\\fold{0..4}.npz，
    与已完成的 P95 v5 / INT8-A 五折完全同一组划分。
  * 训练命令取自 ..\\C7补充验证_20260925\\fp32_command.json（P95 v5 实际用过的那条），
    只做两处改动：去掉全部 --physics-* 参数、改为 --physics-teacher none；输出到本目录。
    其余（360 epoch、patience 80、ALA、轨迹一致性、检查点平均 k=5、seeds 42–46）一字不改。
  * 已跑完的折会自动跳过（本目录已有 5 条结果时不再训练），可重复执行。
  * 训练完调用 summarize_c6.py，生成 C6_physics_gain.csv 与 C6_quant_cost.csv。

预计 15–20 分钟（RTX 5060）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C7 = HERE.parent / "C7补充验证_20260925"


def build_command() -> list[str]:
    base = json.loads((C7 / "fp32_command.json").read_text(encoding="utf-8"))
    args, i = [], 2                                    # base[0] = python, base[1] = train_p95.py
    while i < len(base):
        token = base[i]
        if token.startswith("--physics"):
            i += 1
            while i < len(base) and not base[i].startswith("--"):
                i += 1                                 # 跳过该参数的取值
            continue
        if token in ("--output", "--ablate-export-dir"):
            i += 2                                     # 输出位置改到本目录
            continue
        args.append(token)
        i += 1
    if "--physics-teacher" in args:
        raise SystemExit("物理参数未去干净")
    return [sys.executable, str(C7 / "train_p95.py"), *args,
            "--physics-teacher", "none",
            "--output", str(HERE / "training"),
            "--ablate-export-dir", str(HERE / "models")]


def done_seeds() -> set[int]:
    raw = HERE / "training" / "ablation" / "raw_runs.csv"
    if not raw.exists():
        return set()
    import csv
    with raw.open(encoding="utf-8-sig") as handle:
        return {int(r["seed"]) for r in csv.DictReader(handle)}


def main() -> int:
    if not (C7 / "train_p95.py").exists():
        print(f"找不到 {C7 / 'train_p95.py'}", file=sys.stderr)
        return 2
    splits = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((C7 / "splits").glob("fold*.npz"))}
    if len(splits) != 5:
        print("P95 划分文件不是 5 个", file=sys.stderr)
        return 2
    command = build_command()
    (HERE / "fp32_nophys_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    (HERE / "split_hashes.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    if done_seeds() >= {42, 43, 44, 45, 46}:
        print("5 折无物理对照已存在，跳过训练")
    else:
        print("开始训练：P95 × 5 折，同配方、关闭物理（约 15–20 分钟）")
        print(" ".join(command[2:]))
        with (HERE / "fp32_nophys.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"训练失败，见 {HERE / 'fp32_nophys.log'}", file=sys.stderr)
            return result.returncode
        if done_seeds() < {42, 43, 44, 45, 46}:
            print("训练结束但结果不足 5 折，见日志", file=sys.stderr)
            return 1

    return subprocess.run([sys.executable, str(HERE / "summarize_c6.py")]).returncode


if __name__ == "__main__":
    sys.exit(main())
