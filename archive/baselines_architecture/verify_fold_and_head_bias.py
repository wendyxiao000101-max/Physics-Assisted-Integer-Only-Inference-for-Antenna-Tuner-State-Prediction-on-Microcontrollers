# -*- coding: utf-8 -*-
"""核实两件事：折叠对 alpha 与头部偏置做了什么；交付 blob 里那 15 个偏置从哪来。

    python C5\\scripts\\verify_fold_and_head_bias.py --project <iat_fp32_optimized 绝对路径>

只读，不改任何文件，不需要 GPU / torch。做三件事：

  1. **静态核对折叠代码**：在 `qat_fold_pipeline.py::fold_cosine_head` 与
     `iatopt/quantization.py::FoldedResidual32` 里逐条确认——头是不是 bias=False、
     权重是不是被行归一化、alpha 是不是只做门限检查然后丢弃。
  2. **解析 blob 布局**：把交付的 4,656 B 逐字段拆开，数清 111 个 int32 偏置怎么分层。
  3. **测那 15 个头部偏置值不值**：把它们清零再跑全集，对比 Top-1；
     同时对比 QAT 前的快照（应当全零）。

任何一项核对失败都会**以非零退出码结束**（不是只打印）。参与对比的两个 blob：
    evidence/deployed_int8/model_int8.bin       交付版，sha256 154100ef…
    evidence/qat_snapshot/qat_integer_snapshot.bin  分类头 QAT 之前的快照
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHIFT = 24
SHAPES = [(5, 32), (32, 32), (32, 32), (32, 15)]
HEAD_BIAS_OFFSET, HEAD_BIAS_COUNT = 4532, 15     # 由 §2 的布局推出，脚本里会自校验
RAW5_COLUMNS = [0, 1, 2, 4, 5]                   # 与 verify_audit.py L126 一致
DEPLOYED_SHA256 = "154100ef8d7f6125b671cc0dd977acd2531e5a15e6d32186b212f40a77e39642"
FAILURES: list[str] = []


def check(ok: bool, message: str) -> bool:
    if not ok:
        FAILURES.append(message)
    return ok


class IntegerModel:
    """纯 numpy 整数推理，与交付包的 integer_numpy.py 逐位一致。"""

    def __init__(self, blob: bytes):
        position = 8
        if blob[:8] != b"I8R32V02":
            raise ValueError("不是 I8R32V02 blob")

        def take(count: int, dtype: str) -> np.ndarray:
            nonlocal position
            array = np.frombuffer(blob, dtype=dtype, count=count, offset=position).copy()
            position += array.nbytes
            return array

        self.fields = []
        self.offset = take(5, "<i4"); self.fields.append(("input offset  int32", 5, 20))
        self.step = take(5, "<i4"); self.fields.append(("input step    int32", 5, 20))
        self.hz = int(take(1, "<i4")[0]); self.fields.append(("h zero point  int32", 1, 4))
        self.lut = take(1024, "i1"); self.fields.append(("GELU LUT      int8", 1024, 1024))
        self.layers, self.bias_offsets = [], []
        for index, (fan_in, fan_out) in enumerate(SHAPES):
            weight = take(fan_in * fan_out, "i1").reshape(fan_out, fan_in)
            self.fields.append((f"L{index} weight     int8", fan_in * fan_out, fan_in * fan_out))
            self.bias_offsets.append(position)
            bias = take(fan_out, "<i4"); self.fields.append((f"L{index} bias       int32", fan_out, fan_out * 4))
            multiplier = take(fan_out, "<i4"); self.fields.append((f"L{index} multiplier int32", fan_out, fan_out * 4))
            self.layers.append([weight, bias, multiplier])
        self.skip = int(take(1, "<i4")[0]); self.fields.append(("skip mult     int32", 1, 4))
        if position != len(blob):
            raise ValueError(f"解析后剩余 {len(blob) - position} 字节")

    def predict(self, raw5: np.ndarray) -> np.ndarray:
        x = np.clip((2 * (raw5.astype(np.int64) - self.offset) + self.step)
                    // (2 * self.step), 0, 255).astype(np.int16) - 128

        def linear(q, index, zero_point):
            weight, bias, multiplier = self.layers[index]
            accumulator = (q.astype(np.int32) - zero_point) @ weight.astype(np.int32).T + bias
            return accumulator.astype(np.int64) * multiplier

        def requant(value):
            return (value + (1 << (SHIFT - 1))) >> SHIFT

        h = self.lut[np.clip(requant(linear(x, 0, -128)), -512, 511).astype(np.int32) + 512]
        a = (np.clip(requant(linear(h, 1, self.hz - 128)), 0, 255) - 128).astype(np.int8)
        z = (np.clip(requant(linear(a, 2, -128)
                             + (h.astype(np.int64) - (self.hz - 128)) * self.skip),
                     0, 255) - 128).astype(np.int8)
        return requant(linear(z, 3, -128)).argmax(1)


def inspect_source(project: Path) -> None:
    print("=" * 78)
    print("§1  折叠代码静态核对")
    print("=" * 78)
    checks = [
        (project / "iatopt" / "quantization.py", "FoldedResidual32 的分类头无偏置",
         r"self\.head\s*=\s*nn\.Linear\([^)]*bias\s*=\s*False"),
        (project / "qat_fold_pipeline.py", "折叠只拷贝**归一化后的**头部权重",
         r"model\.head\.weight\.copy_\(F\.normalize\(classifier\.weight"),
        (project / "qat_fold_pipeline.py", "折叠对 alpha 只做门限检查（不满足就报错）",
         r"if alpha < 1\.0 - tolerance:\s*\n\s*raise ValueError"),
    ]
    for path, label, pattern in checks:
        if not check(path.exists(), f"缺文件 {path}"):
            print(f"  [缺文件] {label}  ({path})")
            continue
        text = path.read_text(encoding="utf-8")
        hit = re.search(pattern, text)
        check(bool(hit), f"静态核对失败：{label}")
        print(f"  [{'✓' if hit else '✗'}] {label}")
        if hit:
            print(f"        {hit.group().strip().splitlines()[0]}")
    # 有没有任何一处把头部偏置搬过去
    fold = (project / "qat_fold_pipeline.py").read_text(encoding="utf-8")
    body = fold[fold.index("def fold_cosine_head"):]
    body = body[:body.index("\n\n\n")] if "\n\n\n" in body else body
    clean = check("head.bias" not in body, "fold_cosine_head 里出现了 head.bias")
    print(f"  [{'✓' if clean else '✗'}] fold_cosine_head 里没有任何 head.bias 拷贝")


def inspect_blob(path: Path) -> IntegerModel:
    blob = path.read_bytes()
    print("\n" + "=" * 78)
    print(f"§2  blob 布局   {path.name}   {len(blob):,} B   sha256 {hashlib.sha256(blob).hexdigest()[:8]}")
    print("=" * 78)
    model = IntegerModel(blob)
    print(f"  {'字段':24s} {'元素':>6s} {'字节':>7s}")
    for label, count, size in model.fields:
        print(f"  {label:24s} {count:6,d} {size:7,d}")
    biases = [len(b) for _, b, _ in model.layers]
    print(f"\n  权重元素合计 {sum(w.size for w, _, _ in model.layers):,}"
          f"   偏置元素合计 {sum(biases)}  = {' + '.join(map(str, biases))}")
    same = check(model.bias_offsets[3] == HEAD_BIAS_OFFSET,
                 f"头部偏置偏移 {model.bias_offsets[3]} ≠ 常量 {HEAD_BIAS_OFFSET}")
    print(f"  头部偏置在文件里的偏移 = {model.bias_offsets[3]}"
          f"（脚本常量 {HEAD_BIAS_OFFSET}，{'一致' if same else '不一致！'}）")
    check(len(blob) == 4656, f"blob 长度 {len(blob)} ≠ 4656")
    check(sum(biases) == 111 and biases == [32, 32, 32, 15], f"偏置分层 {biases} ≠ 32+32+32+15")
    return model


def head_bias_ablation(path: Path, raw5: np.ndarray, labels: np.ndarray):
    """返回 (原样正确数, 清零后正确数, 偏置值, 清零后 对→错, 错→对, 预测改变的行数)。"""
    blob = bytearray(path.read_bytes())
    model = IntegerModel(bytes(blob))
    head = np.frombuffer(bytes(blob), dtype="<i4",
                         count=HEAD_BIAS_COUNT, offset=model.bias_offsets[3]).copy()
    before = model.predict(raw5)
    start = model.bias_offsets[3]
    blob[start:start + 4 * HEAD_BIAS_COUNT] = np.zeros(HEAD_BIAS_COUNT, dtype="<i4").tobytes()
    after = IntegerModel(bytes(blob)).predict(raw5)
    right_to_wrong = int(((before == labels) & (after != labels)).sum())
    wrong_to_right = int(((before != labels) & (after == labels)).sum())
    changed = int((before != after).sum())
    return (int((before == labels).sum()), int((after == labels).sum()), head,
            right_to_wrong, wrong_to_right, changed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, default=None,
                    help="evidence/ 目录，默认取 C5/evidence（随包附带）")
    ap.add_argument("--data", type=Path, default=None,
                    help="含 x_train.csv / y_train.csv 的目录，默认 <project>/../esp32p4_iat_demo/input/data")
    args = ap.parse_args()

    project = args.project.resolve()
    data = args.data
    if data is None:
        for candidate in (project.parent / "esp32p4_iat_demo" / "input" / "data", project / "input" / "data"):
            if (candidate / "x_train.csv").exists():
                data = candidate
                break
    if data is None:
        raise SystemExit("找不到 x_train.csv，请用 --data 指定目录")
    evidence = args.evidence
    if evidence is None:
        for candidate in (Path(__file__).resolve().parent.parent / "evidence",
                          project / "evidence", project / "paper_exp" / "evidence",
                          project.parent / "paper_exp" / "evidence"):
            if candidate.exists():
                evidence = candidate
                break
    if evidence is None or not evidence.exists():
        raise SystemExit("找不到 evidence 目录，请用 --evidence 指定")

    inspect_source(project)
    deployed = evidence / "deployed_int8" / "model_int8.bin"
    snapshot = evidence / "qat_snapshot" / "qat_integer_snapshot.bin"
    if not check(deployed.exists(), f"缺交付 blob {deployed}"):
        report()
    digest = hashlib.sha256(deployed.read_bytes()).hexdigest()
    check(digest == DEPLOYED_SHA256, f"交付 blob sha256 {digest[:12]}… ≠ 154100ef…")
    inspect_blob(deployed)

    raw5 = pd.read_csv(data / "x_train.csv", encoding="gbk").values[:, RAW5_COLUMNS].astype(np.float64)
    labels = pd.read_csv(data / "y_train.csv").to_numpy().ravel()

    print("\n" + "=" * 78)
    print("§3  那 15 个头部偏置从哪来、值不值")
    print("=" * 78)
    total = len(labels)
    for tag, path in (("交付版（分类头 QAT 之后）", deployed),
                      ("快照（分类头 QAT 之前）", snapshot)):
        if not check(path.exists(), f"缺文件 {path}"):
            print(f"  {tag:26s} 缺文件 {path}")
            continue
        correct, zeroed, head, r2w, w2r, changed = head_bias_ablation(path, raw5, labels)
        print(f"  {tag:26s} 原样 {correct:,}/{total:,} = {correct / total * 100:.4f}%"
              f"   清零后 {zeroed:,} = {zeroed / total * 100:.4f}%"
              f"   净 Δ = {(correct - zeroed) / total * 100:+.4f} pp")
        print(f"  {'':26s} 清零后：对→错 {r2w}，错→对 {w2r}，净 {r2w - w2r:+d}；预测改变 {changed} 行"
              f"（净值掩盖了双向变化，报告时请同时给出两个方向）")
        check(correct - zeroed == r2w - w2r, "净变化与双向计数不一致")
        print(f"  {'':26s} 头部偏置全零？ {bool(np.all(head == 0))}"
              f"   非零 {int((head != 0).sum())}/15"
              f"   |b| 中位 {int(np.median(np.abs(head)))}  最大 {int(np.abs(head).max())}")


def report() -> None:
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"核对失败 {len(FAILURES)} 项：")
        for item in FAILURES:
            print(f"  ✗ {item}")
        sys.exit(1)
    print("全部核对通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
    report()
