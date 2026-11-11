# -*- coding: utf-8 -*-
"""把 sklearn 树模型编码成真实的紧凑字节表示，并**只用这些字节**做推理。

回应核查意见 §三.6：此前 Pareto 图的森林精度来自 sklearn `predict`（软投票：
各树类概率取平均），而体积按"每叶只存 1 个类别"（硬投票）估计——两者不是同一个实现。
这里给出三种真实编码，每种都有自己的解码器，精度与字节数出自同一份表示：

  hard      叶子只存类别（1 B），各树多数投票，平票取最小类号
  soft_u8   叶子存 15 个 uint8 类概率（round(p·255)），累加后取 argmax
  gbdt_f32  HistGradientBoosting：叶子存 float32 增量，按轮次×类别累加

字节布局（小端）：
  文件头   8 B   magic 'TREE' | u16 树数 | u8 类数 | u8 编码类型
  偏移表   4 B × 树数（每棵树在节点区的起始偏移，u32）
  [gbdt]   4 B × 类数（float32 基线）
  节点区   先序遍历；左孩子紧跟父节点
    内部节点  u8 特征号(<0x80) | i16 阈值 | u16 右孩子相对偏移    = 5 B
    叶  hard     u8 (0x80 | 类别)                                   = 1 B
        soft_u8  u8 0x80 | 15 × u8                                  = 16 B
        gbdt_f32 u8 0x80 | f32                                      = 5 B

阈值：输入全为整数（Q15 反射系数与 MHz 频点），sklearn 的阈值 t 是两个整数间的中点，
对整数 x 有 x ≤ t ⟺ x ≤ ⌊t⌋，所以存 int16(⌊t⌋) 是**无损**的；编码时断言范围。
"""
from __future__ import annotations

import struct

import numpy as np

HARD, SOFT_U8, GBDT_F32 = 0, 1, 2
LEAF = 0x80


def _check_int16(value: float) -> int:
    q = int(np.floor(value))
    if not -32768 <= q <= 32767:
        raise ValueError(f"阈值 {value} 超出 int16")
    return q


def _encode_sklearn_tree(tree, mode: int, classes: int) -> bytes:
    """sklearn DecisionTree 的 tree_ 对象 -> 先序字节。"""
    left, right = tree.children_left, tree.children_right
    feature, threshold, value = tree.feature, tree.threshold, tree.value
    out = bytearray()

    def emit(node: int) -> None:
        if left[node] == -1:
            dist = value[node].reshape(-1)[:classes]
            dist = dist / dist.sum() if dist.sum() > 0 else dist
            if mode == HARD:
                out.append(LEAF | int(np.argmax(dist)))
            else:
                out.append(LEAF)
                out.extend(np.round(dist * 255).astype(np.uint8).tobytes())
            return
        start = len(out)
        out.extend(struct.pack("<Bh", int(feature[node]), _check_int16(threshold[node])))
        patch = len(out)
        out.extend(b"\x00\x00")
        emit(left[node])
        offset = len(out) - start
        if offset > 0xFFFF:
            raise ValueError("单棵树超过 64 KiB，u16 偏移放不下")
        out[patch:patch + 2] = struct.pack("<H", offset)
        emit(right[node])

    emit(0)
    return bytes(out)


def _encode_hgb_predictor(nodes) -> bytes:
    out = bytearray()

    def emit(node: int) -> None:
        if nodes["is_leaf"][node]:
            out.append(LEAF)
            out.extend(struct.pack("<f", float(nodes["value"][node])))
            return
        start = len(out)
        out.extend(struct.pack("<Bh", int(nodes["feature_idx"][node]),
                               _check_int16(nodes["num_threshold"][node])))
        patch = len(out)
        out.extend(b"\x00\x00")
        emit(int(nodes["left"][node]))
        offset = len(out) - start
        if offset > 0xFFFF:
            raise ValueError("单棵树超过 64 KiB")
        out[patch:patch + 2] = struct.pack("<H", offset)
        emit(int(nodes["right"][node]))

    emit(0)
    return bytes(out)


def _emit_generic(node, out: bytearray) -> None:
    """node = ('leaf', value) 或 ('split', 特征号, 整数阈值, 左, 右)；x[特征] <= 阈值 走左。"""
    if node[0] == "leaf":
        out.append(LEAF)
        out.extend(struct.pack("<f", float(node[1])))
        return
    _, feature, thr, left, right = node
    start = len(out)
    out.extend(struct.pack("<Bh", int(feature), _check_int16(thr)))
    patch = len(out)
    out.extend(b"\x00\x00")
    _emit_generic(left, out)
    offset = len(out) - start
    if offset > 0xFFFF:
        raise ValueError("单棵树超过 64 KiB")
    out[patch:patch + 2] = struct.pack("<H", offset)
    _emit_generic(right, out)


def encode_generic_gbdt(trees, classes: int, baseline) -> bytes:
    """XGBoost / LightGBM 等外部 GBDT：第 t 棵树属于类 t % classes，布局与 GBDT_F32 相同。"""
    blobs = []
    for tree in trees:
        out = bytearray()
        _emit_generic(tree, out)
        blobs.append(bytes(out))
    header = b"TREE" + struct.pack("<HBB", len(blobs), classes, GBDT_F32)
    offsets, cursor = [], 0
    for t in blobs:
        offsets.append(cursor)
        cursor += len(t)
    extra = np.asarray(baseline, dtype=np.float32).reshape(classes).tobytes()
    return header + struct.pack(f"<{len(blobs)}I", *offsets) + extra + b"".join(blobs)


def leaf_sums(blob: bytes, x: np.ndarray) -> np.ndarray:
    """GBDT_F32 blob 不加基线的逐类叶值和（用来从原生 margin 反推基线）。"""
    n_trees, classes, mode = struct.unpack_from("<HBB", blob, 4)
    if mode != GBDT_F32:
        raise ValueError("只用于 GBDT_F32")
    zero = bytearray(blob)
    base = 8 + 4 * n_trees
    zero[base:base + 4 * classes] = bytes(4 * classes)
    return predict(bytes(zero), x, return_scores=True)


def encode(model, mode: int, classes: int = 15) -> bytes:
    if mode == GBDT_F32:
        trees = [_encode_hgb_predictor(p.nodes) for stage in model._predictors for p in stage]
        baseline = np.asarray(model._baseline_prediction, dtype=np.float32).reshape(-1)
        extra = baseline.tobytes()
    else:
        estimators = getattr(model, "estimators_", [model])
        trees = [_encode_sklearn_tree(e.tree_, mode, classes) for e in estimators]
        extra = b""
    header = b"TREE" + struct.pack("<HBB", len(trees), classes, mode)
    offsets, cursor = [], 0
    for t in trees:
        offsets.append(cursor)
        cursor += len(t)
    return header + struct.pack(f"<{len(trees)}I", *offsets) + extra + b"".join(trees)


def predict(blob: bytes, x: np.ndarray, return_scores: bool = False) -> np.ndarray:
    """只读字节做推理。x 必须是与训练时同列序的整数输入。"""
    if blob[:4] != b"TREE":
        raise ValueError("不是 TREE blob")
    n_trees, classes, mode = struct.unpack_from("<HBB", blob, 4)
    offsets = np.array(struct.unpack_from(f"<{n_trees}I", blob, 8), dtype=np.int64)
    base = 8 + 4 * n_trees
    baseline = None
    if mode == GBDT_F32:
        baseline = np.frombuffer(blob, dtype="<f4", count=classes, offset=base).astype(np.float64)
        base += 4 * classes
    data = np.frombuffer(blob, dtype=np.uint8, offset=base)
    x = np.asarray(x)
    xi = np.floor(x).astype(np.int64)
    if not np.array_equal(xi, x):
        raise ValueError("输入必须是整数")
    n = len(x)
    rows = np.arange(n)
    score = np.zeros((n, classes), dtype=np.float64 if mode == GBDT_F32 else np.int64)

    # 预解析：每个字节位置上若是内部节点，其阈值与右偏移（向量化，非节点位置的值不会被访问）
    padded = np.concatenate([data, np.zeros(8, dtype=np.uint8)]).astype(np.int64)
    thr = (padded[1:-7] | (padded[2:-6] << 8))
    thr = np.where(thr >= 32768, thr - 65536, thr)
    off = padded[3:-5] | (padded[4:-4] << 8)

    for t, start in enumerate(offsets):
        pos = np.full(n, start, dtype=np.int64)
        active = data[pos] < LEAF
        while active.any():
            p = pos[active]
            f = data[p].astype(np.int64)
            go_left = xi[rows[active], f] <= thr[p]
            pos[active] = np.where(go_left, p + 5, p + off[p])
            active = data[pos] < LEAF
        if mode == HARD:
            score[rows, data[pos] & 0x7F] += 1
        elif mode == SOFT_U8:
            idx = pos[:, None] + 1 + np.arange(classes)[None, :]
            score += data[idx].astype(np.int64)
        else:
            four = padded[(pos + 1)[:, None] + np.arange(4)[None, :]].astype(np.uint8)
            score[:, t % classes] += four.copy().view("<f4").reshape(-1).astype(np.float64)
    if mode == GBDT_F32:
        score += baseline
    if return_scores:
        return score
    return score.argmax(1)       # 平票取最小类号


class ScalarRunner:
    """纯 Python 逐样本推理（只读字节，不用 numpy），用于与 INT8 纯 Python 参考实现同口径计时。

    predict_one(x) 返回 (类别, 本样本访问的内部节点数)。
    """

    def __init__(self, blob: bytes):
        if blob[:4] != b"TREE":
            raise ValueError("不是 TREE blob")
        self.n_trees, self.classes, self.mode = struct.unpack_from("<HBB", blob, 4)
        self.roots = list(struct.unpack_from(f"<{self.n_trees}I", blob, 8))
        base = 8 + 4 * self.n_trees
        self.baseline = [0.0] * self.classes
        if self.mode == GBDT_F32:
            self.baseline = list(struct.unpack_from(f"<{self.classes}f", blob, base))
            base += 4 * self.classes
        self.data = bytes(blob[base:])

    def predict_one(self, x):
        d, classes, mode = self.data, self.classes, self.mode
        score = list(self.baseline) if mode == GBDT_F32 else [0] * classes
        visited = 0
        for t, p in enumerate(self.roots):
            while d[p] < LEAF:
                thr = d[p + 1] | (d[p + 2] << 8)
                if thr >= 32768:
                    thr -= 65536
                visited += 1
                if x[d[p]] <= thr:
                    p += 5
                else:
                    p += d[p + 3] | (d[p + 4] << 8)
            if mode == HARD:
                score[d[p] & 0x7F] += 1
            elif mode == SOFT_U8:
                for c in range(classes):
                    score[c] += d[p + 1 + c]
            else:
                score[t % classes] += struct.unpack_from("<f", d, p + 1)[0]
        best = 0
        for c in range(1, classes):
            if score[c] > score[best]:
                best = c
        return best, visited
