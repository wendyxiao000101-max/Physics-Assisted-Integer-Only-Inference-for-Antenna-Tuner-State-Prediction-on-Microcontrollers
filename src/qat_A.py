# -*- coding: utf-8 -*-
"""P95 sensitivity experiment, INT8-A only.
Network QAT and integer export; no additional head QAT.
Training/calibration use train rows, checkpoint selection uses valid rows.
Test predictions are inspected for conversion diagnostics and final evaluation.
Author-approved exception: seed43 alpha tolerance0.006; other folds0.001.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from iatopt.core import load_dataset, metrics                       # noqa: E402
from iatopt.quantization import (FoldedResidual32, cross_layer_equalize_block,  # noqa: E402
                                 load_fp32_teacher)

SH = 24


# ----------------------------------------------------------------- 折叠（带容差）

def fold_cosine_head(teacher, logit_scale: float, tolerance: float) -> tuple[FoldedResidual32, float]:
    """把 alpha=1 的混合头折叠成无偏置 Linear；返回模型与实际的 alpha。

    ``iatopt.quantization.fold_cosine_head`` 要求 alpha >= 1-1e-6，个别折会差一点点
    （实测最差 1-alpha = 1.16e-05），所以这里放开成可配置容差并把 alpha 记进结果。
    """
    classifier = teacher.network[-1]
    alpha = float(classifier.alpha.detach())
    if alpha < 1.0 - tolerance:
        raise ValueError(f"hybrid alpha {alpha:.9f} 偏离 1 太多（容差 {tolerance}）")
    model = FoldedResidual32(teacher.input_dim, 32, classifier.out_features, logit_scale)
    with torch.no_grad():
        model.stem.weight.copy_(teacher.network[0].weight)
        model.stem.bias.copy_(teacher.network[0].bias)
        block = teacher.network[2].block
        model.block_fc1.weight.copy_(block[0].weight)
        model.block_fc1.bias.copy_(block[0].bias)
        model.block_fc2.weight.copy_(block[3].weight)
        model.block_fc2.bias.copy_(block[3].bias)
        model.head.weight.copy_(F.normalize(classifier.weight, dim=1))
    return model.eval(), alpha


# ----------------------------------------------------------------- 整数运行时

class IntegerModel:
    """纯 numpy 整数推理，与交付包的 integer_numpy.py 逐位一致。"""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        assert raw[:8] == b"I8R32V02"
        pos = 8

        def get(n, dtype):
            nonlocal pos
            a = np.frombuffer(raw, dtype=dtype, count=n, offset=pos).copy()
            pos += a.nbytes
            return a

        self.offset = get(5, "<i4"); self.step = get(5, "<i4")
        self.hz = int(get(1, "<i4")[0]); self.lut = get(1024, "i1")
        self.layers = []
        for ni, no in [(5, 32), (32, 32), (32, 32), (32, 15)]:
            self.layers.append((get(ni * no, "i1").reshape(no, ni), get(no, "<i4"), get(no, "<i4")))
        self.skip = int(get(1, "<i4")[0])
        assert pos == len(raw)

    def predict(self, raw, return_logits=False, return_features=False):
        x = np.clip((2 * (raw.astype(np.int64) - self.offset) + self.step) // (2 * self.step),
                    0, 255).astype(np.int16) - 128
        maxacc = 0

        def linear(q, i, zp):
            nonlocal maxacc
            w, b, m = self.layers[i]
            acc = (q.astype(np.int32) - zp) @ w.astype(np.int32).T + b
            maxacc = max(maxacc, int(np.abs(acc.astype(np.int64)).max()))
            return acc.astype(np.int64) * m

        def rq(v):
            return (v + (1 << (SH - 1))) >> SH

        index = np.clip(rq(linear(x, 0, -128)), -512, 511).astype(np.int32) + 512
        h = self.lut[index]
        a = (np.clip(rq(linear(h, 1, self.hz - 128)), 0, 255) - 128).astype(np.int8)
        z = (np.clip(rq(linear(a, 2, -128) + (h.astype(np.int64) - (self.hz - 128)) * self.skip),
                     0, 255) - 128).astype(np.int8)
        if return_features:
            return z
        logits = rq(linear(z, 3, -128))
        return (logits if return_logits else logits.argmax(1)), maxacc


def batched_predict(model: IntegerModel, raw, chunks: int = 24, **kw):
    parts = [model.predict(b, **kw) for b in np.array_split(raw, chunks)]
    if kw.get("return_features"):
        return np.concatenate(parts)
    return np.concatenate([p[0] for p in parts]), max(p[1] for p in parts)


# ----------------------------------------------------------------- 整数 W8A8 QAT

def rnd(x):
    return x + (torch.floor(x + 0.5) - x).detach()


def quant(x, s, lo, hi):
    return rnd(x / s).clamp(lo, hi) * s


class IntegerQATNet(nn.Module):
    """折叠模型的 W8A8 QAT，量化边界与整数运行时一一对应（含 GELU 查表边界）。"""

    def __init__(self, teacher: FoldedResidual32, calibration: torch.Tensor):
        super().__init__()
        self.net = teacher
        with torch.no_grad():
            p = teacher.stem(calibration)
            h = F.gelu(p)
            a = F.relu(teacher.block_fc1(h))
            z = F.relu(h + teacher.block_fc2(a))
            steps = [p.abs().max() / 511,
                     (h.max() - torch.minimum(h.min(), torch.zeros_like(h.min()))) / 255,
                     a.max() / 255, z.max() / 255]
            self.scales = nn.Parameter(torch.stack(steps))
            self.register_buffer("hz", torch.floor(-h.min() / steps[1] + 0.5))
        self.ws = nn.ParameterList([
            nn.Parameter(layer.weight.detach().abs().amax(1, keepdim=True).clamp_min(1e-9).div(127).log())
            for layer in self.layers()
        ])

    def layers(self):
        return [self.net.stem, self.net.block_fc1, self.net.block_fc2, self.net.head]

    def linear(self, x, i, ins):
        layer = self.layers()[i]
        s = self.ws[i].exp()
        factor = (layer.weight.shape[1] * 127) ** -0.5
        s = s * factor + (s * (1 - factor)).detach()
        w = quant(layer.weight, s, -127, 127)
        b = None if layer.bias is None else quant(layer.bias, ins * s[:, 0], -2147483648, 2147483647)
        return F.linear(x, w, b)

    def forward(self, x):
        s = self.scales * 0.001 + (self.scales * 0.999).detach()
        p = quant(self.linear(x, 0, 1.0), s[0], -512, 511)
        h = quant(F.gelu(p), s[1], -self.hz, 255 - self.hz)
        a = quant(F.relu(self.linear(h, 1, s[1])), s[2], 0, 255)
        z = quant(F.relu(h + self.linear(a, 2, s[2])), s[3], 0, 255)
        return self.linear(z, 3, s[3]) * 20


def export_integer_blob(state: dict, offset, step) -> bytes:
    """把 QAT 状态写成 4,656 字节的 I8R32V02 二进制。"""
    scales = state["scales"].detach().cpu().numpy()
    hz = int(state["hz"])
    data = bytearray(b"I8R32V02")

    def put(v, dtype):
        data.extend(np.asarray(v, dtype=dtype).tobytes())

    put(offset, "<i4"); put(step, "<i4"); put([hz], "<i4")
    grid = torch.arange(-512, 512) * float(scales[0])
    lut = torch.floor(F.gelu(grid) / float(scales[1]) + 0.5).clamp(-hz, 255 - hz).numpy()
    put(lut + hz - 128, "i1")
    for i, name in enumerate(["stem", "block_fc1", "block_fc2", "head"]):
        w = state[f"net.{name}.weight"].detach().cpu().numpy()
        ws = state[f"ws.{i}"].detach().exp().cpu().numpy().ravel()
        ins = [1.0, scales[1], scales[2], scales[3]][i]
        qw = np.clip(np.floor(w / ws[:, None] + 0.5), -127, 127)
        bias = state.get(f"net.{name}.bias")
        b = np.zeros(len(w)) if bias is None else bias.detach().cpu().numpy()
        qb = np.floor(b / (ws * ins) + 0.5)
        outs = [scales[0], scales[2], scales[3], 1e-5][i]
        mult = np.floor(ws * ins / outs * (1 << SH) + 0.5)
        assert np.max(np.abs(qb)) < 2 ** 31 and np.max(mult) < 2 ** 31, "整数溢出"
        put(qw, "i1"); put(qb, "<i4"); put(mult, "<i4")
    put([round(float(scales[1] / scales[3]) * (1 << SH))], "<i4")
    assert len(data) == 4656, len(data)
    return bytes(data)


# ----------------------------------------------------------------- 分类头 QAT

class QuantisedHead(nn.Module):
    def __init__(self, model: IntegerModel, device):
        super().__init__()
        w, b, m = model.layers[-1]
        self.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32, device=device))
        self.register_buffer("mult", torch.tensor(m, dtype=torch.float64, device=device))

    def forward(self, x):
        acc = F.linear(x, rnd(self.weight).clamp(-127, 127), rnd(self.bias))
        return rnd(acc.double() * self.mult / (1 << SH)).float()

    def export(self, source: bytes) -> bytes:
        data = bytearray(source)
        start = len(data) - 604
        data[start:start + 480] = torch.floor(self.weight.detach() + 0.5).clamp(-127, 127) \
            .cpu().numpy().astype("i1").tobytes()
        data[start + 480:start + 540] = torch.floor(self.bias.detach() + 0.5) \
            .cpu().numpy().astype("<i4").tobytes()
        return bytes(data)


# ----------------------------------------------------------------- 选择准则


def selection_score(pred: torch.Tensor, target: torch.Tensor, mode: str,
                    classes: int = 15) -> float:
    """检查点选择准则。

    ``accuracy`` 是未加权准确率：占比 0.6% 的稀有类整体失效只让它掉约 0.6 pp，
    选择器察觉不到，实测导致过一折的第 10 类从 53.85% 塌到 10.44%。
    ``balanced`` 是各类召回的平均，每个类等权，同样的失效会让它掉 1/15 = 6.7 pp。
    """
    if mode == "accuracy":
        return float((pred == target).float().mean())
    total = torch.zeros(classes, device=pred.device)
    correct = torch.zeros(classes, device=pred.device)
    ones = torch.ones(len(target), device=pred.device)
    total.index_add_(0, target, ones)
    correct.index_add_(0, target, (pred == target).float())
    present = total > 0
    return float((correct[present] / total[present]).mean())


# ----------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold_dir", type=Path, help="--ablate-export-dir 导出的某一个折目录")
    ap.add_argument("--x", type=Path, default=ROOT.parent / "esp32p4_iat_demo/input/data/x_train.csv")
    ap.add_argument("--y", type=Path, default=ROOT.parent / "esp32p4_iat_demo/input/data/y_train.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone-epochs", type=int, default=300)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-batch", type=int, default=512)
    ap.add_argument("--backbone-focal", type=float, default=1.0)
    ap.add_argument("--head-epochs", type=int, default=2000)
    ap.add_argument("--head-lr", type=float, default=0.1)
    ap.add_argument("--head-focal", type=float, default=1.0)
    ap.add_argument("--alpha-tolerance", type=float, default=1e-3)
    ap.add_argument("--selection", choices=["accuracy", "balanced"], default="balanced",
                    help="骨干与头部 QAT 的检查点选择准则；balanced 用各类召回均值，"
                         "避免稀有类整体失效被未加权准确率掩盖")
    ap.add_argument("--reuse-backbone", type=Path, default=None,
                    help="复用已导出的骨干整数模型（跳过骨干 QAT），传 model_int8_backbone.bin 的路径；"
                         "给目录名时自动找 <fold>/<该名>/model_int8_backbone.bin")
    ap.add_argument("--out-name", default="int8", help="输出子目录名")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(20260908)
    device = torch.device(args.device)
    fold_dir = args.fold_dir.resolve()
    out = fold_dir / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    split = np.load(fold_dir / "split.npz")
    train_idx, valid_idx, test_idx = split["train"], split["valid"], split["test"]
    bundle = load_dataset(args.x, args.y)
    raw5 = bundle.raw[:, [0, 1, 2, 4, 5]].astype(np.float64)
    labels = bundle.labels
    assert np.equal(raw5, np.round(raw5)).all(), "原始特征应为整数"
    meta = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    print(f"折 {meta['protocol']} seed={meta['seed']}   "
          f"train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}", flush=True)

    # ---- 1. 折叠 + 跨层均衡
    teacher, scaler, teacher_meta = load_fp32_teacher(fold_dir)
    if meta["seed"] == 43:
        # Author approved this specifically disclosed P95 fold exception.
        args.alpha_tolerance = 0.006
    folded, alpha = fold_cosine_head(teacher, float(teacher_meta.get("head_scale", 20)),
                                     args.alpha_tolerance)
    cross_layer_equalize_block(folded)
    normalized = torch.tensor((raw5 - scaler.mean) / scaler.scale, dtype=torch.float32)
    with torch.no_grad():
        fp32_pred = folded(normalized).argmax(1).numpy()
    with torch.no_grad():
        original_pred = teacher(normalized).argmax(1).numpy()
    np.savez_compressed(out / "predictions.npz", fp32=original_pred, folded=fp32_pred, test=test_idx)
    fold_changes = int((original_pred != fp32_pred).sum())
    fp32_test = metrics(labels[test_idx], fp32_pred[test_idx])
    print(f"  alpha={alpha:.9f}   FP32(折叠) 留出集 acc={fp32_test['accuracy']*100:.4f}%", flush=True)

    # ---- 2. 输入整数化（只用训练行统计）
    offset = raw5[train_idx].min(0).astype(np.int32)
    step = np.maximum(np.ceil((raw5[train_idx].max(0) - offset) / 255), 1).astype(np.int32)
    q = np.clip(np.floor((raw5 - offset) / step + 0.5), 0, 255).astype(np.float32)
    clipped = int(((raw5 - offset) / step < 0).any(1).sum() + ((raw5 - offset) / step > 255).any(1).sum())
    with torch.no_grad():
        folded.stem.bias.add_(folded.stem.weight @ torch.tensor(
            (offset - scaler.mean) / scaler.scale, dtype=torch.float32))
        folded.stem.weight.mul_(torch.tensor(step / scaler.scale, dtype=torch.float32))

    x = torch.tensor(q, device=device)
    y = torch.tensor(labels, device=device)
    tr = torch.tensor(train_idx, device=device)
    va = torch.tensor(valid_idx if len(valid_idx) else train_idx, device=device)

    # ---- 3. 整数 W8A8 QAT（骨干），只在训练行上训练、验证行上选点
    reuse = args.reuse_backbone
    if reuse is not None:
        if reuse.is_dir() or not reuse.suffix:
            reuse = fold_dir / reuse.name / "model_int8_backbone.bin"
        blob = Path(reuse).read_bytes()
        (out / "model_int8_backbone.bin").write_bytes(blob)
        print(f"  复用骨干整数模型 {reuse}（跳过骨干 QAT）", flush=True)
        started = time.time()
    else:
        blob, started = None, None
    if blob is None:
     model = IntegerQATNet(folded.to(device), x[tr][:8192]).to(device)
     slow = [model.net.stem.weight]
     other = [p for p in model.parameters() if p is not model.net.stem.weight]
     opt = torch.optim.Adam([{"params": slow, "lr": args.backbone_lr * 0.01},
                             {"params": other, "lr": args.backbone_lr}])
     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.backbone_epochs,
                                                        eta_min=args.backbone_lr * 1e-4)
     best, best_state, started = -1.0, None, time.time()
     for epoch in range(args.backbone_epochs + 1):
         if epoch:
             if epoch == int(args.backbone_epochs * 0.65):
                 model.scales.requires_grad_(False)
                 for p in model.ws:
                     p.requires_grad_(False)
             model.train()
             perm = tr[torch.randperm(len(tr), device=device)]
             for idx in perm.split(args.backbone_batch):
                 opt.zero_grad(set_to_none=True)
                 ce = F.cross_entropy(model(x[idx]), y[idx], reduction="none")
                 (((1 - torch.exp(-ce)) ** args.backbone_focal) * ce).mean().backward()
                 opt.step()
             sched.step()
         model.eval()
         with torch.no_grad():
             pred = torch.cat([model(b).argmax(1) for b in x[va].split(8192)])
             score = selection_score(pred, y[va], args.selection)
         if score > best:
             best = score
             best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
         if epoch % 25 == 0:
             print(f"  骨干 QAT {epoch}/{args.backbone_epochs}  valid={score*100:.4f}%  "
                   f"best={best*100:.4f}%", flush=True)
     blob = export_integer_blob(best_state, offset, step)
     (out / "model_int8_backbone.bin").write_bytes(blob)

    backbone = IntegerModel(out / "model_int8_backbone.bin")
    pred_backbone, _ = batched_predict(backbone, raw5)
    print(f"  骨干整数模型 留出集 acc={float((pred_backbone[test_idx]==labels[test_idx]).mean())*100:.4f}%",
          flush=True)

    # Variant A: no additional head refinement.
    hbest_blob = blob
    (out / "model_int8.bin").write_bytes(hbest_blob)

    # ---- 5. 只在这一步碰 test
    final = IntegerModel(out / "model_int8.bin")
    pred, maxacc = batched_predict(final, raw5)
    np.savez_compressed(out / "predictions.npz", fp32=original_pred, folded=fp32_pred, int8=pred, test=test_idx)
    int8_test = metrics(labels[test_idx], pred[test_idx])
    report = {
        "protocol": meta["protocol"], "seed": meta["seed"], "split_run": meta["split_run"],
        "rows_train": int(len(train_idx)), "rows_valid": int(len(valid_idx)),
        "rows_test": int(len(test_idx)),
        "hybrid_alpha": alpha,
        "fold_top1_changes_all_rows": fold_changes,
        "selection_criterion": args.selection,
        "alpha_tolerance_used": args.alpha_tolerance,
        "backbone_reused": str(args.reuse_backbone) if args.reuse_backbone else "",
        "input_rows_clipped_by_train_range": clipped,
        "int32_accumulator_max": maxacc,
        "model_bytes": len(hbest_blob),
        "model_sha256": hashlib.sha256(hbest_blob).hexdigest(),
        "seconds": time.time() - started,
        "fp32_holdout": {k: v for k, v in fp32_test.items()
                         if k not in {"confusion_matrix", "per_class_recall"}},
        "int8_backbone_holdout_accuracy": float((pred_backbone[test_idx] == labels[test_idx]).mean()),
        "int8_holdout": {k: v for k, v in int8_test.items()
                         if k not in {"confusion_matrix", "per_class_recall"}},
        "int8_holdout_per_class_recall": int8_test["per_class_recall"],
        "quantisation_cost_pp": (int8_test["accuracy"] - fp32_test["accuracy"]) * 100,
        "agreement_with_fp32_on_holdout": float((pred[test_idx] == fp32_pred[test_idx]).mean()),
        "scope": "Training/calibration use train; checkpoint selection uses valid; test is used for conversion diagnostics and final evaluation, not gradient updates or checkpoint selection",
    }
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  FP32 留出集 {fp32_test['accuracy']*100:.4f}%   "
          f"INT8 留出集 {int8_test['accuracy']*100:.4f}%   "
          f"量化代价 {report['quantisation_cost_pp']:+.4f} pp", flush=True)
    print(f"  写入 {out/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
