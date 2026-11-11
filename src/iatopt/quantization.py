from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import IATMLP, Standardizer


class FoldedResidual32(nn.Module):
    """Quantization-friendly equivalent of residual32_gelu_relu_hybrid.

    The trained hybrid gate is expected to be saturated at the cosine branch.
    For Top-1 inference, the positive sample-wise cosine denominator can be
    removed. Class weights are normalized once and the classifier becomes a
    regular bias-free Linear. ``logit_scale`` is also positive and therefore
    has no effect on argmax; it keeps QAT logits in a convenient CE/KD range.
    """

    def __init__(self, input_dim: int = 5, width: int = 32, classes: int = 15, logit_scale: float = 20.0,
                 branch_clip_min: float | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.classes = int(classes)
        self.logit_scale = float(logit_scale)
        self.branch_clip_min = None if branch_clip_min is None else float(branch_clip_min)
        self.stem = nn.Linear(input_dim, width)
        self.gelu = nn.GELU()
        self.block_fc1 = nn.Linear(width, width)
        self.block_relu = nn.ReLU()
        self.block_fc2 = nn.Linear(width, width)
        self.residual_relu = nn.ReLU()
        self.head = nn.Linear(width, classes, bias=False)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        stem = self.gelu(self.stem(value))
        branch = self.block_fc2(self.block_relu(self.block_fc1(stem)))
        if self.branch_clip_min is not None:
            branch = torch.clamp_min(branch, self.branch_clip_min)
        return self.residual_relu(stem + branch)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(value)) * self.logit_scale


def _ste_round(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


def fake_quant_dynamic_activation(value: torch.Tensor) -> torch.Tensor:
    """Row-wise affine fake quantization matching batch-1 oneDNN Linear.

    oneDNN's dynamic Linear path uses the reduced unsigned activation range
    [0, 127] on this backend, even though its placeholder observer advertises
    quint8. Weights remain signed 8-bit qint8.
    """
    if value.ndim != 2:
        raise ValueError("dynamic activation fake quantization expects a 2-D tensor")
    minimum = value.detach().amin(dim=1, keepdim=True).clamp_max(0.0)
    maximum = value.detach().amax(dim=1, keepdim=True).clamp_min(0.0)
    scale = ((maximum - minimum) / 127.0).clamp_min(1e-8)
    zero_point = (-minimum / scale).round().clamp(0, 127)
    quantized = _ste_round(value / scale + zero_point).clamp(0, 127)
    return (quantized - zero_point) * scale


def fake_quant_per_channel_weight(weight: torch.Tensor) -> torch.Tensor:
    """Output-channel symmetric INT8 fake quantization matching oneDNN.

    PyTorch's symmetric qint8 observer uses the midpoint of the [-128, 127]
    range, hence an effective denominator of 127.5 rather than 127.
    """
    absmax = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (absmax[:, 0] / 127.5).to(dtype=torch.float32)
    zero_point = torch.zeros(len(scale), dtype=torch.int32, device=weight.device)
    return torch.fake_quantize_per_channel_affine(
        weight, scale, zero_point, 0, -128, 127,
    )


class FullDynamicQATResidual32(FoldedResidual32):
    """All-linear W8A8 QAT model with per-sample dynamic activation scales.

    Every Linear weight is fake-quantized to per-output-channel INT8. Each
    Linear input is fake-quantized independently per sample, matching the
    deployed batch-1 oneDNN DynamicQuantizedLinear semantics. Nonlinear and
    residual elementwise operators remain floating point; parameter and MAC
    budgets are unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fake_quant_enabled = True
        self.activation_fake_quant_enabled = True

    def _linear(self, value: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        inputs = fake_quant_dynamic_activation(value) if self.activation_fake_quant_enabled else value
        weight = fake_quant_per_channel_weight(layer.weight) if self.weight_fake_quant_enabled else layer.weight
        return F.linear(inputs, weight, layer.bias)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        stem = self.gelu(self._linear(value, self.stem))
        branch = self._linear(self.block_relu(self._linear(stem, self.block_fc1)), self.block_fc2)
        if self.branch_clip_min is not None:
            branch = torch.clamp_min(branch, self.branch_clip_min)
        return self.residual_relu(stem + branch)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._linear(self.forward_features(value), self.head) * self.logit_scale

    @classmethod
    def from_folded(cls, source: FoldedResidual32) -> "FullDynamicQATResidual32":
        model = cls(
            source.input_dim, source.width, source.classes, source.logit_scale,
            branch_clip_min=source.branch_clip_min,
        )
        model.load_state_dict(source.state_dict())
        return model

    def to_folded(self) -> FoldedResidual32:
        model = FoldedResidual32(
            self.input_dim, self.width, self.classes, self.logit_scale,
            branch_clip_min=self.branch_clip_min,
        )
        model.load_state_dict(self.state_dict())
        return model.eval()


@dataclass(frozen=True)
class FoldReport:
    alpha: float
    prediction_agreement: float
    max_logit_argmax_margin_error: float
    original_params: int
    folded_params: int
    original_macs: int
    folded_macs: int


def linear_macs(model: nn.Module) -> int:
    return int(sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear)))


def load_fp32_teacher(model_dir: str | Path) -> tuple[IATMLP, Standardizer, dict]:
    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = IATMLP(
        input_dim=int(metadata["input_dim"]),
        architecture=str(metadata["architecture"]),
        classes=int(metadata["num_classes"]),
        dropout=float(metadata.get("dropout", 0.0)),
        head_scale=float(metadata.get("head_scale", 16.0)),
        hybrid_alpha=float(metadata.get("hybrid_alpha_init", 0.5)),
    )
    state = torch.load(directory / "model_fp32.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, Standardizer.load(directory / "scaler.pkl"), metadata


def fold_cosine_head(teacher: IATMLP, logit_scale: float = 20.0) -> FoldedResidual32:
    if teacher.architecture != "residual32_gelu_relu_hybrid":
        raise ValueError(f"unsupported teacher architecture: {teacher.architecture}")
    classifier = teacher.network[-1]
    if not hasattr(classifier, "alpha") or float(classifier.alpha.detach()) < 1.0 - 1e-6:
        raise ValueError("hybrid classifier is not saturated at the cosine branch")
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
    return model.eval()


def cross_layer_equalize_block(model: FoldedResidual32, max_scale: float = 16.0) -> np.ndarray:
    """Exactly equalize block_fc1 -> ReLU -> block_fc2 channel ranges."""
    with torch.no_grad():
        first_range = model.block_fc1.weight.abs().amax(dim=1)
        second_range = model.block_fc2.weight.abs().amax(dim=0)
        valid = (first_range > 1e-12) & (second_range > 1e-12)
        scales = torch.ones_like(first_range)
        scales[valid] = torch.sqrt(second_range[valid] / first_range[valid])
        scales.clamp_(1.0 / max_scale, max_scale)
        model.block_fc1.weight.mul_(scales[:, None])
        model.block_fc1.bias.mul_(scales)
        model.block_fc2.weight.div_(scales[None, :])
    return scales.cpu().numpy()


def verify_fold(
    teacher: IATMLP,
    folded: FoldedResidual32,
    values: torch.Tensor,
    batch_size: int = 4096,
) -> FoldReport:
    teacher.eval()
    folded.eval()
    matches = 0
    count = 0
    max_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            teacher_logits = teacher(batch)
            folded_logits = folded(batch)
            teacher_pred = teacher_logits.argmax(1)
            folded_pred = folded_logits.argmax(1)
            matches += int((teacher_pred == folded_pred).sum())
            count += len(batch)
            # Both models preserve the same class ordering; compare the top-two
            # class margin after removing the teacher's sample-wise scale.
            t_sorted = teacher_logits.topk(2, dim=1).values
            f_sorted = folded_logits.topk(2, dim=1).values
            t_margin = t_sorted[:, 0] - t_sorted[:, 1]
            f_margin = f_sorted[:, 0] - f_sorted[:, 1]
            normalized_error = (t_margin / t_margin.abs().mean().clamp_min(1e-12) -
                                f_margin / f_margin.abs().mean().clamp_min(1e-12)).abs().max()
            max_error = max(max_error, float(normalized_error))
    original_params = sum(p.numel() for p in teacher.parameters())
    folded_params = sum(p.numel() for p in folded.parameters())
    # HybridLinear evaluates two 32x15 projections.
    original_macs = 5 * 32 + 32 * 32 + 32 * 32 + 2 * 32 * 15
    return FoldReport(
        alpha=float(teacher.network[-1].alpha.detach()),
        prediction_agreement=matches / max(count, 1),
        max_logit_argmax_margin_error=max_error,
        original_params=original_params,
        folded_params=folded_params,
        original_macs=original_macs,
        folded_macs=linear_macs(folded),
    )


def clone_for_conversion(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).cpu().eval()
