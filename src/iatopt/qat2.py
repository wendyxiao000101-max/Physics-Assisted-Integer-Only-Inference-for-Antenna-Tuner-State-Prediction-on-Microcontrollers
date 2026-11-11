"""LSQ+ style INT8 quantisation-aware training for the residual32 hybrid network.

Everything here stays strictly 8-bit and quantises exactly the tensors
``iatopt.core.attach_fake_quant`` already quantises: symmetric per-output-channel
INT8 weights on the four Linear layers, asymmetric UINT8 activations at the four
operator boundaries (input, stem output, residual inner activation, block output).

The difference is the step size.  The min/max scheme reads the step off an EMA of
the batch extrema, so a single outlier weight or activation burns the whole grid
and the network can never argue back.  Here every step size is a *trained*
parameter (Esser et al., "Learned Step Size Quantization", ICLR 2020; Bhalgat et
al., "LSQ+", CVPRW 2020), initialised by a per-channel MSE-optimal clipping search
rather than by max|w|.  The zero point is snapped to an integer inside the forward
pass, so what training sees is bit-for-bit what ``model_int8.npz`` will contain.

Exported artefact format is unchanged, so ``infer_int8.py`` reads it as-is.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return torch.round(value)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _round_ste(value: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(value)


def _grad_scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    """Pass ``value`` through unchanged but multiply its gradient by ``factor``."""
    return (value - value * factor).detach() + value * factor


class LSQWeightFakeQuant(nn.Module):
    """Symmetric per-output-channel INT8 weight quantiser with a learned step.

    ``levels`` stays at 127 (not 128) so the exported integers match the symmetric
    convention ``infer_int8.py`` already assumes.
    """

    def __init__(self, out_features: int, bits: int = 8, search_points: int = 33):
        super().__init__()
        self.levels = 2 ** (bits - 1) - 1
        self.search_points = int(search_points)
        self.log_step = nn.Parameter(torch.zeros(out_features, 1))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialise(self, weight: torch.Tensor) -> None:
        """Per-channel MSE-optimal clipping search over fractions of max|w|."""
        amax = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        best_step = amax / self.levels
        best_error = None
        for fraction in torch.linspace(0.30, 1.0, self.search_points, device=weight.device):
            step = (amax * fraction) / self.levels
            recovered = torch.clamp(torch.round(weight / step), -self.levels, self.levels) * step
            error = (recovered - weight).pow(2).mean(dim=1, keepdim=True)
            if best_error is None:
                best_error, best_step = error, step
            else:
                better = error < best_error
                best_step = torch.where(better, step, best_step)
                best_error = torch.where(better, error, best_error)
        self.log_step.copy_(best_step.clamp_min(1e-12).log())
        self.ready.fill_(True)

    def step_size(self) -> torch.Tensor:
        return self.log_step.exp()

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.log_step.device != weight.device:
            self.to(weight.device)
        if not bool(self.ready):
            self.initialise(weight.detach())
        step = self.step_size()
        if bool(self.frozen):
            step = step.detach()
        else:
            per_channel = max(weight.numel() // max(weight.shape[0], 1), 1)
            step = _grad_scale(step, 1.0 / math.sqrt(per_channel * self.levels))
        return torch.clamp(_round_ste(weight / step), -self.levels, self.levels) * step

    @torch.no_grad()
    def export(self, weight: torch.Tensor):
        step = self.step_size()
        integers = torch.clamp(torch.round(weight / step), -self.levels, self.levels)
        return integers, step


class LSQActFakeQuant(nn.Module):
    """Asymmetric UINT8 activation quantiser with a learned step and offset.

    The integer zero point is produced inside the forward pass, so the training
    graph and the exported artefact use the identical grid.
    """

    def __init__(self, bits: int = 8, percentile: float = 0.999):
        super().__init__()
        self.levels = 2 ** bits - 1
        self.percentile = float(percentile)
        self.log_step = nn.Parameter(torch.zeros(()))
        self.offset = nn.Parameter(torch.zeros(()))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialise(self, value: torch.Tensor) -> None:
        flat = value.detach().reshape(-1).float()
        if 0.0 < self.percentile < 1.0 and flat.numel() <= 4_000_000:
            high = torch.quantile(flat, self.percentile)
            low = torch.quantile(flat, 1.0 - self.percentile)
        else:
            high, low = flat.amax(), flat.amin()
        low = torch.minimum(low, torch.zeros_like(low))       # keep exact zero representable
        high = torch.maximum(high, low + 1e-6)
        self.log_step.copy_(((high - low) / self.levels).clamp_min(1e-12).log())
        self.offset.copy_(low)
        self.ready.fill_(True)

    def step_size(self) -> torch.Tensor:
        return self.log_step.exp()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.log_step.device != value.device:
            self.to(value.device)
        if not bool(self.ready):
            self.initialise(value)
        step, offset = self.step_size(), self.offset
        if bool(self.frozen):
            step, offset = step.detach(), offset.detach()
        else:
            factor = 1.0 / math.sqrt(max(value.numel(), 1) * self.levels)
            step, offset = _grad_scale(step, factor), _grad_scale(offset, factor)
        zero_point = torch.clamp(_round_ste(-offset / step), 0, self.levels)
        quantised = torch.clamp(_round_ste(value / step) + zero_point, 0, self.levels)
        return (quantised - zero_point) * step

    @torch.no_grad()
    def export(self) -> dict:
        step = float(self.step_size())
        zero_point = float(min(max(round(-float(self.offset) / step), 0), self.levels))
        return {"scale": step, "zero_point": zero_point, "levels": self.levels}


def _quantised_linears(model):
    network = model.network
    block = network[2]
    return network[0], block.block[0], block.block[3], network[3]


def attach_lsq_fake_quant(model, weight_bits: int = 8, activation_bits: int = 8,
                          act_percentile: float = 0.999) -> dict:
    """Drop-in replacement for ``attach_fake_quant`` with learned step sizes."""
    from torch.nn.utils import parametrize

    if not model.architecture.startswith("residual"):
        raise ValueError("LSQ fake quantisation targets the residual architectures")
    network = model.network
    block = network[2]
    device = next(model.parameters()).device
    for linear in _quantised_linears(model):
        # register_parametrization runs the quantiser once on the spot, so it has to be
        # on the weight's device already - building it on the CPU raises a device mismatch.
        parametrize.register_parametrization(
            linear, "weight",
            LSQWeightFakeQuant(linear.out_features, weight_bits).to(linear.weight.device),
        )

    observers = nn.ModuleDict({
        name: LSQActFakeQuant(activation_bits, act_percentile)
        for name in ("input", "stem", "inner", "feature")
    }).to(device)
    model.activation_observers = observers

    network[0].register_forward_pre_hook(lambda m, args: (observers["input"](args[0]),))
    network[1].register_forward_hook(lambda m, args, out: observers["stem"](out))
    block.block[1].register_forward_hook(lambda m, args, out: observers["inner"](out))
    block.register_forward_hook(lambda m, args, out: observers["feature"](out))
    return {"weight_bits": weight_bits, "activation_bits": activation_bits,
            "scheme": "lsq", "act_percentile": act_percentile,
            "quantisation_points": ["input", "stem", "inner", "feature"]}


def freeze_quantisers(model) -> int:
    """Stop training the step sizes so the weights settle on a fixed integer grid."""
    frozen = 0
    for module in model.modules():
        if isinstance(module, (LSQWeightFakeQuant, LSQActFakeQuant)):
            module.frozen.fill_(True)
            module.log_step.requires_grad_(False)
            if isinstance(module, LSQActFakeQuant):
                module.offset.requires_grad_(False)
            frozen += 1
    return frozen


def quantiser_parameters(model):
    """The step/offset parameters, so they can get their own optimiser group."""
    out = []
    for module in model.modules():
        if isinstance(module, LSQWeightFakeQuant):
            out.append(module.log_step)
        elif isinstance(module, LSQActFakeQuant):
            out.extend([module.log_step, module.offset])
    return out


def quantiser_parameter_count(model) -> int:
    return sum(p.numel() for p in quantiser_parameters(model))


def split_parameter_groups(model, base_lr: float, step_lr: float, weight_decay: float):
    """Two groups: network weights (decayed) and quantiser steps (never decayed)."""
    step_params = quantiser_parameters(model)
    step_ids = {id(p) for p in step_params}
    weights = [p for p in model.parameters() if id(p) not in step_ids]
    groups = [{"params": weights, "lr": base_lr, "weight_decay": weight_decay}]
    if step_params:
        groups.append({"params": step_params, "lr": step_lr, "weight_decay": 0.0})
    return groups


def export_int8_artifact_lsq(model, model_dir: Path, quant_info: dict) -> None:
    """Write ``model_int8.npz`` using the learned step sizes.

    Same layout as the min/max exporter, so ``infer_int8.py`` needs no change.
    """
    names = ("stem", "res1", "res2", "head")
    payload: dict[str, np.ndarray] = {}
    for name, linear in zip(names, _quantised_linears(model)):
        quantiser = linear.parametrizations.weight[0]
        original = linear.parametrizations.weight.original.detach()
        if isinstance(quantiser, LSQWeightFakeQuant):
            integers, step = quantiser.export(original)
        else:                                    # min/max fallback
            levels = 2 ** (quant_info.get("weight_bits", 8) - 1) - 1
            step = original.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
            integers = torch.clamp(torch.round(original / step), -levels, levels)
        payload[f"{name}_w_int"] = integers.cpu().numpy().astype(np.int8)
        payload[f"{name}_w_scale"] = step.cpu().numpy().astype(np.float32).reshape(-1)
        if linear.bias is not None:
            payload[f"{name}_b"] = linear.bias.detach().cpu().numpy().astype(np.float32)
    for name, observer in model.activation_observers.items():
        spec = observer.export()
        payload[f"act_{name}_scale"] = np.float32(spec["scale"])
        payload[f"act_{name}_zero_point"] = np.float32(spec["zero_point"])
        payload[f"act_{name}_levels"] = np.int32(spec["levels"])
    payload["head_scale"] = np.float32(model.network[-1].scale)
    payload["alpha"] = np.float32(torch.sigmoid(model.network[-1].alpha_logit).item())
    model_dir = Path(model_dir)
    np.savez(model_dir / "model_int8.npz", **payload)
    print(f"INT8 artefact (LSQ) written to {model_dir/'model_int8.npz'}", flush=True)


class FrozenTeacher:
    """An FP32 checkpoint held in eval mode to distil the quantised student against."""

    def __init__(self, model):
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


def load_teacher(checkpoint: Path, input_dim: int, architecture: str, head_scale: float,
                 hybrid_alpha: float, device) -> FrozenTeacher:
    from .core import IATMLP

    model = IATMLP(input_dim, architecture, dropout=0.0,
                   head_scale=head_scale, hybrid_alpha=hybrid_alpha).to(device)
    state = torch.load(Path(checkpoint), map_location=device)
    model.load_state_dict(state)
    print(f"KD teacher loaded from {checkpoint}", flush=True)
    return FrozenTeacher(model)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      temperature: float) -> torch.Tensor:
    import torch.nn.functional as F

    temperature = max(float(temperature), 1e-3)
    return (temperature ** 2) * F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
