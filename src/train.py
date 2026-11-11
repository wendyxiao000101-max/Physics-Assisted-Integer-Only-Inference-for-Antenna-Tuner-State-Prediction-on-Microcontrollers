#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from iatopt.core import (  # noqa: E402
    ARCHITECTURES,
    attach_fake_quant,
    FEATURE_SPECS,
    GAMMA_SCALE,
    FrequencyConditionalStandardizer,
    IATMLP,
    PhysicsTeacher,
    QuantileNormalStandardizer,
    Standardizer,
    benchmark_latency,
    build_features,
    complexity,
    frequency_block_indices,
    load_dataset,
    metrics,
    predict_logits,
    protocol_groups,
    save_json,
    serialized_size,
    set_seed,
    split_indices,
    sweep_fold_indices,
    synthesize_physics_rows,
)
from iatopt.qat2 import (  # noqa: E402
    attach_lsq_fake_quant,
    distillation_loss,
    export_int8_artifact_lsq,
    freeze_quantisers,
    load_teacher,
    quantiser_parameter_count,
    split_parameter_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware FP32 optimization for Huawei IAT")
    parser.add_argument("--mode", choices=["audit", "features", "trees", "ablate", "finalize"], default="ablate")
    parser.add_argument("--x", type=Path, default=ROOT.parent / "esp32p4_iat_demo/input/data/x_train.csv")
    parser.add_argument("--y", type=Path, default=ROOT.parent / "esp32p4_iat_demo/input/data/y_train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--features", nargs="+", choices=FEATURE_SPECS, default=["raw12", "raw5", "polar4", "gamma3", "impedance3", "gamma2"])
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=["mlp64_32", "mlp96_48", "mlp128_64", "residual64"])
    parser.add_argument("--protocols", nargs="+", choices=["random", "point_group", "sweep_group", "frequency_block"], default=["random", "point_group", "sweep_group"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--head-scale", type=float, default=16.0)
    parser.add_argument("--hybrid-alpha-init", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--loss", choices=["ce", "focal", "poly1", "ldam", "ala", "ala_linear"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--poly-epsilon", type=float, default=1.0)
    parser.add_argument("--ldam-max-margin", type=float, default=0.3)
    parser.add_argument("--ala-scale", type=float, default=16.0)
    parser.add_argument("--ala-linear-strength", type=float, default=0.2)
    parser.add_argument("--ala-schedule", choices=["constant", "warmup_decay"], default="constant")
    parser.add_argument("--ala-warmup-fraction", type=float, default=0.15)
    parser.add_argument("--ala-peak-fraction", type=float, default=0.65)
    parser.add_argument("--ala-final-factor", type=float, default=0.15)
    parser.add_argument("--pairwise-ala-strength", type=float, default=0.0)
    parser.add_argument("--effective-number-beta", type=float, default=0.0)
    parser.add_argument("--deferred-reweight-fraction", type=float, default=0.0)
    parser.add_argument("--balanced-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.15)
    parser.add_argument("--crt-epochs", type=int, default=0,
                        help="classifier-only balanced retraining epochs")
    parser.add_argument("--crt-learning-rate", type=float, default=2e-3)
    parser.add_argument("--crt-reinitialize", action="store_true")
    parser.add_argument("--difficulty-margin-epochs", type=int, default=0)
    parser.add_argument("--difficulty-margin-strength", type=float, default=0.2)
    parser.add_argument("--difficulty-margin-gamma", type=float, default=1.5)
    parser.add_argument("--difficulty-margin-learning-rate", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--scene-mix-alpha", type=float, default=0.0,
                        help="same-label, same-frequency, cross-sweep mixup strength")
    parser.add_argument("--scene-mix-prob", type=float, default=0.5)
    parser.add_argument("--trajectory-mix-alpha", type=float, default=0.0,
                        help="same-label nearest-frequency mixup inside one reconstructed sweep")
    parser.add_argument("--trajectory-consistency-weight", type=float, default=0.0,
                        help="train-only KL consistency with a same-label neighboring frequency")
    parser.add_argument("--scene-cvar-weight", type=float, default=0.0)
    parser.add_argument("--scene-cvar-tail-fraction", type=float, default=0.25)
    parser.add_argument("--scene-rex-weight", type=float, default=0.0,
                        help="REx penalty on variance of reconstructed-scene losses")
    parser.add_argument("--group-dro-eta", type=float, default=0.0,
                        help="GroupDRO multiplicative scene-weight step; zero disables it")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--sam-rho", type=float, default=0.0,
                        help="SAM neighborhood radius; zero disables SAM")
    parser.add_argument("--asam-rho", type=float, default=0.0,
                        help="ASAM neighborhood radius; zero disables ASAM")
    parser.add_argument("--asam-eta", type=float, default=0.01,
                        help="ASAM parameter-scale floor")
    parser.add_argument("--swa", action="store_true", help="validate/export epoch-wise averaged weights")
    parser.add_argument("--swa-start", type=int, default=40)
    parser.add_argument("--swad-window", type=int, default=0,
                        help="average the best contiguous validation window; zero disables SWAD")
    parser.add_argument("--swad-force", action="store_true",
                        help="select the best SWAD window even if its validation score is lower")
    parser.add_argument("--checkpoint-average-k", type=int, default=1,
                        help="average the top-k validation checkpoints into one static model")
    parser.add_argument("--frequency-conditional-normalization", action="store_true",
                        help="fit per-frequency raw5 normalization on the training fold only")
    parser.add_argument("--quantile-normalization", action="store_true",
                        help="fit a train-fold-only marginal quantile-to-normal transform")
    parser.add_argument("--keep-default-duplicates", action="store_true",
                        help="retain repeated switch-state rows for default-state-only inputs")
    parser.add_argument("--sweep-fold", type=int, choices=range(5), default=None,
                        help="override sweep fold; useful for validation-selected restarts")
    parser.add_argument("--logit-adjustment", type=float, default=0.0,
                        help="training-time log-prior adjustment strength")
    parser.add_argument("--physics-teacher", choices=["none", "argmin", "lr"], default="none",
                        help="train-only Moebius tuner model that soft-labels synthetic default-state points")
    parser.add_argument("--physics-per-point", type=int, default=8,
                        help="synthetic points generated per unique training default-state point")
    parser.add_argument("--physics-sigma", type=float, default=0.06,
                        help="complex Gaussian jitter of synthetic Γ₂ (unit reflection scale)")
    parser.add_argument("--physics-temperature", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1.0,
                        help="weight of the KL term on synthetic points relative to the supervised loss")
    parser.add_argument("--physics-batch-ratio", type=float, default=1.0,
                        help="synthetic minibatch size as a fraction of --batch-size")
    parser.add_argument("--physics-mode", choices=["jitter", "pair"], default="jitter")
    parser.add_argument("--physics-teacher-noise", type=float, default=0.0,
                        help="教师质量调节：Möbius 系数乘性复高斯噪声强度 eta")
    parser.add_argument("--physics-teacher-shuffle", action="store_true",
                        help="教师质量调节：同一开关态内随机重排频率列（物理形状对、内容错的对照）")
    parser.add_argument("--physics-teacher-seed", type=int, default=20260918,
                        help="教师劣化的随机种子，与训练种子无关")
    parser.add_argument("--physics-sigmas", type=float, nargs="+", default=None,
                        help="multi-scale jitter radii for synthetic Γ₂; overrides --physics-sigma when given")
    parser.add_argument("--physics-pair-fraction", type=float, default=None,
                        help="fraction of synthetic points drawn by same-frequency interpolation instead of jitter; "
                             "overrides --physics-mode when given")
    parser.add_argument("--state-permutation-prob", type=float, default=0.0,
                        help="probability of replacing Γ₁ with that of a sibling row in the same (frequency, Γ₂) "
                             "point group; label preserving and training-only")
    parser.add_argument("--state-invariance-weight", type=float, default=0.0,
                        help="symmetric KL between a sample and a sibling row of the same point group; training-only")
    parser.add_argument("--verify-folded-head", action="store_true",
                        help="also evaluate the alpha=1 folded head (plain Linear with row-normalised "
                             "weights, no bias, no L2 normalisation, no scale) and report per-fold agreement")
    parser.add_argument("--qat", action="store_true",
                        help="train with INT8 fake quantisation in the loop (int8 per-channel weights, "
                             "uint8 asymmetric activations at the operator boundaries)")
    parser.add_argument("--qat-weight-bits", type=int, default=8)
    parser.add_argument("--qat-activation-bits", type=int, default=8)
    parser.add_argument("--qat-scheme", choices=["minmax", "lsq"], default="minmax",
                        help="minmax reproduces the EMA observer; lsq learns every step size "
                             "(LSQ/LSQ+), initialised by a per-channel MSE clipping search")
    parser.add_argument("--qat-step-lr", type=float, default=None,
                        help="learning rate for the quantiser step/offset parameters "
                             "(default: same as --learning-rate); lsq only")
    parser.add_argument("--qat-act-percentile", type=float, default=0.999,
                        help="quantile used to initialise the activation range; 1.0 uses the raw extrema")
    parser.add_argument("--qat-freeze-at", type=float, default=0.0,
                        help="fraction of the schedule after which the step sizes stop training so the "
                             "weights settle on a fixed integer grid; 0 disables")
    parser.add_argument("--qat-init-from", type=Path, default=None,
                        help="FP32 checkpoint to initialise the quantised network from (model_fp32.pth)")
    parser.add_argument("--qat-kd-from", type=Path, default=None,
                        help="FP32 checkpoint used as a frozen teacher; the quantised student is distilled "
                             "against its logits, which is the standard way to recover the INT8 gap")
    parser.add_argument("--qat-kd-weight", type=float, default=0.0,
                        help="weight of the FP32-teacher KL term")
    parser.add_argument("--qat-kd-temperature", type=float, default=2.0,
                        help="softmax temperature of the FP32-teacher KL term")
    parser.add_argument("--physics-min-distance", type=float, default=0.0,
                        help="discard synthetic points whose nearest same-frequency training Γ₂ is closer than "
                             "this radius, so the teacher only supervises what the measured labels do not cover")
    parser.add_argument("--physics-oversample", type=int, default=3,
                        help="candidate-pool multiplier used when --physics-min-distance gates the synthetic cloud")
    parser.add_argument("--physics-weight-schedule", choices=["constant", "linear_decay", "cosine_decay"],
                        default="constant",
                        help="anneal the physics KL weight across training instead of holding it fixed")
    parser.add_argument("--physics-final-weight", type=float, default=0.0,
                        help="physics KL weight reached at the end of the decay schedule")
    parser.add_argument("--physics-decay-end", type=float, default=1.0,
                        help="fraction of training after which the physics weight has fully decayed")
    parser.add_argument("--physics-real-weight", type=float, default=0.0,
                        help="KL towards the Moebius teacher on the measured training rows themselves, in addition "
                             "to the synthetic points; training-only")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--multi-teacher-distill", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--teacher-folds", type=int, default=3)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--teacher-trees", type=int, default=120)
    parser.add_argument("--teacher-neural-weight", type=float, default=0.5)
    parser.add_argument("--teacher-cache-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--final-feature", choices=FEATURE_SPECS, default="gamma3")
    parser.add_argument("--final-architecture", choices=ARCHITECTURES, default="mlp128_64")
    parser.add_argument("--final-epochs", type=int, default=120)
    parser.add_argument("--model-output", type=Path, default=ROOT / "model")
    parser.add_argument("--ablate-export-dir", type=Path, default=None,
                        help="in --mode ablate, also export each run's checkpoint, scaler and split "
                             "so the quantisation pipeline can be replayed per fold")
    return parser.parse_args()


def choose_split(bundle, protocol: str, run: int, seed: int):
    if protocol == "sweep_group":
        return sweep_fold_indices(bundle.labels, bundle.sweep_groups, run % 5)
    if protocol == "frequency_block":
        return frequency_block_indices(bundle.raw, run % 5)
    return split_indices(bundle.labels, protocol_groups(bundle, protocol), seed)


def class_weight_tensor(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=15).astype(np.float64)
    weights = np.sqrt(counts.mean() / np.maximum(counts, 1))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def effective_number_weights(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=15).astype(np.float64)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def ala_schedule_factor(
    epoch: int,
    total_epochs: int,
    schedule: str,
    warmup_fraction: float,
    peak_fraction: float,
    final_factor: float,
) -> float:
    """Return the train-only ALA margin multiplier for one epoch."""
    if schedule == "constant":
        return 1.0
    if not 0.0 <= warmup_fraction < peak_fraction < 1.0:
        raise ValueError("ALA schedule requires 0 <= warmup < peak < 1")
    if not 0.0 <= final_factor <= 1.0:
        raise ValueError("ALA final factor must be in [0, 1]")
    progress = min(max((epoch - 1) / max(total_epochs - 1, 1), 0.0), 1.0)
    if progress < warmup_fraction:
        return progress / max(warmup_fraction, 1e-12)
    if progress <= peak_fraction:
        return 1.0
    decay_progress = (progress - peak_fraction) / (1.0 - peak_fraction)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return final_factor + (1.0 - final_factor) * cosine


def pairwise_confusion_factors(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build a train-OOF class-pair difficulty matrix without validation labels."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), 15):
        raise ValueError("teacher probabilities must be Nx15")
    matrix = np.zeros((15, 15), dtype=np.float64)
    for label in range(15):
        members = labels == label
        if np.any(members):
            matrix[label] = probabilities[members].mean(axis=0)
        matrix[label, label] = 0.0
        maximum = matrix[label].max()
        if maximum > 0:
            matrix[label] /= maximum
    return matrix.astype(np.float32)


def scene_cvar_loss(
    per_sample_loss: torch.Tensor,
    scene_ids: torch.Tensor | None,
    weight: float,
    tail_fraction: float,
) -> torch.Tensor:
    """Blend mean risk with the worst-tail mean over reconstructed scenes."""
    mean_loss = per_sample_loss.mean()
    if weight <= 0.0 or scene_ids is None:
        return mean_loss
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("scene CVaR tail fraction must be in (0, 1]")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("scene CVaR weight must be in [0, 1]")
    _, inverse = torch.unique(scene_ids, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    sums = torch.zeros(group_count, dtype=per_sample_loss.dtype, device=per_sample_loss.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, per_sample_loss)
    counts.scatter_add_(0, inverse, torch.ones_like(per_sample_loss))
    group_losses = sums / counts.clamp_min(1.0)
    count = max(1, int(math.ceil(len(group_losses) * tail_fraction)))
    tail_loss = torch.topk(group_losses, count, largest=True).values.mean()
    return (1.0 - weight) * mean_loss + weight * tail_loss


def scene_rex_groupdro_loss(
    per_sample_loss: torch.Tensor,
    scene_ids: torch.Tensor | None,
    rex_weight: float,
    group_weights: torch.Tensor | None,
    group_dro_eta: float,
    update_group_weights: bool = True,
) -> torch.Tensor:
    """Apply scene-level REx and/or GroupDRO without changing inference capacity."""
    mean_loss = per_sample_loss.mean()
    if scene_ids is None or (rex_weight <= 0.0 and group_dro_eta <= 0.0):
        return mean_loss
    if rex_weight < 0.0 or group_dro_eta < 0.0:
        raise ValueError("REx weight and GroupDRO eta must be non-negative")
    unique, inverse = torch.unique(scene_ids, sorted=True, return_inverse=True)
    sums = torch.zeros(len(unique), dtype=per_sample_loss.dtype, device=per_sample_loss.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, per_sample_loss)
    counts.scatter_add_(0, inverse, torch.ones_like(per_sample_loss))
    group_losses = sums / counts.clamp_min(1.0)
    robust_loss = mean_loss
    if group_dro_eta > 0.0:
        if group_weights is None:
            raise ValueError("GroupDRO requires a persistent scene-weight tensor")
        with torch.no_grad():
            if update_group_weights:
                updated = group_weights[unique] * torch.exp(
                    group_dro_eta * group_losses.detach().clamp(max=20.0)
                )
                group_weights[unique] = updated
                group_weights.clamp_(min=1e-6, max=1e6)
                group_weights.div_(group_weights.mean().clamp_min(1e-12))
            active_weights = group_weights[unique]
        robust_loss = (active_weights * group_losses).sum() / active_weights.sum().clamp_min(1e-12)
    if rex_weight > 0.0 and len(group_losses) > 1:
        robust_loss = robust_loss + rex_weight * group_losses.var(unbiased=False)
    return robust_loss


def balanced_contrastive_loss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Class-averaged supervised contrastive loss for long-tailed batches."""
    embedding = F.normalize(embedding, dim=1)
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    similarity = similarity.masked_fill(identity, -torch.inf)
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positives = targets[:, None].eq(targets[None, :]) & ~identity
    positive_count = positives.sum(1)
    valid = positive_count > 0
    anchor_loss = -(log_prob.masked_fill(~positives, 0.0).sum(1) / positive_count.clamp_min(1))
    class_losses = [anchor_loss[valid & targets.eq(label)].mean()
                    for label in targets.unique() if torch.any(valid & targets.eq(label))]
    if not class_losses:
        return embedding.sum() * 0.0
    return torch.stack(class_losses).mean()


def scene_mix_partners(
    labels: np.ndarray,
    frequencies: np.ndarray,
    sweep_groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Choose same-class/same-frequency partners from another reconstructed sweep."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], list[int]] = {}
    for pos, (label, frequency) in enumerate(zip(labels, frequencies)):
        buckets.setdefault((int(label), int(frequency)), []).append(pos)
    partners = np.arange(len(labels), dtype=np.int64)
    for pos, (label, frequency, sweep) in enumerate(zip(labels, frequencies, sweep_groups)):
        candidates = buckets[(int(label), int(frequency))]
        if len(candidates) <= 1:
            continue
        start = int(rng.integers(len(candidates)))
        for offset in range(len(candidates)):
            candidate = candidates[(start + offset) % len(candidates)]
            if sweep_groups[candidate] != sweep:
                partners[pos] = candidate
                break
    return partners


def trajectory_neighbor_partners(
    labels: np.ndarray,
    frequencies: np.ndarray,
    sweep_groups: np.ndarray,
    point_groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Select a closest-frequency, same-label point from the same sweep."""

    labels = np.asarray(labels)
    frequencies = np.asarray(frequencies)
    sweep_groups = np.asarray(sweep_groups)
    point_groups = np.asarray(point_groups)
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], list[int]] = {}
    for position, (label, sweep) in enumerate(zip(labels, sweep_groups)):
        buckets.setdefault((int(label), int(sweep)), []).append(position)
    partners = np.arange(len(labels), dtype=np.int64)
    for position, (label, sweep, frequency, point) in enumerate(
        zip(labels, sweep_groups, frequencies, point_groups)
    ):
        candidates = np.asarray(buckets[(int(label), int(sweep))], dtype=np.int64)
        candidates = candidates[point_groups[candidates] != point]
        if len(candidates) == 0:
            continue
        distance = np.abs(frequencies[candidates] - frequency)
        nearest = candidates[distance == distance.min()]
        partners[position] = int(rng.choice(nearest))
    return partners


def state_sibling_partners(point_groups: np.ndarray, seed: int) -> np.ndarray:
    """Pick another row of the same (frequency, Γ₂) point group.

    The public data labels a whole point group with a single optimal switch
    state, so every row inside a group carries the same target and differs only
    in ``itState`` and therefore in the measured closed-loop Γ₁.  Γ₁ is a
    nuisance variable: it is a deterministic Moebius image of Γ₂ under the
    measured state, and the sweep-group audit shows the network can use it as a
    scene fingerprint that does not transfer to held-out sweeps.  A sibling row
    is therefore an exactly label-preserving view of the same physical point and
    can be used both as an input augmentation and as an invariance target.  The
    partner index is computed on training rows only.
    """

    point_groups = np.asarray(point_groups, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = np.argsort(point_groups, kind="stable")
    partners = np.arange(len(point_groups), dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(point_groups[order])) + 1
    for block in np.split(order, boundaries):
        if len(block) < 2:
            continue
        shift = rng.integers(1, len(block), len(block))
        partners[block] = block[(np.arange(len(block)) + shift) % len(block)]
    return partners


def build_physics_augmentation(
    raw: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    feature_name: str,
    scaler,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    report_idx: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict] | None:
    """Fit the train-only physics teacher and return normalized synthetic inputs + soft labels."""
    if args.physics_teacher == "none":
        return None
    if feature_name.startswith("context_"):
        raise ValueError("physics augmentation does not support trajectory context features")
    if len(FEATURE_SPECS[feature_name]) > 5:
        raise ValueError("physics augmentation keeps the deployed network input at most 5-D")
    if not 0.0 <= args.physics_weight:
        raise ValueError("physics weight must be non-negative")
    if not 0.0 < args.physics_batch_ratio <= 4.0:
        raise ValueError("physics batch ratio must be in (0, 4]")
    teacher = PhysicsTeacher.fit(
        raw, labels, train_idx, kind=args.physics_teacher,
        noise=getattr(args, "physics_teacher_noise", 0.0),
        shuffle=getattr(args, "physics_teacher_shuffle", False),
        corrupt_seed=getattr(args, "physics_teacher_seed", 20260918),
    )
    synthetic_raw, soft = synthesize_physics_rows(
        raw, train_idx, teacher, seed,
        per_point=args.physics_per_point, sigma=args.physics_sigma,
        temperature=args.physics_temperature, mode=args.physics_mode,
        sigmas=args.physics_sigmas, pair_fraction=args.physics_pair_fraction,
        min_distance=args.physics_min_distance, oversample=args.physics_oversample,
    )
    synthetic_x = scaler.transform(build_features(synthetic_raw, feature_name))
    if synthetic_x.shape[1] > 5:
        raise ValueError(f"physics synthetic features must stay 5-D, got {synthetic_x.shape[1]}")
    info = {
        "physics_teacher": args.physics_teacher,
        "physics_teacher_noise": float(getattr(args, "physics_teacher_noise", 0.0)),
        "physics_teacher_shuffle": bool(getattr(args, "physics_teacher_shuffle", False)),
        "physics_teacher_seed": int(getattr(args, "physics_teacher_seed", 20260918)),
        "physics_synthetic_rows": int(len(soft)),
        "physics_mobius_residual_median": teacher.physics.residual_median,
        "physics_teacher_train_accuracy": float(np.mean(teacher.predict_raw(raw[train_idx]) == labels[train_idx])),
    }
    if report_idx is not None and len(report_idx) > 0:
        info["physics_teacher_validation_accuracy"] = float(
            np.mean(teacher.predict_raw(raw[report_idx]) == labels[report_idx])
        )
    return torch.from_numpy(synthetic_x).to(device), torch.from_numpy(soft).to(device), info, teacher


def physics_real_soft_labels(
    teacher: PhysicsTeacher, raw: np.ndarray, rows: np.ndarray, temperature: float,
) -> np.ndarray:
    """Teacher posterior on the measured training rows themselves.

    The Moebius teacher is a function of (Γ₂, frequency) alone, so its decision surface
    carries no sweep identity.  Distilling it on the real training rows - not only on the
    synthetic ones - transfers that scene-independent surface directly and costs nothing
    at inference.  Only training rows are passed in, so no held-out sweep leaks.
    """

    gamma2 = (raw[rows, 4] + 1j * raw[rows, 5]) / GAMMA_SCALE
    return teacher.probabilities(gamma2, raw[rows, 2], temperature)


@torch.inference_mode()
def folded_head_prediction(model, features: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Predict with the deployable alpha=1 head: a plain Linear whose rows are unit norm.

    With alpha == 1 the logit is s * (h . w_c / ||w_c||) / ||h||, and both s and ||h||
    are positive constants shared by every class, so the argmax - and the whole Top-k
    ordering - is unchanged by dropping them together with the linear branch and the
    bias.  Only the softmax probabilities differ, which is why the confidence readout
    still needs ||h||.
    """
    head = model.network[-1]
    weight = F.normalize(head.weight.detach(), dim=1)
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(features), batch):
        chunk = torch.from_numpy(features[start:start + batch]).to(device)
        out.append((model.network[:-1](chunk) @ weight.T).argmax(1).cpu().numpy())
    return np.concatenate(out)


def export_int8_artifact(model, model_dir: Path, quant_info: dict) -> None:
    """Write a self-contained INT8 artefact: integer weights, per-channel scales,
    activation scales and zero points.  Runs on a PC with numpy alone."""
    network = model.network
    names = ("stem", "res1", "res2", "head")
    linears = (network[0], network[2].block[0], network[2].block[3], network[3])
    payload: dict[str, np.ndarray] = {}
    levels = 2 ** (quant_info.get("weight_bits", 8) - 1) - 1
    for name, linear in zip(names, linears):
        original = linear.parametrizations.weight.original.detach().cpu()
        scale = original.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
        payload[f"{name}_w_int"] = torch.clamp(torch.round(original / scale), -levels, levels).numpy().astype(np.int8)
        payload[f"{name}_w_scale"] = scale.numpy().astype(np.float32).reshape(-1)
        if linear.bias is not None:
            payload[f"{name}_b"] = linear.bias.detach().cpu().numpy().astype(np.float32)
    for name, observer in model.activation_observers.items():
        spec = observer.export()
        payload[f"act_{name}_scale"] = np.float32(spec["scale"])
        payload[f"act_{name}_zero_point"] = np.float32(spec["zero_point"])
        payload[f"act_{name}_levels"] = np.int32(spec["levels"])
    payload["head_scale"] = np.float32(model.network[-1].scale)
    payload["alpha"] = np.float32(torch.sigmoid(model.network[-1].alpha_logit).item())
    np.savez(model_dir / "model_int8.npz", **payload)
    print(f"INT8 artefact written to {model_dir/'model_int8.npz'}", flush=True)


def physics_weight_at(epoch: int, total_epochs: int, args: argparse.Namespace) -> float:
    """Anneal the physics KL weight; ``constant`` reproduces the previous behaviour."""
    start, final = args.physics_weight, args.physics_final_weight
    if args.physics_weight_schedule == "constant":
        return start
    if not 0.0 < args.physics_decay_end <= 1.0:
        raise ValueError("physics decay end must lie in (0, 1]")
    progress = min(max((epoch - 1) / max(total_epochs - 1, 1), 0.0), 1.0) / args.physics_decay_end
    progress = min(progress, 1.0)
    factor = (
        1.0 - progress if args.physics_weight_schedule == "linear_decay"
        else 0.5 * (1.0 + math.cos(math.pi * progress))
    )
    return final + (start - final) * factor


def physics_consistency_loss(
    model: torch.nn.Module,
    synthetic_x: torch.Tensor,
    synthetic_soft: torch.Tensor,
    batch_size: int,
    generator_device: torch.device,
) -> torch.Tensor:
    index = torch.randint(0, len(synthetic_x), (batch_size,), device=generator_device)
    log_probability = F.log_softmax(model(synthetic_x[index]), dim=1)
    return F.kl_div(log_probability, synthetic_soft[index], reduction="batchmean")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def oof_teacher_probabilities(
    teacher_x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    trees: int = 160,
) -> np.ndarray:
    result = np.zeros((len(labels), 15), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, valid in splitter.split(teacher_x, labels, groups):
        teacher = ExtraTreesClassifier(
            n_estimators=trees,
            max_features=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        teacher.fit(teacher_x[train], labels[train])
        result[valid] = teacher.predict_proba(teacher_x[valid]).astype(np.float32)
    if np.any(result.sum(axis=1) < 0.99):
        raise AssertionError("OOF teacher did not predict every training row")
    return result


def oof_multiteacher_probabilities(
    teacher_x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    device: torch.device,
    folds: int,
    epochs: int,
    batch_size: int,
    trees: int,
    neural_weight: float,
) -> np.ndarray:
    """OOF blend of ExtraTrees and an ALA-cosine neural teacher.

    Every prediction is produced by teachers that did not fit that row or its
    reconstructed sweep group.  The returned labels belong only to the outer
    training fold and are safe for student distillation.
    """
    if not 2 <= folds <= 5:
        raise ValueError("teacher folds must be between 2 and 5")
    if not 0.0 <= neural_weight <= 1.0:
        raise ValueError("teacher neural weight must be in [0, 1]")
    tree_result = np.zeros((len(labels), 15), dtype=np.float32)
    neural_result = np.zeros_like(tree_result)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(splitter.split(teacher_x, labels, groups)):
        inner_scaler = Standardizer.fit(teacher_x[train])
        inner_train = inner_scaler.transform(teacher_x[train])
        inner_valid = inner_scaler.transform(teacher_x[valid])
        tree = ExtraTreesClassifier(
            n_estimators=trees,
            max_features=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed + fold,
            n_jobs=-1,
        )
        tree.fit(inner_train, labels[train])
        tree_probability = tree.predict_proba(inner_valid).astype(np.float32)
        tree_result[valid[:, None], tree.classes_[None, :]] = tree_probability

        set_seed(seed + 1009 + fold)
        teacher = IATMLP(teacher_x.shape[1], "residual48_mish_relu_cosine", dropout=0.05).to(device)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
        train_tensor = torch.from_numpy(inner_train).to(device)
        target_tensor = torch.from_numpy(labels[train].astype(np.int64)).to(device)
        counts = np.bincount(labels[train], minlength=15).astype(np.float32)
        quantity = 1.0 / np.log(np.maximum(counts, 1) / max(float(counts.min()), 1.0) + 1.0)
        quantity_tensor = torch.from_numpy(quantity.astype(np.float32)).to(device)
        for _ in range(epochs):
            teacher.train()
            order = torch.randperm(len(train), device=device)
            for index in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = teacher(train_tensor[index])
                targets = target_tensor[index]
                true_cosine = (logits.detach() / 16.0).gather(1, targets[:, None]).squeeze(1)
                difficulty = (1.0 - true_cosine).clamp(0.0, 2.0) / 2.0
                adjusted = logits.clone()
                adjusted[torch.arange(len(index), device=device), targets] -= (
                    16.0 * difficulty * quantity_tensor[targets]
                )
                loss = F.cross_entropy(adjusted, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
        teacher.eval()
        neural_logits = predict_logits(teacher, inner_valid)
        neural_result[valid] = torch.softmax(torch.from_numpy(neural_logits), dim=1).numpy()
        del teacher, train_tensor, target_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if np.any(tree_result.sum(axis=1) < 0.99) or np.any(neural_result.sum(axis=1) < 0.99):
        raise AssertionError("multi-teacher OOF prediction is incomplete")
    result = (1.0 - neural_weight) * tree_result + neural_weight * neural_result
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def train_mlp(
    x: np.ndarray,
    teacher_x: np.ndarray,
    labels: np.ndarray,
    point_groups: np.ndarray,
    sweep_groups: np.ndarray,
    frequencies: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_name: str,
    architecture: str,
    protocol: str,
    seed: int,
    args: argparse.Namespace,
    raw: np.ndarray | None = None,
) -> tuple[dict, dict[str, torch.Tensor], Standardizer]:
    set_seed(seed)
    if args.physics_teacher != "none" and raw is None:
        raise ValueError("physics augmentation requires the raw six-column measurements")
    if args.rdrop_alpha > 0 and args.mixup_alpha > 0:
        raise ValueError("R-Drop and mixup cannot be enabled together")
    device = resolve_device(args.device)
    if args.frequency_conditional_normalization and args.quantile_normalization:
        raise ValueError("choose only one normalization method")
    if args.frequency_conditional_normalization:
        if feature_name != "raw5":
            raise ValueError("frequency-conditional normalization currently supports raw5 only")
        scaler = FrequencyConditionalStandardizer.fit(x[train_idx])
    elif args.quantile_normalization:
        scaler = QuantileNormalStandardizer.fit(x[train_idx])
    else:
        scaler = Standardizer.fit(x[train_idx])
    train_x = scaler.transform(x[train_idx])
    valid_x = scaler.transform(x[valid_idx])
    test_x = scaler.transform(x[test_idx])
    physics = build_physics_augmentation(
        raw, labels, train_idx, feature_name, scaler, seed, args, device, report_idx=valid_idx,
    ) if raw is not None else None
    physics_info = physics[2] if physics is not None else {}
    physics_batch = int(round(args.batch_size * args.physics_batch_ratio))
    soft = None
    teacher_oof = None
    pairwise_oof = None
    if args.distill and args.multi_teacher_distill:
        raise ValueError("choose either --distill or --multi-teacher-distill")
    need_teacher = args.distill or args.multi_teacher_distill or args.pairwise_ala_strength > 0
    if need_teacher:
        teacher_kind = "multi" if args.multi_teacher_distill else "tree"
        digest = hashlib.sha1(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        cache_root = args.teacher_cache_dir if args.teacher_cache_dir is not None else args.output / "teacher_cache"
        cache = cache_root / (
            f"{teacher_kind}_{protocol}_{seed}_{digest}_f{args.teacher_folds}_"
            f"e{args.teacher_epochs}_t{args.teacher_trees}.npz"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            teacher_oof = np.load(cache)["probabilities"].astype(np.float32)
        elif args.multi_teacher_distill:
            teacher_oof = oof_multiteacher_probabilities(
                teacher_x[train_idx], labels[train_idx], sweep_groups[train_idx], seed,
                device, args.teacher_folds, args.teacher_epochs, args.batch_size,
                args.teacher_trees, args.teacher_neural_weight,
            )
            np.savez_compressed(cache, probabilities=teacher_oof)
        else:
            teacher_scaler = Standardizer.fit(teacher_x[train_idx])
            teacher_normalized = teacher_scaler.transform(teacher_x[train_idx])
            teacher_oof = oof_teacher_probabilities(
                teacher_normalized, labels[train_idx], sweep_groups[train_idx], seed,
                trees=args.teacher_trees,
            )
            np.savez_compressed(cache, probabilities=teacher_oof)
        if args.distill or args.multi_teacher_distill:
            soft = teacher_oof
        pairwise_oof = teacher_oof
        if args.multi_teacher_distill and args.pairwise_ala_strength > 0:
            tree_cache = cache_root / (
                f"tree_{protocol}_{seed}_{digest}_f{args.teacher_folds}_"
                f"e{args.teacher_epochs}_t{args.teacher_trees}.npz"
            )
            if tree_cache.exists():
                pairwise_oof = np.load(tree_cache)["probabilities"].astype(np.float32)
            else:
                teacher_scaler = Standardizer.fit(teacher_x[train_idx])
                pairwise_oof = oof_teacher_probabilities(
                    teacher_scaler.transform(teacher_x[train_idx]), labels[train_idx],
                    sweep_groups[train_idx], seed, trees=args.teacher_trees,
                )
                np.savez_compressed(tree_cache, probabilities=pairwise_oof)
    set_seed(seed)

    # For inputs derived only from the default-state response, rows that differ
    # solely by itState are exact duplicate observations with the same label.
    # Keeping one copy per point avoids weighting a scene by how many switch-state
    # measurements happened to be present and makes the effective sample unit explicit.
    fit_x = train_x
    fit_y = labels[train_idx]
    fit_source_idx = train_idx
    default_only_features = {
        "gamma2", "gamma3", "polar4", "impedance3",
        "default_polar5", "default_logpolar5", "default_phase0_polar5",
    }
    deduplicated_default = feature_name in default_only_features and not args.keep_default_duplicates
    if deduplicated_default:
        _, unique_positions = np.unique(train_x, axis=0, return_index=True)
        unique_positions = np.sort(unique_positions)
        fit_x = train_x[unique_positions]
        fit_y = fit_y[unique_positions]
        fit_source_idx = train_idx[unique_positions]
        if soft is not None:
            soft = soft[unique_positions]

    pairwise_matrix = None
    if args.pairwise_ala_strength > 0:
        assert pairwise_oof is not None
        pairwise_source = pairwise_oof
        if deduplicated_default:
            pairwise_source = pairwise_source[unique_positions]
        pairwise_matrix = torch.from_numpy(pairwise_confusion_factors(pairwise_source, fit_y)).to(device)

    model = IATMLP(
        train_x.shape[1], architecture, dropout=args.dropout,
        head_scale=args.head_scale, hybrid_alpha=args.hybrid_alpha_init,
    ).to(device)
    quant_info = {}
    if args.qat:
        quant_info = (
            attach_lsq_fake_quant(model, args.qat_weight_bits, args.qat_activation_bits,
                                  args.qat_act_percentile)
            if args.qat_scheme == "lsq"
            else attach_fake_quant(model, args.qat_weight_bits, args.qat_activation_bits)
        )
    projection = None
    if args.balanced_contrastive_weight > 0:
        feature_dim = int(model.network[-1].in_features)
        projection = torch.nn.Linear(feature_dim, min(32, feature_dim), bias=False).to(device)
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer_parameters = list(model.parameters()) + ([] if projection is None else list(projection.parameters()))
    optimizer = optimizer_class(
        optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, args.adam_beta2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 50) if args.scheduler == "cosine" else None
    weight = class_weight_tensor(fit_y).to(device) if args.class_weights else None
    if args.effective_number_beta > 0:
        weight = effective_number_weights(fit_y, args.effective_number_beta).to(device)
    counts = np.bincount(fit_y, minlength=15).astype(np.float32)
    ldam_margin = args.ldam_max_margin * np.power(np.maximum(counts, 1), -0.25)
    ldam_margin /= max(float(ldam_margin.max()), 1e-12)
    ldam_margin *= args.ldam_max_margin
    ldam_margin = torch.from_numpy(ldam_margin).to(device)
    quantity_factor = 1.0 / np.log(np.maximum(counts, 1) / max(float(counts.min()), 1.0) + 1.0)
    quantity_factor = torch.from_numpy(quantity_factor.astype(np.float32)).to(device)
    reweight_start = max(1, int(round(args.epochs * args.deferred_reweight_fraction)))
    tensors = [torch.from_numpy(fit_x), torch.from_numpy(fit_y.astype(np.int64))]
    teacher_position = None
    if soft is not None:
        teacher_position = len(tensors)
        tensors.append(torch.from_numpy(soft))
    scene_position = None
    if args.scene_cvar_weight > 0 or args.scene_rex_weight > 0 or args.group_dro_eta > 0:
        scene_position = len(tensors)
        tensors.append(torch.from_numpy(sweep_groups[fit_source_idx].astype(np.int64)))
    partner_position = None
    partner_fraction = 0.0
    if args.scene_mix_alpha > 0 and (args.trajectory_mix_alpha > 0 or args.trajectory_consistency_weight > 0):
        raise ValueError("scene mix and trajectory context training are mutually exclusive")
    if args.scene_mix_alpha > 0:
        partners = scene_mix_partners(
            fit_y,
            frequencies[fit_source_idx],
            sweep_groups[fit_source_idx],
            seed,
        )
        partner_position = len(tensors)
        tensors.append(torch.from_numpy(fit_x[partners]))
        partner_fraction = float(np.mean(partners != np.arange(len(partners))))
    elif args.trajectory_mix_alpha > 0 or args.trajectory_consistency_weight > 0:
        partners = trajectory_neighbor_partners(
            fit_y,
            frequencies[fit_source_idx],
            sweep_groups[fit_source_idx],
            point_groups[fit_source_idx],
            seed,
        )
        partner_position = len(tensors)
        tensors.append(torch.from_numpy(fit_x[partners]))
        partner_fraction = float(np.mean(partners != np.arange(len(partners))))
    sibling_position = None
    sibling_fraction = 0.0
    if args.state_permutation_prob > 0 or args.state_invariance_weight > 0:
        if not 0.0 <= args.state_permutation_prob <= 1.0:
            raise ValueError("state permutation probability must lie in [0, 1]")
        siblings = state_sibling_partners(point_groups[fit_source_idx], seed)
        sibling_position = len(tensors)
        tensors.append(torch.from_numpy(fit_x[siblings]))
        sibling_fraction = float(np.mean(siblings != np.arange(len(siblings))))
    physics_real_position = None
    if physics is not None and args.physics_real_weight > 0:
        physics_real_position = len(tensors)
        tensors.append(torch.from_numpy(
            physics_real_soft_labels(physics[3], raw, fit_source_idx, args.physics_temperature)
        ))
    if device.type == "cuda":
        resident = [tensor.to(device) for tensor in tensors]
        loader = None
    else:
        resident = None
        loader = DataLoader(
            TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
    if args.checkpoint_average_k < 1:
        raise ValueError("checkpoint-average-k must be at least 1")
    if args.swad_window == 1 or args.swad_window < 0:
        raise ValueError("swad-window must be zero or at least 2")
    if args.sam_rho > 0 and args.asam_rho > 0:
        raise ValueError("SAM and ASAM are mutually exclusive")
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    best_source = "base"
    wait = 0
    history = []
    top_checkpoint_states = []
    checkpoint_average_members = []
    swad_recent = []
    best_swad_window = None
    best_swad_score = -1.0
    started = time.perf_counter()
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa else None
    group_dro_weights = None
    if args.group_dro_eta > 0:
        group_dro_weights = torch.ones(
            int(sweep_groups.max()) + 1, dtype=torch.float32, device=device,
        )

    log_prior = None
    if args.logit_adjustment > 0:
        counts = np.bincount(fit_y, minlength=15).astype(np.float32)
        log_prior = torch.log(torch.from_numpy(counts / counts.sum()).to(device).clamp_min(1e-8))

    current_epoch = 0

    def objective(current_model, inputs, targets, teacher_prob, scene_ids, update_group_weights=True):
        features = current_model.forward_features(inputs)
        logits = current_model.classifier(features)
        adjusted = logits if log_prior is None else logits + args.logit_adjustment * log_prior
        schedule_factor = ala_schedule_factor(
            current_epoch, args.epochs, args.ala_schedule,
            args.ala_warmup_fraction, args.ala_peak_fraction, args.ala_final_factor,
        )
        if args.loss == "ldam":
            adjusted = adjusted.clone()
            adjusted[torch.arange(len(targets), device=device), targets] -= ldam_margin[targets]
        elif args.loss == "ala":
            if not architecture.endswith("_cosine"):
                raise ValueError("ALA loss requires a cosine architecture")
            adjusted = adjusted.clone()
            true_cosine = (logits.detach() / args.ala_scale).gather(1, targets[:, None]).squeeze(1)
            difficulty = (1.0 - true_cosine).clamp(0.0, 2.0) / 2.0
            adjustment = schedule_factor * args.ala_scale * difficulty * quantity_factor[targets]
            if pairwise_matrix is not None:
                rivals = logits.detach().clone()
                rivals[torch.arange(len(targets), device=device), targets] = -torch.inf
                rival_class = rivals.argmax(1)
                adjustment *= 1.0 + args.pairwise_ala_strength * pairwise_matrix[targets, rival_class]
            adjusted[torch.arange(len(targets), device=device), targets] -= adjustment
        elif args.loss == "ala_linear":
            true_probability = F.softmax(logits.detach(), dim=1).gather(1, targets[:, None]).squeeze(1)
            difficulty = 1.0 - true_probability
            adjustment = schedule_factor * args.ala_linear_strength * difficulty * quantity_factor[targets]
            if pairwise_matrix is not None:
                rivals = logits.detach().clone()
                rivals[torch.arange(len(targets), device=device), targets] = -torch.inf
                rival_class = rivals.argmax(1)
                adjustment *= 1.0 + args.pairwise_ala_strength * pairwise_matrix[targets, rival_class]
            adjusted = adjusted.clone()
            adjusted[torch.arange(len(targets), device=device), targets] -= adjustment
        active_weight = weight if weight is not None and current_epoch >= reweight_start else None
        if args.loss == "focal":
            ce = F.cross_entropy(adjusted, targets, weight=active_weight, reduction="none")
            probability = F.softmax(adjusted, dim=1).gather(1, targets[:, None]).squeeze(1)
            per_sample_loss = (1.0 - probability).pow(args.focal_gamma) * ce
        elif args.loss == "poly1":
            ce = F.cross_entropy(
                adjusted, targets, weight=active_weight,
                label_smoothing=args.label_smoothing, reduction="none",
            )
            probability = F.softmax(adjusted, dim=1).gather(1, targets[:, None]).squeeze(1)
            per_sample_loss = ce + args.poly_epsilon * (1.0 - probability)
        else:
            per_sample_loss = F.cross_entropy(
                adjusted, targets, weight=active_weight,
                label_smoothing=args.label_smoothing, reduction="none",
            )
        if teacher_prob is not None:
            temp = args.temperature
            softened = teacher_prob.clamp_min(1e-8).pow(1/temp)
            softened = softened / softened.sum(1, keepdim=True)
            kd = F.kl_div(
                F.log_softmax(logits/temp, dim=1), softened, reduction="none",
            ).sum(1) * temp * temp
            per_sample_loss = (1-args.distill_weight) * per_sample_loss + args.distill_weight * kd
        loss = scene_cvar_loss(
            per_sample_loss, scene_ids,
            args.scene_cvar_weight, args.scene_cvar_tail_fraction,
        )
        if args.scene_rex_weight > 0 or args.group_dro_eta > 0:
            loss = scene_rex_groupdro_loss(
                per_sample_loss, scene_ids, args.scene_rex_weight,
                group_dro_weights, args.group_dro_eta, update_group_weights,
            )
        if projection is not None:
            contrastive = balanced_contrastive_loss(
                projection(features), targets, args.contrastive_temperature,
            )
            loss = loss + args.balanced_contrastive_weight * contrastive
        return logits, loss

    for epoch in range(1, args.epochs + 1):
        current_epoch = epoch
        physics_weight_now = physics_weight_at(epoch, args.epochs, args)
        model.train()
        if projection is not None:
            projection.train()
        running = 0.0
        if resident is not None:
            order = torch.randperm(len(fit_y), device=device)
            batches = ([tensor[index] for tensor in resident] for index in order.split(args.batch_size))
        else:
            assert loader is not None
            batches = loader
        for batch in batches:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[1].to(device, non_blocking=True)
            teacher_prob = batch[teacher_position].to(device, non_blocking=True) if teacher_position is not None else None
            scene_ids = batch[scene_position].to(device, non_blocking=True) if scene_position is not None else None
            partner = batch[partner_position].to(device, non_blocking=True) if partner_position is not None else None
            sibling = batch[sibling_position].to(device, non_blocking=True) if sibling_position is not None else None
            physics_real = (
                batch[physics_real_position].to(device, non_blocking=True)
                if physics_real_position is not None else None
            )
            clean_xb = xb
            if sibling is not None and args.state_permutation_prob > 0:
                swap = torch.rand(len(xb), 1, device=xb.device) < args.state_permutation_prob
                xb = torch.where(swap, sibling, xb)
            if partner is not None and args.scene_mix_alpha > 0 and np.random.random() < args.scene_mix_prob:
                lam = float(np.random.beta(args.scene_mix_alpha, args.scene_mix_alpha))
                xb = lam * xb + (1-lam) * partner
            elif partner is not None and args.trajectory_mix_alpha > 0 and np.random.random() < args.scene_mix_prob:
                lam = float(np.random.beta(args.trajectory_mix_alpha, args.trajectory_mix_alpha))
                xb = lam * xb + (1-lam) * partner
            if args.noise_std > 0:
                xb = xb + torch.randn_like(xb) * args.noise_std
            optimizer.zero_grad(set_to_none=True)
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                order = torch.randperm(len(xb))
                mixed = lam * xb + (1-lam) * xb[order]
                logits, first_loss = objective(model, mixed, yb, teacher_prob, scene_ids)
                second_teacher = teacher_prob[order] if teacher_prob is not None else None
                second_scenes = scene_ids[order] if scene_ids is not None else None
                _, second_loss = objective(model, mixed, yb[order], second_teacher, second_scenes)
                loss = lam * first_loss + (1-lam) * second_loss
            elif args.rdrop_alpha > 0:
                logits_a, loss_a = objective(model, xb, yb, teacher_prob, scene_ids)
                logits_b, loss_b = objective(model, xb, yb, teacher_prob, scene_ids)
                log_a = F.log_softmax(logits_a, dim=1)
                log_b = F.log_softmax(logits_b, dim=1)
                probability_a = log_a.exp()
                probability_b = log_b.exp()
                consistency = 0.5 * (
                    F.kl_div(log_a, probability_b, reduction="batchmean")
                    + F.kl_div(log_b, probability_a, reduction="batchmean")
                )
                logits = 0.5 * (logits_a + logits_b)
                loss = 0.5 * (loss_a + loss_b) + args.rdrop_alpha * consistency
            elif args.trajectory_consistency_weight > 0:
                assert partner is not None
                logits, primary_loss = objective(model, xb, yb, teacher_prob, scene_ids)
                partner_logits, partner_loss = objective(model, partner, yb, None, scene_ids)
                log_primary = F.log_softmax(logits, dim=1)
                log_partner = F.log_softmax(partner_logits, dim=1)
                consistency = 0.5 * (
                    F.kl_div(log_primary, log_partner.exp(), reduction="batchmean")
                    + F.kl_div(log_partner, log_primary.exp(), reduction="batchmean")
                )
                loss = 0.5 * (primary_loss + partner_loss) + args.trajectory_consistency_weight * consistency
            else:
                logits, loss = objective(model, xb, yb, teacher_prob, scene_ids)
            if sibling is not None and args.state_invariance_weight > 0:
                anchor_logits = logits if (clean_xb is xb and args.mixup_alpha <= 0) else model(clean_xb)
                sibling_logits = model(sibling)
                log_sample = F.log_softmax(anchor_logits, dim=1)
                log_sibling = F.log_softmax(sibling_logits, dim=1)
                invariance = 0.5 * (
                    F.kl_div(log_sample, log_sibling.exp(), reduction="batchmean")
                    + F.kl_div(log_sibling, log_sample.exp(), reduction="batchmean")
                )
                loss = loss + args.state_invariance_weight * invariance
            if physics_real is not None and args.physics_real_weight > 0:
                real_logits = logits if clean_xb is xb else model(clean_xb)
                loss = loss + args.physics_real_weight * F.kl_div(
                    F.log_softmax(real_logits, dim=1), physics_real, reduction="batchmean",
                )
            if physics is not None and physics_weight_now > 0:
                loss = loss + physics_weight_now * physics_consistency_loss(
                    model, physics[0], physics[1], physics_batch, device,
                )
            loss.backward()
            if args.sam_rho > 0 or args.asam_rho > 0:
                adaptive = args.asam_rho > 0
                rho = args.asam_rho if adaptive else args.sam_rho
                gradient_terms = []
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    scale_tensor = parameter.detach().abs() + args.asam_eta if adaptive else 1.0
                    gradient_terms.append((scale_tensor * parameter.grad).norm(p=2))
                grad_norm = torch.norm(torch.stack(gradient_terms), p=2)
                perturbations = []
                scale = rho / (grad_norm + 1e-12)
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is None:
                            perturbation = None
                        elif adaptive:
                            perturbation = (parameter.detach().abs() + args.asam_eta).pow(2) * parameter.grad * scale
                        else:
                            perturbation = parameter.grad * scale
                        perturbations.append(perturbation)
                        if perturbation is not None:
                            parameter.add_(perturbation)
                optimizer.zero_grad(set_to_none=True)
                _, second_loss = objective(
                    model, xb, yb, teacher_prob, scene_ids, update_group_weights=False,
                )
                second_loss.backward()
                with torch.no_grad():
                    for parameter, perturbation in zip(model.parameters(), perturbations):
                        if perturbation is not None:
                            parameter.sub_(perturbation)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(yb)
        if scheduler is not None:
            scheduler.step()
        if swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            evaluation_model = swa_model
            source = "swa"
        else:
            evaluation_model = model
            source = "base"
        valid_prediction = predict_logits(evaluation_model, valid_x).argmax(1)
        valid_accuracy = float(accuracy_score(labels[valid_idx], valid_prediction))
        history.append({"epoch": epoch, "train_loss": running/len(fit_y), "valid_accuracy": valid_accuracy})
        if epoch == 1 or epoch % 10 == 0:
            save_json(args.output / "live.json", dict(epoch=epoch, valid_accuracy=valid_accuracy, seconds=time.perf_counter()-started))
            print(f"epoch {epoch}: val={valid_accuracy:.6f}", flush=True)
        candidate_state = {
            name.removeprefix("module."): value.detach().clone()
            for name, value in evaluation_model.state_dict().items()
            if name != "n_averaged"
        }
        if args.checkpoint_average_k > 1:
            top_checkpoint_states.append((valid_accuracy, epoch, candidate_state))
            top_checkpoint_states.sort(key=lambda item: (item[0], item[1]), reverse=True)
            del top_checkpoint_states[args.checkpoint_average_k:]
        if args.swad_window >= 2:
            swad_recent.append((valid_accuracy, epoch, candidate_state))
            if len(swad_recent) > args.swad_window:
                swad_recent.pop(0)
            if len(swad_recent) == args.swad_window:
                window_score = float(np.mean([item[0] for item in swad_recent]))
                if window_score > best_swad_score + 1e-12:
                    best_swad_score = window_score
                    best_swad_window = list(swad_recent)
        if valid_accuracy > best_accuracy + 1e-8:
            best_accuracy = valid_accuracy
            best_epoch = epoch
            best_state = candidate_state
            best_source = source
            checkpoint_average_members = [epoch]
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    assert best_state is not None
    if args.checkpoint_average_k > 1:
        for count in range(2, len(top_checkpoint_states) + 1):
            selected = top_checkpoint_states[:count]
            averaged_state = {}
            for name in selected[0][2]:
                values = [item[2][name] for item in selected]
                averaged_state[name] = (
                    torch.stack(values).mean(0)
                    if values[0].is_floating_point() else values[0].clone()
                )
            model.load_state_dict(averaged_state)
            averaged_prediction = predict_logits(model, valid_x).argmax(1)
            averaged_accuracy = float(accuracy_score(labels[valid_idx], averaged_prediction))
            if averaged_accuracy > best_accuracy + 1e-8:
                best_accuracy = averaged_accuracy
                best_state = averaged_state
                best_source = f"checkpoint_average_{count}"
                checkpoint_average_members = [item[1] for item in selected]
        model.load_state_dict(best_state)
    if best_swad_window is not None:
        swad_state = {}
        for name in best_swad_window[0][2]:
            values = [item[2][name] for item in best_swad_window]
            swad_state[name] = (
                torch.stack(values).mean(0)
                if values[0].is_floating_point() else values[0].clone()
            )
        model.load_state_dict(swad_state)
        swad_prediction = predict_logits(model, valid_x).argmax(1)
        swad_accuracy = float(accuracy_score(labels[valid_idx], swad_prediction))
        if args.swad_force or swad_accuracy > best_accuracy + 1e-8:
            best_accuracy = swad_accuracy
            best_state = swad_state
            best_source = f"swad_window_{args.swad_window}"
            checkpoint_average_members = [item[1] for item in best_swad_window]
    model.load_state_dict(best_state)
    difficulty_margin = np.zeros(15, dtype=np.float32)
    if args.difficulty_margin_epochs > 0:
        base_prediction = predict_logits(model, valid_x).argmax(1)
        validation_recall = np.asarray(metrics(labels[valid_idx], base_prediction)["per_class_recall"])
        raw_margin = np.power(1.0-validation_recall, args.difficulty_margin_gamma)
        raw_margin *= quantity_factor.detach().cpu().numpy()
        if raw_margin.max() > 0:
            difficulty_margin = (args.difficulty_margin_strength * raw_margin/raw_margin.max()).astype(np.float32)
        margin_tensor = torch.from_numpy(difficulty_margin).to(device)
        margin_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.difficulty_margin_learning_rate,
            weight_decay=args.weight_decay, betas=(0.9,args.adam_beta2),
        )
        margin_x = torch.from_numpy(fit_x).to(device)
        margin_y = torch.from_numpy(fit_y.astype(np.int64)).to(device)
        chosen_state = {name:value.detach().clone() for name,value in best_state.items()}
        chosen_accuracy = best_accuracy
        for _ in range(args.difficulty_margin_epochs):
            model.train()
            order = torch.randperm(len(fit_y),device=device)
            for index in order.split(args.batch_size):
                margin_optimizer.zero_grad(set_to_none=True)
                margin_logits = model(margin_x[index])
                margin_logits[torch.arange(len(index),device=device),margin_y[index]] -= margin_tensor[margin_y[index]]
                margin_loss = F.cross_entropy(margin_logits,margin_y[index])
                margin_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                margin_optimizer.step()
            prediction = predict_logits(model,valid_x).argmax(1)
            candidate_accuracy = float(accuracy_score(labels[valid_idx],prediction))
            if candidate_accuracy > chosen_accuracy + 1e-8:
                chosen_accuracy = candidate_accuracy
                chosen_state = {name:value.detach().clone() for name,value in model.state_dict().items()}
        best_state = chosen_state
        best_accuracy = chosen_accuracy
        if any(not torch.equal(best_state[name], value) for name,value in model.state_dict().items()):
            model.load_state_dict(best_state)
        if chosen_accuracy > history[best_epoch-1]["valid_accuracy"] + 1e-8:
            best_source = "validation_difficulty_margin"
    crt_alpha = 0.0
    if args.crt_epochs > 0:
        base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        head = model.network[-1]
        if not isinstance(head, torch.nn.Linear):
            raise TypeError("cRT requires a final Linear classifier")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        if args.crt_reinitialize:
            head.reset_parameters()
        crt_optimizer = torch.optim.AdamW(head.parameters(), lr=args.crt_learning_rate, weight_decay=0.0)
        crt_x = torch.from_numpy(fit_x).to(device)
        crt_y = torch.from_numpy(fit_y.astype(np.int64)).to(device)
        inverse_frequency = torch.from_numpy((1.0 / np.maximum(counts, 1))[fit_y]).float().to(device)
        model.eval()
        head.train()
        for _ in range(args.crt_epochs):
            sampled = torch.multinomial(inverse_frequency, len(fit_y), replacement=True)
            for index in sampled.split(args.batch_size):
                crt_optimizer.zero_grad(set_to_none=True)
                crt_logits = model(crt_x[index])
                crt_loss = F.cross_entropy(crt_logits, crt_y[index])
                crt_loss.backward()
                crt_optimizer.step()
        crt_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        head_prefix = f"network.{len(model.network)-1}."
        chosen_state = base_state
        chosen_accuracy = best_accuracy
        for alpha in np.linspace(0.1, 1.0, 10):
            blended = {name: value.detach().clone() for name, value in base_state.items()}
            for name in blended:
                if name.startswith(head_prefix):
                    blended[name] = (1.0-alpha) * base_state[name] + alpha * crt_state[name]
            model.load_state_dict(blended)
            prediction = predict_logits(model, valid_x).argmax(1)
            candidate_accuracy = float(accuracy_score(labels[valid_idx], prediction))
            if candidate_accuracy > chosen_accuracy + 1e-8:
                chosen_accuracy = candidate_accuracy
                chosen_state = blended
                crt_alpha = float(alpha)
        best_state = chosen_state
        best_accuracy = chosen_accuracy
        if crt_alpha > 0:
            best_source = "crt_interpolation"
        model.load_state_dict(best_state)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    test_prediction = predict_logits(model, test_x).argmax(1)
    test_metrics = metrics(labels[test_idx], test_prediction)
    folded_metrics = {}
    if args.verify_folded_head:
        folded_prediction = folded_head_prediction(model, test_x)
        folded = metrics(labels[test_idx], folded_prediction)
        folded_metrics = {
            "folded_test_accuracy": folded["accuracy"],
            "folded_test_balanced_accuracy": folded["balanced_accuracy"],
            "folded_test_macro_f1": folded["macro_f1"],
            "folded_test_minimum_class_recall": folded["minimum_class_recall"],
            "folded_agreement": float(np.mean(folded_prediction == test_prediction)),
            "folded_disagreements": int(np.sum(folded_prediction != test_prediction)),
        }
        print(f"  folded head: acc={folded['accuracy']*100:.4f}%  "
              f"agreement={folded_metrics['folded_agreement']*100:.6f}%  "
              f"disagreements={folded_metrics['folded_disagreements']}/{len(test_idx)}", flush=True)
    model.cpu()
    best_state = {name: value.cpu() for name, value in best_state.items()}
    params, macs, flops = complexity(model)
    row = {
        "model_type": "mlp",
        "feature": feature_name,
        "input_dim": int(x.shape[1]),
        "architecture": architecture,
        "protocol": protocol,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "validation_accuracy": best_accuracy,
        **{f"test_{key}": value for key, value in test_metrics.items() if key not in {"confusion_matrix", "per_class_recall"}},
        "params": params,
        "macs": macs,
        "flops": flops,
        "state_dict_kib": serialized_size(model)/1024,
        "latency_us_cpu_1thread": benchmark_latency(model, x.shape[1]),
        "train_seconds": time.perf_counter()-started,
        "effective_training_samples": int(len(fit_y)),
        "distilled": bool(args.distill),
        "mixup_alpha": args.mixup_alpha,
        "scene_mix_alpha": args.scene_mix_alpha,
        "scene_mix_prob": args.scene_mix_prob,
        "scene_partner_fraction": partner_fraction,
        "state_permutation_prob": args.state_permutation_prob,
        "state_invariance_weight": args.state_invariance_weight,
        "state_sibling_fraction": sibling_fraction,
        "trajectory_mix_alpha": args.trajectory_mix_alpha,
        "trajectory_consistency_weight": args.trajectory_consistency_weight,
        "noise_std": args.noise_std,
        "head_scale": args.head_scale,
        "hybrid_alpha_init": args.hybrid_alpha_init,
        "rdrop_alpha": args.rdrop_alpha,
        "sam_rho": args.sam_rho,
        "asam_rho": args.asam_rho,
        "asam_eta": args.asam_eta,
        "swa": bool(args.swa),
        "swad_window": args.swad_window,
        "swad_force": bool(args.swad_force),
        "checkpoint_average_k": args.checkpoint_average_k,
        "frequency_conditional_normalization": bool(args.frequency_conditional_normalization),
        "quantile_normalization": bool(args.quantile_normalization),
        "keep_default_duplicates": bool(args.keep_default_duplicates),
        "checkpoint_average_members": checkpoint_average_members,
        "best_weight_source": best_source,
        "logit_adjustment": args.logit_adjustment,
        "loss": args.loss,
        "ldam_max_margin": args.ldam_max_margin,
        "ala_scale": args.ala_scale,
        "ala_linear_strength": args.ala_linear_strength,
        "ala_schedule": args.ala_schedule,
        "ala_warmup_fraction": args.ala_warmup_fraction,
        "ala_peak_fraction": args.ala_peak_fraction,
        "ala_final_factor": args.ala_final_factor,
        "pairwise_ala_strength": args.pairwise_ala_strength,
        "focal_gamma": args.focal_gamma,
        "effective_number_beta": args.effective_number_beta,
        "deferred_reweight_fraction": args.deferred_reweight_fraction,
        "balanced_contrastive_weight": args.balanced_contrastive_weight,
        "scene_cvar_weight": args.scene_cvar_weight,
        "scene_cvar_tail_fraction": args.scene_cvar_tail_fraction,
        "scene_rex_weight": args.scene_rex_weight,
        "group_dro_eta": args.group_dro_eta,
        "multi_teacher_distill": bool(args.multi_teacher_distill),
        "teacher_folds": args.teacher_folds,
        "teacher_epochs": args.teacher_epochs,
        "teacher_neural_weight": args.teacher_neural_weight,
        "crt_epochs": args.crt_epochs,
        "crt_alpha": crt_alpha,
        "difficulty_margin_epochs": args.difficulty_margin_epochs,
        "difficulty_margin": difficulty_margin.tolist(),
        "training_device": str(device),
        "optimizer": args.optimizer,
        "adam_beta2": args.adam_beta2,
        "scheduler": args.scheduler,
        "hybrid_alpha": (
            float(model.network[-1].alpha.detach())
            if hasattr(model.network[-1], "alpha") else None
        ),
        "physics_teacher": args.physics_teacher,
        "physics_per_point": args.physics_per_point,
        "physics_sigma": args.physics_sigma,
        "physics_temperature": args.physics_temperature,
        "physics_weight": args.physics_weight,
        "physics_batch_ratio": args.physics_batch_ratio,
        "physics_mode": args.physics_mode,
        "physics_sigmas": "|".join(f"{value:g}" for value in args.physics_sigmas) if args.physics_sigmas else "",
        "physics_pair_fraction": (
            args.physics_pair_fraction if args.physics_pair_fraction is not None
            else (1.0 if args.physics_mode == "pair" else 0.0)
        ),
        "physics_real_weight": args.physics_real_weight,
        **folded_metrics,
        "qat": bool(args.qat),
        "qat_weight_bits": args.qat_weight_bits if args.qat else 0,
        "qat_activation_bits": args.qat_activation_bits if args.qat else 0,
        "physics_min_distance": args.physics_min_distance,
        "physics_oversample": args.physics_oversample,
        "physics_weight_schedule": args.physics_weight_schedule,
        "physics_final_weight": args.physics_final_weight,
        "physics_decay_end": args.physics_decay_end,
        **{key: value for key, value in physics_info.items() if key != "physics_teacher"},
    }
    detail = {
        "summary": row,
        "history": history,
        "test": test_metrics,
        "pairwise_confusion_factors": (
            pairwise_matrix.detach().cpu().numpy().tolist() if pairwise_matrix is not None else None
        ),
    }
    return detail, best_state, scaler


def export_ablate_fold(directory: Path, state, scaler, feature: str, architecture: str,
                       protocol: str, seed: int, split_run: int, x_columns: int,
                       train_idx, valid_idx, test_idx, detail: dict, args) -> None:
    """Write one fold in the same layout ``load_fp32_teacher`` expects, plus the split."""
    directory = Path(directory) / f"{protocol}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in state.items()}, directory / "model_fp32.pth")
    scaler.dump(directory / "scaler.pkl")
    np.savez(directory / "split.npz", train=np.asarray(train_idx), valid=np.asarray(valid_idx),
             test=np.asarray(test_idx))
    summary = detail.get("summary", {})
    save_json(directory / "metadata.json", {
        "feature_spec": feature, "feature_names": FEATURE_SPECS[feature], "input_dim": int(x_columns),
        "architecture": architecture, "hidden": ARCHITECTURES[architecture], "num_classes": 15,
        "dropout": args.dropout, "head_scale": args.head_scale,
        "hybrid_alpha_init": args.hybrid_alpha_init,
        "protocol": protocol, "seed": int(seed), "split_run": int(split_run),
        "train_rows": int(len(train_idx)), "valid_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "fp32_holdout": {key[5:]: value for key, value in summary.items() if key.startswith("test_")},
        "scope": "single fold of the ablate protocol; fitted on train only, test never seen",
    })
    print(f"  exported fold -> {directory}", flush=True)


def run_ablation(bundle, args: argparse.Namespace) -> None:
    output = args.output / "ablation"
    output.mkdir(parents=True, exist_ok=True)
    teacher_x = build_features(bundle.raw, "raw5")
    rows = []
    for feature in args.features:
        x = build_features(bundle.raw, feature, bundle.point_groups, bundle.sweep_groups)
        for architecture in args.architectures:
            for protocol in args.protocols:
                for run, seed in enumerate(args.seeds):
                    split_run = args.sweep_fold if protocol == "sweep_group" and args.sweep_fold is not None else run
                    train_idx, valid_idx, test_idx = choose_split(bundle, protocol, split_run, seed)
                    print(f"MLP feature={feature:10s} arch={architecture:11s} protocol={protocol:15s} seed={seed}", flush=True)
                    detail, fold_state, fold_scaler = train_mlp(
                        x, teacher_x, bundle.labels, bundle.point_groups, bundle.sweep_groups,
                        bundle.raw[:, 2], train_idx, valid_idx, test_idx,
                        feature, architecture, protocol, seed, args, raw=bundle.raw,
                    )
                    if args.ablate_export_dir is not None:
                        export_ablate_fold(args.ablate_export_dir, fold_state, fold_scaler,
                                           feature, architecture, protocol, seed, split_run,
                                           x.shape[1], train_idx, valid_idx, test_idx, detail, args)
                    rows.append(detail["summary"])
                    save_json(output / "runs" / f"{feature}__{architecture}__{protocol}__{seed}.json", detail)
                    pd.DataFrame(rows).to_csv(output / "raw_runs.csv", index=False, encoding="utf-8-sig")
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["feature", "input_dim", "architecture", "protocol"], as_index=False).agg(
        accuracy_mean=("test_accuracy", "mean"), accuracy_std=("test_accuracy", "std"),
        balanced_accuracy_mean=("test_balanced_accuracy", "mean"), macro_f1_mean=("test_macro_f1", "mean"),
        minimum_class_recall_mean=("test_minimum_class_recall", "mean"), params=("params", "first"),
        macs=("macs", "first"), state_dict_kib=("state_dict_kib", "mean"),
        latency_us=("latency_us_cpu_1thread", "mean"), best_epoch_median=("best_epoch", "median"),
    )
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(summary, bundle.audit, output / "report.md")


def run_trees(bundle, args: argparse.Namespace) -> None:
    rows = []
    output = args.output / "trees"
    output.mkdir(parents=True, exist_ok=True)
    for feature in args.features:
        x = build_features(bundle.raw, feature, bundle.point_groups, bundle.sweep_groups)
        for protocol in args.protocols:
            for run, seed in enumerate(args.seeds):
                train, valid, test = choose_split(bundle, protocol, run, seed)
                fit = np.r_[train, valid]
                for name, model in (
                    ("extra_trees", ExtraTreesClassifier(n_estimators=200, max_features=None, class_weight="balanced", n_jobs=-1, random_state=seed)),
                    ("hist_gb", HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, l2_regularization=1e-3, random_state=seed)),
                ):
                    print(f"TREE feature={feature:10s} model={name:11s} protocol={protocol:15s} seed={seed}", flush=True)
                    started = time.perf_counter()
                    model.fit(x[fit], bundle.labels[fit])
                    pred = model.predict(x[test])
                    result = metrics(bundle.labels[test], pred)
                    blob = pickle.dumps(model, protocol=5)
                    if name == "extra_trees":
                        nodes = int(sum(tree.tree_.node_count for tree in model.estimators_))
                    else:
                        nodes = None
                    sample = x[test[:1]]
                    begin = time.perf_counter_ns()
                    for _ in range(100):
                        model.predict(sample)
                    latency = (time.perf_counter_ns()-begin)/100/1000
                    rows.append({"model_type": name, "feature": feature, "input_dim":x.shape[1], "protocol":protocol, "seed":seed,
                                 **{key:value for key,value in result.items() if key not in {"confusion_matrix","per_class_recall"}},
                                 "serialized_kib":len(blob)/1024, "nodes":nodes, "latency_us":latency, "train_seconds":time.perf_counter()-started})
                    pd.DataFrame(rows).to_csv(output / "raw_runs.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows).groupby(["model_type","feature","input_dim","protocol"],as_index=False).agg(
        accuracy_mean=("accuracy","mean"), accuracy_std=("accuracy","std"), balanced_accuracy_mean=("balanced_accuracy","mean"),
        macro_f1_mean=("macro_f1","mean"), minimum_class_recall_mean=("minimum_class_recall","mean"), serialized_kib=("serialized_kib","mean"),
        nodes=("nodes","mean"), latency_us=("latency_us","mean"))
    summary.to_csv(output / "summary.csv",index=False,encoding="utf-8-sig")


def run_feature_analysis(bundle, args: argparse.Namespace) -> None:
    all_features = build_features(bundle.raw, "raw12")
    names = FEATURE_SPECS["raw12"]
    train, _, test = split_indices(bundle.labels, bundle.sweep_groups, 42)
    scaler = Standardizer.fit(all_features[train])
    train_x, test_x = scaler.transform(all_features[train]), scaler.transform(all_features[test])
    mi = mutual_info_classif(train_x, bundle.labels[train], random_state=42)
    forest = ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42).fit(train_x, bundle.labels[train])
    perm = permutation_importance(forest, test_x, bundle.labels[test], n_repeats=5, random_state=42, n_jobs=-1)
    frame = pd.DataFrame({"feature":names,"mutual_information":mi,"tree_importance":forest.feature_importances_,
                          "permutation_importance_mean":perm.importances_mean,"permutation_importance_std":perm.importances_std})
    frame.sort_values("permutation_importance_mean",ascending=False).to_csv(args.output/"feature_analysis.csv",index=False,encoding="utf-8-sig")


def train_full(bundle, args: argparse.Namespace) -> None:
    feature = args.final_feature
    architecture = args.final_architecture
    x = build_features(bundle.raw, feature, bundle.point_groups, bundle.sweep_groups)
    scaler = Standardizer.fit(x)
    normalized = scaler.transform(x)
    set_seed(20260823)
    device = resolve_device(args.device)
    if args.distill and args.multi_teacher_distill:
        raise ValueError("choose either --distill or --multi-teacher-distill")
    teacher_oof = None
    pairwise_oof = None
    if args.distill or args.multi_teacher_distill or args.pairwise_ala_strength > 0:
        raw_teacher = build_features(bundle.raw, "raw5")
        if args.multi_teacher_distill:
            teacher_oof = oof_multiteacher_probabilities(
                raw_teacher, bundle.labels, bundle.sweep_groups, 20260823,
                device, args.teacher_folds, args.teacher_epochs, args.batch_size,
                args.teacher_trees, args.teacher_neural_weight,
            )
        else:
            teacher_scaler = Standardizer.fit(raw_teacher)
            teacher_oof = oof_teacher_probabilities(
                teacher_scaler.transform(raw_teacher), bundle.labels,
                bundle.sweep_groups, 20260823, trees=args.teacher_trees,
            )
        pairwise_oof = teacher_oof
        if args.multi_teacher_distill and args.pairwise_ala_strength > 0:
            teacher_scaler = Standardizer.fit(raw_teacher)
            pairwise_oof = oof_teacher_probabilities(
                teacher_scaler.transform(raw_teacher), bundle.labels,
                bundle.sweep_groups, 20260823, trees=args.teacher_trees,
            )
        set_seed(20260823)
    model = IATMLP(
        x.shape[1], architecture, dropout=args.dropout,
        head_scale=args.head_scale, hybrid_alpha=args.hybrid_alpha_init,
    ).to(device)
    if args.qat and args.qat_init_from is not None:
        model.load_state_dict(torch.load(Path(args.qat_init_from), map_location=device))
        print(f"QAT initialised from {args.qat_init_from}", flush=True)
    final_quant_info = {}
    if args.qat:
        final_quant_info = (
            attach_lsq_fake_quant(model, args.qat_weight_bits, args.qat_activation_bits,
                                  args.qat_act_percentile)
            if args.qat_scheme == "lsq"
            else attach_fake_quant(model, args.qat_weight_bits, args.qat_activation_bits)
        )
    kd_teacher = None
    if args.qat and args.qat_kd_from is not None and args.qat_kd_weight > 0:
        kd_teacher = load_teacher(args.qat_kd_from, x.shape[1], architecture,
                                  args.head_scale, args.hybrid_alpha_init, device)
    if args.qat and args.qat_scheme == "lsq":
        model.eval()
        with torch.no_grad():   # trigger the lazy step-size initialisation before the optimiser is built
            model(torch.from_numpy(normalized[:min(4096, len(normalized))].astype(np.float32)).to(device))
        model.train()
        groups = split_parameter_groups(
            model, args.learning_rate,
            args.qat_step_lr if args.qat_step_lr is not None else args.learning_rate,
            args.weight_decay,
        )
        optimizer = torch.optim.AdamW(groups, lr=args.learning_rate,
                                      weight_decay=args.weight_decay, betas=(0.9, args.adam_beta2))
        print(f"LSQ quantiser parameters: {quantiser_parameter_count(model)} "
              f"(excluded from the reported parameter count)", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(0.9, args.adam_beta2),
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.final_epochs, eta_min=args.learning_rate/50)
    weight = class_weight_tensor(bundle.labels).to(device) if args.class_weights else None
    counts = np.bincount(bundle.labels, minlength=15).astype(np.float32)
    final_ldam_margin = args.ldam_max_margin * np.power(np.maximum(counts, 1), -0.25)
    final_ldam_margin /= max(float(final_ldam_margin.max()), 1e-12)
    final_ldam_margin *= args.ldam_max_margin
    final_ldam_margin = torch.from_numpy(final_ldam_margin).to(device)
    final_quantity = 1.0 / np.log(np.maximum(counts, 1) / max(float(counts.min()), 1.0) + 1.0)
    final_quantity = torch.from_numpy(final_quantity.astype(np.float32)).to(device)
    final_pairwise = None
    if args.pairwise_ala_strength > 0:
        assert pairwise_oof is not None
        final_pairwise = torch.from_numpy(
            pairwise_confusion_factors(pairwise_oof, bundle.labels)
        ).to(device)
    final_physics = build_physics_augmentation(
        bundle.raw, bundle.labels, np.arange(len(bundle.labels)), feature, scaler,
        20260823, args, device,
    )
    final_physics_batch = int(round(args.batch_size * args.physics_batch_ratio))
    if final_physics is not None:
        print(f"physics teacher: {json.dumps(final_physics[2], ensure_ascii=False)}", flush=True)
    full_x = torch.from_numpy(normalized)
    full_y = torch.from_numpy(bundle.labels)
    full_soft = (
        torch.from_numpy(teacher_oof)
        if teacher_oof is not None and (args.distill or args.multi_teacher_distill) else None
    )
    full_scene = torch.from_numpy(bundle.sweep_groups.astype(np.int64)) if args.scene_cvar_weight > 0 else None
    full_partner = None
    if args.trajectory_consistency_weight > 0:
        full_partners = trajectory_neighbor_partners(
            bundle.labels,
            bundle.raw[:, 2],
            bundle.sweep_groups,
            bundle.point_groups,
            20260823,
        )
        full_partner = torch.from_numpy(normalized[full_partners])
    full_sibling = None
    if args.state_permutation_prob > 0 or args.state_invariance_weight > 0:
        full_siblings = state_sibling_partners(bundle.point_groups, 20260823)
        full_sibling = torch.from_numpy(normalized[full_siblings])
    full_physics_real = None
    if final_physics is not None and args.physics_real_weight > 0:
        full_physics_real = torch.from_numpy(physics_real_soft_labels(
            final_physics[3], bundle.raw, np.arange(len(bundle.labels)), args.physics_temperature,
        ))
    if device.type == "cuda":
        full_x = full_x.to(device); full_y = full_y.to(device)
        if full_soft is not None: full_soft = full_soft.to(device)
        if full_scene is not None: full_scene = full_scene.to(device)
        if full_partner is not None: full_partner = full_partner.to(device)
        if full_sibling is not None: full_sibling = full_sibling.to(device)
        if full_physics_real is not None: full_physics_real = full_physics_real.to(device)
        loader = None
    else:
        full_tensors = [full_x, full_y]
        if full_soft is not None: full_tensors.append(full_soft)
        if full_scene is not None: full_tensors.append(full_scene)
        if full_partner is not None: full_tensors.append(full_partner)
        if full_sibling is not None: full_tensors.append(full_sibling)
        if full_physics_real is not None: full_tensors.append(full_physics_real)
        loader = DataLoader(TensorDataset(*full_tensors),batch_size=args.batch_size,shuffle=True,
                            generator=torch.Generator().manual_seed(20260823),num_workers=0)
    history=[]

    def final_adjusted_logits(logits, targets, epoch):
        adjusted = logits
        schedule_factor = ala_schedule_factor(
            epoch, args.final_epochs, args.ala_schedule,
            args.ala_warmup_fraction, args.ala_peak_fraction, args.ala_final_factor,
        )
        if args.loss == "ldam":
            adjusted = logits.clone()
            adjusted[torch.arange(len(targets), device=device), targets] -= final_ldam_margin[targets]
        elif args.loss == "ala":
            if not architecture.endswith("_cosine"):
                raise ValueError("ALA loss requires a cosine architecture")
            true_cosine = (logits.detach()/args.ala_scale).gather(1,targets[:,None]).squeeze(1)
            difficulty = (1.0-true_cosine).clamp(0.0,2.0)/2.0
            adjusted = logits.clone()
            adjustment = schedule_factor*args.ala_scale*difficulty*final_quantity[targets]
            if final_pairwise is not None:
                rivals=logits.detach().clone(); rivals[torch.arange(len(targets),device=device),targets]=-torch.inf
                adjustment *= 1.0+args.pairwise_ala_strength*final_pairwise[targets,rivals.argmax(1)]
            adjusted[torch.arange(len(targets),device=device),targets] -= adjustment
        elif args.loss == "ala_linear":
            probability = F.softmax(logits.detach(),dim=1).gather(1,targets[:,None]).squeeze(1)
            adjusted = logits.clone()
            adjustment = schedule_factor*args.ala_linear_strength*(1.0-probability)*final_quantity[targets]
            if final_pairwise is not None:
                rivals=logits.detach().clone(); rivals[torch.arange(len(targets),device=device),targets]=-torch.inf
                adjustment *= 1.0+args.pairwise_ala_strength*final_pairwise[targets,rivals.argmax(1)]
            adjusted[torch.arange(len(targets),device=device),targets] -= adjustment
        return adjusted

    for epoch in range(1,args.final_epochs+1):
        physics_weight_now = physics_weight_at(epoch, args.final_epochs, args)
        if (args.qat and args.qat_scheme == "lsq" and args.qat_freeze_at > 0
                and epoch == max(1, int(round(args.final_epochs * args.qat_freeze_at)))):
            print(f"epoch {epoch}: froze {freeze_quantisers(model)} quantiser step sizes", flush=True)
        model.train(); running=0.0
        if loader is None:
            order = torch.randperm(len(bundle.labels),device=device)
            batches = (
                (
                    full_x[index], full_y[index],
                    full_soft[index] if full_soft is not None else None,
                    full_scene[index] if full_scene is not None else None,
                    full_partner[index] if full_partner is not None else None,
                    full_sibling[index] if full_sibling is not None else None,
                    full_physics_real[index] if full_physics_real is not None else None,
                )
                for index in order.split(args.batch_size)
            )
        else:
            def cpu_batches():
                for batch in loader:
                    position = 2
                    soft_batch = batch[position] if full_soft is not None else None
                    position += int(full_soft is not None)
                    scene_batch = batch[position] if full_scene is not None else None
                    position += int(full_scene is not None)
                    partner_batch = batch[position] if full_partner is not None else None
                    position += int(full_partner is not None)
                    sibling_batch = batch[position] if full_sibling is not None else None
                    position += int(full_sibling is not None)
                    physics_real_batch = batch[position] if full_physics_real is not None else None
                    yield (batch[0], batch[1], soft_batch, scene_batch, partner_batch,
                           sibling_batch, physics_real_batch)
            batches = cpu_batches()
        for xb,yb,teacher_prob,scene_ids,partner_x,sibling_x,physics_real in batches:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            if teacher_prob is not None: teacher_prob = teacher_prob.to(device, non_blocking=True)
            if scene_ids is not None: scene_ids = scene_ids.to(device, non_blocking=True)
            if partner_x is not None: partner_x = partner_x.to(device, non_blocking=True)
            if physics_real is not None: physics_real = physics_real.to(device, non_blocking=True)
            clean_xb = xb
            if sibling_x is not None:
                sibling_x = sibling_x.to(device, non_blocking=True)
                if args.state_permutation_prob > 0:
                    swap = torch.rand(len(xb), 1, device=xb.device) < args.state_permutation_prob
                    xb = torch.where(swap, sibling_x, xb)
            optimizer.zero_grad(set_to_none=True); logits=model(xb)
            adjusted = final_adjusted_logits(logits, yb, epoch)
            per_sample_loss=F.cross_entropy(
                adjusted,yb,weight=weight,label_smoothing=args.label_smoothing,reduction="none"
            )
            if teacher_prob is not None:
                temp=args.temperature
                softened=teacher_prob.clamp_min(1e-8).pow(1/temp)
                softened=softened/softened.sum(1,keepdim=True)
                kd=F.kl_div(F.log_softmax(logits/temp,dim=1),softened,reduction="none").sum(1)*temp*temp
                per_sample_loss=(1-args.distill_weight)*per_sample_loss+args.distill_weight*kd
            loss=scene_cvar_loss(
                per_sample_loss,scene_ids,args.scene_cvar_weight,args.scene_cvar_tail_fraction
            )
            if partner_x is not None:
                partner_logits = model(partner_x)
                partner_adjusted = final_adjusted_logits(partner_logits, yb, epoch)
                partner_per_sample = F.cross_entropy(
                    partner_adjusted,yb,weight=weight,label_smoothing=args.label_smoothing,reduction="none"
                )
                partner_loss = scene_cvar_loss(
                    partner_per_sample,scene_ids,args.scene_cvar_weight,args.scene_cvar_tail_fraction
                )
                log_primary = F.log_softmax(logits,dim=1)
                log_partner = F.log_softmax(partner_logits,dim=1)
                consistency = 0.5*(
                    F.kl_div(log_primary,log_partner.exp(),reduction="batchmean")
                    + F.kl_div(log_partner,log_primary.exp(),reduction="batchmean")
                )
                loss = 0.5*(loss+partner_loss)+args.trajectory_consistency_weight*consistency
            if sibling_x is not None and args.state_invariance_weight > 0:
                sibling_logits = model(sibling_x)
                log_sample = F.log_softmax(logits,dim=1)
                log_sibling = F.log_softmax(sibling_logits,dim=1)
                invariance = 0.5*(
                    F.kl_div(log_sample,log_sibling.exp(),reduction="batchmean")
                    + F.kl_div(log_sibling,log_sample.exp(),reduction="batchmean")
                )
                loss = loss + args.state_invariance_weight * invariance
            if physics_real is not None and args.physics_real_weight > 0:
                real_logits = logits if clean_xb is xb else model(clean_xb)
                loss = loss + args.physics_real_weight * F.kl_div(
                    F.log_softmax(real_logits,dim=1), physics_real, reduction="batchmean",
                )
            if kd_teacher is not None:
                loss = loss + args.qat_kd_weight * distillation_loss(
                    logits, kd_teacher(clean_xb), args.qat_kd_temperature,
                )
            if final_physics is not None and physics_weight_now > 0:
                loss = loss + physics_weight_now * physics_consistency_loss(
                    model, final_physics[0], final_physics[1], final_physics_batch, device,
                )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); running+=float(loss.detach())*len(yb)
        scheduler.step(); history.append({"epoch":epoch,"train_loss":running/len(bundle.labels)})
        if epoch == 1 or epoch % 10 == 0: print(f"final epoch {epoch}/{args.final_epochs} loss={history[-1]['train_loss']:.6f}",flush=True)
    model.eval(); pred=predict_logits(model,normalized).argmax(1); full_metrics=metrics(bundle.labels,pred); model.cpu()
    params,macs,flops=complexity(model)
    if args.qat and args.qat_scheme == "lsq":
        params -= quantiser_parameter_count(model)
    model_dir=args.model_output.resolve(); model_dir.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),model_dir/"model_fp32.pth"); scaler.dump(model_dir/"scaler.pkl")
    if args.qat:
        if args.qat_scheme == "lsq":
            export_int8_artifact_lsq(model, model_dir, final_quant_info)
        else:
            export_int8_artifact(model, model_dir, final_quant_info)
    metadata={"feature_spec":feature,"feature_names":FEATURE_SPECS[feature],"input_dim":x.shape[1],"architecture":architecture,
              "hidden":ARCHITECTURES[architecture],"num_classes":15,"dropout":args.dropout,"params":params,"macs":macs,"flops":flops,
              "head_scale":args.head_scale,"hybrid_alpha_init":args.hybrid_alpha_init,"rdrop_alpha":args.rdrop_alpha,
              "state_dict_kib":(model_dir/"model_fp32.pth").stat().st_size/1024,"training_seed":20260823,"training_epochs":args.final_epochs,
              "optimizer":"adamw","scheduler":"cosine","learning_rate":args.learning_rate,
              "weight_decay":args.weight_decay,"adam_beta2":args.adam_beta2,
              "training_loss":args.loss,"ldam_max_margin":args.ldam_max_margin,
              "ala_linear_strength":args.ala_linear_strength,
              "ala_schedule":args.ala_schedule,"ala_warmup_fraction":args.ala_warmup_fraction,
              "ala_peak_fraction":args.ala_peak_fraction,"ala_final_factor":args.ala_final_factor,
              "pairwise_ala_strength":args.pairwise_ala_strength,
              "scene_cvar_weight":args.scene_cvar_weight,"scene_cvar_tail_fraction":args.scene_cvar_tail_fraction,
              "trajectory_consistency_weight":args.trajectory_consistency_weight,
              "state_permutation_prob":args.state_permutation_prob,
              "state_invariance_weight":args.state_invariance_weight,
              "physics_sigmas":list(args.physics_sigmas) if args.physics_sigmas else [],
              "physics_pair_fraction":(
                  args.physics_pair_fraction if args.physics_pair_fraction is not None
                  else (1.0 if args.physics_mode == "pair" else 0.0)
              ),
              "physics_real_weight":args.physics_real_weight,
              "qat":bool(args.qat),
              "qat_weight_bits":args.qat_weight_bits if args.qat else 0,
              "qat_activation_bits":args.qat_activation_bits if args.qat else 0,
              "qat_scheme":args.qat_scheme if args.qat else "",
              "qat_step_lr":(args.qat_step_lr if args.qat_step_lr is not None else args.learning_rate) if args.qat and args.qat_scheme=="lsq" else 0.0,
              "qat_act_percentile":args.qat_act_percentile if args.qat and args.qat_scheme=="lsq" else 0.0,
              "qat_freeze_at":args.qat_freeze_at if args.qat else 0.0,
              "qat_init_from":str(args.qat_init_from) if args.qat and args.qat_init_from else "",
              "qat_kd_from":str(args.qat_kd_from) if args.qat and args.qat_kd_from else "",
              "qat_kd_weight":args.qat_kd_weight if args.qat else 0.0,
              "qat_kd_temperature":args.qat_kd_temperature if args.qat else 0.0,
              "physics_min_distance":args.physics_min_distance,
              "physics_weight_schedule":args.physics_weight_schedule,
              "physics_final_weight":args.physics_final_weight,
              "physics_decay_end":args.physics_decay_end,
              "physics_teacher":args.physics_teacher,"physics_per_point":args.physics_per_point,
              "physics_sigma":args.physics_sigma,"physics_temperature":args.physics_temperature,
              "physics_weight":args.physics_weight,"physics_batch_ratio":args.physics_batch_ratio,
              "physics_mode":args.physics_mode,
              "training_only_note":(
                  "Moebius teacher/synthetic rows and trajectory partners are training-only; inference input stays raw5"
                  if final_physics is not None else
                  "Trajectory partners and consistency loss are training-only; inference input stays raw5"
              ),
              **({key:value for key,value in final_physics[2].items() if key!="physics_teacher"} if final_physics is not None else {}),
              "distilled":bool(args.distill or args.multi_teacher_distill),
              "multi_teacher_distill":bool(args.multi_teacher_distill),
              "teacher_folds":args.teacher_folds,"teacher_epochs":args.teacher_epochs,
              "teacher_trees":args.teacher_trees,"teacher_neural_weight":args.teacher_neural_weight,
              "hybrid_alpha":float(model.network[-1].alpha.detach()) if hasattr(model.network[-1],"alpha") else None,
              "training_device":str(device),"torch_version":torch.__version__,
              "public_full_fit_metrics_descriptive_only":full_metrics,"data_audit":bundle.audit,
              "transform_note":"gamma values are signed Q15-like integers; impedance3 divides gamma by 32768 before z=(1+gamma)/(1-gamma)"}
    save_json(model_dir/"metadata.json",metadata); save_json(args.output/"final_training_history.json",history)
    print(json.dumps({"model":str(model_dir/"model_fp32.pth"),"accuracy":full_metrics["accuracy"],"input_dim":x.shape[1],"size_kib":metadata["state_dict_kib"]},ensure_ascii=False,indent=2))


def write_markdown_report(summary: pd.DataFrame, audit: dict, path: Path) -> None:
    lines=["# IAT FP32 消融实验报告","",f"- 样本：{audit['rows']:,}，类别：{audit['classes']}，单点物理组：{audit['unique_point_groups']:,}。",
           f"- 推断扫描轨迹：{audit['sweep_groups']}（两个频段分别独立重建）。","- random 用于同口径比较；最终选型以 sweep_group 为主。","",
           "|特征|维数|结构|协议|准确率|Balanced Acc|Macro-F1|参数|MACs|模型 KiB|","|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in summary.sort_values(["protocol","accuracy_mean"],ascending=[True,False]).iterrows():
        lines.append(f"|{r.feature}|{int(r.input_dim)}|{r.architecture}|{r.protocol}|{r.accuracy_mean:.5f} ± {0 if pd.isna(r.accuracy_std) else r.accuracy_std:.5f}|{r.balanced_accuracy_mean:.5f}|{r.macro_f1_mean:.5f}|{int(r.params)}|{int(r.macs)}|{r.state_dict_kib:.2f}|")
    lines += ["","## 解释原则","","随机划分中同一默认态/频点组会跨集合，因此 99% 以上结果只用于和常见公开口径比较；未知用户场景能力必须结合 sweep_group 与 frequency_block 结果判断。"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    args=parse_args(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    if args.teacher_cache_dir is not None:
        args.teacher_cache_dir = args.teacher_cache_dir.resolve()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    device = resolve_device(args.device)
    print(f"training device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    bundle=load_dataset(args.x.resolve(),args.y.resolve()); save_json(args.output/"data_audit.json",bundle.audit)
    if args.mode=="audit": print(json.dumps(bundle.audit,ensure_ascii=False,indent=2))
    elif args.mode=="features": run_feature_analysis(bundle,args)
    elif args.mode=="trees": run_trees(bundle,args)
    elif args.mode=="ablate": run_ablation(bundle,args)
    else: train_full(bundle,args)


if __name__=="__main__": main()
