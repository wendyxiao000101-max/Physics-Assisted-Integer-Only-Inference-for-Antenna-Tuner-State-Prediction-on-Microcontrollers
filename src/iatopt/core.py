from __future__ import annotations

import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

RAW_NAMES = (
    "gammaIn1Re",
    "gammaIn1Im",
    "closeFreqMHz",
    "itState",
    "gammaIn2Re",
    "gammaIn2Im",
)

FEATURE_SPECS: dict[str, tuple[str, ...]] = {
    "raw12": (
        "g1_re", "g1_im", "freq", "g2_re", "g2_im", "g1_mag", "g1_phase",
        "g2_mag", "g2_phase", "delta_re", "delta_im", "delta_mag",
    ),
    "raw6": RAW_NAMES,
    "raw5": ("g1_re", "g1_im", "freq", "g2_re", "g2_im"),
    "polar4": ("freq", "g2_mag", "g2_sin_phase", "g2_cos_phase"),
    "gamma3": ("freq", "g2_re", "g2_im"),
    "impedance3": ("freq", "z2_re", "z2_im"),
    "gamma2": ("g2_re", "g2_im"),
    "g1_only3": ("freq", "g1_re", "g1_im"),
    "g1_state4": ("freq", "g1_re", "g1_im", "state"),
    "default_polar5": ("freq", "g2_re", "g2_im", "g2_mag", "g2_phase"),
    "default_logpolar5": ("freq", "g2_re", "g2_im", "log1p_g2_mag", "g2_phase"),
    "default_phase0_polar5": ("freq", "g2_re", "g2_im", "g2_mag", "g2_phase_0_2pi"),
    "delta5": ("freq", "g2_re", "g2_im", "delta_re", "delta_im"),
    "ratio5": ("freq", "g2_re", "g2_im", "ratio_re", "ratio_im"),
    "smith5": ("freq", "z2_re", "z2_im", "delta_z_re", "delta_z_im"),
    "relative_polar5": ("freq", "g1_mag", "g2_mag", "relative_phase_sin", "relative_phase_cos"),
    "relative_ratio5": ("freq", "g1_mag", "magnitude_ratio", "relative_phase_sin", "relative_phase_cos"),
    "context_slope5": ("freq", "g2_re", "g2_im", "g2_slope_re", "g2_slope_im"),
    "context_tangent5": ("freq", "g2_re", "g2_im", "delta_tangent", "delta_normal"),
    "context_frenet5": ("freq", "g2_residual_tangent", "g2_residual_normal", "delta_tangent", "delta_normal"),
    "context_rotated5": ("freq", "g1_tangent", "g1_normal", "g2_tangent", "g2_normal"),
}

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "single32": (32,),
    "single64": (64,),
    "mlp32_16": (32, 16),
    "mlp48_24": (48, 24),
    "mlp48_24_silu": (48, 24),
    "mlp48_24_gelu": (48, 24),
    "mlp64_32": (64, 32),
    "original64_32": (64, 32),
    "mlp64_32_silu": (64, 32),
    "mlp64_32_gelu": (64, 32),
    "mlp64_32_leaky": (64, 32),
    "mlp64_32_bn": (64, 32),
    "mlp80_40": (80, 40),
    "mlp96_48": (96, 48),
    "mlp128_64": (128, 64),
    "mlp128_64_silu": (128, 64),
    "mlp32_16_gelu_hybrid": (32, 16),
    "mlp64_32_gelu_hybrid": (64, 32),
    "mlp128_64_gelu_hybrid": (128, 64),
    "mlp64_32_gelu_cosine": (64, 32),
    "residual32_gelu_relu_cosine": (32, 32),
    "residual64": (64, 64),
    "residual70": (70, 70),
    "residual32": (32, 32),
    "residual32_gelu_relu_hybrid": (32, 32),
    "residual32_noskip_gelu_relu_hybrid": (32, 32),
    "residual40": (40, 40),
    "residual40_gelu_relu_hybrid": (40, 40),
    "residual46_gelu_relu_hybrid": (46, 46),
    "residual48": (48, 48),
    "residual48_gelu_relu": (48, 48),
    "residual48_gelu_relu_cosine": (48, 48),
    "residual48_silu": (48, 48),
    "residual48_mish": (48, 48),
    "residual48_mish_relu": (48, 48),
    "residual48_mish_relu_cosine": (48, 48),
    "residual48_mish_relu_hybrid": (48, 48),
    "residual48_silu_relu_hybrid": (48, 48),
    "residual48_gelu_relu_hybrid": (48, 48),
    "residual48_mish_silu_hybrid": (48, 48),
    "residual48_relu_silu": (48, 48),
    "physics_residual48": (48, 48),
}


@dataclass(frozen=True)
class DatasetBundle:
    raw: np.ndarray
    labels: np.ndarray
    point_groups: np.ndarray
    sweep_groups: np.ndarray
    feature_path: Path
    label_path: Path
    audit: dict


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def dump(self, path: Path) -> None:
        joblib.dump({"mean": self.mean, "scale": self.scale}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        payload = joblib.load(path)
        if payload.get("kind") == "frequency_conditional":
            return FrequencyConditionalStandardizer(
                frequencies=np.asarray(payload["frequencies"], dtype=np.float64),
                conditional_mean=np.asarray(payload["conditional_mean"], dtype=np.float64),
                conditional_scale=np.asarray(payload["conditional_scale"], dtype=np.float64),
                frequency_mean=float(payload["frequency_mean"]),
                frequency_scale=float(payload["frequency_scale"]),
            )
        if payload.get("kind") == "quantile_normal":
            return QuantileNormalStandardizer(payload["transformer"])
        return cls(np.asarray(payload["mean"]), np.asarray(payload["scale"]))


@dataclass
class FrequencyConditionalStandardizer:
    """Normalize raw5 RF coordinates using train-fold statistics per frequency."""

    frequencies: np.ndarray
    conditional_mean: np.ndarray
    conditional_scale: np.ndarray
    frequency_mean: float
    frequency_scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "FrequencyConditionalStandardizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("frequency-conditional normalization requires Nx5 raw5 features")
        frequencies = np.sort(np.unique(values[:, 2]))
        rf_columns = (0, 1, 3, 4)
        conditional_mean = np.empty((len(frequencies), len(rf_columns)), dtype=np.float64)
        conditional_scale = np.empty_like(conditional_mean)
        for row, frequency in enumerate(frequencies):
            subset = values[values[:, 2] == frequency][:, rf_columns]
            conditional_mean[row] = subset.mean(axis=0)
            conditional_scale[row] = subset.std(axis=0)
        conditional_scale[conditional_scale < 1e-12] = 1.0
        frequency_scale = float(values[:, 2].std())
        if frequency_scale < 1e-12:
            frequency_scale = 1.0
        return cls(
            frequencies=frequencies,
            conditional_mean=conditional_mean,
            conditional_scale=conditional_scale,
            frequency_mean=float(values[:, 2].mean()),
            frequency_scale=frequency_scale,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("frequency-conditional normalization requires Nx5 raw5 features")
        result = values.copy()
        query_frequency = values[:, 2]
        for table_column, value_column in enumerate((0, 1, 3, 4)):
            mean = np.interp(query_frequency, self.frequencies, self.conditional_mean[:, table_column])
            scale = np.interp(query_frequency, self.frequencies, self.conditional_scale[:, table_column])
            result[:, value_column] = (values[:, value_column] - mean) / np.maximum(scale, 1e-12)
        result[:, 2] = (query_frequency - self.frequency_mean) / self.frequency_scale
        return result.astype(np.float32)

    def dump(self, path: Path) -> None:
        joblib.dump(
            {
                "kind": "frequency_conditional",
                "frequencies": self.frequencies,
                "conditional_mean": self.conditional_mean,
                "conditional_scale": self.conditional_scale,
                "frequency_mean": self.frequency_mean,
                "frequency_scale": self.frequency_scale,
            },
            path,
            compress=3,
        )


@dataclass
class QuantileNormalStandardizer:
    """Train-only marginal quantile mapping to a standard normal distribution."""

    transformer: QuantileTransformer

    @classmethod
    def fit(cls, values: np.ndarray, n_quantiles: int = 256) -> "QuantileNormalStandardizer":
        values = np.asarray(values, dtype=np.float64)
        transformer = QuantileTransformer(
            n_quantiles=min(int(n_quantiles), len(values)),
            output_distribution="normal",
            subsample=None,
            random_state=0,
        )
        transformer.fit(values)
        return cls(transformer)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        result = self.transformer.transform(values)
        if not np.isfinite(result).all():
            raise ValueError("quantile normalization produced non-finite values")
        return result.astype(np.float32)

    def dump(self, path: Path) -> None:
        joblib.dump(
            {"kind": "quantile_normal", "transformer": self.transformer},
            path,
            compress=3,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(feature_path: Path, label_path: Path) -> DatasetBundle:
    frame = pd.read_csv(feature_path, encoding="gbk")
    labels = pd.read_csv(label_path, encoding="gbk").values.reshape(-1).astype(np.int64)
    if frame.shape[1] != 6:
        raise ValueError(f"expected six raw features, got {frame.shape}")
    if len(frame) != len(labels):
        raise ValueError("feature/label row count mismatch")
    frame.columns = RAW_NAMES
    raw = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("raw features contain NaN or infinity")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(15)):
        raise ValueError(f"expected labels 0..14, got {classes.tolist()}")

    point_index = pd.MultiIndex.from_arrays(
        [raw[:, 2].astype(np.int64), raw[:, 4].astype(np.int64), raw[:, 5].astype(np.int64)]
    )
    point_groups, uniques = pd.factorize(point_index, sort=True)
    group_label_counts = pd.DataFrame({"group": point_groups, "label": labels}).groupby("group").label.nunique()
    if int(group_label_counts.max()) != 1:
        raise ValueError("a default-state/frequency group contains conflicting labels")
    sweep_groups, sweep_audit = reconstruct_sweep_groups(raw, point_groups)
    sizes = np.bincount(point_groups)
    audit = {
        "rows": int(len(raw)),
        "raw_dimensions": 6,
        "classes": 15,
        "class_counts": np.bincount(labels, minlength=15).astype(int).tolist(),
        "unique_point_groups": int(len(uniques)),
        "point_group_size_min": int(sizes.min()),
        "point_group_size_max": int(sizes.max()),
        "duplicate_raw_rows": int(pd.DataFrame(raw).duplicated().sum()),
        **sweep_audit,
    }
    return DatasetBundle(raw, labels, point_groups.astype(np.int64), sweep_groups, feature_path, label_path, audit)


def reconstruct_sweep_groups(raw: np.ndarray, point_groups: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reconstruct smooth default-state trajectories without using labels.

    The public data contains 105 unique default-state points at every frequency.
    Tracks are matched independently inside each contiguous frequency band.
    """
    points = pd.DataFrame(
        {"freq": raw[:, 2].astype(np.int64), "g2_re": raw[:, 4], "g2_im": raw[:, 5], "point": point_groups}
    ).drop_duplicates("point")
    freqs = np.sort(points.freq.unique())
    bands: list[list[int]] = []
    for freq in freqs:
        if not bands or freq != bands[-1][-1] + 1:
            bands.append([int(freq)])
        else:
            bands[-1].append(int(freq))

    point_to_sweep: dict[int, int] = {}
    match_distances: list[float] = []
    next_group = 0
    tracks_per_band: list[int] = []
    for band in bands:
        first = points[points.freq == band[0]].sort_values(["g2_re", "g2_im"])
        n_tracks = len(first)
        tracks_per_band.append(n_tracks)
        previous_xy = first[["g2_re", "g2_im"]].to_numpy()
        previous_tracks = np.arange(next_group, next_group + n_tracks)
        for point, track in zip(first.point.to_numpy(), previous_tracks):
            point_to_sweep[int(point)] = int(track)
        for freq in band[1:]:
            current = points[points.freq == freq].sort_values(["g2_re", "g2_im"])
            current_xy = current[["g2_re", "g2_im"]].to_numpy()
            if len(current_xy) != n_tracks:
                raise ValueError(f"frequency {freq} has {len(current_xy)} points, expected {n_tracks}")
            rows, cols = linear_sum_assignment(cdist(previous_xy, current_xy))
            assigned_tracks = np.empty(n_tracks, dtype=np.int64)
            assigned_tracks[cols] = previous_tracks[rows]
            match_distances.extend(cdist(previous_xy, current_xy)[rows, cols].tolist())
            for point, track in zip(current.point.to_numpy(), assigned_tracks):
                point_to_sweep[int(point)] = int(track)
            previous_xy = current_xy
            previous_tracks = assigned_tracks
        next_group += n_tracks
    sweep = np.fromiter((point_to_sweep[int(group)] for group in point_groups), dtype=np.int64)
    return sweep, {
        "frequency_bands": [[band[0], band[-1]] for band in bands],
        "sweep_groups": int(next_group),
        "tracks_per_band": tracks_per_band,
        "trajectory_match_distance_median": float(np.median(match_distances)),
        "trajectory_match_distance_p95": float(np.percentile(match_distances, 95)),
        "trajectory_reconstruction_uses_labels": False,
    }


def build_features(
    raw: np.ndarray,
    spec: str,
    point_groups: np.ndarray | None = None,
    sweep_groups: np.ndarray | None = None,
) -> np.ndarray:
    if spec not in FEATURE_SPECS:
        raise KeyError(f"unknown feature spec {spec!r}")
    if spec.startswith("context_"):
        if point_groups is None or sweep_groups is None:
            raise ValueError(f"feature {spec} requires point_groups and sweep_groups")
        return build_context_features(raw, point_groups, sweep_groups, spec)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 6:
        raise ValueError(f"raw input must be Nx6, got {values.shape}")
    g1_re, g1_im, freq, state, g2_re, g2_im = values.T
    g1_mag = np.hypot(g1_re, g1_im)
    g2_mag = np.hypot(g2_re, g2_im)
    g1_phase = np.arctan2(g1_im, g1_re)
    g2_phase = np.arctan2(g2_im, g2_re)
    if spec == "raw12":
        result = np.column_stack((g1_re, g1_im, freq, g2_re, g2_im, g1_mag, g1_phase,
                                  g2_mag, g2_phase, g1_re-g2_re, g1_im-g2_im, g1_mag-g2_mag))
    elif spec == "raw6":
        result = values.copy()
    elif spec == "raw5":
        result = np.column_stack((g1_re, g1_im, freq, g2_re, g2_im))
    elif spec == "default_polar5":
        result = np.column_stack((freq, g2_re, g2_im, g2_mag, g2_phase))
    elif spec == "default_logpolar5":
        result = np.column_stack((freq, g2_re, g2_im, np.log1p(g2_mag), g2_phase))
    elif spec == "default_phase0_polar5":
        result = np.column_stack((freq, g2_re, g2_im, g2_mag, np.mod(g2_phase, 2.0 * np.pi)))
    elif spec == "polar4":
        result = np.column_stack((freq, g2_mag, np.sin(g2_phase), np.cos(g2_phase)))
    elif spec == "gamma3":
        result = np.column_stack((freq, g2_re, g2_im))
    elif spec == "g1_only3":
        result = np.column_stack((freq, g1_re, g1_im))
    elif spec == "g1_state4":
        result = np.column_stack((freq, g1_re, g1_im, state))
    elif spec == "impedance3":
        gamma = (g2_re + 1j * g2_im) / 32768.0
        denominator = 1.0 - gamma
        small = np.abs(denominator) < 1e-6
        denominator[small] = 1e-6 * np.exp(1j * np.angle(denominator[small] + 1e-12))
        impedance = (1.0 + gamma) / denominator
        result = np.column_stack((freq, np.clip(impedance.real, -50, 50), np.clip(impedance.imag, -50, 50)))
    elif spec == "delta5":
        result = np.column_stack((freq, g2_re, g2_im, g1_re - g2_re, g1_im - g2_im))
    elif spec == "ratio5":
        g1 = (g1_re + 1j * g1_im) / 32768.0
        g2 = (g2_re + 1j * g2_im) / 32768.0
        denominator = g2.copy()
        small = np.abs(denominator) < 1e-5
        denominator[small] = 1e-5 * np.exp(1j * np.angle(denominator[small] + 1e-12))
        ratio = np.clip(((g1 / denominator).real), -20, 20) + 1j * np.clip(((g1 / denominator).imag), -20, 20)
        result = np.column_stack((freq, g2_re, g2_im, ratio.real, ratio.imag))
    elif spec == "smith5":
        g1 = (g1_re + 1j * g1_im) / 32768.0
        g2 = (g2_re + 1j * g2_im) / 32768.0
        d1 = 1.0 - g1
        d2 = 1.0 - g2
        d1[np.abs(d1) < 1e-6] = 1e-6
        d2[np.abs(d2) < 1e-6] = 1e-6
        z1 = np.clip(((1 + g1) / d1).real, -50, 50) + 1j * np.clip(((1 + g1) / d1).imag, -50, 50)
        z2 = np.clip(((1 + g2) / d2).real, -50, 50) + 1j * np.clip(((1 + g2) / d2).imag, -50, 50)
        delta_z = np.clip((z1-z2).real, -50, 50) + 1j * np.clip((z1-z2).imag, -50, 50)
        result = np.column_stack((freq, z2.real, z2.imag, delta_z.real, delta_z.imag))
    elif spec in {"relative_polar5", "relative_ratio5"}:
        relative_phase = g2_phase - g1_phase
        if spec == "relative_polar5":
            second_magnitude = g2_mag
        else:
            second_magnitude = np.clip(g2_mag / np.maximum(g1_mag, 1e-6), 0.0, 20.0)
        result = np.column_stack((
            freq, g1_mag, second_magnitude,
            np.sin(relative_phase), np.cos(relative_phase),
        ))
    else:
        result = np.column_stack((g2_re, g2_im))
    if not np.isfinite(result).all():
        raise ValueError(f"feature transform {spec} produced non-finite values")
    return result.astype(np.float64)


def build_context_features(
    raw: np.ndarray,
    point_groups: np.ndarray,
    sweep_groups: np.ndarray,
    spec: str,
) -> np.ndarray:
    """Create five-dimensional local-spectrum features without using labels.

    A single default-state point is shared by multiple switch-state rows. Local
    derivatives are therefore estimated once per unique point, independently
    inside each reconstructed sweep and contiguous frequency band, and then
    broadcast back to its rows. Savitzky-Golay smoothing uses only measurements
    from the same sweep available at inference time.
    """

    if spec not in {"context_slope5", "context_tangent5", "context_frenet5", "context_rotated5"}:
        raise KeyError(f"unknown context feature spec {spec!r}")
    values = np.asarray(raw, dtype=np.float64)
    point_groups = np.asarray(point_groups, dtype=np.int64)
    sweep_groups = np.asarray(sweep_groups, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"raw input must be Nx6, got {values.shape}")
    if len(values) != len(point_groups) or len(values) != len(sweep_groups):
        raise ValueError("context group arrays must match raw row count")

    unique_points, first_positions = np.unique(point_groups, return_index=True)
    point_sweeps = sweep_groups[first_positions]
    point_frequency = values[first_positions, 2]
    point_g2 = values[first_positions, 4:6]
    table_size = int(unique_points.max()) + 1
    smooth_table = np.zeros((table_size, 2), dtype=np.float64)
    slope_table = np.zeros((table_size, 2), dtype=np.float64)
    assigned = np.zeros(table_size, dtype=bool)

    for sweep in np.unique(point_sweeps):
        positions = np.flatnonzero(point_sweeps == sweep)
        order = positions[np.argsort(point_frequency[positions])]
        frequency = point_frequency[order]
        starts = np.r_[0, np.flatnonzero(np.diff(frequency) > 1.0) + 1]
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            segment = order[start:end]
            segment_frequency = point_frequency[segment]
            signal = point_g2[segment]
            count = len(segment)
            if count >= 5:
                window = min(7, count if count % 2 else count - 1)
                polynomial = min(2, window - 1)
                smooth = np.column_stack([
                    savgol_filter(signal[:, column], window, polynomial, mode="interp")
                    for column in range(2)
                ])
                slope = np.column_stack([
                    savgol_filter(signal[:, column], window, polynomial, deriv=1, delta=1.0, mode="interp")
                    for column in range(2)
                ])
            elif count >= 2:
                smooth = signal.copy()
                slope = np.column_stack([
                    np.gradient(signal[:, column], segment_frequency)
                    for column in range(2)
                ])
            else:
                smooth = signal.copy()
                slope = np.zeros_like(signal)
            ids = unique_points[segment]
            smooth_table[ids] = smooth
            slope_table[ids] = slope
            assigned[ids] = True

    if not assigned[unique_points].all():
        raise AssertionError("context construction left point groups unassigned")
    smooth = smooth_table[point_groups]
    slope = slope_table[point_groups]
    tangent_norm = np.linalg.norm(slope, axis=1, keepdims=True)
    tangent = slope / np.maximum(tangent_norm, 1e-12)
    zero_tangent = tangent_norm[:, 0] < 1e-12
    tangent[zero_tangent] = np.asarray([1.0, 0.0])
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    g1 = values[:, 0:2]
    g2 = values[:, 4:6]
    delta = g1 - g2
    delta_tangent = np.sum(delta * tangent, axis=1)
    delta_normal = np.sum(delta * normal, axis=1)
    if spec == "context_slope5":
        result = np.column_stack((values[:, 2], g2, slope))
    elif spec == "context_tangent5":
        result = np.column_stack((values[:, 2], g2, delta_tangent, delta_normal))
    elif spec == "context_frenet5":
        residual = g2 - smooth
        residual_tangent = np.sum(residual * tangent, axis=1)
        residual_normal = np.sum(residual * normal, axis=1)
        result = np.column_stack((values[:, 2], residual_tangent, residual_normal, delta_tangent, delta_normal))
    else:
        g1_tangent = np.sum(g1 * tangent, axis=1)
        g1_normal = np.sum(g1 * normal, axis=1)
        g2_tangent = np.sum(g2 * tangent, axis=1)
        g2_normal = np.sum(g2 * normal, axis=1)
        result = np.column_stack((values[:, 2], g1_tangent, g1_normal, g2_tangent, g2_normal))
    if not np.isfinite(result).all():
        raise ValueError(f"context transform {spec} produced non-finite values")
    return result.astype(np.float64)


GAMMA_SCALE = 32768.0
SWITCH_STATES = 16


@dataclass
class MobiusPhysics:
    """Per-(switch state, frequency) bilinear tuner model fitted on training rows only.

    A passive matching network maps the default-state reflection coefficient to
    the closed-loop one through a Moebius transform Γ₁ = (aΓ₂ + b) / (cΓ₂ + 1).
    Fitting the three complex coefficients for every (itState, frequency) pair
    is a linear least-squares problem, so the model needs no labels at all.
    It is used exclusively during training to label synthetic default-state
    points; the deployed network still consumes the raw five-dimensional input.
    """

    frequencies: np.ndarray
    coefficients: np.ndarray  # (SWITCH_STATES, len(frequencies), 3) complex
    fitted: np.ndarray  # (SWITCH_STATES, len(frequencies)) bool
    residual_median: float

    @classmethod
    def fit(cls, raw: np.ndarray, rows: np.ndarray, min_rows: int = 4) -> "MobiusPhysics":
        raw = np.asarray(raw, dtype=np.float64)
        rows = np.asarray(rows)
        subset = raw[rows]
        g1 = (subset[:, 0] + 1j * subset[:, 1]) / GAMMA_SCALE
        g2 = (subset[:, 4] + 1j * subset[:, 5]) / GAMMA_SCALE
        frequency = subset[:, 2].astype(np.int64)
        state = subset[:, 3].astype(np.int64)
        frequencies = np.sort(np.unique(frequency))
        coefficients = np.zeros((SWITCH_STATES, len(frequencies), 3), dtype=np.complex128)
        fitted = np.zeros((SWITCH_STATES, len(frequencies)), dtype=bool)
        residuals: list[np.ndarray] = []
        for column, value in enumerate(frequencies):
            at_frequency = frequency == value
            for s in range(SWITCH_STATES):
                mask = at_frequency & (state == s)
                if int(mask.sum()) < min_rows:
                    continue
                z, w = g2[mask], g1[mask]
                design = np.column_stack((z, np.ones_like(z), -z * w))
                solution, *_ = np.linalg.lstsq(design, w, rcond=None)
                coefficients[s, column] = solution
                fitted[s, column] = True
                a, b, c = solution
                residuals.append(np.abs((a * z + b) / (c * z + 1.0) - w))
        # A missing (state, frequency) cell borrows the nearest fitted frequency of the same state.
        for s in range(SWITCH_STATES):
            available = np.flatnonzero(fitted[s])
            if len(available) == 0:
                raise ValueError(f"switch state {s} has no rows to fit the physics model")
            for column in np.flatnonzero(~fitted[s]):
                coefficients[s, column] = coefficients[s, available[np.argmin(np.abs(available - column))]]
        median = float(np.median(np.concatenate(residuals))) if residuals else float("nan")
        return cls(frequencies, coefficients, fitted, median)

    def _columns(self, frequency: np.ndarray) -> np.ndarray:
        frequency = np.asarray(frequency, dtype=np.float64)
        columns = np.searchsorted(self.frequencies, frequency)
        columns = np.clip(columns, 0, len(self.frequencies) - 1)
        left = np.clip(columns - 1, 0, len(self.frequencies) - 1)
        use_left = np.abs(self.frequencies[left] - frequency) < np.abs(self.frequencies[columns] - frequency)
        return np.where(use_left, left, columns)

    def closed_loop(self, gamma2: np.ndarray, frequency: np.ndarray, state: np.ndarray | int) -> np.ndarray:
        """Predict the complex closed-loop reflection coefficient (unit scale)."""
        columns = self._columns(frequency)
        state = np.broadcast_to(np.asarray(state, dtype=np.int64), columns.shape)
        a, b, c = self.coefficients[state, columns].T
        return (a * gamma2 + b) / (c * gamma2 + 1.0)

    def magnitudes(self, gamma2: np.ndarray, frequency: np.ndarray, classes: int = 15) -> np.ndarray:
        """|Γ₁| predicted for every label state -> (N, classes)."""
        return np.stack(
            [np.abs(self.closed_loop(gamma2, frequency, s)) for s in range(classes)], axis=1
        )


class PhysicsTeacher:
    """Label arbitrary default-state points with the fitted Moebius model.

    ``kind="argmin"`` uses the pure physics rule (best state minimises |Γ₁|).
    ``kind="lr"`` additionally fits a multinomial logistic regression from the
    fifteen predicted magnitudes to the training labels, which absorbs the
    unknown selection criterion of the labelling procedure.  Both variants are
    fitted on training rows only and used only to generate training targets.
    """

    def __init__(self, physics: MobiusPhysics, kind: str = "lr"):
        if kind not in {"argmin", "lr"}:
            raise ValueError(f"unknown physics teacher kind {kind!r}")
        self.physics = physics
        self.kind = kind
        self.scaler: Standardizer | None = None
        self.model = None

    @classmethod
    def fit(cls, raw: np.ndarray, labels: np.ndarray, rows: np.ndarray, kind: str = "lr",
            noise: float = 0.0, shuffle: bool = False, corrupt_seed: int = 20260918) -> "PhysicsTeacher":
        physics = MobiusPhysics.fit(raw, rows)
        if noise or shuffle:                                  # 教师质量调节（学生配方不变）
            from .teacher_quality import corrupt_coefficients
            physics = MobiusPhysics(
                physics.frequencies,
                corrupt_coefficients(physics.coefficients, noise, shuffle, corrupt_seed),
                physics.fitted,
                physics.residual_median,
            )
        teacher = cls(physics, kind)
        if kind == "lr":
            from sklearn.linear_model import LogisticRegression

            raw = np.asarray(raw, dtype=np.float64)
            g2 = (raw[rows, 4] + 1j * raw[rows, 5]) / GAMMA_SCALE
            features = physics.magnitudes(g2, raw[rows, 2])
            teacher.scaler = Standardizer.fit(features)
            teacher.model = LogisticRegression(max_iter=3000, C=10.0)
            teacher.model.fit(teacher.scaler.transform(features), np.asarray(labels)[rows])
        return teacher

    def logits(self, gamma2: np.ndarray, frequency: np.ndarray) -> np.ndarray:
        magnitudes = self.physics.magnitudes(gamma2, frequency)
        if self.kind == "argmin":
            return -magnitudes
        assert self.model is not None and self.scaler is not None
        scores = self.model.decision_function(self.scaler.transform(magnitudes))
        full = np.full((len(scores), 15), -30.0)
        full[:, self.model.classes_] = scores
        return full

    def probabilities(self, gamma2: np.ndarray, frequency: np.ndarray, temperature: float) -> np.ndarray:
        logits = self.logits(gamma2, frequency) / float(temperature)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities.astype(np.float32)

    def predict_raw(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        g2 = (raw[:, 4] + 1j * raw[:, 5]) / GAMMA_SCALE
        return self.logits(g2, raw[:, 2]).argmax(axis=1)


def synthesize_physics_rows(
    raw: np.ndarray,
    rows: np.ndarray,
    teacher: PhysicsTeacher,
    seed: int,
    per_point: int = 8,
    sigma: float = 0.06,
    temperature: float = 1.0,
    mode: str = "jitter",
    sigmas: Sequence[float] | None = None,
    pair_fraction: float | None = None,
    min_distance: float = 0.0,
    oversample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create physics-consistent raw6 rows around training default-state points.

    Default-state points are perturbed in the complex Γ₂ plane (``jitter``) or
    interpolated/extrapolated between two training points of the same frequency
    (``pair``).  The closed-loop measurement of a random switch state is
    generated with the fitted Moebius map so that the synthetic row has the same
    physical structure as a real measurement, and the teacher supplies a soft
    label.  Only training rows are used, so no validation or test information
    leaks into the augmentation.

    ``sigmas`` replaces the single jitter radius by a multi-scale mixture: each
    synthetic point draws its radius uniformly from the list, so the synthetic
    cloud covers both the near field and the wide gaps that separate held-out
    sweep trajectories.  ``pair_fraction`` generalises the binary ``mode`` switch
    into the fraction of points produced by same-frequency interpolation, making
    ``jitter`` and ``pair`` two ends of one continuum.  Both default to the
    previous behaviour when left unset.

    ``min_distance`` gates the synthetic cloud by coverage: a point whose nearest
    same-frequency training Γ₂ is closer than this radius is discarded, so the
    teacher only supervises the region the measured labels do not already cover.
    The sweep-fold audit shows accuracy falling from 97% to 74% as that distance
    grows, i.e. the near field is exactly where the real labels are the better
    signal and the teacher's own 3% error rate is pure contamination.
    ``oversample`` multiplies the candidate pool so that gating does not shrink
    the surviving set.
    """
    if mode not in {"jitter", "pair"}:
        raise ValueError(f"unknown physics synthesis mode {mode!r}")
    if per_point < 1:
        raise ValueError("per_point must be positive")
    radii = np.asarray(list(sigmas) if sigmas else [sigma], dtype=np.float64)
    if radii.size == 0 or not np.all(radii >= 0.0):
        raise ValueError("physics jitter radii must be non-negative and non-empty")
    if pair_fraction is None:
        pair_fraction = 1.0 if mode == "pair" else 0.0
    if not 0.0 <= float(pair_fraction) <= 1.0:
        raise ValueError("physics pair fraction must lie in [0, 1]")
    if min_distance < 0.0:
        raise ValueError("physics minimum distance must be non-negative")
    if oversample < 1:
        raise ValueError("physics oversample must be at least 1")
    raw = np.asarray(raw, dtype=np.float64)
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    subset = raw[rows]
    frequency = subset[:, 2].astype(np.int64)
    g2 = (subset[:, 4] + 1j * subset[:, 5]) / GAMMA_SCALE
    point_key = pd.MultiIndex.from_arrays([frequency, subset[:, 4], subset[:, 5]])
    _, first_positions = np.unique(point_key.to_numpy(), return_index=True)
    points = np.sort(first_positions)
    count = len(points) * per_point * (oversample if min_distance > 0.0 else 1)
    candidate_count = count
    anchor = rng.choice(points, count)
    anchor_frequency = frequency[anchor]
    scale = radii[rng.integers(0, len(radii), count)]
    noise = scale * (rng.standard_normal(count) + 1j * rng.standard_normal(count))
    gamma2 = g2[anchor] + noise
    interpolated = np.flatnonzero(rng.random(count) < float(pair_fraction))
    if len(interpolated) > 0:
        for value in np.unique(anchor_frequency[interpolated]):
            selected = interpolated[anchor_frequency[interpolated] == value]
            candidates = points[frequency[points] == value]
            if len(candidates) == 0:
                continue
            partner = rng.choice(candidates, len(selected))
            weight = rng.uniform(-0.25, 1.25, len(selected))
            gamma2[selected] = (
                weight * g2[anchor[selected]] + (1.0 - weight) * g2[partner] + 0.3 * noise[selected]
            )
    magnitude = np.abs(gamma2)
    outside = magnitude > 0.98
    gamma2[outside] = gamma2[outside] / magnitude[outside] * 0.98
    if min_distance > 0.0:
        keep = np.zeros(len(gamma2), dtype=bool)
        for value in np.unique(anchor_frequency):
            selected = np.flatnonzero(anchor_frequency == value)
            reference = g2[frequency == value]
            if len(reference) == 0:
                keep[selected] = True
                continue
            nearest = np.abs(gamma2[selected][:, None] - reference[None, :]).min(axis=1)
            keep[selected[nearest >= min_distance]] = True
        gamma2 = gamma2[keep]
        anchor_frequency = anchor_frequency[keep]
        count = len(gamma2)
        if count == 0:
            raise ValueError("physics minimum distance discarded every synthetic point")
    soft = teacher.probabilities(gamma2, anchor_frequency, temperature)
    measured_state = rng.integers(0, SWITCH_STATES, count)
    gamma1 = teacher.physics.closed_loop(gamma2, anchor_frequency, measured_state)
    synthetic = np.column_stack((
        gamma1.real * GAMMA_SCALE, gamma1.imag * GAMMA_SCALE, anchor_frequency.astype(np.float64),
        measured_state.astype(np.float64), gamma2.real * GAMMA_SCALE, gamma2.imag * GAMMA_SCALE,
    ))
    finite = np.isfinite(synthetic).all(axis=1)
    global LAST_SYNTHESIS_AUDIT
    LAST_SYNTHESIS_AUDIT = dict(anchors=int(len(points)), candidates=int(candidate_count), after_gate=int(count), finite_rows=int(finite.sum()), acceptance_rate=float(finite.sum()/candidate_count), sigma=float(sigma), min_distance=float(min_distance))
    return synthetic[finite], soft[finite]


def split_indices(labels: np.ndarray, groups: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if groups is None:
        train, temp = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
        valid, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=labels[temp])
        return np.sort(train), np.sort(valid), np.sort(test)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    keep, test = next(outer.split(indices, labels, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed + 1009)
    train_rel, valid_rel = next(inner.split(keep, labels[keep], groups[keep]))
    train, valid = keep[train_rel], keep[valid_rel]
    if set(groups[train]) & set(groups[valid]) or set(groups[train]) & set(groups[test]) or set(groups[valid]) & set(groups[test]):
        raise AssertionError("group leakage detected")
    return np.sort(train), np.sort(valid), np.sort(test)


def sweep_fold_indices(labels: np.ndarray, groups: np.ndarray, fold: int, seed: int = 20260823) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    keep, test = folds[fold]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=seed + fold)
    train_rel, valid_rel = next(inner.split(keep, labels[keep], groups[keep]))
    train, valid = keep[train_rel], keep[valid_rel]
    if set(groups[train]) & set(groups[valid]) or set(groups[train]) & set(groups[test]) or set(groups[valid]) & set(groups[test]):
        raise AssertionError("sweep group leakage detected")
    return np.sort(train), np.sort(valid), np.sort(test)


def frequency_block_indices(raw: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = raw[:, 2].astype(np.int64)
    unique = np.sort(np.unique(frequency))
    bands: list[np.ndarray] = []
    starts = np.r_[0, np.flatnonzero(np.diff(unique) > 1) + 1]
    ends = np.r_[starts[1:], len(unique)]
    for start, end in zip(starts, ends):
        bands.append(unique[start:end])
    chunks = [np.array_split(band, 5) for band in bands]
    test_freq = np.concatenate([parts[fold] for parts in chunks])
    valid_freq = np.concatenate([parts[(fold + 1) % 5] for parts in chunks])
    test = np.flatnonzero(np.isin(frequency, test_freq))
    valid = np.flatnonzero(np.isin(frequency, valid_freq))
    train = np.flatnonzero(~np.isin(frequency, np.r_[test_freq, valid_freq]))
    return train, valid, test


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float, activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, width), activation(), nn.Dropout(dropout), nn.Linear(width, width))
        self.activation = activation()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class _RoundSTE(torch.autograd.Function):
    """Round with a straight-through gradient, the standard QAT estimator."""

    @staticmethod
    def forward(ctx, value):
        return torch.round(value)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _round_ste(value: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(value)


class WeightFakeQuant(nn.Module):
    """Symmetric per-output-channel INT8 fake quantisation for a Linear weight.

    Per-channel is not a refinement here but a requirement: on this network the
    per-tensor variant costs a further 1.7 points, because the residual block's
    second projection has a much wider row spread than the other matrices.
    """

    def __init__(self, bits: int = 8):
        super().__init__()
        self.levels = 2 ** (bits - 1) - 1

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        scale = weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / self.levels
        return torch.clamp(_round_ste(weight / scale), -self.levels, self.levels) * scale


class ActivationFakeQuant(nn.Module):
    """Asymmetric UINT8 fake quantisation with an EMA min/max observer.

    Two of the four quantised tensors are post-ReLU and therefore non-negative;
    a signed symmetric grid would throw away half of its levels on them, which
    measured 0.73 points on this model.  The observer is frozen at evaluation
    time so the exported scales are the ones used for calibration.
    """

    def __init__(self, bits: int = 8, momentum: float = 0.01):
        super().__init__()
        self.levels = 2 ** bits - 1
        self.momentum = float(momentum)
        self.register_buffer("minimum", torch.tensor(float("nan")))
        self.register_buffer("maximum", torch.tensor(float("nan")))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            low = value.detach().amin()
            high = value.detach().amax()
            if bool(torch.isnan(self.minimum)):
                self.minimum.copy_(low)
                self.maximum.copy_(high)
            else:
                self.minimum.mul_(1 - self.momentum).add_(self.momentum * low)
                self.maximum.mul_(1 - self.momentum).add_(self.momentum * high)
        if bool(torch.isnan(self.minimum)):
            return value
        low = torch.minimum(self.minimum, torch.zeros_like(self.minimum))
        scale = ((self.maximum - low) / self.levels).clamp_min(1e-12)
        zero_point = torch.round(-low / scale)
        quantised = torch.clamp(_round_ste(value / scale) + zero_point, 0, self.levels)
        return (quantised - zero_point) * scale

    def export(self) -> dict:
        low = float(min(self.minimum.item(), 0.0))
        scale = max((self.maximum.item() - low) / self.levels, 1e-12)
        return {"scale": scale, "zero_point": float(round(-low / scale)), "levels": self.levels}


def attach_fake_quant(model: "IATMLP", weight_bits: int = 8, activation_bits: int = 8) -> dict:
    """Insert INT8 fake quantisation at the operator boundaries of a residual*_hybrid net.

    Quantisation points are the tensors a real integer kernel would hand between
    operators: the network input, the stem activation, the residual block's inner
    activation and the block's output.  The pre-activation sum is deliberately NOT
    a quantisation point - it carries a 28x wider range than the tensor ReLU lets
    through, and quantising it there measured a 31 point collapse.
    """

    from torch.nn.utils import parametrize

    if not model.architecture.startswith("residual"):
        raise ValueError("fake quantisation currently targets the residual architectures")
    network = model.network
    block = network[2]
    for linear in (network[0], block.block[0], block.block[3], network[3]):
        parametrize.register_parametrization(linear, "weight", WeightFakeQuant(weight_bits))

    device = next(model.parameters()).device
    observers = nn.ModuleDict({
        name: ActivationFakeQuant(activation_bits)
        for name in ("input", "stem", "inner", "feature")
    }).to(device)
    model.activation_observers = observers

    network[0].register_forward_pre_hook(lambda m, args: (observers["input"](args[0]),))
    network[1].register_forward_hook(lambda m, args, out: observers["stem"](out))
    block.block[1].register_forward_hook(lambda m, args, out: observers["inner"](out))
    block.register_forward_hook(lambda m, args, out: observers["feature"](out))
    return {"weight_bits": weight_bits, "activation_bits": activation_bits,
            "quantisation_points": ["input", "stem", "inner", "feature"]}


class PlainBlock(ResidualBlock):
    """与 ResidualBlock 参数和构造顺序完全相同，只去掉跳连：activation(F(x))。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(value))


class CosineLinear(nn.Linear):
    def __init__(self, input_dim: int, output_dim: int, scale: float = 16.0):
        super().__init__(input_dim, output_dim, bias=False)
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(value, dim=1), F.normalize(self.weight, dim=1))


class HybridLinear(nn.Module):
    """Learnable interpolation between magnitude-aware and angular logits.

    One weight matrix is shared by both branches, so the parameter overhead is
    one scalar even though inference evaluates two projections.
    """

    def __init__(self, input_dim: int, output_dim: int, scale: float = 16.0, alpha: float = 0.5):
        super().__init__()
        self.in_features = int(input_dim)
        self.out_features = int(output_dim)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        initial = min(max(float(alpha), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(initial / (1.0 - initial))))
        self.scale = float(scale)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        linear = F.linear(value, self.weight, self.bias)
        angular = self.scale * F.linear(F.normalize(value, dim=1), F.normalize(self.weight, dim=1))
        alpha = self.alpha
        return (1.0 - alpha) * linear + alpha * angular


class PhysicsInteraction(nn.Module):
    """Parameter-free complex interaction expansion for standardized raw5 input.

    The public interface stays five-dimensional.  The network additionally sees
    bounded complex differences, inner/cross products and reflection energies.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 5:
            raise ValueError("PhysicsInteraction requires raw5 input")
        g1_re, g1_im, _, g2_re, g2_im = value.unbind(dim=-1)
        interactions = torch.stack(
            (
                g1_re - g2_re,
                g1_im - g2_im,
                g1_re * g2_re + g1_im * g2_im,
                g1_im * g2_re - g1_re * g2_im,
                g1_re.square() + g1_im.square(),
                g2_re.square() + g2_im.square(),
            ),
            dim=-1,
        )
        return torch.cat((value, torch.tanh(interactions)), dim=-1)


class IATMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        architecture: str,
        classes: int = 15,
        dropout: float = 0.10,
        head_scale: float = 16.0,
        hybrid_alpha: float = 0.5,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise KeyError(architecture)
        hidden = ARCHITECTURES[architecture]
        self.input_dim = input_dim
        self.architecture = architecture
        if architecture == "physics_residual48":
            if input_dim != 5:
                raise ValueError("physics_residual48 requires five-dimensional raw5 input")
            width = hidden[0]
            self.network = nn.Sequential(
                PhysicsInteraction(), nn.Linear(11, width), nn.ReLU(),
                ResidualBlock(width, dropout), nn.Linear(width, classes),
            )
        elif architecture.startswith("residual"):
            width = hidden[0]
            if "_silu_relu" in architecture:
                stem_activation = nn.SiLU
                residual_activation = nn.ReLU
            elif "_gelu_relu" in architecture:
                stem_activation = nn.GELU
                residual_activation = nn.ReLU
            elif "_mish_silu" in architecture:
                stem_activation = nn.Mish
                residual_activation = nn.SiLU
            elif "_mish_relu" in architecture:
                stem_activation: type[nn.Module] = nn.Mish
                residual_activation: type[nn.Module] = nn.ReLU
            elif architecture.endswith("_relu_silu"):
                stem_activation = nn.ReLU
                residual_activation = nn.SiLU
            elif architecture.endswith("_silu"):
                stem_activation = residual_activation = nn.SiLU
            elif architecture.endswith("_mish"):
                stem_activation = residual_activation = nn.Mish
            else:
                stem_activation = residual_activation = nn.ReLU
            if architecture.endswith("_cosine"):
                classifier: nn.Module = CosineLinear(width, classes, scale=head_scale)
            elif architecture.endswith("_hybrid"):
                classifier = HybridLinear(width, classes, scale=head_scale, alpha=hybrid_alpha)
            else:
                classifier = nn.Linear(width, classes)
            self.network = nn.Sequential(
                nn.Linear(input_dim, width), stem_activation(),
                (PlainBlock if "_noskip" in architecture else ResidualBlock)(
                    width, dropout, residual_activation), classifier,
            )
        elif architecture == "original64_32":
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes),
            )
        else:
            # 先剥掉分类头后缀，否则 mlp64_32_gelu_hybrid 的 _gelu 认不出来
            head_kind, stem_name = "", architecture
            for suffix in ("_hybrid", "_cosine"):
                if stem_name.endswith(suffix):
                    head_kind, stem_name = suffix, stem_name[: -len(suffix)]
                    break
            dims = (input_dim, *hidden, classes)
            layers: list[nn.Module] = []
            if stem_name.endswith("_silu"):
                activation: type[nn.Module] = nn.SiLU
            elif stem_name.endswith("_gelu"):
                activation = nn.GELU
            elif stem_name.endswith("_leaky"):
                activation = nn.LeakyReLU
            else:
                activation = nn.ReLU
            use_batch_norm = stem_name.endswith("_bn")
            for layer_index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                if layer_index == len(hidden) and head_kind == "_hybrid":
                    layers.append(HybridLinear(left, right, scale=head_scale, alpha=hybrid_alpha))
                elif layer_index == len(hidden) and head_kind == "_cosine":
                    layers.append(CosineLinear(left, right, scale=head_scale))
                else:
                    layers.append(nn.Linear(left, right))
                if layer_index < len(hidden):
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(right))
                    layers.append(activation())
                    if dropout > 0 and layer_index == 0:
                        layers.append(nn.Dropout(dropout))
            self.network = nn.Sequential(*layers)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        return self.network[:-1](value)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.network[-1](features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(value))


def complexity(model: nn.Module) -> tuple[int, int, int]:
    params = sum(parameter.numel() for parameter in model.parameters())
    macs = 0
    for module in model.modules():
        if isinstance(module, HybridLinear):
            macs += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features
    return params, macs, macs * 2


def serialized_size(model: nn.Module) -> int:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return len(stream.getvalue())


@torch.inference_mode()
def predict_logits(model: nn.Module, values: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    device = next(model.parameters()).device
    return np.concatenate([
        model(tensor[start:start+batch_size].to(device, non_blocking=True)).cpu().numpy()
        for start in range(0, len(tensor), batch_size)
    ])


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    matrix = confusion_matrix(truth, prediction, labels=np.arange(15))
    per_class = recall_score(truth, prediction, labels=np.arange(15), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "minimum_class_recall": float(per_class.min()),
        "per_class_recall": per_class.tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def benchmark_latency(model: nn.Module, input_dim: int, loops: int = 3000) -> float:
    model.eval()
    sample = torch.zeros(1, input_dim)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            for _ in range(100):
                model(sample)
            begin = time.perf_counter_ns()
            for _ in range(loops):
                model(sample)
        return (time.perf_counter_ns() - begin) / loops / 1000.0
    finally:
        torch.set_num_threads(previous_threads)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protocol_groups(bundle: DatasetBundle, protocol: str) -> np.ndarray | None:
    if protocol == "random":
        return None
    if protocol == "point_group":
        return bundle.point_groups
    if protocol == "sweep_group":
        return bundle.sweep_groups
    raise KeyError(protocol)
