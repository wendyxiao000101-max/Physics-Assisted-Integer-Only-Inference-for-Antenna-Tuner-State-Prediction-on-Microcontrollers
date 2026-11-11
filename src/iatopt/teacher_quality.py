# -*- coding: utf-8 -*-
"""教师质量调节器：在**不改变学生训练配方**的前提下，可控地劣化物理教师。

本文件不依赖 torch，被两处共用，保证两边用的是**同一段随机数代码**：
  * `iatopt/teacher_quality.py`（由 patch_teacher_quality.py 复制过去，供 train.py 调用）
  * `teacher_same_protocol.py`（离线评估教师本身的准确率）

只提供一个函数 `corrupt_coefficients`，作用在 Möbius 系数 (a, b, c) 上：

  shuffle=True  把同一开关态下的 72 个频率列随机重排 —— 系数的**分布**完全保留，
                但每个频点拿到的是别的频点的传输参数。这是"形状像物理、内容是错的"
                的零假设：如果学生在这一档仍然获益，那收益来自数据增广的正则化，
                而不是物理。

  noise=η>0     每个系数乘以 (1 + η·g)，g 为标准复高斯（实虚部各 N(0, 1/2)）。
                随后把 c 按 |c| ≤ 0.95 收缩，保证 Möbius 极点 Γ₂ = −1/c 始终落在
                单位圆外，劣化后的映射仍是无源网络能实现的形状，不会出现发散。

两者可叠加；先重排后加噪。随机数只由 `seed` 决定，与训练随机种子无关，因此
同一档位在五折里拿到的是**同一个**被劣化的物理模型，折间差异只来自划分本身。
"""
from __future__ import annotations

import numpy as np

POLE_MARGIN = 0.95   # |c| 的上限：极点 −1/c 至少落在 |Γ₂| = 1/0.95 ≈ 1.053 之外


def corrupt_coefficients(coefficients: np.ndarray,
                         noise: float = 0.0,
                         shuffle: bool = False,
                         seed: int = 0) -> np.ndarray:
    """(states, freqs, 3) 复系数 -> 同形状的劣化副本。noise=0 且 shuffle=False 时原样返回。"""
    coefficients = np.array(coefficients, dtype=np.complex128, copy=True)
    if not shuffle and not noise:
        return coefficients
    if noise < 0:
        raise ValueError("teacher noise must be non-negative")
    rng = np.random.default_rng(int(seed))
    if shuffle:
        for state in range(coefficients.shape[0]):
            coefficients[state] = coefficients[state][rng.permutation(coefficients.shape[1])]
    if noise:
        gaussian = (rng.standard_normal(coefficients.shape)
                    + 1j * rng.standard_normal(coefficients.shape)) / np.sqrt(2.0)
        coefficients = coefficients * (1.0 + float(noise) * gaussian)
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = np.abs(coefficients[:, :, 2])
    excess = magnitude > POLE_MARGIN
    if excess.any():                       # 把极点推回单位圆外，保持"无源网络"形状
        scale = np.ones_like(magnitude)
        scale[excess] = POLE_MARGIN / magnitude[excess]
        coefficients[:, :, 2] = coefficients[:, :, 2] * scale
    return coefficients


def describe(noise: float = 0.0, shuffle: bool = False, seed: int = 0) -> str:
    if shuffle and noise:
        return f"shuffle+eta{noise:g}(seed{seed})"
    if shuffle:
        return f"shuffle(seed{seed})"
    if noise:
        return f"eta{noise:g}(seed{seed})"
    return "clean"
