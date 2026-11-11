# -*- coding: utf-8 -*-
"""加一个**只去掉跳连**的对照架构 `residual32_noskip_gelu_relu_hybrid`（幂等，带备份与自检）。

    python C5\\scripts\\patch_noskip.py --project "<iat_fp32_optimized 绝对路径>"

回应核查意见 §三.1：此前的 "residual32 vs mlp64_32" 同时改变了宽度（32 vs 64/32）、
线性层数（4 vs 3）与跳连，+0.874 pp 不能归因于跳连本身。本补丁构造的对照与
`residual32_gelu_relu_hybrid` **逐层完全相同**：

    Linear(5,32) → GELU → [Linear(32,32) → ReLU → Dropout → Linear(32,32)] → ReLU → HybridLinear(32,15)

唯一区别是残差块输出由  ReLU(x + F(x))  改为  ReLU(F(x))。
参数量相同（2,800），且模块构造顺序相同 → 同一随机种子下**初始权重逐元素相同**，
这比"参数量接近"的对照更干净。补丁会实际构造两个模型并断言这一点。

改动三处，均为新增，不改原有逻辑：
  1. 在 CosineLinear 之前新增 `PlainBlock(ResidualBlock)`，只覆写 forward；
  2. residual 分支按名字里是否含 `_noskip` 选择 PlainBlock / ResidualBlock；
  3. ARCHITECTURES 注册新名字。
备份为 core.py.bak.noskip。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BLOCK_OLD = '''class CosineLinear(nn.Linear):'''
BLOCK_NEW = '''class PlainBlock(ResidualBlock):
    """与 ResidualBlock 参数和构造顺序完全相同，只去掉跳连：activation(F(x))。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(value))


class CosineLinear(nn.Linear):'''

USE_OLD = '''                ResidualBlock(width, dropout, residual_activation), classifier,'''
USE_NEW = '''                (PlainBlock if "_noskip" in architecture else ResidualBlock)(
                    width, dropout, residual_activation), classifier,'''

ARCH_OLD = '''    "residual32_gelu_relu_hybrid": (32, 32),'''
ARCH_NEW = '''    "residual32_gelu_relu_hybrid": (32, 32),
    "residual32_noskip_gelu_relu_hybrid": (32, 32),'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    core = project / "iatopt" / "core.py"
    if not core.exists():
        print(f"找不到 {core}", file=sys.stderr)
        return 2
    text = core.read_text(encoding="utf-8")
    pending = [(o, n, name) for o, n, name in ((BLOCK_OLD, BLOCK_NEW, "PlainBlock"),
                                              (USE_OLD, USE_NEW, "residual 分支选择块"),
                                              (ARCH_OLD, ARCH_NEW, "注册架构名")) if n not in text]
    if pending:
        backup = core.with_suffix(core.suffix + ".bak.noskip")
        if not backup.exists():
            shutil.copy2(core, backup)
            print(f"已备份到 {backup.name}")
        for old, new, name in pending:
            if text.count(old) != 1:
                print(f"锚点 [{name}] 出现 {text.count(old)} 次（应为 1），已中止，未改动文件。", file=sys.stderr)
                return 1
            text = text.replace(old, new)
            print(f"已插入 [{name}]")
        core.write_text(text, encoding="utf-8")
    else:
        print("已经打过补丁，跳过")

    sys.path.insert(0, str(project))
    import importlib
    import torch
    module = importlib.import_module("iatopt.core")
    importlib.reload(module)
    assert "residual32_noskip_gelu_relu_hybrid" in module.ARCHITECTURES, "架构名未注册"
    torch.manual_seed(20260923)
    base = module.IATMLP(5, "residual32_gelu_relu_hybrid", classes=15, dropout=0.02, head_scale=20.0, hybrid_alpha=0.5)
    torch.manual_seed(20260923)
    plain = module.IATMLP(5, "residual32_noskip_gelu_relu_hybrid", classes=15, dropout=0.02, head_scale=20.0, hybrid_alpha=0.5)
    assert type(plain.network[2]).__name__ == "PlainBlock", "noskip 没有用上 PlainBlock"
    assert type(base.network[2]).__name__ == "ResidualBlock", "原架构被意外改动"
    same_init = all(torch.equal(a, b) for a, b in zip(base.state_dict().values(), plain.state_dict().values()))
    assert same_init, "同种子下初始权重不一致"
    x = torch.randn(64, 5)
    base.eval(); plain.eval()
    with torch.no_grad():
        stem = base.network[1](base.network[0](x))
        expected = base.network[2].activation(base.network[2].block(stem))
        got = plain.network[2](stem)
    assert torch.allclose(expected, got), "PlainBlock 输出不等于 ReLU(F(x))"
    params = sum(p.numel() for p in plain.parameters())
    print(f"自检通过：PlainBlock 生效、原架构未变、同种子初始权重逐元素相同、参数 {params:,}（应为 2,800）")
    return 0 if params == 2800 else 1


if __name__ == "__main__":
    sys.exit(main())
