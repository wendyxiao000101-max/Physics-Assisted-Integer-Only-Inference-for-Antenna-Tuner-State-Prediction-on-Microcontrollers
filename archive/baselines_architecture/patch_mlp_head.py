# -*- coding: utf-8 -*-
"""让无残差 MLP 也能用与残差网络相同的分类头（幂等，可重复执行）。

    python scripts\\patch_mlp_head.py --project <iat_fp32_optimized 绝对路径>

**为什么需要这个补丁。** `HybridLinear.forward` 是

    (1 − α) · Linear(x)  +  α · scale · cos(x, W)

交付模型学到的 α 收敛到 **0.99989**，也就是说它实际上是一个 **scale = 20 的余弦
（角度）分类头**，普通线性分支几乎不起作用。余弦头对长尾类的召回有众所周知的
巨大帮助——实测残差网络最小类召回 68.6 %，而换成普通 `nn.Linear` 头的 MLP 是 0.0 %。

所以"残差 vs 无残差"的对比里混进了"余弦头 vs 线性头"，而且后者很可能是主因。
`core.py` 原来只在 `residual*` 分支里认 `_hybrid` / `_cosine` 后缀，MLP 分支
写死了 `nn.Linear`，所以这个对照跑不出来。补丁做两件事：

  1. ARCHITECTURES 里加 `mlp32_16_gelu_hybrid` / `mlp64_32_gelu_hybrid` /
     `mlp128_64_gelu_hybrid` 三个名字；
  2. MLP 分支先剥掉 `_hybrid` / `_cosine` 后缀再判断激活函数（否则
     `mlp64_32_gelu_hybrid` 的 `_gelu` 认不出来），并把最后一层换成对应的头。

打完补丁就能跑出干净的分解：

    residual32_gelu_relu_hybrid  vs  mlp64_32_gelu_hybrid   -> 只差一条跳连
    mlp64_32_gelu_hybrid         vs  mlp64_32_gelu          -> 只差一个分类头

被改的文件先备份成 `*.bak.mlphead`，回滚就是覆盖回去。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ARCH_OLD = '''    "mlp128_64_silu": (128, 64),
'''

ARCH_NEW = '''    "mlp128_64_silu": (128, 64),
    "mlp32_16_gelu_hybrid": (32, 16),
    "mlp64_32_gelu_hybrid": (64, 32),
    "mlp128_64_gelu_hybrid": (128, 64),
    "mlp64_32_gelu_cosine": (64, 32),
'''

BODY_OLD = '''        else:
            dims = (input_dim, *hidden, classes)
            layers: list[nn.Module] = []
            if architecture.endswith("_silu"):
                activation: type[nn.Module] = nn.SiLU
            elif architecture.endswith("_gelu"):
                activation = nn.GELU
            elif architecture.endswith("_leaky"):
                activation = nn.LeakyReLU
            else:
                activation = nn.ReLU
            use_batch_norm = architecture.endswith("_bn")
            for layer_index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                layers.append(nn.Linear(left, right))
'''

BODY_NEW = '''        else:
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
'''


def apply(path: Path, pairs: list[tuple[str, str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    todo = [(old, new, name) for old, new, name in pairs if new not in text]
    if not todo:
        print(f"  {path.name}: 已经打过补丁，跳过")
        return
    backup = path.with_suffix(path.suffix + ".bak.mlphead")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  {path.name}: 已备份到 {backup.name}")
    for old, new, name in todo:
        count = text.count(old)
        if count != 1:
            sys.exit(f"错误：{path} 中锚点 [{name}] 出现 {count} 次（应为 1 次），已中止，未改动文件。")
        text = text.replace(old, new)
        print(f"  {path.name}: 已插入 [{name}]")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True,
                    help="iat_fp32_optimized 目录的**绝对路径**")
    args = ap.parse_args()
    project = args.project.resolve()
    core = project / "iatopt" / "core.py"
    if not core.exists():
        sys.exit(f"找不到 {core}")

    print(f"项目：{project}")
    apply(core, [(ARCH_OLD, ARCH_NEW, "四个新架构名"),
                 (BODY_OLD, BODY_NEW, "MLP 分支的分类头支持")])

    print("\n验证：")
    sys.path.insert(0, str(project))
    import importlib
    core_module = importlib.import_module("iatopt.core")
    importlib.reload(core_module)
    print(f"  iatopt.core -> {core_module.__file__}")
    for name in ("mlp64_32_gelu_hybrid", "mlp128_64_gelu_hybrid", "mlp32_16_gelu_hybrid"):
        print(f"  ARCHITECTURES 含 {name}: {name in core_module.ARCHITECTURES}")
    # 真正构一个模型出来，确认最后一层是 HybridLinear 且参数量对得上
    for name in ("residual32_gelu_relu_hybrid", "mlp64_32_gelu_hybrid", "mlp64_32_gelu"):
        model = core_module.IATMLP(5, name, classes=15, dropout=0.02,
                                   head_scale=20.0, hybrid_alpha=0.5)
        params, macs, _ = core_module.complexity(model)
        print(f"  {name:30s} 末层 {type(model.network[-1]).__name__:13s} "
              f"参数 {params:6,d}  MACs {macs:6,d}")
    print("\n补丁完成。回滚：把 core.py.bak.mlphead 覆盖回 core.py。")


if __name__ == "__main__":
    main()
