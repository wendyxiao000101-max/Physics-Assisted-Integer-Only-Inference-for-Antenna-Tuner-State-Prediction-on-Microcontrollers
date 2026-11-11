# -*- coding: utf-8 -*-
"""再加一个架构名：交付骨干 + **纯余弦头**（幂等，可重复执行）。

    python C5\\scripts\\patch_cosine_head.py --project <iat_fp32_optimized 绝对路径>

**必须先跑 `patch_mlp_head.py`**，本脚本的锚点是那个补丁插入的行。

为什么要这一档：`HybridLinear` 的可学习插值权重 α 在 raw5 的 60 次运行里 **100 %
收敛到 > 0.99**（中位 0.999964，alpha_logit 中位 +10.24，且是顶着 AdamW 的
weight_decay 跑上去的）。α → 1 意味着：

  * 线性分支 `Wx + b` 的贡献只剩余弦分支的 ~3.6e-5；
  * **偏置只挂在线性分支上**（余弦分支 `scale · cos(x, W)` 没有偏置），
    所以头部那 15 个偏置实际上是死参数——这正解释了折叠时 2,800 → 2,784
    少掉的恰好是 15 个头部偏置。

`CosineLinear` 把 α 固定为 1，**不带 alpha_logit、不带偏置**，比 `HybridLinear`
少 16 个参数，推理也只算一个分支而不是两个（MACs 减半：3,168 → 2,688）。
如果它和 hybrid 打平，那么可学习的 α 就是可以省掉的设计冗余，INT8 模型还能再瘦一点。

`core.py` 的 residual 分支本来就认 `_cosine` 后缀（`endswith("_cosine")` →
`CosineLinear`），只是 ARCHITECTURES 里没有 `residual32_gelu_relu_cosine`
这个名字，所以命令行传不进去。补丁只加这一个名字，不动任何逻辑。

备份为 `core.py.bak.cosine`。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = '''    "mlp64_32_gelu_cosine": (64, 32),
'''

REPLACEMENT = '''    "mlp64_32_gelu_cosine": (64, 32),
    "residual32_gelu_relu_cosine": (32, 32),
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True,
                    help="iat_fp32_optimized 目录的**绝对路径**")
    args = ap.parse_args()
    project = args.project.resolve()
    core = project / "iatopt" / "core.py"
    if not core.exists():
        sys.exit(f"找不到 {core}")

    text = core.read_text(encoding="utf-8")
    if REPLACEMENT in text:
        print("已经打过补丁，跳过")
    else:
        count = text.count(ANCHOR)
        if count != 1:
            sys.exit("错误：找不到锚点（出现 %d 次，应为 1 次）。\n"
                     "多半是还没跑 patch_mlp_head.py——请先执行：\n"
                     "  python C5\\scripts\\patch_mlp_head.py --project \"%s\"" % (count, project))
        backup = core.with_suffix(core.suffix + ".bak.cosine")
        if not backup.exists():
            shutil.copy2(core, backup)
            print(f"已备份到 {backup.name}")
        core.write_text(text.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
        print("已插入 residual32_gelu_relu_cosine")

    print("\n验证：")
    sys.path.insert(0, str(project))
    import importlib
    module = importlib.import_module("iatopt.core")
    importlib.reload(module)
    print(f"  iatopt.core -> {module.__file__}")
    for name in ("residual32_gelu_relu_hybrid", "residual32_gelu_relu_cosine",
                 "mlp64_32_gelu_hybrid", "mlp64_32_gelu_cosine"):
        present = name in module.ARCHITECTURES
        print(f"  ARCHITECTURES 含 {name:30s}: {present}")
        if not present:
            continue
        model = module.IATMLP(5, name, classes=15, dropout=0.02,
                              head_scale=20.0, hybrid_alpha=0.5)
        params, macs, _ = module.complexity(model)
        print(f"      末层 {type(model.network[-1]).__name__:13s} 参数 {params:6,d}  MACs {macs:6,d}")
    print("\n补丁完成。回滚：把 core.py.bak.cosine 覆盖回 core.py。")


if __name__ == "__main__":
    main()
