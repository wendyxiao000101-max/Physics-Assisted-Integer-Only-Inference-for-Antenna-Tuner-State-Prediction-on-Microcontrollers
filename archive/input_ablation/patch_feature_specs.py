# -*- coding: utf-8 -*-
"""给 iatopt/core.py 增加两个输入消融必需的对照臂（幂等，可重复执行）。

    g1_only3  = (freq, g1_re, g1_im)          只有当前态反射系数，没有默认态
    g1_state4 = (freq, g1_re, g1_im, state)   当前态反射系数 + 开关态索引

这两个臂是证明「默认态反射系数 Γ₂ 是最佳判别依据」的关键对照：
没有它们，消融只能说明"加维度没用"，说明不了"去掉 Γ₂ 会塌"。

用法：
    python patch_feature_specs.py <PROJECT_ROOT>\\iat_fp32_optimized\\iatopt\\core.py

会先写一个 .bak 备份，再原地插入。已经打过补丁则什么都不做。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SPEC_ANCHOR = '    "gamma2": ("g2_re", "g2_im"),\n'
SPEC_ADDITION = (
    '    "g1_only3": ("freq", "g1_re", "g1_im"),\n'
    '    "g1_state4": ("freq", "g1_re", "g1_im", "state"),\n'
)

BUILD_ANCHOR = (
    '    elif spec == "gamma3":\n'
    '        result = np.column_stack((freq, g2_re, g2_im))\n'
)
BUILD_ADDITION = (
    '    elif spec == "g1_only3":\n'
    '        result = np.column_stack((freq, g1_re, g1_im))\n'
    '    elif spec == "g1_state4":\n'
    '        result = np.column_stack((freq, g1_re, g1_im, state))\n'
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    if '"g1_only3"' in text and 'spec == "g1_only3"' in text:
        print(f"{path} 已包含 g1_only3 / g1_state4，无需修改。")
        return

    for anchor, name in ((SPEC_ANCHOR, "FEATURE_SPECS 里的 gamma2 条目"),
                         (BUILD_ANCHOR, "build_features 里的 gamma3 分支")):
        if text.count(anchor) != 1:
            raise SystemExit(f"找不到唯一的锚点（{name}），出现 {text.count(anchor)} 次。"
                             f"请手工添加，内容见本脚本顶部。")

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")

    text = text.replace(SPEC_ANCHOR, SPEC_ANCHOR + SPEC_ADDITION, 1)
    text = text.replace(BUILD_ANCHOR, BUILD_ANCHOR + BUILD_ADDITION, 1)
    path.write_text(text, encoding="utf-8")
    print(f"已给 {path} 增加 g1_only3 / g1_state4 两个特征谱。")
    print("自检：python -c \"from iatopt.core import FEATURE_SPECS; "
          "print(FEATURE_SPECS['g1_only3'], FEATURE_SPECS['g1_state4'])\"")


if __name__ == "__main__":
    main()
