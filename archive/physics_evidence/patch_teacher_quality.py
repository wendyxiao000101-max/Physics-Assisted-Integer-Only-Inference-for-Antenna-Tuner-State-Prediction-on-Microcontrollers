# -*- coding: utf-8 -*-
"""给 iat_fp32_optimized 加上"教师质量调节"三个开关（幂等，可重复执行）。

    python scripts\\patch_teacher_quality.py --project <iat_fp32_optimized 绝对路径>

做三件事：
  1. 把 teacher_quality.py 复制进 iatopt/ —— 与离线评估脚本共用同一段随机数代码；
  2. core.py::PhysicsTeacher.fit 增加 noise / shuffle / corrupt_seed 三个参数，
     **在拟合逻辑回归头之前**劣化 Möbius 系数，使教师的分类头总是对自己拿到的
     那个（可能已被劣化的）物理模型做到最好 —— 这样调节的是"物理模型有多准"，
     而不是"分类头有没有收敛"；
  3. train.py 增加 --physics-teacher-noise / --physics-teacher-shuffle /
     --physics-teacher-seed，并把三个值写进 raw_runs.csv 的记录里。

学生侧一行没动：网络结构、优化器、轮数、增广、损失全部不变。

每个被改的文件都会先存一份 .bak.teacherq。改完会自行验证一遍并打印结果。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "teacher_quality.py"

CORE_OLD = '''    @classmethod
    def fit(cls, raw: np.ndarray, labels: np.ndarray, rows: np.ndarray, kind: str = "lr") -> "PhysicsTeacher":
        physics = MobiusPhysics.fit(raw, rows)
        teacher = cls(physics, kind)
'''

CORE_NEW = '''    @classmethod
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
'''

ARGS_OLD = '''    parser.add_argument("--physics-mode", choices=["jitter", "pair"], default="jitter")
'''

ARGS_NEW = '''    parser.add_argument("--physics-mode", choices=["jitter", "pair"], default="jitter")
    parser.add_argument("--physics-teacher-noise", type=float, default=0.0,
                        help="教师质量调节：Möbius 系数乘性复高斯噪声强度 eta")
    parser.add_argument("--physics-teacher-shuffle", action="store_true",
                        help="教师质量调节：同一开关态内随机重排频率列（物理形状对、内容错的对照）")
    parser.add_argument("--physics-teacher-seed", type=int, default=20260918,
                        help="教师劣化的随机种子，与训练种子无关")
'''

CALL_OLD = '''    teacher = PhysicsTeacher.fit(raw, labels, train_idx, kind=args.physics_teacher)
'''

CALL_NEW = '''    teacher = PhysicsTeacher.fit(
        raw, labels, train_idx, kind=args.physics_teacher,
        noise=getattr(args, "physics_teacher_noise", 0.0),
        shuffle=getattr(args, "physics_teacher_shuffle", False),
        corrupt_seed=getattr(args, "physics_teacher_seed", 20260918),
    )
'''

INFO_OLD = '''    info = {
        "physics_teacher": args.physics_teacher,
        "physics_synthetic_rows": int(len(soft)),
'''

INFO_NEW = '''    info = {
        "physics_teacher": args.physics_teacher,
        "physics_teacher_noise": float(getattr(args, "physics_teacher_noise", 0.0)),
        "physics_teacher_shuffle": bool(getattr(args, "physics_teacher_shuffle", False)),
        "physics_teacher_seed": int(getattr(args, "physics_teacher_seed", 20260918)),
        "physics_synthetic_rows": int(len(soft)),
'''


def apply(path: Path, pairs: list[tuple[str, str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    todo = [(old, new, name) for old, new, name in pairs if new not in text]
    if not todo:
        print(f"  {path.name}: 已经打过补丁，跳过")
        return
    backup = path.with_suffix(path.suffix + ".bak.teacherq")
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
                    help="iat_fp32_optimized 目录的**绝对路径**（里面有 train.py 和 iatopt/）")
    args = ap.parse_args()

    project = args.project.resolve()
    core, train = project / "iatopt" / "core.py", project / "train.py"
    for path in (core, train, HELPER):
        if not path.exists():
            sys.exit(f"找不到 {path}")

    print(f"项目：{project}")
    target = project / "iatopt" / "teacher_quality.py"
    shutil.copy2(HELPER, target)
    print(f"  已复制 {target}")
    apply(core, [(CORE_OLD, CORE_NEW, "PhysicsTeacher.fit 劣化钩子")])
    apply(train, [(ARGS_OLD, ARGS_NEW, "三个命令行开关"),
                  (CALL_OLD, CALL_NEW, "教师构造处传参"),
                  (INFO_OLD, INFO_NEW, "raw_runs.csv 记录字段")])

    print("\n验证：")
    sys.path.insert(0, str(project))
    import importlib
    tq = importlib.import_module("iatopt.teacher_quality")
    print(f"  iatopt.teacher_quality -> {tq.__file__}")
    text = train.read_text(encoding="utf-8")
    for flag in ("--physics-teacher-noise", "--physics-teacher-shuffle", "--physics-teacher-seed"):
        print(f"  train.py 含 {flag}: {flag in text}")
    print("\n补丁完成。回滚：把 *.bak.teacherq 覆盖回原文件即可。")


if __name__ == "__main__":
    main()
