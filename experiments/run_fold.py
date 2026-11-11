"""Single-fold FP32 training entry point.

Wraps the frozen training code in ``src/`` so that every run uses the archived
grouped split indices in ``data/splits/<name>/fold<k>.npz`` instead of drawing a
new split.  Called by ``run_all.py``; all command-line options are forwarded to
``src/train.py``.
"""
from pathlib import Path
import json
import os
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import train  # noqa: E402
from iatopt import core  # noqa: E402

SPLITS = Path(os.environ.get("IAT_SPLIT_DIR", REPO / "data/splits/original"))


def choose_split(bundle, protocol, run, seed):
    assert protocol == "sweep_group" and run == seed - 42
    z = np.load(SPLITS / f"fold{run}.npz")
    if SPLITS.name == "original":  # the archived folds must keep the 210 reconstructed groups disjoint
        groups = [set(bundle.sweep_groups[z[k]]) for k in ("train", "valid", "test")]
        assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    return z["train"], z["valid"], z["test"]


train.choose_split = choose_split
_build = train.build_physics_augmentation


def traced(*args, **kwargs):
    """Record how many synthetic rows survived the distance gate (logging only)."""
    result = _build(*args, **kwargs)
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    audit = getattr(core, "LAST_SYNTHESIS_AUDIT", None) if result is not None else None
    (out / "synthesis_audit.json").write_text(json.dumps(audit or {"physics": "none"}, indent=2), encoding="utf8")
    return result


train.build_physics_augmentation = traced
train.main()
