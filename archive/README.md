# Archived experiment scripts

These are the scripts that produced the remaining numbers of the paper, copied without
functional changes. They are kept for provenance and are not wired to this repository's layout;
absolute paths of the authors' machine were replaced by `<PROJECT_ROOT>` or by the
`IAT_PROJECT_ROOT` environment variable. Their outputs are in `../results/`, and
`../docs/TABLE_MAP.md` lists which script belongs to which table.

They were run from a working tree with this layout:

```
<PROJECT_ROOT>/
  iat_fp32_optimized/          training code (now ../src) plus C5/scripts (now baselines_architecture/)
  esp32p4_iat_demo/input/data/ x_train.csv, y_train.csv (now ../data)
  output/<experiment>/         one folder per experiment, each holding its own script and results
```

Most scripts locate inputs relative to their own file (`Path(__file__).resolve().parents[...]`),
so re-running one requires either recreating that layout or editing the few path constants at
the top of the script. The `.bat` files were run inside `iat_fp32_optimized/` and call
`train.py` / `qat_fold_pipeline.py`, which are identical to `../src/`.

| Folder | Original location | Purpose |
|---|---|---|
| `main_fp32_int8/` | `iat_fp32_optimized/` | Primary FP32 folds, QAT, integer export, A/B/C head variants (Tables V, XII) |
| `input_ablation/` | `iat_fp32_optimized/` | Input ablation (Table II) |
| `physics_evidence/` | `output/C1.../scripts`, `iat_fp32_optimized/` | Held-out surrogate error and teacher-quality arms (Fig. 4) |
| `p95_grouping/` | `output/C7...`, `output/C6...` | Alternative P95 grouping, matched physics-off control, grouping-rule probe (Section VI-C) |
| `sensitivity_original/` | `output/C13...` | Original runner of Table VI; superseded by `../experiments/` |
| `baselines_architecture/` | `iat_fp32_optimized/C5/scripts` | Tree baselines, storage encodings, architecture and head controls (Tables VIII, IX; Fig. 6) |
| `rf_analysis/` | `output/...` | Conditional RF analysis (Table XI; Figs. 7, 8) |

The `patch_*.py` scripts apply small, reversible edits to `iatopt/core.py` for one control
experiment each (for example a plain-MLP head or a no-skip block); `../src/` is the unpatched code.
