# Physics-Assisted Integer-Only Inference for Antenna Tuner-State Prediction on Microcontrollers

Code, data splits and result tables for the paper

> L. Xiao, X. Yang, Y. Shi, J. Wang, and X. Chen, "Physics-Assisted Integer-Only Inference for
> Antenna Tuner-State Prediction on Microcontrollers," submitted to *IEEE Access*, 2026.

The repository contains the training code for the physics-assisted residual classifier, the
quantization-aware-training (QAT) and integer-export pipeline, the archived grouped
cross-validation splits, the exported FP32 and INT8-A models with a bit-exact PC reference
runtime, ESP32-S3 firmware, and the result tables behind every table and figure of the paper.

## Repository layout

| Path | Content |
|---|---|
| `data/` | The 96,768-record dataset (`x_train.csv`, `y_train.csv`) and the split indices used in the paper (`splits/original`, `splits/p95`). See `data/README.md`. |
| `src/` | Frozen training code: `train.py` (FP32 training, surrogate, teacher, losses), `qat_fold_pipeline.py` and `qat_A.py` (classifier folding, cross-layer equalization, QAT, integer export), `iatopt/` (library). |
| `experiments/` | Runner for the matched five-fold FP32 experiments (Tables VI and VII) and the physical-teacher baselines. Paths are relative to the repository. |
| `deploy/pc_reference/` | Exported full-data FP32 and INT8-A models (`model_int8.bin`, 4,656 bytes) and the PC reference runtimes. |
| `deploy/esp32s3_baseline_160mhz/` | ESP-IDF firmware for FP32 and INT8-A replay on the ESP32-S3 (baseline build; see the note below). |
| `results/` | Archived result tables, one folder per paper table or figure. |
| `archive/` | Scripts of the remaining experiments exactly as they were run, kept for provenance. See `archive/README.md`. |
| `docs/TABLE_MAP.md` | Which script and which result file produce each table and figure. |

## Environment

Experiments were run with Python 3.12 on Windows 11 with one CUDA GPU.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # install the torch build that matches your CUDA version
```

`requirements.txt` pins the versions used for the paper. The logistic teacher is fitted with
scikit-learn's L-BFGS solver, and its per-fold accuracy moves by up to about 0.25 percentage
points across scikit-learn/BLAS builds; the surrogate, the argmin rule, the splits and the
aggregation are deterministic.

## Quick start

All commands are run from the repository root.

```bash
# 1. Smoke test (CPU, 2 epochs, one fold). Delete runs/ afterwards: completed folds are skipped on later calls
python experiments/run_all.py --variants baseline --folds 0 --epochs 2 --device cpu

# 2. Table VII: matched single-component ablation (25 runs, about 30 min on one GPU with 3 workers)
python experiments/run_all.py --table VII

# 3. Table VI: hyperparameter sensitivity (45 runs, about 1.5 h; the reference run is shared with Table VII)
python experiments/run_all.py --table VI

# 4. Physical baselines of Table VIII: logistic teacher and surrogate argmin rule (CPU, 1-2 min)
python experiments/teacher_baseline.py

# 5. Five-fold means, sample SDs, paired intervals, t and exact Wilcoxon tests
python experiments/aggregate.py
```

Outputs are written to `runs/` (ignored by git); the archived outputs of the paper are in
`results/`. Each variant changes only the options listed in
`experiments/configs/variants.json` relative to `experiments/configs/main_recipe.json`.

Full-pipeline reproduction of the primary FP32 and INT8-A results (Tables V and XII) uses
`src/train.py` followed by `src/qat_fold_pipeline.py`; the exact command sequence is in
`archive/main_fp32_int8/run_int8_generalization.bat`.

## Inference with the exported models

```bash
cd deploy/pc_reference
pip install -r requirements.txt
python infer_int8.py      # integer-only reference; reproduces the on-device INT8-A scores bit-exactly
python infer_fp32.py
```

## Reproducibility status

`docs/TABLE_MAP.md` lists, for every table and figure, the script and result file behind it.

* **`experiments/`** (Tables VI and VII, and the physical baselines of Table VIII): runs from
  this repository with the commands above.
* **`archive/`** (all other experiments): the scripts as they were run. They expect the original
  working-tree layout described in `archive/README.md`; their outputs are in `results/`.
* **Firmware**: `deploy/esp32s3_baseline_160mhz/` is the 160-MHz build with scalar kernels. The
  optimized 240-MHz build measured in Table X of the paper (ESP-DSP FP32 kernel, custom PIE INT8
  kernel, IRAM placement) is not part of this package.

All accuracies in the paper are development results on reconstructed-trajectory groups; see
Section VI-H of the paper for the scope and limitations.

## License and citation

Code is released under the MIT License (`LICENSE`). The dataset in `data/` was released by
Huawei for the "IAT" enterprise challenge of the 21st China Graduate Electronics Design
Contest; see `data/README.md`. If you use this repository, please cite the paper
(`CITATION.cff`).
