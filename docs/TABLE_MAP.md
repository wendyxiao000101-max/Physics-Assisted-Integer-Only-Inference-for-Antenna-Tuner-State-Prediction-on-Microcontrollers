# Table and figure map

Numbering follows the submitted manuscript. Status codes:

* **R** - repository-relative: runs from this repository with the commands shown.
* **A** - archived as run: the unmodified script that produced the numbers is in `archive/`
  and expects the original working tree (`archive/README.md`); its outputs are in `results/`.
* **N** - not included yet.

All result paths are relative to `results/`; all archive paths are relative to `archive/`.

## Tables

| Item | Content | Script(s) | Result file(s) | Status |
|---|---|---|---|---|
| Table I | Data composition, reconstructed groups | `p95_grouping/prepare.py` (trajectory reconstruction and label audit) | `p95_grouping/group_audit.json`, `p95_grouping/group_sizes.csv`; splits in `data/splits/` | A |
| Table II | Grouped FP32 input ablation | `input_ablation/run_input_ablation_A.bat`, `run_input_ablation_B.bat`, `patch_feature_specs.py`, `summarize_input_ablation.py` | `table02_input_ablation/T15_input_ablation_network.csv` (network), `T13_input_ablation_probe.csv` (probe) | A |
| Table III | Training and quantization settings | `experiments/configs/main_recipe.json`; `main_fp32_int8/run_int8_generalization.bat` | `deploy/pc_reference/provenance/*.json` | - |
| Table IV | ESP32-S3 build configuration | `deploy/esp32s3_baseline_160mhz/*/sdkconfig.defaults` (baseline build only) | - | N (optimized build) |
| Table V | Grouped FP32, INT8-A/B/C | `main_fp32_int8/run_int8_generalization.bat` (FP32 folds, then QAT), `run_int8_balsel.bat` (variant C), `summarize_fold_int8.py`; code in `src/train.py`, `src/qat_fold_pipeline.py`, `src/qat_A.py` | `table05_12_main_fp32_int8/int8_generalisation.csv` (`protocol = sweep_group`; FP32 = `fp32_*`, A = `int8_backbone_acc`, B = `int8_*`), `int8_generalisation_balanced.csv` (C), `T7_head_qat_three_way.csv`, `T7b_head_qat_per_fold.csv` | A |
| Table VI | FP32 hyperparameter sensitivity | `python experiments/run_all.py --table VI`, then `python experiments/aggregate.py` | `table06_sensitivity/C13_summary.csv`, `C13_per_fold.csv`, `C13_per_class.csv` | R |
| Table VII | FP32 single-component ablation (physics, ALA, class-pair factor, trajectory loss) | `python experiments/run_all.py --table VII`, then `python experiments/aggregate.py` | `table07_component_ablation/supplement_summary.csv`, `supplement_paired_contrasts.csv`, `supplement_per_fold.csv` | R |
| Table VIII | Baseline accuracy and storage | Trees: `baselines_architecture/baseline_pareto_v2.py`, `baseline_gbm_ext.py`, `tree_codec.py`. Physical baselines: `python experiments/teacher_baseline.py` | `table08_09_baselines_architecture/T22v2_budget_table.csv`, `T22c_budget_with_gbm.csv`, `T22v2_paired_vs_int8.csv`, `T22v2_encoding_checks.csv`; `table07_component_ablation/teacher_per_fold.csv`, `teacher_cost.json` | A (trees), R (teacher) |
| Table IX | Separate cross-entropy architecture controls | `baselines_architecture/patch_mlp_head.py`, `patch_cosine_head.py`, `patch_noskip.py`, `run_c5_head_decomposition.bat`, `run_c5_cosine_ablation.bat`, `run_c5_noskip.bat`, `c5_stats.py` | `table08_09_baselines_architecture/T23_c5_architecture_head.csv`, `T23b_*`, `T24_*`, `T24b_*`, `T27_*`, `T27b_*`, `T27c_*` | A |
| Table X | ESP32-S3 latency and resources | Optimized 240-MHz firmware | `table10_board_replay/pc_reference_verification.json` (PC side of the replay check only) | N |
| Table XI | Conditional RF metrics | `rf_analysis/compute_three_model_rf.py`, `compare_ab.py` | `table11_rf_analysis/rf_summary.csv`, `rf_per_fold.csv`, `coverage.csv`, `summary.json`, `AB_*.csv` | A |
| Table XII | Secondary random-split results | Same scripts as Table V | `table05_12_main_fp32_int8/int8_generalisation.csv` (`protocol = random`), `T1_main_results.csv` | A |
| Table XIII | Notation | - | - | - |

## Figures

| Item | Content | Script(s) | Result file(s) | Status |
|---|---|---|---|---|
| Fig. 4(a) | Held-out surrogate reflection error | `physics_evidence/run_c1_validation.py`, `verify_outputs.py`, `reproduce_figures.py` | `fig4_physics_evidence/C1_01_reflection_error_per_fold.csv`, `C1_07_test_pooled_by_band.csv`, `figure2_error_summary.csv` | A |
| Fig. 4(b) | Student gain versus teacher quality (separate four-arm recipe) | `physics_evidence/patch_teacher_quality.py`, `teacher_quality.py`, `run_teacher_quality_T1.bat`, `run_teacher_quality_T2.bat`, `summarize_teacher_quality.py` | `fig4_physics_evidence/T21_teacher_quality_vs_student_gain.csv`, `T21c_teacher_arm_contrasts.csv`, `figure1_*.csv` | A |
| Fig. 5 | Paired INT8-A minus FP32 accuracy | Same runs as Tables V and XII | `table05_12_main_fp32_int8/T3_quantisation_cost_per_fold.csv`, `T3b_significance.csv` | A |
| Fig. 6 | Model size versus accuracy | `baselines_architecture/baseline_pareto_v2.py`, `make_c5_figures.py` | `table08_09_baselines_architecture/T22v2_baseline_pareto.csv`, `T22v2_valid_frontier.csv` | A |
| Figs. 7, 8 | Conditional return-loss CDFs and confusion costs | `rf_analysis/compare_ab.py`, `reproduce_figure7_data.py` | `table11_rf_analysis/AB_RL_thresholds.csv`, `AB_confusion_cost_pairs.csv` | A |

## Results quoted in the text

| Section | Content | Script(s) | Result file(s) | Status |
|---|---|---|---|---|
| VI-B | Teacher and argmin baselines, teacher cost estimate | `python experiments/teacher_baseline.py` | `table07_component_ablation/teacher_per_fold.csv`, `teacher_cost.json`, `supplement_paired_contrasts.csv`; `fig4_physics_evidence/C1_03_*`, `C1_06_*` | R |
| VI-C | P95 regrouping, FP32/INT8-A under P95, matched physics-off control | `p95_grouping/prepare.py`, `train_p95.py`, `run_experiments.py`, `prepare_qat.py`, `summarize.py`, `diagnose_head.py`; `run_c6_p95_nophys.py`, `summarize_c6.py` | `p95_grouping/classification_summary.csv`, `classification_folds.csv`, `C6_physics_gain.csv`, `C6_per_fold.csv`, `quantization_delta_*.csv`, `head_gate_diagnostics.json` | A |
| VI-C | ExtraTrees grouping-rule probe and RF comparison | `p95_grouping/auxiliary.py` | `p95_grouping/ExtraTrees_sensitivity_folds.csv`, `ExtraTrees_config.json`, `RF_sensitivity_common.csv` | A |
| VI-D | Quantization cost and head refinement | Same runs as Table V | `table05_12_main_fp32_int8/T3_*`, `T3b_*`, `T7_*`, `T7b_*` | A |
| VI-E | Desktop scalar-Python latency | `baselines_architecture/bench_c5_latency.py` | `table08_09_baselines_architecture/T26_desktop_latency.csv` | A |
| IV-B | Classifier-folding and CLE equivalence checks | `baselines_architecture/verify_fold_consistency.py`, `verify_fold_and_head_bias.py` | `table05_12_main_fp32_int8/T8_head_folding_equivalence.csv`, `table08_09_baselines_architecture/T25_fold_consistency.csv` | A |
| App. C | Class-pair factor construction | `src/iatopt/core.py` (out-of-fold ExtraTrees inside training) | cached under `runs/teacher_cache/` when training | R |
