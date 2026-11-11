@echo off
setlocal
cd /d "%~dp0\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  C5 - non-residual MLP baseline, FAIR recipe  (replaces run_c5_mlp_baselines)
REM
REM  WHY THIS EXISTS: the first attempt ran the MLPs under the delivered v5
REM  recipe (ala_linear + LDAM margin + pairwise-ALA + head-scale 20).  That
REM  recipe was co-tuned with the residual net's HybridLinear head; a plain
REM  nn.Linear head does not train under it.  Every MLP run peaked at epoch
REM  3-13 of 347 and early-stopped, with minimum class recall 0.000.  Those
REM  numbers measure a broken optimisation, not an architecture, and would be
REM  a strawman baseline if published.
REM
REM  FIX: give all three architectures the SAME plain cross-entropy recipe,
REM  including the residual net.  The residual point under the v5 recipe
REM  (94.6755) stays in the paper as the delivered system; this block is the
REM  like-for-like architecture comparison.  Everything except --loss and the
REM  ALA/LDAM switches is unchanged, and the physics teacher stays on for all
REM  three, so the only thing that differs between arms is the backbone.
REM
REM  arch                         matrix weights   role
REM  ---------------------------  --------------   ----------------------------
REM  residual32_gelu_relu_hybrid           2,688   anchor - the paper's backbone
REM  mlp64_32_gelu                         2,848   1.06x, parameter-matched
REM  mlp128_64                             9,792   3.64x, deliberately larger
REM
REM  Stage 1 smoke: 2 runs, ~12 min, checks that training is healthy.
REM  Stage 2 full:  3 arms x 5 folds = 15 runs, ~87 min.
REM ============================================================================

set BASE=--mode ablate --features raw5 --protocols sweep_group --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ce --trajectory-consistency-weight 0.05 --device cuda --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05

echo ==== STAGE 1  smoke (seed 42 only, ~12 min) ====
%PY% train.py %BASE% --seeds 42 --architectures residual32_gelu_relu_hybrid mlp64_32_gelu --output results_c5_fair_smoke
if errorlevel 1 goto fail

%PY% -c "import pandas as pd;d=pd.read_csv(r'results_c5_fair_smoke\ablation\raw_runs.csv');print();print(d[['architecture','best_epoch','epochs_ran','test_accuracy','test_minimum_class_recall']].to_string(index=False));print();bad=(d.best_epoch<30).any();print('SMOKE FAILED: best_epoch still collapses, do not run stage 2' if bad else 'smoke OK: training is healthy, stage 2 starting');raise SystemExit(1 if bad else 0)"
if errorlevel 1 goto smokefail

echo.
echo ==== STAGE 2  full (15 runs, ~87 min) ====
%PY% train.py %BASE% --seeds 42 43 44 45 46 --architectures residual32_gelu_relu_hybrid mlp64_32_gelu mlp128_64 --output results_c5_fair
if errorlevel 1 goto fail

echo.
echo ==== C5 FAIR BASELINES DONE ====
echo results_c5_fair\ablation\raw_runs.csv
goto end

:smokefail
echo.
echo ==== SMOKE FAILED - send me results_c5_fair_smoke\ablation\raw_runs.csv, do not run stage 2 ====
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
