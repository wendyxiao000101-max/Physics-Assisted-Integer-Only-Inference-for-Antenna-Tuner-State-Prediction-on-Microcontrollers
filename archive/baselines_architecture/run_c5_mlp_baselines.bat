@echo off
setlocal
cd /d "%~dp0\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  C5 - the reviewer's "3-layer MLP without residual" baseline.
REM
REM  mlp* names take the plain branch in core.py: Linear -> act -> Linear ->
REM  act -> Linear, no skip connection.  mlp64_32 is 5-64-32-15, i.e. exactly
REM  three weight matrices.
REM
REM  Everything else is the delivered v5 recipe, byte-identical to the arm that
REM  produced results_teacher_B0_clean (94.6755 +- 1.0680), so the residual
REM  point on the Pareto plot is already in hand and is NOT re-run here.
REM  Same 5 sweep_group folds (seeds 42-46) -> every comparison is paired.
REM
REM  arch              structure      matrix weights   vs residual32
REM  ----------------  -------------  --------------   -------------
REM  mlp64_32_gelu     5-64-32-15              2,848          1.06x   <- key control
REM  mlp64_32          5-64-32-15              2,848          1.06x   <- literal ask
REM  mlp128_64         5-128-64-15             9,792          3.64x
REM  mlp32_16          5-32-16-15                912          0.34x
REM
REM  4 arms x 5 folds = 20 runs, ~347 s each -> about 2 hours.
REM  raw_runs.csv is rewritten after every run, so stopping early is fine:
REM  the arms are ordered so the ones the argument needs finish first.
REM ============================================================================

set BASE=--mode ablate --features raw5 --protocols sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --device cuda --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05

%PY% train.py %BASE% --architectures mlp64_32_gelu mlp64_32 mlp128_64 mlp32_16 --output results_c5_mlp
if errorlevel 1 goto fail

echo.
echo ==== C5 MLP BASELINES DONE ====
echo results_c5_mlp\ablation\raw_runs.csv
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
