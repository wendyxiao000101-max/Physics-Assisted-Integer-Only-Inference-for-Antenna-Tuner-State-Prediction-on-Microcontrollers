@echo off
setlocal
cd /d "%~dp0\..\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  C5 - same-backbone "skip connection only" control (audit item 3.1).
REM
REM  residual32_noskip_gelu_relu_hybrid is layer-for-layer identical to the
REM  deployed residual32_gelu_relu_hybrid: same widths, same 2,800 parameters,
REM  same module order (so identical initial weights under the same seed).
REM  The ONLY change: the block output is ReLU(F(x)) instead of ReLU(x + F(x)).
REM
REM  Part A (default): plain-CE recipe, lr 1.5e-3, physics teacher lr with
REM  cosine_decay to weight 0 at 70 %% of training - the same recipe as
REM  results_c5_head_lr15 and results_c5_cosine.  The residual arm is re-run
REM  as an in-block anchor; it reproduced 95.0544 exactly (all 5 folds) in the
REM  cosine block, so any difference here would flag nondeterminism.
REM  10 runs, roughly 20 minutes.
REM
REM  Part B (only with argument  v5 ): the ALA-linear main recipe, paired with
REM  results_teacher_B0_clean (residual, same recipe, 94.6755).  5 runs.
REM
REM  PREREQUISITE (once):
REM      %PY% C5\scripts\patch_noskip.py --project "<absolute path of this folder>"
REM ============================================================================

set BASE=--mode ablate --features raw5 --protocols sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --trajectory-consistency-weight 0.05 --device cuda
set PHYS=--physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05

%PY% -c "import iatopt.core as c;ok='residual32_noskip_gelu_relu_hybrid' in c.ARCHITECTURES and hasattr(c,'PlainBlock');print('preflight OK' if ok else 'PREFLIGHT MISSING: residual32_noskip_gelu_relu_hybrid');raise SystemExit(0 if ok else 1)"
if errorlevel 1 goto nopatch

echo ==== A  noskip vs residual, plain CE recipe ====
%PY% train.py %BASE% %PHYS% --loss ce --architectures residual32_noskip_gelu_relu_hybrid residual32_gelu_relu_hybrid --output results_c5_noskip
if errorlevel 1 goto fail

if /i not "%~1"=="v5" goto done
echo ==== B  noskip, ALA-linear main recipe (pairs with results_teacher_B0_clean) ====
%PY% train.py %BASE% %PHYS% --physics-teacher-seed 20260918 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --architectures residual32_noskip_gelu_relu_hybrid --output results_c5_noskip_v5
if errorlevel 1 goto fail

:done
echo.
echo ==== C5 NOSKIP DONE ====
echo results_c5_noskip\ablation\raw_runs.csv
if /i "%~1"=="v5" echo results_c5_noskip_v5\ablation\raw_runs.csv
goto end

:nopatch
echo.
echo ==== core.py has not been patched.  Run this first: ====
echo   %PY% C5\scripts\patch_noskip.py --project "%CD%"
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
