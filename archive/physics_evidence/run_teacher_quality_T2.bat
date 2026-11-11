@echo off
setlocal
cd /d "%~dp0\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  TEACHER QUALITY vs STUDENT GAIN  --  TIER 2  (optional: 3 more points so the trend can be regressed)
REM
REM  One knob is turned: how good the physics teacher is.
REM  The student recipe below is byte-identical across every arm and identical
REM  to the delivered v5 recipe.  Same 5 sweep_group folds (seeds 42-46) in
REM  every arm, so every comparison is paired fold by fold.
REM
REM  Run this ONLY after Tier 1.  It fills in the middle of the quality ladder.
REM
REM  arm  teacher                       teacher test Acc (measured offline)
REM  ---  ----------------------------  ----------------------------------
REM  A1   argmin (pure physics rule)    92.01 %   <- label-free teacher, the other shipped variant
REM  B1   lr, eta=0.05                  87.73 %
REM  B3   lr, eta=0.40                  60.43 %
REM
REM  3 arms x 5 folds = 15 runs, ~347 s each -> about 1.5 hours.
REM  Each arm writes its own results_teacher_* folder, so a partial run is
REM  still usable: analyse whatever finished.
REM
REM  PREREQUISITE (run once):
REM      %PY% scripts\patch_teacher_quality.py --project "<absolute path to this folder>"
REM ============================================================================

set BASE=--mode ablate --features raw5 --architectures residual32_gelu_relu_hybrid --protocols sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --device cuda
set PHYS=--physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05 --physics-teacher-seed 20260918

REM ---- preflight: the patch must be in place, otherwise every arm is silently identical
%PY% -c "import io,sys;s=io.open('train.py',encoding='utf-8').read();m=[f for f in ['--physics-teacher-noise','--physics-teacher-shuffle','--physics-teacher-seed','--physics-decay-end'] if f not in s];print('PREFLIGHT MISSING: '+', '.join(m)) if m else print('preflight OK');sys.exit(1 if m else 0)"
if errorlevel 1 goto nopatch
%PY% -c "import iatopt.teacher_quality as t;print('teacher_quality ->',t.__file__)"
if errorlevel 1 goto nopatch

echo ==== A1  argmin teacher (pure physics rule, label-free) ====
%PY% train.py %BASE% --physics-teacher argmin %PHYS% --output results_teacher_A1_argmin
if errorlevel 1 goto fail

echo ==== B1  lr teacher, eta = 0.05 ====
%PY% train.py %BASE% --physics-teacher lr %PHYS% --physics-teacher-noise 0.05 --output results_teacher_B1_eta005
if errorlevel 1 goto fail

echo ==== B3  lr teacher, eta = 0.40 ====
%PY% train.py %BASE% --physics-teacher lr %PHYS% --physics-teacher-noise 0.40 --output results_teacher_B3_eta040
if errorlevel 1 goto fail

echo.
echo ==== TIER 2 DONE ====
echo Now run:
echo   %PY% summarize_teacher_quality.py results_teacher_*\ablation\raw_runs.csv --out tables
goto end

:nopatch
echo.
echo ==== train.py has not been patched.  Run this first: ====
echo   %PY% scripts\patch_teacher_quality.py --project "%CD%"
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
