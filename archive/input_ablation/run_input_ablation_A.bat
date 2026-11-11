@echo off
setlocal
cd /d "%~dp0\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  Block A - input-space ablation WITHOUT physics distillation.
REM  Every arm gets an identical recipe, so the only variable is the input space.
REM  This is the only block where all seven arms are comparable: the physics
REM  augmentation refuses inputs wider than 5-D, so raw12 / raw6 can only run here.
REM  7 features x 2 protocols x 5 seeds = 70 runs.  This is the paper's main table.
REM ============================================================================
%PY% -c "import io,sys;s=io.open('train.py',encoding='utf-8').read();m=[f for f in ['--physics-weight-schedule','--physics-final-weight','--physics-decay-end','--physics-min-distance'] if f not in s];print('PREFLIGHT MISSING: '+', '.join(m)) if m else print('preflight OK');sys.exit(1 if m else 0)"
if errorlevel 1 goto oldtrain

%PY% "%~dp0patch_feature_specs.py" iatopt\core.py
if errorlevel 1 goto fail

%PY% train.py --mode ablate ^
  --features raw12 raw6 raw5 gamma3 gamma2 g1_only3 g1_state4 ^
  --architectures residual32_gelu_relu_hybrid --protocols random sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --device cuda --physics-teacher none ^
  --output results_input_ablation_A
if errorlevel 1 goto fail

echo.
echo ==== BLOCK A DONE ====
echo results_input_ablation_A\ablation\raw_runs.csv
goto end

:oldtrain
echo.
echo ==== WRONG train.py  (see run_input_ablation_smoke.bat) ====
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
