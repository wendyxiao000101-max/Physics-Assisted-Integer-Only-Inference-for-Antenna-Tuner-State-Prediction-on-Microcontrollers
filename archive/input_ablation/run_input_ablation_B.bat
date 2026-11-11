@echo off
setlocal
cd /d "%~dp0\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  Block B - the same comparison under the FULL delivered v5 recipe
REM  (physics teacher + cosine-decay weight annealing + distance gating).
REM  Restricted to <=5-D default-state inputs, because build_physics_augmentation
REM  rejects feature specs wider than 5-D (train.py L396).
REM  5 features x 2 protocols x 5 seeds = 50 runs.  Confirmation, not the main table.
REM ============================================================================
%PY% -c "import io,sys;s=io.open('train.py',encoding='utf-8').read();m=[f for f in ['--physics-weight-schedule','--physics-final-weight','--physics-decay-end','--physics-min-distance'] if f not in s];print('PREFLIGHT MISSING: '+', '.join(m)) if m else print('preflight OK');sys.exit(1 if m else 0)"
if errorlevel 1 goto oldtrain

%PY% train.py --mode ablate ^
  --features raw5 gamma3 impedance3 polar4 gamma2 ^
  --architectures residual32_gelu_relu_hybrid --protocols random sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --device cuda --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05 ^
  --output results_input_ablation_B
if errorlevel 1 goto fail

echo.
echo ==== BLOCK B DONE ====
echo results_input_ablation_B\ablation\raw_runs.csv
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
