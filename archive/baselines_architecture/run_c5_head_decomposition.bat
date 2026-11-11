@echo off
setlocal
cd /d "%~dp0\..\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  C5 - decompose the residual-vs-MLP gap into SKIP CONNECTION vs HEAD.
REM
REM  What the smoke run revealed: the delivered model's HybridLinear head learns
REM  alpha = 0.99989, i.e. it collapses to a pure cosine classifier at scale 20.
REM  Cosine heads are known to rescue minority classes - measured min-class
REM  recall was 68.6 % for the residual net and 0.0 % for a plain-Linear-head
REM  MLP.  So the earlier "residual vs MLP" runs conflated two changes.  This
REM  block separates them:
REM
REM    residual32_gelu_relu_hybrid  vs  mlp64_32_gelu_hybrid  -> whole-architecture
REM        contrast (width 32 vs 64/32, 4 vs 3 linear layers AND the skip all
REM        differ - NOT attributable to the skip alone; see run_c5_noskip.bat)
REM    mlp64_32_gelu_hybrid         vs  mlp64_32_gelu         -> the head alone
REM
REM  Plain cross-entropy for every arm, same 5 sweep_group folds.  Physics
REM  teacher: lr, weight schedule cosine_decay from 1.0 to 0 at 70 %% of
REM  training (NOT on throughout).  Measured wall time on the RTX 5060 was
REM  about 2 min per residual run and 1-2 min per MLP run.
REM
REM  Learning rate is also checked: the MLP arms are run at the tuned 1.5e-3 and
REM  at 5e-4, because the tuned value was chosen for the residual backbone and a
REM  baseline that is handicapped by someone else's learning rate is a strawman.
REM  Separate --output per learning rate, since raw_runs.csv does not record it.
REM
REM  PREREQUISITE (run once):
REM      %PY% C5\scripts\patch_mlp_head.py --project "%CD%"
REM ============================================================================

set BASE=--mode ablate --features raw5 --protocols sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ce --trajectory-consistency-weight 0.05 --device cuda --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05

%PY% -c "import iatopt.core as c;missing=[n for n in ['mlp64_32_gelu_hybrid','mlp128_64_gelu_hybrid','mlp32_16_gelu_hybrid'] if n not in c.ARCHITECTURES];print('PREFLIGHT MISSING: '+', '.join(missing)) if missing else print('preflight OK');raise SystemExit(1 if missing else 0)"
if errorlevel 1 goto nopatch

echo ==== A  lr = 1.5e-3  (the tuned value) ====
%PY% train.py %BASE% --learning-rate 0.0015 --architectures residual32_gelu_relu_hybrid mlp64_32_gelu_hybrid mlp64_32_gelu mlp128_64_gelu_hybrid mlp32_16_gelu_hybrid --output results_c5_head_lr15
if errorlevel 1 goto fail

echo ==== B  lr = 5e-4  (fairer chance for the MLP arms) ====
%PY% train.py %BASE% --learning-rate 0.0005 --architectures mlp64_32_gelu_hybrid mlp64_32_gelu mlp128_64_gelu_hybrid --output results_c5_head_lr05
if errorlevel 1 goto fail

echo.
echo ==== C5 HEAD DECOMPOSITION DONE ====
echo results_c5_head_lr15\ablation\raw_runs.csv
echo results_c5_head_lr05\ablation\raw_runs.csv
goto end

:nopatch
echo.
echo ==== core.py has not been patched.  Run this first: ====
echo   %PY% C5\scripts\patch_mlp_head.py --project "%CD%"
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
