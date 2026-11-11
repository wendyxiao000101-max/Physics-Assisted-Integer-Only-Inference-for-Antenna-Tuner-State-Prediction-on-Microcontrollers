@echo off
setlocal
cd /d "%~dp0\..\.."
set PY=.\.venv-gpu\Scripts\python.exe

REM ============================================================================
REM  C5 - is the LEARNABLE alpha worth its 16 parameters?
REM
REM  HybridLinear computes  (1 - a) * (W x + b)  +  a * scale * cos(x, W),
REM  with a = sigmoid(alpha_logit) learned from a = 0.5.  Across the 60 raw5
REM  runs on record it converged above 0.99 every time (median 0.999964,
REM  alpha_logit median +10.24).  NOTE: above 0.99 is not the folding gate -
REM  qat_fold_pipeline requires 1 - a <= 1e-3, and one C5 run (seed 45 of the
REM  CE hybrid arm, a = 0.997738) does NOT pass it.  At a -> 1:
REM    * the linear branch contributes ~3.6e-5 of the cosine branch;
REM    * the bias lives ONLY on the linear branch, so the head's 15 biases are
REM      dead weight - which is exactly the 2,800 -> 2,784 drop at folding.
REM
REM  CosineLinear pins a = 1 and carries no alpha_logit and no bias: 16 fewer
REM  TRAINING parameters and one projection instead of two during training
REM  (3,168 -> 2,688 MACs).  The deployed INT8 model is already folded to a
REM  single projection, so there is NO extra saving at deployment.
REM  If it ties with the hybrid head, the learnable alpha is design redundancy
REM  and the deployed INT8 model can shed it.
REM
REM  Four arms, same 5 sweep folds, plain CE, lr 1.5e-3, physics teacher lr
REM  with cosine_decay to weight 0 at 70 %% of training -
REM  everything identical to results_c5_head_lr15 so the two blocks are
REM  directly comparable fold by fold:
REM
REM    residual32_gelu_relu_cosine   deployed backbone, pure cosine head
REM    residual32_gelu_relu_hybrid   deployed backbone, hybrid head  (re-run as
REM                                  an in-block anchor; must reproduce 95.054)
REM    mlp64_32_gelu_cosine          MLP backbone, pure cosine head
REM    mlp64_32_gelu_hybrid          MLP backbone, hybrid head
REM
REM  ~20 runs, roughly 70 minutes.
REM
REM  PREREQUISITES (in this order, once each):
REM      %PY% C5\scripts\patch_mlp_head.py    --project "%CD%"
REM      %PY% C5\scripts\patch_cosine_head.py --project "%CD%"
REM ============================================================================

set BASE=--mode ablate --features raw5 --protocols sweep_group --seeds 42 43 44 45 46 --epochs 347 --patience 60 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ce --trajectory-consistency-weight 0.05 --device cuda --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05

%PY% -c "import iatopt.core as c;m=[n for n in ['residual32_gelu_relu_cosine','mlp64_32_gelu_cosine','mlp64_32_gelu_hybrid'] if n not in c.ARCHITECTURES];print('PREFLIGHT MISSING: '+', '.join(m)) if m else print('preflight OK');raise SystemExit(1 if m else 0)"
if errorlevel 1 goto nopatch

%PY% train.py %BASE% --architectures residual32_gelu_relu_cosine residual32_gelu_relu_hybrid mlp64_32_gelu_cosine mlp64_32_gelu_hybrid --output results_c5_cosine
if errorlevel 1 goto fail

echo.
echo ==== C5 COSINE ABLATION DONE ====
echo results_c5_cosine\ablation\raw_runs.csv
goto end

:nopatch
echo.
echo ==== core.py is missing the architecture names.  Run both patches first: ====
echo   %PY% C5\scripts\patch_mlp_head.py    --project "%CD%"
echo   %PY% C5\scripts\patch_cosine_head.py --project "%CD%"
goto end

:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
