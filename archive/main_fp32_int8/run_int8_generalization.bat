@echo off
setlocal
cd /d "%~dp0"
set PY=.\.venv-gpu\Scripts\python.exe
set COMMON=--mode ablate --epochs 360 --patience 80 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --optimizer adamw --scheduler cosine --device cuda
set NEWNET=--features raw5 --architectures residual32_gelu_relu_hybrid --head-scale 20 --hybrid-alpha-init 0.5 --checkpoint-average-k 5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05
set PHYS=--physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05
set EXPORT=model/foldexp

echo ##############################################################
echo #  STAGE 1/3   FP32 per-fold training + checkpoint export
echo ##############################################################
echo ##### 1/2  RANDOM 5 seeds
%PY% train.py %COMMON% %NEWNET% %PHYS% --protocols random --seeds 42 43 44 45 46 --output results_int8gen_random --ablate-export-dir %EXPORT%
if errorlevel 1 goto fail
echo ##### 2/2  SWEEP 5-fold
%PY% train.py %COMMON% %NEWNET% %PHYS% --protocols sweep_group --seeds 42 43 44 45 46 --output results_int8gen_sweep --ablate-export-dir %EXPORT%
if errorlevel 1 goto fail

echo ##############################################################
echo #  STAGE 2/3   per-fold integer INT8 pipeline
echo ##############################################################
echo ##### fold 1/10  random_seed42
%PY% qat_fold_pipeline.py %EXPORT%/random_seed42 --device cuda
if errorlevel 1 goto fail
echo ##### fold 2/10  random_seed43
%PY% qat_fold_pipeline.py %EXPORT%/random_seed43 --device cuda
if errorlevel 1 goto fail
echo ##### fold 3/10  random_seed44
%PY% qat_fold_pipeline.py %EXPORT%/random_seed44 --device cuda
if errorlevel 1 goto fail
echo ##### fold 4/10  random_seed45
%PY% qat_fold_pipeline.py %EXPORT%/random_seed45 --device cuda
if errorlevel 1 goto fail
echo ##### fold 5/10  random_seed46
%PY% qat_fold_pipeline.py %EXPORT%/random_seed46 --device cuda
if errorlevel 1 goto fail
echo ##### fold 6/10  sweep_group_seed42
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed42 --device cuda
if errorlevel 1 goto fail
echo ##### fold 7/10  sweep_group_seed43
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed43 --device cuda
if errorlevel 1 goto fail
echo ##### fold 8/10  sweep_group_seed44
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed44 --device cuda
if errorlevel 1 goto fail
echo ##### fold 9/10  sweep_group_seed45
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed45 --device cuda
if errorlevel 1 goto fail
echo ##### fold 10/10  sweep_group_seed46
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed46 --device cuda
if errorlevel 1 goto fail

echo ##############################################################
echo #  STAGE 3/3   summary
echo ##############################################################
%PY% summarize_fold_int8.py %EXPORT% --csv reports/int8_generalisation.csv
if errorlevel 1 goto fail

echo.
echo ==== DONE ====
goto end
:fail
echo.
echo ==== FAILED ====
:end
endlocal
pause
