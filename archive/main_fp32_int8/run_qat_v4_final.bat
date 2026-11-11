@echo off
setlocal
cd /d "%~dp0"
set PY=.\.venv-gpu\Scripts\python.exe
set CFG=--final-feature raw5 --final-architecture residual32_gelu_relu_hybrid --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --scheduler cosine --head-scale 20 --hybrid-alpha-init 0.5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05 --device cuda
set FP32=model/residual32_physics_dual_champion/model_fp32.pth
set BASE=--qat --qat-scheme lsq --qat-init-from %FP32%

echo ##### FINAL 1/4  qv4_F1_B347  (347 epochs)
%PY% train.py --mode finalize %CFG% %BASE% --final-epochs 347 --model-output model/qv4_F1_B347
if errorlevel 1 goto fail
%PY% infer_int8.py model/qv4_F1_B347 --compare
if errorlevel 1 goto fail

echo ##### FINAL 2/4  qv4_F2_C347_kd05  (347 epochs)
%PY% train.py --mode finalize %CFG% %BASE% --qat-kd-from %FP32% --qat-kd-weight 0.5 --qat-kd-temperature 2.0 --final-epochs 347 --model-output model/qv4_F2_C347_kd05
if errorlevel 1 goto fail
%PY% infer_int8.py model/qv4_F2_C347_kd05 --compare
if errorlevel 1 goto fail

echo ##### FINAL 3/4  qv4_F3_B347_bs256  (347 epochs)
%PY% train.py --mode finalize %CFG% %BASE% --batch-size 256 --final-epochs 347 --model-output model/qv4_F3_B347_bs256
if errorlevel 1 goto fail
%PY% infer_int8.py model/qv4_F3_B347_bs256 --compare
if errorlevel 1 goto fail

echo ##### FINAL 4/4  qv4_F4_B500  (500 epochs)
%PY% train.py --mode finalize %CFG% %BASE% --final-epochs 500 --model-output model/qv4_F4_B500
if errorlevel 1 goto fail
%PY% infer_int8.py model/qv4_F4_B500 --compare
if errorlevel 1 goto fail

echo ##### SUMMARY  (qv2 + qv3 + qv4)
%PY% summarize_qat.py --prefix model/qv --csv reports/qat_v4_final.csv
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
