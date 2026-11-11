@echo off
setlocal
cd /d "%~dp0"
set PY=.\.venv-gpu\Scripts\python.exe
set EXPORT=model/foldexp

echo ##### 1/10  random_seed42   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/random_seed42 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 2/10  random_seed43   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/random_seed43 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 3/10  random_seed44   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/random_seed44 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 4/10  random_seed45   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/random_seed45 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 5/10  random_seed46   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/random_seed46 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 6/10  sweep_group_seed42   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed42 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 7/10  sweep_group_seed43   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed43 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 8/10  sweep_group_seed44   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed44 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 9/10  sweep_group_seed45   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed45 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail
echo ##### 10/10  sweep_group_seed46   balanced selection (reuse backbone)
%PY% qat_fold_pipeline.py %EXPORT%/sweep_group_seed46 --device cuda --selection balanced --reuse-backbone int8 --out-name int8_bal --alpha-tolerance 1.0
if errorlevel 1 goto fail

echo ##### SUMMARY  (new vs old selection)
%PY% summarize_fold_int8.py %EXPORT% --subdir int8_bal --compare int8 --csv reports/int8_generalisation_balanced.csv
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
