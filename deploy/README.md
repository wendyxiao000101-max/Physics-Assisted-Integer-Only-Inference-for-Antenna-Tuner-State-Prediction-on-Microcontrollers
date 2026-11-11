# Deployment

## `pc_reference/`

Full-data FP32 model (folded) and the primary integer model INT8-A (`models/int8/model_int8.bin`,
4,656 bytes, format `I8R32V02`), with reference runtimes:

* `integer_runtime.py` / `fast_integer_runtime.py` - integer-only inference; the on-device
  INT8-A scores must match this reference exactly.
* `infer_int8.py`, `infer_fp32.py` - run a model over `../../data/x_train.csv` and `y_train.csv`.
* `compare_board.py` - compares predictions returned by the board with the PC reference.
* `model_hashes.json`, `verification.json`, `provenance/` - hashes and the QAT configuration
  of the exported models.

These are fitting-data models: replaying the full dataset verifies implementation agreement,
not generalization.

## `esp32s3_baseline_160mhz/`

ESP-IDF projects (`esp-fp32`, `esp-int8`) that replay the dataset on an ESP32-S3 and report
accuracy and per-sample latency. `esp-int8/main/model_int8.bin` is byte-identical to the
PC reference model. Build with ESP-IDF v5.2 (`idf.py set-target esp32s3 && idf.py build`);
`gen_dataset.py` creates the dataset partition image from the CSV files.

This is the 160-MHz build with scalar kernels. Table X of the paper reports a later optimized
build (240 MHz, ESP-DSP dot product for FP32, custom PIE kernel and IRAM placement for INT8-A)
that is not part of this package.
