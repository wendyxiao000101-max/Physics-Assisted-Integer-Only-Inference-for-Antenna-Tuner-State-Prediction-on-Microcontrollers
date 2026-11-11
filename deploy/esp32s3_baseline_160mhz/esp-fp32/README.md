# ESP32-S3 FP32 纯推理延迟

本工程将仓库中的折叠 FP32 模型部署到 ESP32-S3。输入数据保留在 Flash 的
`dataset` 分区；Cycle Counter 包围每帧标准化和 `fp32_predict()`，后者
包含四层线性计算、GELU、ReLU、残差和 argmax。

## 生成文件

```bash
python3 gen_model_blob.py
python3 gen_dataset.py
```

`gen_model_blob.py` 从 `models/fp32/model_fp32_folded.pt` 原样提取 FP32
权重；`gen_dataset.py` 生成与 INT8 工程相同的 Flash 数据集。

## 编译与烧录

```bash
. $HOME/esp/esp-idf/export.sh
idf.py build
idf.py -p COM端口 flash
python -m esptool --chip esp32s3 -p COM端口 write_flash 0x90000 main/dataset.bin
idf.py -p COM端口 monitor
```

若此前已经烧录过相同的 `dataset.bin`，只需烧录应用程序。

正确结果应包含：

```text
RESULT rows=96768 correct=95965 errors=803 accuracy=99.170180 ...
LATENCY_CYCLES pure_inference ...
LATENCY_US pure_inference ...
```

延迟统计先预热 100 帧，扣除 Cycle Counter 读取开销，包含标准化，但不包含
Flash 读取、串口输出和循环间的任务让出时间。百分位采用 64 周期
（160 MHz 下为 0.4 微秒）宽的在线直方图。

## PC 一致性检查

```bash
cc -O2 -std=c11 -I main -o /tmp/fp32_pc_check \
  main/pc_check.c main/fp32_runtime.c -lm
/tmp/fp32_pc_check main/fp32_model.bin main/dataset.bin \
  ../results/fp32/full/predictions.csv
```

应输出 `correct=95965 agreement=96768`。
