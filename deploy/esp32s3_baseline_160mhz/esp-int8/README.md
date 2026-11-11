# ESP32-S3 INT8 UART 逐帧推理

将仓库中的 INT8-A 模型（`I8R32V02`，4,656 字节）放到 ESP32-S3 上，经控制台串口与上位机逐帧问答，最后统计准确率。

片上核与仓库根目录 `integer_runtime.py` 对齐。输入仍是 5 个**原始整数** `[g1Re, g1Im, freq, g2Re, g2Im]`；`offset/step` 量化在固件内部完成，不要再套 `scaler.pkl`。

## 协议（ASCII，一行一帧）

波特率默认 115200。ESP32-S3-DevKit 的 USB 口一般为 USB-Serial-JTAG，波特率可被忽略，但仍按 115200 打开。

上电或收到 `HELLO` 后，设备打印：

```
READY I8R32V02 sha256=07f8ff773e70fb92070f6e205987f546860ceceea0371efacaadbdea8b85de54 bytes=4656
```

主机发送（5 维；若误发 6 维则设备丢弃 `itState`）：

```
9021 -3716 737 531 18969
```

设备回复（类别与片上推理微秒）：

```
OK 6 42
```

解析失败为 `ERR parse` / `ERR infer`。金标：上述样本类别为 **6**。

## 编译烧录（ESP-IDF 5.x）

```bash
cd esp32s3
. $HOME/esp/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build flash
```

若数据口是板载 USB-UART（GPIO43/44）而不是 USB-Serial-JTAG，在 `sdkconfig.defaults` 中改为：

```
CONFIG_ESP_CONSOLE_UART_DEFAULT=y
```

然后 `idf.py fullclean && idf.py build flash`。

## 上位机

```bash
python3 -m pip install -r esp32s3/requirements.txt
python3 host_eval_uart.py --port /dev/cu.usbmodemXXXX --limit 20
python3 host_eval_uart.py --port /dev/cu.usbmodemXXXX
```

`--limit 20` 用于通断；不加则跑完整 `data/x_train.csv`（96,768 条）。结束写入 `esp32s3/results/uart/metrics.json` 与 `predictions.csv`。

报告中请同时看：

- `pc_agreement_percent`：与 PC 整数核一致率，部署正确时应为 **100%**
- `accuracy_percent`：相对训练集标签，应接近 **98.888%**（回代，不是留出测试）
- `uart_errors`：传输失败帧数，不应计入“模型不准”

打开串口时脚本会拉低 DTR/RTS 并等待 `READY`，避免和 boot log 搅在一起。

无开发板时可用同一套 C 核在电脑上对照 Python：

```bash
cc -O2 -std=c11 -o /tmp/int8_pc_check main/pc_check.c main/integer_runtime.c main/model_blob.c
/tmp/int8_pc_check
```

首行应为 `6`（与 `9021 -3716 737 531 18969` 的 Python 结果一致）。
