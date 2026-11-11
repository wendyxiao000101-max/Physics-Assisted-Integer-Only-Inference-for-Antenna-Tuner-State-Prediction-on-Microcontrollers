# 最终推理模型：FP32 + INT8 variant A

本包基于用户提供的《PC双模型推理与训练 (2).zip》生成。FP32模型及标准化文件逐字节保留；默认INT8换为原包内网络级QAT完成、额外head QAT之前的全量模型快照。没有使用任何一折模型冒充最终全量模型，没有把旧B的偏置简单清零，也没有重新训练。

## 结果与区别

| 当前交付模型 | 全量拟合正确数/总数 | 全量拟合准确率 |
|---|---:|---:|
| FP32（原模型） | 95,965 / 96,768 | 99.170180% |
| INT8-A（本包默认） | 95,692 / 96,768 | 98.888062% |

旧INT8-B为99.050306%，不再是本包默认模型。A的五折分组均值94.510060%来自五个分别训练的模型，不能写成此单个二进制的验证成绩。选A简化流程并不意味着全量拟合精度也提高。

INT8-A文件为4,656字节，每样本2,688次线性MAC；15个头部偏置槽位保留但为零。INT8权重/激活存储、INT32偏置与累加、INT64定点临时乘积及整数GELU查表不变。不是可直接加载的ESP-DL模型。

## 使用

解压后进入本README所在目录：

```powershell
python -m pip install -r requirements.txt
python infer_fp32.py
python infer_int8.py
```

已有环境可直接运行，不必重复安装依赖。两个入口默认CPU执行，读取data中的公开CSV，生成results目录下的metrics.json与逐条predictions.csv。本包已包含重新运行的结果。

单条输入与自定义CSV：

```powershell
python infer_fp32.py --input 9021 -3716 737 531 18969
python infer_int8.py --input 9021 -3716 737 10 531 18969
python infer_int8.py --x data/x_train.csv --no-labels
python benchmark_int8.py
```

五维原始整数顺序：[g1Re,g1Im,freq,g2Re,g2Im]；六维：[g1Re,g1Im,freq,itState,g2Re,g2Im]，忽略itState。不要预先标准化。类别0至14。CSV保留原公开数据的表头及GBK编码。FP32通过模型旁的scaler.pkl标准化，INT8量化常量已包含在二进制中。

## 文件与验证

- models/fp32：TorchScript .pt、状态字典.pth、scaler及原始metadata；均保持原包字节。
- models/int8/model_int8.bin：新的默认A模型。
- integer_runtime.py与fast_integer_runtime.py：标量参考和NumPy批量整数实现；96,768条预测已经逐条一致性验证。
- model_hashes.json、verification.json：来源、哈希与当前结果。
- provenance：历史网络QAT配置/日志，以及旧B训练配置。后者只用于证实A快照是旧B的输入，不表示本包执行head QAT。

这是最终模型与推理交付包，不是完整从头训练复现包。旧训练、head QAT和历史导出脚本留在原压缩包中，未复制为本包默认入口，避免再次生成B或其他格式覆盖A。原压缩包保持不变。

INT8-A SHA256：07f8ff773e70fb92070f6e205987f546860ceceea0371efacaadbdea8b85de54。
