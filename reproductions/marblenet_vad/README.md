# MarbleNet-3x2x64 复现

这是 MarbleNet 论文中 `MarbleNet-3x2x64` 的纯 PyTorch 复现，不依赖
NeMo。实现放在 `reproductions/marblenet_vad/`，可以直接使用项目已经生成
的 LibriVAD-style 数据。

## 模型结构

`MarbleNet-BxRxC` 的配置为 `B=3`、`R=2`、`C=64`：

| 阶段 | 输出通道 | 卷积核 | 重复次数 | 残差 |
| --- | ---: | ---: | ---: | --- |
| Conv1 | 128 | 11 | 1 | 否 |
| B1 | 64 | 13 | 2 | 是 |
| B2 | 64 | 15 | 2 | 是 |
| B3 | 64 | 17 | 2 | 是 |
| Conv2 | 128 | 29, dilation=2 | 1 | 否 |
| Conv3 | 128 | 1 | 1 | 否 |
| 平均池化 + Linear | 2 | - | - | - |

每个时间通道可分离卷积块依次使用 depthwise Conv1d、pointwise Conv1d、
BatchNorm、ReLU 和 Dropout。输入为 `[B, 64, 64]` 的 MFCC，输出为
`[B, 2]` 的 logits。当前实现的可训练参数为 **89,154 个**，与论文报告的
约 88K 量级一致。

## 论文协议与 LibriVAD 适配

论文的 SCF 训练集由 Google Speech Commands 语音和 Freesound 非语音片段
组成；本项目改用已经生成的 LibriVAD-style mixtures：

| 项目 | 论文 / NeMo 配置 | 本目录实现 |
| --- | --- | --- |
| 输入特征 | 64 维 MFCC | 16 kHz、25 ms 窗、10 ms 帧移、`n_fft=512`、64 Mel、64 MFCC |
| 训练段长 | 0.63 s（约 64 帧） | 0.63 s，10,080 采样点 |
| 训练标签 | 每段一个语音/非语音标签 | 窗内逐采样标签多数投票 |
| 训练采样 | 语音段/非语音段 1:1 | `LibriVADSegments(balance=True)` |
| 波形增强 | 80% 概率：±5 ms 时移、白噪声 -90 到 -46 dB | 默认 `0.8`，可配置 |
| 频谱增强 | 2 个时间 mask（≤25）、2 个频率 mask（≤15）、5 个 cutout | `SpecAugment` + `SpecCutout` |
| 优化器 | SGD，momentum 0.9，weight decay 0.001 | 相同 |
| 学习率 | 0.01 → 0.001，Warmup-Hold-Decay，5%/45%/50%，二阶多项式 | 相同 |
| 训练规模 | 150 epochs，batch 128/GPU | 默认 150 epochs，batch 128 |
| 推理重叠 | 87.5% overlap，median smoothing | 默认相同 |
| 帧化 | AV A-speech 帧级评估 | LibriVAD 25 ms/10 ms，多数投票 |

论文中的 AV A-speech 分数不能和本项目 LibriVAD-style 测试集直接比较：
数据域、负样本定义和帧化协议都不同。这里的目标是复现模型、训练配方和
滑窗推理，并得到同一数据协议下的可重复基线。

当前本机同时生成了 `medium` 档的 LibriSpeech 与 LibriSpeechConcat 数据。
完整的 two-variant `large` 训练集估算约 1.31 TB，超出本机 658 GB 可用空间；
正式实验采用 `medium` manifest，并在全部 9 类噪声 × 6 个 SNR 上做确定性
分层行抽样。这样既保留完整条件覆盖，也不会在启动时扫描几十万条混合音频。

## 冒烟训练

先使用已经验证过的 smoke manifests：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke `
  --epochs 3 --limit 4 --batch-size 16
```

只想验证前向、反向和 checkpoint 落盘时：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke `
  --epochs 1 --limit 4 --batch-size 16 --max-steps 1
```

训练产物默认写到 `results/marblenet_vad/`：

```text
best.pt
last.pt
history.json
```

## Causal 训练与续训

加上 `--causal` 后，编码器全部改成左侧补零的因果卷积，MFCC 也使用
`center=False` 并只在左侧补一个 FFT 窗，因此推理不会读取未来帧。
checkpoint 会记录 `model_config.causal=true`，评估脚本会自动恢复对应的
因果特征前端。

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke `
  --causal --results-dir results\marblenet_vad_causal `
  --epochs 30 --batch-size 16
```

从已有 checkpoint 继续训练时，`--epochs` 表示总 epoch 预算，`--resume`
会恢复模型、优化器和历史记录：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke `
  --resume results\marblenet_vad_causal\last.pt `
  --causal --results-dir results\marblenet_vad_causal `
  --epochs 30 --batch-size 16
```

因果性检查可以直接运行：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\test_causal.py
```

上面的 smoke checkpoint 只用于验证训练链路、续训和流式实现，不是正式效果
评估；正式实验使用下一节的 medium 数据。

## Medium 正式训练

先按根目录 README 的命令生成 `LibriSpeech_{train,val,test}_medium.*`。训练
脚本的 `--train-rows` / `--val-rows` 不是 `--limit` 那种取前 N 行，而是
先按 `(noise_name, snr_db)` 分组，再在每组内确定性洗牌并轮询抽样：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\manifests\LibriSpeech_train_medium.tsv `
  --val-manifest data\librivad\manifests\LibriSpeech_val_medium.tsv `
  --data-root data\librivad --causal `
  --resume results\marblenet_vad_causal\last.pt `
  --results-dir results\marblenet_vad_causal_formal `
  --train-rows 8640 --val-rows 432 `
  --train-stride 4800 --val-stride 2400 `
  --epochs 100 --max-steps 28902 `
  --batch-size 128 --num-workers 0 `
  --warmup-ratio 0.03 --hold-ratio 0.25
```

`--epochs` 表示总预算；smoke checkpoint 停在零基 epoch 59，因此这条命令
从 epoch 60 继续并再训练 40 个 formal epoch。8640 条训练行在 0.63 s、
50% overlap 的切窗协议下展开为 90,514 个平衡窗口（45,257 speech /
45,257 silence），共 708 step/epoch。432 条验证行展开为 18,449 个窗口。

本机在 2026-09-16 完成了上述 100 epoch 正式训练。验证集最佳 checkpoint
为零基 epoch 93（日志中的 epoch 94/100）：

| 指标 | 验证集 |
| --- | ---: |
| loss | 0.15936 |
| accuracy | 0.93273 |
| AUROC | 0.98168 |

训练结束后使用同一组条件分层规则抽取 4320 条 medium 测试行：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\evaluate.py `
  --manifest data\librivad\manifests\LibriSpeech_test_medium.tsv `
  --data-root data\librivad `
  --checkpoint results\marblenet_vad_causal_formal\best.pt `
  --row-sample 4320
```

正式测试结果：

| 聚合级别 | accuracy | AUROC | TPR@FPR=0.315 |
| --- | ---: | ---: | ---: |
| sample-level | 0.90635 | 0.91246 | 0.93428 |
| LibriVAD frame-level（25 ms / 10 ms） | 0.90712 | 0.91302 | 0.93510 |

这是 LibriVAD-style medium 数据的确定性分层抽样测试，不是完整 14,148 条
测试集，也不是论文 AV A-speech 协议的分数。此前 smoke checkpoint 约
0.62 的 AUC 只用于验证 causal 前向、数据和续训链路，不能作为正式 VAD
指标。评估脚本使用流式混淆统计和 65,536 桶概率直方图累计全局 AUC，
避免把几十亿逐采样概率同时保存在内存中；结果 JSON 仍保留每条测试音频
的独立指标，便于按噪声和 SNR 复查。

## 滑窗评估

评估脚本默认读取 `best.pt`，使用 87.5% overlap 和 median smoothing：

```powershell
.\.venv\Scripts\python.exe reproductions\marblenet_vad\evaluate.py `
  --manifest data\librivad\smoke\manifests\LibriSpeech_test_small.tsv `
  --data-root data\librivad\smoke `
  --checkpoint results\marblenet_vad\best.pt `
  --limit 4
```

结果写入 `results/marblenet_vad/evaluation.json`，包括 sample-level 和
LibriVAD 25 ms/10 ms frame-level 的 accuracy、AUROC 和
`TPR@FPR=0.315`。`--smoothing mean` 可切换为论文比较过的均值滤波。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `model.py` | MarbleNet/Jasper 风格可分离卷积与残差块 |
| `features.py` | NeMo 对齐的 MFCC 前端 |
| `dataset.py` | LibriVAD TSV、逐采样标签、0.63 s 切窗 |
| `augment.py` | 波形时移/白噪声、SpecAugment、SpecCutout |
| `train.py` | 训练、验证、学习率调度与 checkpoint |
| `evaluate.py` | 87.5% 滑窗推理、median/mean 平滑与指标 |
| `test_causal.py` | 卷积、MFCC、流式缓存的因果性检查 |
| `test_evaluate.py` | 流式指标与 scikit-learn 参考实现的对照 |
