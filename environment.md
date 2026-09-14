# 环境快照（environment）

> 快照日期：2026-09-14
> 记录范围：G 阶段（轻量神经 VAD）开发环境的系统、硬件与依赖版本。
> 用途：复现实验环境、排查「本机可跑、换机报错」类问题。

## 1. 系统与硬件

| 项目 | 值 |
| --- | --- |
| 操作系统 | Windows 11（10.0.26200） |
| CPU | AMD64 Family 25 Model 117 Stepping 2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 显存 | 8 GB（8151 MiB） |
| 计算能力 | 12.0 |
| NVIDIA 驱动 | 591.86 |
| CUDA（torch 内置） | 12.8 |
| cuDNN | 9.10.2 |

## 2. Python 与虚拟环境

| 项目 | 值 |
| --- | --- |
| Python | 3.12.14（AMD64） |
| 虚拟环境 | 项目根目录下的 `.venv` |
| pip | 26.2.1 |
| torch 线程数 | 8 |

## 3. 核心依赖版本

以下版本与 `requirements.txt` 保持一致，是项目直接依赖。

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| torch | 2.9.1+cu128 | 训练 / 推理 |
| torchaudio | 2.9.1+cu128 | 音频加载与重采样 |
| torchmetrics | 1.9.0 | 评测指标 |
| einops | 0.8.2 | 张量重排 |
| d2l | 0.17.0 | 课程练习 |
| librosa | 1.0.0 | 特征提取（STFT / Mel） |
| soundfile | 0.14.0 | 读写 flac / wav |
| webrtcvad-wheels | 2.0.14 | WebRTC VAD 基线 |
| numpy | 2.5.2 | 数值计算 |
| scipy | 1.18.0 | 信号处理 |
| matplotlib | 3.11.1 | 可视化 |
| scikit-learn | 1.9.0 | 指标与工具 |
| onnx | 1.22.0 | 模型导出 |
| onnxruntime | 1.28.0 | ONNX 推理 |
| pyyaml | 6.0.3 | 配置读取 |
| tqdm | 4.70.0 | 进度条 |

## 4. 随依赖安装的工具包

非项目直接依赖，由 d2l / jupyter 等带入，版本仅作参考。

| 依赖 | 版本 |
| --- | --- |
| jupyterlab | 4.6.3 |
| notebook | 7.6.2 |
| ipykernel | 7.3.0 |
| pandas | 3.0.5 |

## 5. 未安装项

- `torchvision`：未安装，本项目不涉及图像任务。
- `onnxruntime-gpu`：未安装，当前 `onnxruntime` 仅有 `AzureExecutionProvider` 和 `CPUExecutionProvider`，ONNX 推理走 CPU。

## 6. 自检结果

自检命令：

```powershell
cd <项目根目录>
.\.venv\Scripts\python.exe scripts\verify_env.py
```

2026-09-14 运行结果：`ALL PASS`。关键输出：

```text
torch: 2.9.1+cu128, CUDA 可用: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
webrtcvad: 16 帧判定输出 [False x 16]
librosa: STFT (1025, 16), Mel (128, 16)
onnxruntime: 1.28.0, providers=['AzureExecutionProvider', 'CPUExecutionProvider']
GPU 矩阵乘法 OK
ALL PASS
```

## 7. 重新生成快照

依赖升级后，用下面命令导出冻结版本，再更新第 3、4 节：

```powershell
.\.venv\Scripts\python.exe -m pip list --format=freeze
```

同步更新 `requirements.txt` 后需重跑一次 `scripts\verify_env.py`，确认 `ALL PASS`。
