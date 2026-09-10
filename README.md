# 轻量神经 VAD 项目（G 阶段）

对应路线：G（研一上 9-12 月）→ A'（研一下）→ D（逃生舱）。
本项目存放代码、脚本与数据，笔记仍放 Obsidian。

## 环境

- 位置：`C:\Users\20547\Desktop\论文p12\轻量VAD项目\.venv`
- Python 3.12.13（独立虚拟环境，不污染系统 Python）
- PyTorch 2.9.1 + cu128（GPU 版，RTX 5060 Laptop 已验证可用）

### 每次开始练习前激活环境

PowerShell 中执行：

```powershell
cd C:\Users\20547\Desktop\论文p12\轻量VAD项目
.\.venv\Scripts\Activate.ps1
```

激活后命令行前缀会变成 `(.venv)`，直接 `python xxx.py` 即可。
不想激活也可以全程用完整路径：

```powershell
C:\Users\20547\Desktop\论文p12\轻量VAD项目\.venv\Scripts\python.exe xxx.py
```

## 已安装组件（全部验证通过）

- 深度学习：torch / torchaudio / torchmetrics / einops / d2l / jupyter
- 音频处理：librosa / soundfile / webrtcvad-wheels
- 数值与可视化：numpy / scipy / matplotlib
- 部署相关：onnx / onnxruntime
- 工具：tqdm / pyyaml

## 常用命令

```powershell
# 1. 自检环境（确认 GPU、各库可用）
python scripts\verify_env.py

# 2. 下载公开数据（LibriSpeech 子集 + MUSAN，约 2.2GB，可选）
python scripts\download_data.py

# 3. 启动 Jupyter（学习 D2L / 实验用）
python -m jupyter lab
```

## 目录结构

```text
轻量VAD项目/
├── .venv/           # Python 虚拟环境（不要手动删）
├── scripts/         # 脚本（验证、下载、练习）
├── data/            # 数据集（LibriSpeech、MUSAN 等）
├── notes/           # 项目内临时笔记
└── README.md
```

## 数据说明

- LibriSpeech dev-clean / test-clean / train-clean-100：干净语音，用于训练/评测
- MUSAN：噪声与音乐库，用于合成带噪语音（压缩包约 10.5GB，下载脚本排在最后，按需保留）
- DNS Challenge 数据太大，先不下载；需要时再补脚本

## 里程碑对照

- 9 月：DL 补课 + 数据准备 + 训练骨架能跑通，WebRTC VAD 基线分数到手
- 10 月：第一个神经 VAD 指标超过能量 VAD 基线
- 11 月：瘦身到 <10K 参数 + 流式化 + INT8 量化不降档
- 12 月：专利交底 + 短文初稿 + 简历条目
