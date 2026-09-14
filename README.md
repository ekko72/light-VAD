# 轻量神经 VAD 项目（G 阶段）

对应路线：G（研一上 9-12 月）→ A'（研一下）→ D（逃生舱）。
本项目存放代码、脚本与数据，笔记仍放 Obsidian。

## 环境

- 位置：项目根目录下的 `.venv`
- Python 3.12.14（独立虚拟环境，不污染系统 Python）
- PyTorch 2.9.1 + cu128（GPU 版，RTX 5060 Laptop 已验证可用）
- 完整环境快照与依赖版本：`environment.md`

### 每次开始练习前激活环境

PowerShell 中执行：

```powershell
cd <项目根目录>
.\.venv\Scripts\Activate.ps1
```

激活后命令行前缀会变成 `(.venv)`，直接 `python xxx.py` 即可。
不想激活也可以全程用完整路径：

```powershell
.\.venv\Scripts\python.exe xxx.py
```

## 已安装组件（全部验证通过）

- 深度学习：torch / torchaudio / torchmetrics / einops / d2l / jupyter
- 音频处理：librosa / soundfile / webrtcvad-wheels
- 数值与可视化：numpy / scipy / matplotlib
- 部署相关：onnx / onnxruntime
- 工具：tqdm / pyyaml / python-docx

## 常用命令

```powershell
# 1. 自检环境（确认 GPU、各库可用）
python scripts\verify_env.py

# 2. 下载公开数据（LibriSpeech 子集 + MUSAN，约 2.2GB，可选）
python scripts\download_data.py

# 只下载需要的条目（可选；默认 8 连接并发分块下载）
python scripts\download_data.py --only train-clean-100,musan

# 3. 启动 Jupyter（学习 D2L / 实验用）
python -m jupyter lab
```

## 目录结构

```text
light-VAD/
├── .venv/           # Python 虚拟环境（不要手动删）
├── scripts/         # 脚本（验证、下载、练习）
├── data/            # 数据集（LibriSpeech、MUSAN 等）
├── notes/           # 项目内临时笔记
└── README.md
```

`nicklashansen/voice-activity-detection` 的独立复现已拆分到同级目录
`..\voice-activity-detection-reproduction`，不再占用本项目代码目录。

## 数据说明

- LibriSpeech dev-clean / test-clean / train-clean-100：干净语音，用于训练/评测
- MUSAN：噪声与音乐库，用于合成带噪语音（压缩包约 10.5GB，全部下载约 16.6GB，下载脚本排在最后）
- DNS Challenge 数据太大，先不下载；需要时再补脚本

## 数据预处理

一次性生成 manifest、固定划分、标签抽查和 SNR 混合验证：

```powershell
# 1. 生成 speech/noise manifest 与 train/val/test 划分
python scripts\prepare_manifests.py

# 2. 随机抽查 10 条语音 + 10 条噪声，输出明细表、拼接音频和波形图
python scripts\inspect_audio.py

# 3. 抽查 10 条语音的帧标签并出图
python scripts\generate_labels.py

# 4. 生成 5 x 3 组 SNR 混合样例并校验实际 SNR
python scripts\mix_noise.py
```

协议固定在 `data_protocol.md`，中间文件在 `data\manifests`、`data\splits`、`data\validation`。

> 自定义 SNR 列表时，PowerShell 请用等号传参，例如
> `python scripts\mix_noise.py --snrs=-10,0,10`。

`inspect_audio.py` 的拼接音频由 3 秒片段加 0.3 秒间隔组成，便于一次听完全部抽查条目；
加 `--play` 可直接播放（Windows），`--count` 和 `--seed` 控制抽查条数与随机性。

## 里程碑对照

- 9 月：DL 补课 + 数据准备 + 训练骨架能跑通，WebRTC VAD 基线分数到手
- 10 月：第一个神经 VAD 指标超过能量 VAD 基线
- 11 月：瘦身到 <10K 参数 + 流式化 + INT8 量化不降档
- 12 月：专利交底 + 短文初稿 + 简历条目
