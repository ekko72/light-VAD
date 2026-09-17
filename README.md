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
│   └── librivad/    # LibriVAD 风格数据管线
├── reproductions/   # 论文/开源项目复现
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

## LibriVAD 风格数据管线（v2）

第二条数据线：用 LibriVAD 的 forced alignment 标签 + 官方 9 类噪声，替换 v1 的
RMS 门限标签与 MUSAN。两条线的中间产物互不覆盖，协议见
`data_protocol_librivad_v2.md`（v1 仍保留在 `data_protocol.md`）。

```powershell
# 1. 下载并校验上游数据（对齐包 + 噪声包，约 4GB）
python scripts\librivad\download.py

# 2. 生成 LibriVAD 风格数据（variant x split x 9 噪声 x 6 SNR）
python scripts\librivad\generate.py --size small

# 完整 medium 实验集（约 27.4 万条混合音频，单变体约 1 小时）
python scripts\librivad\generate.py `
  --size medium --variants LibriSpeech --splits train,val,test `
  --batch-size 2900 --progress-every 1000
python scripts\librivad\generate.py `
  --size medium --variants LibriSpeechConcat --splits train,val,test `
  --batch-size 2900 --progress-every 1000

# 只跑冒烟子集
python scripts\librivad\generate.py --size small --noises Babble_noise --snrs=-5,5

# 3. 逐条校验：重算标签、重混音频、比对 SNR / scale / hash / 选择
python scripts\librivad\verify.py `
  --manifest data\librivad\manifests\LibriSpeech_train_small.tsv `
  --check-selection

# 4. 与上游转录实现做协议一致性检查
python scripts\librivad\parity.py --samples 25
```

产物布局：

```text
data\librivad\
├── raw\          # 上游对齐与噪声（不入库）
├── generated\    # {variant}\{split_dir}\{noise}\{snr}\*.wav
├── labels\       # 逐采样点 int16 标签（每个干净样本一份）
└── manifests\    # {variant}_{split}_{size}.tsv / .json
```

评测沿用上游 25 ms 窗、10 ms 帧移、窗内多数投票；这与 v1 的 30 ms 中心帧协议
不同，两套指标不要混着比。

## MarbleNet 复现

`reproductions/marblenet_vad/` 提供 MarbleNet-3x2x64 的纯 PyTorch 复现，
不依赖 NeMo；模型、MFCC、数据切窗、训练配方和 87.5% 滑窗评估都在该目录。

```powershell
# 冒烟训练（复用 LibriVAD smoke 数据）
python reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke --causal `
  --results-dir results\marblenet_vad_causal `
  --epochs 2 --limit 8 --batch-size 16

# 从 causal checkpoint 续训（--epochs 是总预算）
python reproductions\marblenet_vad\train.py `
  --train-manifest data\librivad\smoke\manifests\LibriSpeech_train_small.tsv `
  --val-manifest data\librivad\smoke\manifests\LibriSpeech_val_small.tsv `
  --data-root data\librivad\smoke --causal `
  --results-dir results\marblenet_vad_causal `
  --resume results\marblenet_vad_causal\last.pt `
  --epochs 60 --batch-size 16

# 正式 medium 训练：按噪声/SNR 确定性分层抽样，避免只取 manifest 前几行
python reproductions\marblenet_vad\train.py `
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

# 滑窗评估
python reproductions\marblenet_vad\evaluate.py `
  --manifest data\librivad\manifests\LibriSpeech_test_medium.tsv `
  --data-root data\librivad `
  --checkpoint results\marblenet_vad_causal_formal\best.pt `
  --row-sample 4320
```

2026-09-16 的正式 medium 运行以零基 epoch 93 为最佳 checkpoint，验证
AUROC `0.98168`；4320 条分层测试的 sample-level AUROC 为 `0.91246`，
LibriVAD 帧级 AUROC 为 `0.91302`。训练产物写入
`results\marblenet_vad_causal_formal\`；完整结果、口径差异和参数说明见
`reproductions\marblenet_vad\README.md`。

## Difficulty-Adaptive Temporal Context（A0-A5）

`reproductions/difficulty_adaptive_context/` 检验 Long-RF 的收益是否集中在
Short-RF 不确定的困难帧。两套 causal MarbleNet 的参数量均为 89,154，仅使用
不同 dilation profile：Short 为 123 帧 / 1.23 s，Long 为 383 帧 / 3.83 s。

2026-09-16 完成的基础 30 轮和 40 轮等额续训结果显示：

| 模型 | Best epoch | Val AUROC | 测试 Delta F1（相对 Short） |
| --- | ---: | ---: | ---: |
| Short-RF | 39 | 0.95547 | - |
| Long-RF | 38 | 0.95526 | Clean +0.0048，-5 dB +0.0110 |

Long 的增益集中在 Short 模型最困难的置信度分桶：该桶 error reduction 为
`+4.709 pp`，整句聚类 bootstrap 的 95% CI 为 `[+3.165, +6.352] pp`。
其他难度桶没有一致收益，unseen 优势也主要由 SSN 驱动，因此当前证据支持
“困难帧更受益”，但还不足以证明应始终运行 Long-RF。

完整协议、A1-A5 结果、稳健性检查和限制见
`reproductions/difficulty_adaptive_context/README.md`。

## 里程碑对照

- 9 月：DL 补课 + 数据准备 + 训练骨架能跑通，WebRTC VAD 基线分数到手
- 10 月：第一个神经 VAD 指标超过能量 VAD 基线
- 11 月：瘦身到 <10K 参数 + 流式化 + INT8 量化不降档
- 12 月：专利交底 + 短文初稿 + 简历条目
