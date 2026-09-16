# 数据协议 v2：LibriVAD-style

> 版本：v2（2026-09-16）
> 适用范围：本项目的 LibriVAD 风格数据生成、标签、噪声混合与评测协议。
> 与 v1 的关系：v1（`data_protocol.md`）基于自建 RMS 门限标签 + MUSAN 噪声，仍然有效；
> v2 是并行的第二条数据线，改用 LibriVAD 的强制对齐标签与官方噪声集。两条线不混用中间产物。

## 1. 上游来源与固定版本

| 项目 | 值 |
| --- | --- |
| 代码仓库 | https://github.com/IoannisStylianou/LibriVAD |
| 固定提交 | `b39c051c08cf3f7b3d748a093484907b0e6e32ec` |
| 数据下载源 | `https://hf-mirror.com/datasets/LibriVAD/LibriVAD/resolve/main/Files` |
| 强制对齐包 SHA-256 | `742D5F68B46CF250AC9647398583AC6555B982BDCEDFA30E7D429A0C46982CFC` |
| 噪声包 SHA-256 | `4D6C5279BA628D8FC1895439033BE7FD06861CB1122B41E1696EA492D4E31BB3` |
| 噪声包大小 | `3,315,377,901` 字节（约 3.09 GiB） |

校验和记录在 `scripts/librivad/config.py`，下载脚本会逐字节比对；不一致直接报错。

## 2. 音频参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 采样率 | 16000 Hz | LibriSpeech 与 LibriVAD 噪声均为 16k，全流程不重采样 |
| 声道 | mono | 上游噪声文件本身是单声道 |
| 存储格式 | PCM_16 WAV | 生成音频落盘格式，与上游 `Results/` 一致 |
| 内存数值 | int16 / float64 | 混合时转 float64 计算，回写前统一缩放回 int16 |
| 标签格式 | int16 `.npy` | 逐采样点标签，`1` = speech，`0` = silence |

## 3. 数据划分

| 本项目 split | LibriSpeech 子集 | 用途 |
| --- | --- | --- |
| train | `train-clean-100` | 训练 |
| val | `dev-clean` | 验证 / 调参 |
| test | `test-clean` | 测试 |

噪声侧不切分文件：每个噪声类别都提供 `<noise>_train.wav`、`<noise>_val.wav`、
`<noise>_test.wav` 三条独立长音频，按 split 直接取对应文件，不存在训练/测试泄漏。

九个噪声类别（`NOISE_NAMES`）：

```text
Babble, SSN, Domestic, Nature, Office, Public, Street, Transport, City
```

### 规模档位

对排好序的干净样本列表做确定性步进抽样，不引入随机数：

| size | 步进 | 说明 |
| --- | --- | --- |
| small | `[::100]` | 冒烟 / 快速迭代 |
| medium | `[::10]` | 中等规模实验 |
| large | `[::]` | 全量 |

干净样本数（乘 9 噪声 × 6 SNR 之前）：

```text
small:  LibriSpeech 286/28/27;    Concat 143/14/14
medium: LibriSpeech 2854/271/262; Concat 1427/136/131
large:  LibriSpeech 28535/2703/2620; Concat 14267/1351/1310
```

顺序为 train/val/test。生成器记录的 `seed` 恒为 `0`：上游选择过程本身无随机性，
该字段只是显式声明「可复现」。

## 4. 标签规则

标签来自 LibriVAD 提供的 forced alignment（TextGrid 的 `words` tier），不是能量门限：

- 文本为空或 `None` 的区间 → `0`；其余 → `1`。
- 时间转采样点：`int(float32(time_s) * float32(16000))`，向上游一样截断而非四舍五入。
- 首段若为静音，`label[:end] = 0`；末段之后到文件结尾同样置 `0`。
- 噪声混合不改变标签：标签只由干净语音（或 Concat 波形）决定，因此每个干净样本
  只存一份 `.npy`，被该样本的所有噪声/SNR 行共用。

`unaligned.txt` 中列出的语音不参与任何划分。

## 5. Concat 变体

`LibriSpeechConcat` 把相邻两条 utterance 拼成一条更长的样本：

```text
[utterance A] + [中间静音] + [utterance B]
```

- 配对：按路径排序后两两相邻配对（奇数条丢弃最后一条），文件名规则与上游一致。
- 中间静音长度：`int((len(audio_A) + len(audio_B)) / 4)` 个采样点。
- 静音来源：`train-clean-100` 所有 utterance 的对齐静音区间，按排序顺序串成一条流。
- 静音增益：`0.001`（即 -60 dB），随后整体转 PCM16。
- 标签：`label_A + zeros(middle) + label_B`。

### 与上游的确定性差异

上游用多线程并发拼接静音池，取到哪一段取决于线程调度，因此同一份输入在不同机器上
可能得到不同的 Concat 波形。本项目保留「区间顺序」但改成**单线程按序消费**，
结果是确定的、可逐字节复现的。这是有意偏离，记录在此以便对照论文/上游数值时心里有数。

## 6. 噪声混合

- 噪声游标按 `(batch, noise, SNR)` 分组重置，在噪声文件上连续前进，读到末尾取模回绕。
- SNR 以**语音区**（`label == 1`）的 RMS 定义，不是整段 RMS：

```text
scale = rms(speech[label == 1]) / (rms(noise) * 10^(snr_db / 20))
output = scale_to_pcm16(speech + scale * noise)
```

- 目标 SNR 档位：`-5, 0, 5, 10, 15, 20` dB。
- `scale_to_pcm16` 只在峰值越界时整体等比缩放，不改变信噪比。
- manifest 同时记录目标 SNR、实测 SNR 和 `noise_scale`，验证脚本会独立重算比对。
- 每条生成音频带 `mix_hash`（SHA-256，覆盖来源、噪声起点、波形与标签），可做完整性抽查。

## 7. 输出结构

```text
data/librivad/
├── downloads/                       # 上游压缩包（不入库）
├── raw/
│   ├── Forced_alignments/librispeech_alignments/{split}/...
│   └── Noises/<noise>/<noise>_<split>.wav
├── generated/{variant}/{split_dir}/{noise}/{snr}/{utterance}.wav
├── labels/{variant}/{split_dir}/{utterance}.npy
└── manifests/
    ├── {variant}_{split}_{size}.tsv
    └── {variant}_{split}_{size}.json
```

`variant` ∈ {`LibriSpeech`, `LibriSpeechConcat`}，`split_dir` ∈
{`train-clean-100`, `dev-clean`, `test-clean`}。

manifest 关键列：

| 列 | 含义 |
| --- | --- |
| `sample_id` | `variant:split:noise:snr:utterance` |
| `sequence_index` / `batch_index` | 确定性选择序号与批号，用于复现游标状态 |
| `source_relative_path` | 干净语音路径（Concat 用 `\|` 连接两条） |
| `speech_utterance_ids` | utterance id（Concat 两条） |
| `clean_duration_s` / `speech_duration_s` / `silence_duration_s` | 时长统计 |
| `noise_name` / `noise_source` / `noise_start_sample` | 噪声来源与起始偏移 |
| `snr_db` / `measured_snr_db` / `noise_scale` | SNR 与缩放系数 |
| `label_relative_path` / `output_audio_path` / `output_frames` | 产物定位 |
| `seed` / `mix_hash` / `protocol_commit` | 可复现性字段 |

## 8. 评测协议

沿用上游 LibriVAD 的帧化规则，**与本项目 v1 / 同门 `voice-activity-detection-reproduction`
的 30 ms 中心帧协议不同**，两套指标不可直接比较：

| 参数 | 值 |
| --- | --- |
| 窗长 | 25 ms（400 采样点） |
| 帧移 | 10 ms（160 采样点） |
| 帧标签 | 窗内逐采样标签的多数投票 |
| 平滑 | 可选，论文侧常用中值/多数后处理 |

## 9. 常用命令

```powershell
# 下载并校验上游数据（对齐包 + 噪声包）
python scripts\librivad\download.py

# 只看确定性选择结果，不落盘
python scripts\librivad\generate.py --size small --dry-run

# 生成 small 规模的 LibriVAD 风格数据（9 噪声 × 6 SNR）
python scripts\librivad\generate.py --size small

# 生成 medium 完整实验集（train/val/test，约 27.4 万条混合音频）
python scripts\librivad\generate.py --size medium `
  --variants LibriSpeech --splits train,val,test
python scripts\librivad\generate.py --size medium `
  --variants LibriSpeechConcat --splits train,val,test

# 只生成指定噪声/SNR，用于快速冒烟
python scripts\librivad\generate.py --size small --noises Babble_noise --snrs -5,5

# 逐条校验：重算标签、重混音频、比对 SNR / scale / hash / 选择
python scripts\librivad\verify.py --manifest data\librivad\manifests\LibriSpeech_train_small.tsv --check-selection

# 与上游转录实现的协议一致性检查
python scripts\librivad\parity.py --samples 25
```

正式训练不要对完整 medium/large manifest 逐行展开窗口。训练脚本提供
`--train-rows` / `--val-rows`（评估为 `--row-sample`）：按
`(noise_name, snr_db)` 分组后确定性轮询抽样，保证 9 类噪声和 6 个 SNR
都进入训练/评估，同时控制本机内存、磁盘扫描和单 epoch 步数。

> PowerShell 传参注意：`--snrs -5,5` 这类以负号开头的值请写成 `--snrs=-5,5`。

冒烟数据（每份 manifest 8 行，共 48 行）生成在 `data/librivad/smoke/`，
已通过标签重建、混音字节比对、SNR、hash 与选择规则的全部校验。
