# 数据协议 v1

> 版本：v1（2026-09-10）  
> 适用范围：本项目的音频读取、标签、噪声混合、评测协议。标签规则一旦确定，后续不再随意修改；需要变更时升级版本并重新生成全部中间文件。

## 1. 音频参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 目标采样率 | 16000 Hz | LibriSpeech 已为 16k；MUSAN 在读取时即时重采样，不落盘 |
| 声道 | mono | 多声道音频读取时取平均 |
| 数值格式 | float32 | 归一化到 [-1, 1) 附近 |
| 帧长 | 480 samples = 30 ms | |
| 帧移 | 160 samples = 10 ms | |
| 帧数 | `(N - frame_len) // frame_shift + 1` | 不足一帧时 pad 到一帧 |

## 2. Ground Truth 标签

- 标签来源：clean speech 波形本身，噪声混合不改变标签。
- 先按帧计算 RMS。
- 阈值：`max(frame_rms) * 0.02`，低于该阈值的帧记为 silence。
- 对初步标签做 2 帧扩张（等价于约 20 ms hangover），减少边界突变。
- 输出为 uint8 一维数组：`1` = speech，`0` = non-speech。

## 3. 噪声混合

- 每种噪声裁剪或循环到与 speech 等长；采样率统一到 16k。
- SNR 定义：`SNR = 10 * log10(P_speech / P_noise)`，其中功率按整段混合长度计算。
- 噪声缩放系数：`scale = rms_speech / (rms_noise * 10^(snr/20))`。
- 混合后若峰值超过 0.99，整体等比缩放，不改变 speech/noise 比值。
- 训练阶段后续改为 on-the-fly augmentation，同一对 speech/noise 不再需要落盘。

## 4. 数据划分

### Speech

| 划分 | 来源 | 用途 |
| --- | --- | --- |
| train | train-clean-100 | 训练 |
| val | dev-clean | 验证 |
| test | test-clean | 测试 |

### Noise

- 训练组子类按文件级 70% / 15% / 15% 切为 train / val / test-seen，保证文件不重复。
- 以下子类整组进入 test-unseen，不出现在训练中：
  - `music/hd-classical`
  - `noise/sound-bible`
  - `speech/librivox`
- 其他已知训练组：
  - music: `fma`、`jamendo`、`rfm`
  - noise: `free-sound`
  - speech: `us-gov`
- test-seen 额外包含整组 `music/fma-western-art`。

## 5. 产物

```text
data/manifests/speech_manifest.tsv
data/manifests/musan_manifest.tsv
data/splits/train_speech.tsv
data/splits/val_speech.tsv
data/splits/test_speech.tsv
data/splits/train_noise.tsv
data/splits/val_noise.tsv
data/splits/test_seen_noise.tsv
data/splits/test_unseen_noise.tsv
data/splits/summary.json
```
