# Difficulty-Adaptive Temporal Context（A0-A5）

本实验检验的不是“Long-RF 是否整体优于 Short-RF”，而是一个更具体的问题：

> Long-RF 的收益是否集中在 Short 模型不确定、困难或接近边界的时间帧？

两套模型只改变五段 dilation profile，保持参数量、特征、标签、数据、损失、
优化器、训练轮数和随机种子一致。

## A0 Protocol

| 项目 | 设置 |
| --- | --- |
| 数据 | LibriVAD-style `LibriSpeech_{train,val,test}_medium.tsv` |
| 训练请求 / 有效 chunk | 8640 / 8286 |
| 验证请求 / 有效 chunk | 432 / 318 |
| 输入 | causal MFCC，64 维，25 ms 窗，10 ms 帧移 |
| 训练上下文 | 4.00 s |
| 目标片段 | 后 4.00 s |
| Batch size | 128 |
| 基础训练预算 | 30 epochs |
| 续训预算 | 从第 30 轮续到第 40 轮，两套模型相同 |
| 优化器 | SGD，momentum 0.9，weight decay 0.001 |
| 基础学习率 | 0.01 -> 0.001，Warmup-Hold-Decay |
| 续训学习率 | 固定 0.001 |
| Seed | 17 |
| 训练排除噪声 | `SSN_noise`, `Street_noise`, `Transport_noise` |

Short-RF 与 Long-RF 的配置：

| 模型 | Dilation profile | 参数量 | Causal | Receptive field |
| --- | --- | ---: | --- | ---: |
| Short-RF | `(1, 1, 1, 1, 1)` | 89,154 | 是 | 123 帧 / 1.23 s |
| Long-RF | `(1, 2, 3, 4, 4)` | 89,154 | 是 | 383 帧 / 3.83 s |

验证集固定；训练 chunk 起点每个 epoch 重采样，但 Short/Long 使用相同的
epoch 起点。基础结果保存在 `results/difficulty_adaptive_context/formal_*`，
40 轮续训结果保存在 `formal_*_ext40`，不会覆盖原始 checkpoint。

## 训练

下列命令可复现 Short。将 `short`、`formal_short` 分别替换为 `long`、
`formal_long` 即可得到对照模型。

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.train_context `
  --train-manifest data\librivad\manifests\LibriSpeech_train_medium.tsv `
  --val-manifest data\librivad\manifests\LibriSpeech_val_medium.tsv `
  --results-dir results\difficulty_adaptive_context\formal_short `
  --rf-profile short `
  --train-rows 8640 --val-rows 432 `
  --context-seconds 4 --target-seconds 4 `
  --batch-size 128 --num-workers 0 `
  --epochs 30 --seed 17 `
  --exclude-noise-names SSN_noise Street_noise Transport_noise
```

续训从各自第 30 轮的 `last.pt` 开始。`--epochs 40` 表示总 epoch 预算；
固定学习率确保两套模型只继续优化，不重新进入高学习率周期：

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.train_context `
  --train-manifest data\librivad\manifests\LibriSpeech_train_medium.tsv `
  --val-manifest data\librivad\manifests\LibriSpeech_val_medium.tsv `
  --results-dir results\difficulty_adaptive_context\formal_short_ext40 `
  --rf-profile short `
  --train-rows 8640 --val-rows 432 `
  --context-seconds 4 --target-seconds 4 `
  --batch-size 128 --num-workers 0 `
  --epochs 40 --seed 17 `
  --max-lr 0.001 --min-lr 0.001 `
  --warmup-ratio 0 --hold-ratio 0 `
  --exclude-noise-names SSN_noise Street_noise Transport_noise `
  --resume results\difficulty_adaptive_context\formal_short\last.pt
```

验证集最佳结果：

| 模型 | Best epoch | Val AUROC | Val F1 | Val error |
| --- | ---: | ---: | ---: | ---: |
| Short, 30 epochs | 30 | 0.95478 | 0.96291 | 6.608% |
| Long, 30 epochs | 27 | 0.95441 | 0.96271 | 6.606% |
| Short, extended 40 | 39 | 0.95547 | 0.96305 | 6.579% |
| Long, extended 40 | 38 | 0.95526 | 0.96273 | 6.617% |

## A1-A5 Evaluation

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.evaluate_context `
  --manifest data\librivad\manifests\LibriSpeech_test_medium.tsv `
  --short-checkpoint results\difficulty_adaptive_context\formal_short_ext40\best.pt `
  --long-checkpoint results\difficulty_adaptive_context\formal_long_ext40\best.pt `
  --row-sample 1080 --seed 17 `
  --output-dir results\difficulty_adaptive_context\formal_eval_ext40
```

本次评估得到 1024 条有效 utterance、553,532 个共同有效帧。结果文件：
`report.md`、`results.json`、`frame_predictions.npz`。

### A1 Overall Gain

| SNR | Short F1 | Long F1 | Delta F1 | Short error | Long error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.9623 | 0.9671 | 0.0048 | 6.257% | 5.509% |
| 20 | 0.9672 | 0.9673 | 0.0001 | 5.532% | 5.506% |
| 10 | 0.9559 | 0.9586 | 0.0027 | 7.496% | 7.018% |
| 5 | 0.9378 | 0.9484 | 0.0105 | 10.557% | 8.814% |
| 0 | 0.9264 | 0.9340 | 0.0076 | 12.609% | 11.313% |
| -5 | 0.9144 | 0.9254 | 0.0110 | 14.468% | 12.728% |

A1 的方向是正面的：六个条件的 Delta F1 都为正，低 SNR 的绝对误差下降
更明显。Clean 条件也受益，说明 Long-RF 不只是噪声平滑器。

### A2 Short-Model Difficulty Buckets

按 Short 模型置信度 `abs(p_short - 0.5)` 的全局五等分分桶：

| Short difficulty | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| Very hard | 110,707 | 32.261% | 27.552% | +4.709 pp |
| Hard | 110,706 | 9.551% | 9.843% | -0.293 pp |
| Medium | 110,704 | 2.144% | 2.272% | -0.128 pp |
| Easy | 110,702 | 0.251% | 0.265% | -0.014 pp |
| Very easy | 110,713 | 0.019% | 0.020% | -0.001 pp |

核心结论是“局部支持，而非单调递增”：只有 Very hard 桶有明显收益，
其余桶略微变差。不能声称 Long-RF 会随难度提升而单调改善全部帧。

Very hard 桶按 SNR 分解后：

| SNR | Frames | Error reduction | Utterance-clustered 95% CI |
| --- | ---: | ---: | ---: |
| Clean | 20,118 | +4.906 pp | [3.079, 6.902] pp |
| 20 | 10,211 | +0.735 pp | [-0.875, 2.306] pp |
| 10 | 13,843 | +3.309 pp | [1.202, 5.452] pp |
| 5 | 14,687 | +8.075 pp | [2.100, 14.483] pp |
| 0 | 17,767 | +5.662 pp | [1.141, 10.695] pp |
| -5 | 20,389 | +6.533 pp | [2.427, 10.890] pp |

除 20 dB 外，其余条件均为正；Clean 与多个中高 SNR 条件也有收益，因此
Very hard 桶的增益不能简单归因于低 SNR。按 utterance 聚类的 bootstrap
给出 Very hard error reduction 为 `4.709 pp`，95% CI
`[3.165, 6.352] pp`，不跨零。

### A3 SNR Trend

`Spearman(SNR rank, Delta F1) = -0.7714, p = 0.0724`。方向支持“噪声越强、
Long-RF 相对收益越大”，但只有六个 SNR 点，不能把它当作单条件下的强
统计证据。

### A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Error reduction |
| --- | ---: | ---: |
| 0-50 ms | 52,469 | +0.516 pp |
| 50-100 ms | 43,196 | +1.239 pp |
| 100-200 ms | 74,230 | +1.309 pp |
| >200 ms | 383,637 | +0.770 pp |

收益不是严格的边界局部效应：0-50 ms 并非最大，100-200 ms 的平均改善
反而最高，远处帧也有收益。

### A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| Seen | 305,307 | 0.9543 | 0.9547 | +0.099 pp |
| Unseen | 142,650 | 0.9221 | 0.9383 | +2.551 pp |

Unseen 减 Seen 的 error reduction 为 `+2.452 pp`，整句聚类 bootstrap 的
95% CI 为 `[+1.174, +3.820] pp`。但逐噪声结果并不一致：

| Noise | Error reduction |
| --- | ---: |
| SSN | +6.301 pp |
| Street | +0.450 pp |
| Transport | -0.242 pp |
| City | -0.037 pp |
| Domestic | -0.185 pp |

因此只能说 unseen 总体受益，主要驱动来自 SSN；不能声称所有 unseen noise
都获得一致提升。

## Robustness Analysis

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.analyze_context_robustness `
  --predictions results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz `
  --results results\difficulty_adaptive_context\formal_eval_ext40\results.json `
  --output-dir results\difficulty_adaptive_context\formal_eval_ext40\robustness `
  --bootstrap-repeats 1000 --seed 17
```

该脚本按 `source_key` 对整句聚类重采样，避免把同一句内高度相关的时间帧
当成独立样本。结果在 `robustness.md` 和 `robustness.json`。

## Phase A2 Confirmation

A1–A5 是探索性和单 seed 证据。正式确认使用独立的 speaker cohort：

1. 固定 speaker 划分，一半用于 calibration，一半用于 final test；
2. calibration 只用于选择 `confidence = abs(p_short - 0.5)` 的固定阈值；
3. final test 不重新调阈值，只报告自然产生的 activation rate；
4. 使用 speaker-level bootstrap，并比较 confidence gating 与同 activation
   budget 的 random gating；
5. 只有多 seed、CI、random 对照和 unseen-noise 四项同时通过，才记为
   `FORMAL_GO`。

当前 smoke 验证命令如下。正式运行时会继续追加 seed：

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.analyze_context_gate `
  --prediction seed17=results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz `
  --prediction seed23=results\difficulty_adaptive_context\seed23_eval_ext40\frame_predictions.npz `
  --prediction seed41=results\difficulty_adaptive_context\seed41_eval_ext40\frame_predictions.npz `
  --output-dir results\difficulty_adaptive_context\phase_a2_confirmatory `
  --primary-calibration-activation 0.10 `
  --activation-budgets 0.05 0.10 0.20 `
  --bootstrap-repeats 5000 --random-repeats 1000
```

`10%` 是 calibration 上的第一版固定操作点，不是要求 final test 必须达到
`10%`。如果 test 自然产生 `8%`、`13%` 或 `17%`，都按实际值报告。

输出包括 `phase_a2_confirmatory.md` 和
`phase_a2_confirmatory.json`，以及每个 seed 的 threshold、test activation、
correction、harm、signed net utility、speaker-cluster CI 和 random-gate
对照。

## Conclusion

当前结果支持以下有限结论：

1. Long-RF 相对 Short-RF 有稳定的整体增益，低 SNR 下更明显。
2. 增益集中在 Short 模型最不确定的 Very hard 帧；其他难度桶没有收益。
3. Very hard 增益不是纯粹的低 SNR 效应，但仍需多随机种子复核。
4. “只在边界受益”的假设不成立，增益分布较宽。
5. unseen 的总体优势主要来自 SSN，跨噪声类别并不一致。

这还不足以证明应无条件把所有帧都送入 Long-RF。当前已进入 Phase A2：
先确认固定 gate 在多 seed、speaker-cluster CI、random 对照和 unseen-noise
下是否稳定为正；确认后才进入 shared encoder + optional stateless
refinement 的架构实现。

## Smoke 与 Formal 的区别

- `smoke_*` 只验证 forward/backward、causal streaming、checkpoint 和
  evaluation 链路，不是正式指标。
- `formal_*` 使用 medium manifest、8640/432 请求规模、固定协议和
  `seed=17`。
- `formal_*_ext40` 是当前文档采用的 40 轮续训结果；原始 30 轮结果仍保留，
  可用于核对续训收益。
- `results/` 和所有 `.pt`、`.npz`、`.log` 都被 `.gitignore` 排除，不会提交
  大体积训练产物。

## Files

| 文件 | 内容 |
| --- | --- |
| `data.py` | 4 s context/label chunk、因果 MFCC frame bounds、确定性采样 |
| `train_context.py` | Short/Long 配对训练和续训 |
| `evaluate_context.py` | A1-A5 帧级评估与报告生成 |
| `analyze_context_robustness.py` | SNR 分层与 utterance-clustered bootstrap |
| `analyze_context_gate.py` | Phase A2 固定 gate、speaker split 与确认性统计 |
| `test_context.py` | RF、参数量、causal streaming、帧边界与采样测试 |
