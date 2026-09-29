# Difficulty-Adaptive Temporal Context（A0-A13）

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

## A9 Temporal Span-Value-Cost Sweep

A9 固定 Short encoder、feature tap、refinement channels/blocks、kernel 5、
训练协议、seed 和 gate threshold，只改变 causal refinement 的 dilation
profile。各 profile 精确对应下列 lookback：

| RF span | Dilation profile | Lookback |
| ---: | --- | ---: |
| 64 | `(1, 2, 4, 8, 16)` | 0.64 s |
| 128 | `(1, 2, 4, 8, 32)` | 1.28 s |
| 256 | `(1, 2, 4, 8, 64)` | 2.56 s |
| 384 | `(1, 2, 4, 8, 96)` | 3.84 s |
| 512 | `(1, 2, 4, 8, 128)` | 5.12 s |

本次 seed 17 扫描复用已有 seed 17 RF64 与 RF384 checkpoints，其余 span
按相同协议重新训练和评估：

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.run_a9_span_sweep `
  --mode all --spans 64 128 256 384 512 --seeds 17 `
  --reuse-checkpoint "17:64:results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025\best.pt" `
  --reuse-checkpoint "17:384:results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384\best.pt"
```

固定阈值 `0.13` 下的主结果：

| RF | Adaptive F1 | Refine-only F1 | Activation | Correction/selected | Harm/selected | Net/selected | Net/frame |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.94447 | 0.94444 | 5.021% | 14.228% | 11.737% | 2.490% | 0.125% |
| 128 | 0.94636 | 0.94685 | 5.021% | 20.029% | 11.831% | 8.199% | 0.412% |
| 256 | 0.94635 | 0.94700 | 5.021% | 20.497% | 11.864% | 8.633% | 0.433% |
| 384 | 0.94637 | 0.94696 | 5.021% | 20.657% | 11.797% | 8.860% | 0.445% |
| 512 | 0.94588 | 0.94630 | 5.021% | 19.148% | 12.064% | 7.084% | 0.356% |

对应成本：

| RF | MACs/selected frame | MACs/call | Cache bytes | CPU median ms/call | CPU p95 ms/call |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3,456 | 221,184 | 67,072 | 5.329 | 7.166 |
| 128 | 3,456 | 221,184 | 99,840 | 7.338 | 9.613 |
| 256 | 3,456 | 221,184 | 165,376 | 14.372 | 16.266 |
| 384 | 3,456 | 221,184 | 230,912 | 16.474 | 18.670 |
| 512 | 3,456 | 221,184 | 296,448 | 33.202 | 39.415 |

RF64 和 RF384 的新评估与旧 `fixed013` 基准在 F1、activation、selected
和 net utility 上完全一致。`U384-U64 = +6.369 pp`；17 个 SNR/噪声/seen
条件单元中有 14 个方向为正，5 个 SNR 点全部为正，9 个噪声类别中 7 个
为正。Clean、Nature 和 Transport 为负，因此这不是“所有 domain 都一致
改善”的结论。

曲线呈现清楚的“最小有效跨度 + 饱和”：64→128 从 `2.490%` 跳到
`8.199%`，128→256→384 只剩小幅增长，512 回落到 `7.084%`，同时 CPU
中位延迟约增至 RF384 的两倍。按文档的 A9 条件，这支持 mechanism-level
GO；RF384 是当前 A10 的候选跨度，不在 A9 阶段冻结最终架构。

这里的三条 cost 指标是固定输入、batch 1、single CPU thread 的 isolated
refinement-call microbenchmark，不是完整 streaming RTF。真实 conditional
compute、router overhead 和端到端 Pareto 结论留给 A11。

完整汇总位于 `results/difficulty_adaptive_context/a9_span_sweep/`：
`a9_span_sweep.md`、`a9_span_sweep.csv`、`a9_span_sweep.json` 和
`a9_span_sweep.png`。

## A10 Temporal Value Predictability

A10 检验一个比“帧是否困难”更具体的问题：能否只用 Short 模型可见的因果
统计量，预测送入 Long refinement 后的 signed marginal value
`v_t = 1[Refiner correct] - 1[Short correct] in {-1, 0, +1}`。

分析固定 A9 的 RF384 prediction bundle，使用
`posterior_entropy`、`posterior_change`、25-frame `short_term_variance` 和
`prediction_switch_rate`。Short hidden-embedding delta 不在冻结 bundle
中，因此没有使用。价值模型是只含 calibration speakers 的三分类 logistic
regression，不带 MLP；阈值按 calibration activation budget 选择，test
只报告自然 activation。主预算为 5%，同时扫描 2%、10% 和 20%：

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.analyze_a10_value_predictability
```

冻结预测集上的 signed value 分布：

| Split | Frames | +1 Correction | 0 No effect | -1 Harm | E[v] | Correction/frame | Harm/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 255,232 | 3,203 | 250,119 | 1,910 | 0.005066 | 1.255% | 0.748% |
| test | 298,300 | 3,981 | 291,924 | 2,395 | 0.005317 | 1.335% | 0.803% |

5% calibration budget 下的 test gate comparison：

| Gate | Test activation | Correction/selected | Harm/selected | Net/selected | Net/frame | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncertainty | 5.050% | 26.427% | 15.899% | 10.528% | 0.532% | 0.94638 |
| value logistic | 5.009% | 26.645% | 16.030% | 10.615% | 0.532% | 0.94646 |
| random mean | 5.006% | 1.336% | 0.804% | 0.532% | 0.027% | 0.94379 |

Value 减 uncertainty 的 net/selected 为 `+0.087 pp`，但 paired
speaker-cluster 95% CI 为 `[-0.055, +0.258] pp`，跨过零。2%、5%、10%
budget 下 value 的点估计略高，20% 时 value 的 net/selected 为 `2.416%`，
低于 uncertainty 的 `2.478%`。因此这里只能说明 value router 有很小的
方向性信号，不能证明它显著优于 uncertainty。

10x10 的 entropy-variance calibration value map 中，最高有限 bin 的
`E[v] = 0.074092`，位于高 entropy、高 short-term variance 区域，包含
10,460 个 calibration frames。由于 `+1/-1` 事件稀少，大多数低值 bin 的
均值恰好为 0；这张图适合作为机制检查，不足以单独作为阈值依据。

A10 状态为 **UNCERTAINTY_PROXY**：方向条件通过，但 paired CI 未通过，因此
保留简单的 uncertainty gate 作为当前低成本的诚实代理。这不是 A-wide
NO-GO，也不否定 A9 的 mechanism-level GO。

完整结果位于 `results/difficulty_adaptive_context/a10_value_predictability/`：
`a10_value_predictability.md`、`a10_value_predictability.json`、
`a10_temporal_value_map.csv`、`a10_temporal_value_map.png` 和
`a10_gate_comparison.png`。

## A11 Conditional Compute Realization

A11 检验 shared Short encoder 在真实 CPU streaming 路径上的条件计算收益。
计时从缓存 causal MFCC 开始，batch=1、single CPU thread、chunk=1 frame，
每个 operating point 使用独立进程和固定 CPU affinity；每个 utterance
先 warm-up 200 frames。主测试为 8 条按 clean/noise/SNR 条件平衡选择的
utterance，共 4,389 frames。阈值只由 calibration-speaker Short scores
选择，test activation 不强制匹配 budget。

```powershell
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.benchmark_a11_conditional_compute
```

正式结果：

| Operating point | Test activation | Benchmark activation | F1 | Mean ms/frame | P95 ms/frame | Analytical MACs/frame (test act.) | Cache bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 0.000% | 0.000% | 0.94363 | 2.195 | 3.523 | 87,232 | 34,304 |
| RF64 AlwaysRefine | 100.000% | 100.000% | 0.94444 | 3.052 | 4.633 | 90,688 | 67,072 |
| RF384 AlwaysRefine | 100.000% | 100.000% | 0.94696 | 3.156 | 4.675 | 90,688 | 230,912 |
| Adaptive 2% | 1.983% | 2.689% | 0.94505 | 2.003 | 3.395 | 87,301 | 230,912 |
| Adaptive 5% | 5.050% | 5.423% | 0.94638 | 2.269 | 3.634 | 87,407 | 230,912 |
| Adaptive 10% | 10.317% | 10.276% | 0.94693 | 2.346 | 3.798 | 87,589 | 230,912 |
| Adaptive 20% | 21.459% | 20.141% | 0.94696 | 2.457 | 3.962 | 87,974 | 230,912 |

主预算 5% 满足文档预先定义的强信号：

`F1_Short < F1_Adaptive < F1_RF384`，同时
`T_Short < T_Adaptive < T_RF384`。Adaptive 5% 的平均延迟为
3.156 -> 2.269 ms/frame，相对 AlwaysRefine 降低 `28.1%`，不是
near-equivalent，因此 A11 的
`NO_GO_EFFICIENCY_OVERHEAD` 失败门槛没有被触发。它恢复了
`(0.94638-0.94363)/(0.94696-0.94363)=82.6%` 的 Long-vs-Short F1
增益，观察到的 benchmark activation 为 `5.423%`。解析 MACs 从
AlwaysRefine 的 90,688 降到 87,407，但 wall-clock 的收益也包含 gate、
history concatenation、dispatch 和 cache 的实现成本；这些成本已经按
`C_avg = C_S + C_G + r C_R + C_scheduling` 单独记录。

该结果只支持 conditional compute 在 CPU latency 与解析 MACs 上的收益。
Adaptive 仍保留 RF384 的完整 streaming history，所以 cache bytes 与
AlwaysRefine 相同，均为 230,912 bytes，不能据此声称内存也按 activation
比例下降。RTF 0.227 描述的是 cached-feature streaming boundary，不包含
MFCC 提取和麦克风采集，也不是完整 audio-to-decision RTF。

A11 状态为 **STRONG_GO**：当前 shared-encoder conditional branch 在真实
CPU 上兑现了 Pareto 收益。限制是 8 条 utterance、每个 utterance 单次
计时、未固定 CPU frequency；高预算点的局部顺序可能受计时噪声影响，主
结论只使用预算 5% 的严格三阶比较。

完整结果位于 `results/difficulty_adaptive_context/a11_conditional_compute/`：
`a11_conditional_compute.md`、`a11_conditional_compute.json`、
`a11_conditional_compute.csv` 和 `a11_accuracy_latency_pareto.png`。

## A12 Routing Robustness

A12 在冻结的 RF384 bundle 和校准阈值 `0.13` 下检查 proposal gate 的
分布外稳健性，测试集为 298,300 frames。SSR sweep 使用
`10/30/50/70/90%`，声学单元格覆盖 seen/unseen 与
`20/10/5/0/-5 dB`。任何 test cell 都不重新选择阈值。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.run_a12_routing_robustness
```

关键结果：

| Cell | Activation | Net utility / selected | F1 |
| --- | ---: | ---: | ---: |
| SSR 10%, all | 12.778% | 19.457% | 0.40309 |
| SSR 50%, all | 8.931% | 20.218% | 0.83806 |
| SSR 90%, all | 4.736% | 22.581% | 0.95638 |
| SSR 90%, seen | 3.604% | -1.156% | 0.96098 |
| Seen, 10 dB | 4.623% | 7.073% | 0.95850 |
| Seen, 0 dB | 5.910% | 3.359% | 0.94088 |
| Seen, -5 dB | 5.533% | -0.925% | 0.93660 |
| Unseen, -5 dB | 10.606% | 20.489% | 0.86308 |

30 个主要 development cells 中有 2 个原始点估计未通过预声明门槛：
seen 的 `90% SSR` 和 `-5 dB`。简单 5% source-rank stabilizer 把两项
都修到正的点估计（`+2.941%` 和 `+0.244%`），但两项修复后的
speaker-cluster CI 仍跨零。主要问题是 selected utility，不是激活失控：
最大 raw activation 为 `14.208%`，低于 `15%` warning 和 `20%`
failure 门槛。

CPU 列是基于 A11 实测 5% endpoint 的解析投影
`base_ms + activation * refinement_ms_per_selected_frame`，不是新的
硬件 benchmark。A12 状态为 **GO_WITH_SOURCE_RANK_STABILIZER**，但这
只是 conditional router robustness，不是 strong GO：修复只通过点估计，
两个 CI 仍包含零。该状态不足以单独支撑最终的 Strong GO 声明。

完整结果位于 `results/difficulty_adaptive_context/a12_routing_robustness/`：
`a12_routing_robustness.md`、`a12_routing_robustness.json`、
`a12_routing_robustness.csv` 和 `a12_routing_robustness.png`。

## A13 Streaming and Segment Audit

A13 只检查 streaming 与分段行为，不重新研究 C 的 boundary supervision，
也不改变 A12 的 condition-level routing 判定。审计内容分为两部分：

- 相同声学前缀接不同未来时，MFCC、padding、Short speech probability、
  refinement residual 和 gate 前缀是否保持一致；
- batch isolation、常见 chunk boundary、full-sequence 等价性、eval-mode
  normalization、recurrent hidden state 与 causal frontend 配置是否成立。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m reproductions.difficulty_adaptive_context.run_a13_streaming_audit
```

Causality audit 固定使用 CPU。GPU 在 chunked 与 full-sequence 卷积路径上
会选用不同的 backend kernel，产生约 `1e-3` 的非确定性数值差；CPU 上可
直接检查实现等价性，不把算子误差误报成未来帧泄漏。正式审计中，
same-prefix/different-future 的 MFCC、probability、residual 和 gate 差异
均为零；batch isolation 最大差为 `1.19e-7`，chunk 与 full sequence 的
最大 logits 差为 `1.43e-6`，gate mismatch 为 0。因此 causality status
为 **PASS**。

Segment metrics 使用固定 posterior decision threshold `0.5`，short speech
定义为 `<=20` frames（200 ms），onset tolerance 为 20 frames。评估集为
298,300 test frames、492 utterances：

| Method | Short recall | Whole-run miss | Onset median | Offset median | Clipping | False activation | Transitions/1k | Runs <=2 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 92.593% | 10 | 3.00 | 2.00 | 165.560 s | 115.120 s | 18.894 | 25.832% |
| Refine-only | 100.000% | 2 | 2.00 | 2.00 | 150.580 s | 114.240 s | 19.930 | 26.729% |
| Adaptive | 98.148% | 5 | 3.00 | 2.00 | 155.300 s | 112.110 s | 19.360 | 25.957% |

Adaptive 相对 Short 把 short-speech recall 从 `92.593%` 提高到
`98.148%`，whole-run misses 从 10 降到 5，clipping duration 减少
`10.260 s`，false activation duration 减少 `3.010 s`。onset 和 offset
median 没有恶化；transitions 增加 `0.466/1k frames`，runs `<=2` frames
增加 `0.125 pp`，均在预声明容差内。seen 组达到 `100%` short-speech
recall 和 0 次 whole-run miss；unseen 组从 Short 的 `89.474%` 提高到
`94.737%`，whole-run misses 从 8 降到 5。Refine-only 仍是 diagnostic
ceiling，Adaptive 没有完全达到它的 `100%` recall，但未触发任何分段
非回归门槛。

A13 状态为 **PASS**，说明当前 selective refinement 在 CPU streaming
路径上没有发现未来帧泄漏，也没有发现明显的 onset/offset、短语音召回、
clipping、false activation 或 decision-flicker 退化。该 PASS 只覆盖
streaming/segment diagnostic，不修复 A12 已发现的 seen-condition routing
失败，因此整体仍不能称为最终 Strong GO。

完整结果位于 `results/difficulty_adaptive_context/a13_streaming_audit/`：
`a13_streaming_audit.md`、`a13_streaming_audit.json`、
`a13_segment_metrics.csv` 和 `a13_segment_metrics.png`。

## Conclusion

当前结果支持以下有限结论：

1. Long-RF 相对 Short-RF 有稳定的整体增益，低 SNR 下更明显。
2. 增益集中在 Short 模型最不确定的 Very hard 帧；其他难度桶没有收益。
3. Very hard 增益不是纯粹的低 SNR 效应，但仍需多随机种子复核。
4. “只在边界受益”的假设不成立，增益分布较宽。
5. unseen 的总体优势主要来自 SSN，跨噪声类别并不一致。
6. A9 显示 temporal refinement value 在 128 frames 开始明显兑现，
   128→384 接近饱和，512 的准确率收益回落而成本继续增长。
7. A10 的 signed-value logistic 在 5% budget 下略高于 uncertainty，但
   paired CI 跨零且 20% budget 时回落；当前没有充分证据用独立 value
   router 替换简单 uncertainty gate。
8. A11 在 single-thread CPU、batch=1、chunk=1 frame 的 cached-MFCC
   streaming 路径上通过 Strong GO：Adaptive 5% 同时满足 F1 与 latency
   的严格三阶排序，并将平均延迟相对 AlwaysRefine 降低 28.1%。
9. A12 的固定 gate 在两个主要 development cells 上出现非正 utility；
   5% source-rank stabilizer 能将两项修到正的点估计，但修复后的
   speaker-cluster CI 仍跨零。A12 因此为 conditional
   `GO_WITH_SOURCE_RANK_STABILIZER`，不能单独支撑最终 Strong GO。
10. A13 在 CPU streaming 路径上通过 same-prefix/different-future、
    batch、chunk、gate 和 cache 等价性审计；Adaptive 相对 Short 也通过
    分段非回归检查。A13 的 PASS 只说明当前机制没有发现 streaming 或
    segment-level 明显退化，不覆盖 A12 条件级 router robustness 仍偏弱
    的事实。

这还不足以证明应无条件把所有帧都送入 Long-RF。Phase A2 已确认固定 gate
在多 seed、speaker-cluster CI、random 对照和 unseen-noise 下稳定为正；
A3-A9 已完成 shared encoder + optional stateless refinement 的跨度机制
验证。A9 结论是 mechanism-level GO；A10 进一步表明 uncertainty 仍是当前
可保留的低成本 value proxy，而不是直接冻结最终架构；A11 则证明该 branch
在 CPU latency 与 MACs 上已经形成有效 Pareto 点，但 cache 仍保持 RF384
级占用。A12 进一步显示，pooled accuracy 和 pooled utility 都不能替代
seen/unseen 条件级检查：当前 absolute posterior gate 在部分已见噪声条件
下选择的帧平均为负收益；source-rank stabilizer 只在点估计层面修复这些
条件，CI 仍跨零，因此 A14 的 untouched OOD 和多 seed 验证仍然必要。

## Limitations and Final-OOD Closure

A 实验阶段已经关闭，不再进行 final confirmation 或后续搜索：

`FINAL_OOD_STATUS=NOT_EXECUTABLE_NO_UNUSED_CONFIRMATORY_DATASET`

`FINAL_OOD_OUTCOME_EXPOSED=false`

`SCIENTIFIC_BLINDNESS_PRESERVED=true`

`A_v2_STATUS=CONDITIONAL_GO`

`M2_FINAL_OOD_SUPERIORITY=UNRESOLVED`

`A_EXPERIMENTATION_COMPLETE=true`

`NEXT_EXPERIMENT_AUTHORIZED=false`

本研究不存在预先保留且未使用的 confirmatory final-OOD dataset。冻结的
final-OOD evaluator 在 runtime preflight 中发现绑定 manifest 的
`candidate_dataset` 为 `null`，且已有 LibriSpeech test speakers 已用于
A-v1 evaluation。执行因此 fail-closed，未加载 confirmatory 数据，未运行
模型，也未产生 final-OOD frame、source、aggregate 或 subgroup outcome。

因此，final untouched-OOD superiority claim 未被检验。该状态既不能解释为
支持 M2，也不能解释为否定 M2；A-v2 的正式状态仍为 `CONDITIONAL_GO`。
不得从已有 development/OOD 数据重新划分或挑选 `NEW_FINAL_OOD`，不得为
final confirmation 新建、筛选或调参数据集，不得重跑 A14/A-v2 OOD，也不得
继续 M2/router/threshold/backbone/RF search 或启动 E5。

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
| `train_adaptive.py` | 训练 sparse shared-encoder refinement；支持 A9 RF span |
| `evaluate_adaptive.py` | 评估 fixed gate、refine-only ceiling 与单次成本 |
| `run_a9_span_sweep.py` | 运行、汇总和绘制 A9 span/value/cost sweep |
| `analyze_a10_value_predictability.py` | 拟合 calibration-only value score，比较 gate、random 与 activation budget |
| `benchmark_a11_conditional_compute.py` | 隔离进程测量 A11 activation、延迟分布、RTF、MACs、cache、RSS 与 Pareto 图 |
| `run_a12_routing_robustness.py` | 冻结阈值下的 SSR/acoustic gate robustness、source-rank stabilizer、成本投影与 conditional GO 判定 |
| `run_a13_streaming_audit.py` | causal prefix/chunk audit、segment metrics、flicker 诊断与 A13 报告 |
| `test_context.py` | RF、参数量、causal streaming、帧边界与采样测试 |
| `test_adaptive_model.py` | RF span、sparse refinement 与 streaming 单元测试 |
| `test_run_a9_span_sweep.py` | A9 runner、GO 判定和报告单元测试 |
| `test_analyze_a10_value_predictability.py` | A10 value、bootstrap、value map 与报告单元测试 |
| `test_benchmark_a11_conditional_compute.py` | A11 阈值隔离、指标、延迟、开销拆分、GO 判定和报告单元测试 |
| `test_run_a12_routing_robustness.py` | A12 bundle、rank stabilizer、成本投影、声学单元格与判定单元测试 |
| `test_run_a13_streaming_audit.py` | A13 segment metrics、因果前缀、chunk、阈值与判定单元测试 |
