# A-ANALYSIS-v1 长上下文价值分析最终报告

## 1. 摘要

本文研究长上下文在语音活动检测（VAD）中的价值来源、有效范围以及是否存在更便宜的替代方案。冻结的 LibriVAD 距离干预实验表明，长上下文的主要收益来自远端来源与环境信息，而不是远端帧的时间顺序：在约 0.40 至 2.00 s 的描述性有效窗口内，替换远端来源持续提高 log-loss，而打乱远端顺序的效应接近零或为负；到 3.00 s，来源替换效应在冻结分析中已接近零。LibriVAD 上 Long-RF（长感受野因果 VAD，约 3.83 s 感受野）的 proper loss 低于 Short（短感受野因果 VAD，约 1.23 s 感受野），整体改善为 0.0321，最难 20% 帧改善为 0.0765，说明收益集中在困难帧。一个仅增加 512 bytes 缓存的 MFCC 因果滑动统计 FiLM 适配器能够部分缩小与 Long-RF 的差距，但没有证明与 Long-RF 等价。

外部验证给出了更严格的边界。AVA-Speech 上，来源替换与顺序打乱的相对方向仍与 LibriVAD 一致，但 Long-RF 的难帧收益没有复现：在最难 20% 帧上，Long-RF 相对 Short 的 ΔF1 为 -0.0764，Δproper loss 为 +0.1685。基于可修复性感知的路由器 M2（条件性分析对象，不是本文主方法）相对不确定性门控的效用增量为 +0.00051674，95% CI 跨零；相对最强简单基线 Shallow GBDT 的效用增量为 -0.00067777，95% CI 同样跨零。VoxConverse 外部验证中，M2 相对不确定性门控的 ΔU 为 -0.0010571，95% CI 完全为负，5 个 seed 均未取得正向结果。总体结论是：长上下文在有限窗口内具有可观测价值，主要来自来源信息；但现有方法尚未把这一机制稳定转化为真实场景中的性能优势。

## 2. 引言

长上下文 VAD 通常被假定能够利用更长的语音、噪声和环境历史来改善决策。然而，上下文变长并不自动意味着收益增加：远端的语音内容可能已经与当前帧无关，真正有用的是录音条件、噪声类别、信噪比或环境来源等较慢变化的统计量。因此，本文不把长上下文模型视为天然更优的方法，而是回答三个问题：

1. 长上下文在哪里有用：收益是否集中在特定距离窗口和困难帧；
2. 为什么有用：远端上下文提供的是时间顺序信息，还是来源与环境信息；
3. 能否用更便宜的方式获得同样收益：轻量长时统计量是否可以替代 Long-RF 的缓存和计算成本。

本文的主要贡献如下。

- 给出有限有效窗口的描述性证据：来源替换效应在 0.40 至 2.00 s 内可见，3.00 s 附近接近零；远端顺序打乱没有显示同等效应。
- 给出来源驱动机制证据：不同噪声类别、不同 SNR 和不同录音来源的替换效应依次减弱，而同实例换时间的对照不明确。
- 在 LibriVAD 上复现难帧收益：Long-RF 相对 Short 的整体 proper-loss 改善为 0.0321，最难 20% 帧改善为 0.0765。
- 评估轻量替代方案：512 bytes 的 MFCC 因果滑动均值/方差 FiLM 适配器在 LibriVAD 上部分追回收益，但不能声称跨域等价。
- 给出方法评估的负结果：M2 未优于不确定性门控或最强简单基线，VoxConverse 外部验证未支持 superiority。
- 在 AVA-Speech 上做外部分布验证：来源驱动机制方向一致，但难帧收益和不确定性门控优势均未复现。

## 3. 实验设置

### 3.1 数据

| 数据集 | 划分与用途 | 规模 | 统计聚类 | 备注 |
| --- | --- | ---: | ---: | --- |
| LibriVAD | 官方 test split，用于距离扫描、机制归因、难帧分析和多 seed 汇总 | 553,532 帧；470,043 语音帧；83,489 静音帧 | 197 个 source cluster | 使用 LibriSpeech_test_medium.tsv；未访问最终独立 OOD 数据 |
| AVA-Speech | 官方 test split，用于真实录音外部分布验证 | 60,002 帧；32,231 语音帧；27,771 静音帧 | 2 个 video/source cluster | 官方 test 视频为 xO4ABy2iOQA 和 y7ncweROe9U；未证明 speaker-disjoint |
| VoxConverse | 冻结的会议录音外部验证 | 7,224,444 帧 | 216 个 recording cluster | 仅用于评估冻结的 M2 与不确定性门控，不做训练或阈值调整 |

LibriVAD 机制实验覆盖 Babble、SSN 和 Domestic 噪声类别。距离干预使用完整因果 MFCC 切片，干预目标固定为 512 个。AVA-Speech 机制结果仅覆盖 2 个官方 test 视频，因此 source-level 不确定性较粗。

### 3.2 模型

| 模型 | 定义 |
| --- | --- |
| Short | 短感受野因果 VAD，约 1.23 s 感受野 |
| Long-RF | 长感受野因果 VAD，约 3.83 s 感受野 |
| Adaptive | 不确定性门控自适应细化模型 |
| AlwaysRefine | 始终细化的诊断模型，用作计算上界 |
| WebRTC VAD | WebRTC 内置 VAD 基线 |
| Silero VAD | Silero 预训练 VAD 基线 |
| NeMo MarbleNet | NeMo 预训练 MarbleNet VAD 基线 |
| Shallow GBDT | 浅层梯度提升树基线 |
| M2 | 基于可修复性感知的路由器；仅作为条件性分析对象，不是本文主方法 |

### 3.3 指标

- F1：固定阈值下的检测 F1。
- AUC：全帧排序指标。
- proper loss：二分类对数损失，用于评估概率预测质量。
- Δlog-loss：干预前后 log-loss 的平均变化；正值表示干预使模型变差。
- utility：在固定预算下，选择细化后正确性变化带来的平均效用。
- MACs/frame、streaming cache、latency：计算、缓存和延迟成本。
- causal output：输出是否只依赖当前及历史帧。

### 3.4 统计

LibriVAD 距离扫描和机制归因使用 source/utterance cluster bootstrap 95% CI。AVA-Speech 使用 video/source cluster bootstrap 95% CI。VoxConverse 使用 recording cluster paired bootstrap 95% CI。多 seed 汇总使用 seed 均值和样本标准差；未根据结果挑选 seed。对于只有 2 个 source cluster 的 AVA-Speech，区间宽度受聚类数量限制，不能等同于 speaker-disjoint 或独立录音总体上的强不确定性估计。

### 3.5 数据来源

本报告只打包已有结果，不重新训练模型、不修改阈值或路由，也不打开最终独立 OOD 数据。主要来源文件如下。

| 内容 | 来源 |
| --- | --- |
| 细粒度距离扫描 | `results/difficulty_adaptive_context/a_analysis_v2_distance/distance_sweep_fine_report.md` |
| 机制归因 | `results/difficulty_adaptive_context/a_analysis_v2_distance/mechanism_attribution_report.md` |
| LibriVAD 难帧收益 | `results/difficulty_adaptive_context/a_analysis_v2_distance/long_rf_by_window_report.md` |
| 长时统计量实验 | `results/difficulty_adaptive_context/a_analysis_v2_distance/long_term_stats_report.md` 与 `long_term_stats_results.csv` |
| E3 最强简单基线比较 | `results/difficulty_adaptive_context/e3_baseline_comparison/baseline_comparison_report.md` |
| VoxConverse 外部验证 | `results/difficulty_adaptive_context/a_ext_val_voxconverse_v1/report.md`、`per_seed.csv`、`bootstrap.csv` |
| AVA-Speech 主结果 | `results/difficulty_adaptive_context/ava_speech_v1/ava_speech_report.md` 与 `ava_speech_main_results.csv` |
| AVA-Speech 机制结果 | `results/difficulty_adaptive_context/ava_speech_v1/ava_speech_mechanism_results.csv` |
| AVA-Speech 轻量替代方案 | `results/difficulty_adaptive_context/ava_speech_v1/ava_speech_lightweight_results.csv` |
| 多 seed 汇总 | `results/difficulty_adaptive_context/a_analysis_v1/multiseed_results.csv` 与 `multiseed_report.md` |
| AVA-Speech 主表 | `results/difficulty_adaptive_context/a_analysis_v1/main_table.csv` 与 `main_table_report.md` |

## 4. 主要结果

### 4.1 有限有效窗口

细粒度距离扫描使用冻结的 LibriVAD 测试清单、冻结的评估 mask 和冻结的 Long-RF 检查点。表 1 给出具有代表性的距离点；完整曲线覆盖 0.10 至 3.00 s。

| d (s) | 来源替换 Δlog-loss，95% CI | 顺序打乱 Δlog-loss，95% CI | 同句换时间 Δlog-loss，95% CI |
|---: | ---: | ---: | ---: |
| 0.40 | 0.056278 [0.032060, 0.080867] | -0.005424 [-0.011406, 0.000399] | 0.004682 [-0.001536, 0.011279] |
| 0.50 | 0.050488 [0.026960, 0.074521] | -0.007764 [-0.014249, -0.001142] | 0.001980 [-0.003874, 0.007857] |
| 1.00 | 0.060033 [0.034628, 0.084084] | -0.004347 [-0.008774, 0.000398] | 0.003648 [-0.001264, 0.008351] |
| 2.00 | 0.021132 [0.009701, 0.032631] | -0.001564 [-0.003318, 0.000107] | 0.000044 [-0.002555, 0.002375] |
| 3.00 | 0.000420 [-0.000066, 0.000888] | 0.000007 [-0.000067, 0.000077] | -0.000116 [-0.000224, -0.000013] |

冻结规则在 0.40 至 2.00 s 找到一个包含 8 个相邻网格点的描述性有效窗口。在该窗口内，来源替换的下 95% CI 均大于零，顺序打乱的上 95% CI 均不超过 0.0005。到 3.00 s，来源替换效应降至 0.000420，95% CI 跨零，说明远端来源信息在该距离上已经不可稳定观测。

需要同时报告一个严格的边界：A-ANALYSIS-v1 的全距离确认性规则没有通过。0.25、1.00 和 2.00 s 的主要问题是对顺序效应的置信区间精度要求过严，而不是出现了较大的顺序效应；3.00 s 则是来源效应本身接近零。因此，0.40 至 2.00 s 应表述为描述性有效窗口，而不是已经由确认性门槛认证的普遍窗口。

AVA-Speech 的两个 test 视频给出同方向的曲线：顺序打乱效应在约 1 s 后消失，来源替换效应在约 1 至 2 s 仍可见，3 s 附近接近零。由于每个视频只有一个 source cluster，表内区间退化为点值，因此该结果只作为方向一致性证据，不作为独立统计显著性证据。

### 4.2 来源驱动机制

在 LibriVAD 的冻结有效窗口内，匹配条件下的平均 Δlog-loss 排序为：不同噪声类别（匹配 SNR）最大，其次为同类别不同 SNR，再次为同类别不同录音来源，同实例换时间不明确。表 2 给出 d=1.00 s 的代表性结果。

| 条件 | 平均 Δlog-loss | 95% CI | 解释 |
| --- | ---: | ---: | --- |
| different_class_matched_snr | 0.143752 | [0.117486, 0.169448] | 正向效应 |
| different_snr_same_class | 0.092778 | [0.075800, 0.110540] | 正向效应 |
| same_class_different_recording | 0.060033 | [0.035443, 0.084008] | 正向效应 |
| same_instance_other_time | 0.003648 | [-0.000888, 0.008262] | 不明确 |

这一排序说明，远端上下文的价值主要与来源、噪声类别和 SNR 等条件变量相关，而不是与远端帧的排列顺序相关。需要保留一个识别边界：LibriVAD 每个噪声类别只有一个噪声源文件，同录音换时间条件同时改变语音和噪声时间段，因此该实验支持录音/来源、噪声类别和 SNR 对比，但不能单独识别“噪声实例身份”这一因素。

AVA-Speech 上，两个 test 视频的平均 Δlog-loss 方向与 LibriVAD 一致：来源替换大于顺序打乱，顺序效应约 1 s 后消失，来源效应到 1 至 2 s 仍有正值，3 s 附近接近零。该结果支持机制方向的外部一致性，不支持 speaker-disjoint 或更大真实录音总体上的强泛化结论。

### 4.3 Long-RF 收益集中在难帧

在 LibriVAD 上，Long-RF 相对 Short 的 proper loss 改善定义为 Short proper loss 减去 Long-RF proper loss。

| 评估群体 | Long-RF 相对 Short 的 proper-loss 改善 |
| --- | ---: |
| 全部帧 | 0.032077 |
| 最难 20% 帧 | 0.076483 |

难帧上的改善约为整体改善的 2.4 倍。距离分层分析进一步显示，d=1.00 s 时高来源替换四分位组的收益为 0.173515，低来源替换四分位组为 -0.076342，高减低的差值为 0.249857，95% CI [0.144517, 0.365112]。不过，按冻结窗口聚合后，高减低差值为 -0.000375，95% CI [-0.013787, 0.015075]；窗口外聚合差值为 -0.006798，95% CI [-0.019270, 0.005752]。因此，更稳妥的表述是：难帧收益在 LibriVAD 上成立，距离分层的高来源组差异在部分距离点上明显，但冻结窗口聚合的高/低来源对比并不显著。

AVA-Speech 没有复现这一结论。最难 20% 帧上，Long-RF 相对 Short 的 ΔF1 为 -0.0764，Δproper loss 为 +0.1685，即 Long-RF 在该外部分布上反而更差。该负结果必须与 LibriVAD 的正结果同时报告。

### 4.4 轻量替代方案

LibriVAD 上评估了一个只增加 512 bytes 缓存的轻量适配器：在冻结 Short 骨干上加入 MFCC 因果指数滑动均值与方差，并通过 FiLM 注入第一个 128 通道编码器块。该实验只训练 FiLM 适配器，不改变骨干、阈值、门控或路由。表 4 给出 3 个 seed 的汇总。

| 时间常数 τ (s) | seeds | F1 | AUC | proper loss | 相对 Short 的 proper-loss 降低 | 与 Long-RF 的差距 | 最难 20% 帧相对 Short 的收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 3 | 0.9499 | 0.9498 | 0.2001 | 0.0093 | 0.0034 | 0.0304 |
| 1.00 | 3 | 0.9474 | 0.9478 | 0.2082 | 0.0012 | 0.0115 | -0.0030 |
| 2.00 | 3 | 0.9502 | 0.9512 | 0.2010 | 0.0084 | 0.0043 | 0.0242 |

该结果支持“512-byte 长时统计量可以部分追回 Long-RF 收益”的弱结论。τ=0.50 s 和 τ=2.00 s 在整体 proper loss 和难帧收益上均给出正向点估计，但不同 seed 间波动明显，且所有设置仍与 Long-RF 存在非零差距。现有冻结结果覆盖 τ=0.50、1.00、2.00 s；协议中提到的 1 至 4 s 全范围尚未完整覆盖，3 和 4 s 为 `not_available`。

AVA-Speech 上的轻量适配器使用 512 bytes 额外缓存，Long-RF 参考缓存为 103,936 bytes。表 5 给出 3 个 seed 的总体与难帧结果。

| seed | 总体 F1 | 总体 AUC | 总体 proper loss | 难帧 F1 | 难帧 AUC | 难帧 proper loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.8292 | 0.8969 | 0.4228 | 0.6231 | 0.7836 | 0.5782 |
| 23 | 0.7624 | 0.8845 | 0.4721 | 0.4582 | 0.7805 | 0.5238 |
| 41 | 0.7501 | 0.8775 | 0.4956 | 0.3576 | 0.7520 | 0.5474 |
| 均值 ± 标准差 | 0.7806 ± 0.0426 | 0.8863 ± 0.0098 | 0.4635 ± 0.0371 | 0.4796 ± 0.1341 | 0.7721 ± 0.0174 | 0.5498 ± 0.0273 |

AVA-Speech 的轻量适配器是在该数据集上重新训练的，因此这些数值不能解释为跨域 zero-shot superiority，也不能单独证明轻量方法优于 Long-RF。它们只说明：在真实录音分布上，小缓存适配器可以形成可训练的正向替代路径，但跨域等价性仍未建立。

### 4.5 M2 方法评估

M2 是基于可修复性感知的路由器，仅作为条件性分析对象。E3 比较使用固定 5% 预算和冻结的内部测试集，主指标为 utility。

| 比较 | 效用差值 | source-cluster 95% CI | 结论 |
| --- | ---: | ---: | --- |
| M2 - 不确定性门控 | +0.00051674 | [-0.00018877, 0.00132076] | 区间跨零，未支持 M2 优于不确定性门控 |
| M2 - Shallow GBDT | -0.00067777 | [-0.00170659, 0.00013932] | 点估计为负，区间跨零，未支持 M2 优于最强简单基线 |

Shallow GBDT 的 utility 为 0.00632648，M2 的 utility 为 0.00564871。Oracle gate 的 utility 为 0.01361742，仅作为诊断上界，不是可部署方法。

VoxConverse 外部验证使用冻结的 5% 预算和 recording cluster paired bootstrap。M2 相对不确定性门控的 ΔU 为 -0.0010571，95% CI [-0.0014702, -0.0006669]，5 个 seed 中 0 个为正；5% 平均激活率为 0.0554929。按标准统计表述，该外部验证未支持 superiority，点估计为负。固定预算和激活率约束本身通过，但不能抵消主效用对比的负结果。

### 4.6 AVA-Speech 外部验证

AVA-Speech 主结果来自官方 test split 的 overall 条件。表 7 给出检测指标，表 8 给出成本与因果性。

| 模型 | F1，95% CI | AUC，95% CI | proper loss，95% CI |
| --- | ---: | ---: | ---: |
| Short | 0.7414 [0.7220, 0.7585] | 0.8009 [0.7563, 0.8586] | 0.7795 [0.7554, 0.8035] |
| Long-RF | 0.7435 [0.7166, 0.7683] | 0.7874 [0.7631, 0.8414] | 0.7562 [0.7074, 0.8051] |
| Adaptive | 0.7366 [0.7074, 0.7622] | 0.7998 [0.7560, 0.8556] | 0.7924 [0.7752, 0.8096] |
| AlwaysRefine | 0.7322 [0.6986, 0.7618] | 0.7950 [0.7598, 0.8390] | 0.8525 [0.8489, 0.8561] |
| WebRTC VAD | 0.7139 [0.6884, 0.7361] | 0.6216 [0.5912, 0.6545] | 10.0039 [9.8842, 10.1237] |
| Silero VAD | 0.8455 [0.8167, 0.8658] | 0.9347 [0.9302, 0.9377] | 0.4715 [0.4376, 0.5053] |
| NeMo MarbleNet | 0.7943 [0.7166, 0.8457] | 0.9121 [0.8987, 0.9161] | 0.6003 [0.5377, 0.6630] |
| Shallow GBDT | 0.8114 [0.7790, 0.8343] | 0.8810 [0.8695, 0.8877] | 0.4291 [0.4235, 0.4348] |

| 模型 | MACs/frame | streaming cache (bytes) | latency (ms/frame) | causal output |
| --- | ---: | ---: | ---: | --- |
| Short | 45312 | 34304 | 0.0224 | 是 |
| Long-RF | 45312 | 103936 | 0.0213 | 是 |
| Adaptive | 45634.60684643845 | 230912 | 0.0339 | 是 |
| AlwaysRefine | 48768 | 230912 | 0.0300 | 是 |
| WebRTC VAD | `not_available` | `not_available` | `not_available` | 是 |
| Silero VAD | `not_available` | `not_available` | 0.2479 | 是 |
| NeMo MarbleNet | `not_available` | `not_available` | 0.0342 | 是 |
| Shallow GBDT | `not_available` | `not_available` | 0.0507 | 是 |

AVA-Speech 机制结果与 LibriVAD 的方向一致：顺序打乱效应约 1 s 后消失，来源替换到 1 至 2 s 仍有影响，3 s 接近零。两个 test 视频的数值范围如下。

| d (s) | 顺序打乱 Δlog-loss 范围 | 来源替换 Δlog-loss 范围 |
|---: | ---: | ---: |
| 0.25 | 0.0885 至 0.1095 | 0.8655 至 1.3807 |
| 0.50 | 0.0095 至 0.0425 | 0.6490 至 0.9987 |
| 1.00 | -0.0021 至 -0.0019 | 0.4646 至 0.6775 |
| 2.00 | -0.0127 至 -0.0087 | 0.1100 至 0.1678 |
| 3.00 | -0.0008 至 -0.0005 | 0.0016 至 0.0028 |

AVA-Speech 的两个主要负结果如下。

- Long-RF 难帧收益未复现：最难 20% 帧上，Long-RF 相对 Short 的 ΔF1 为 -0.0764，Δproper loss 为 +0.1685。
- 不确定性门控优势未复现：Adaptive 不确定性门控相对随机门控的 ΔF1 为 -0.0039，Δproper loss 为 +0.0061，激活率为 0.0933。

AVA-Speech 官方 test split 只有 2 个 video/source cluster，source-level 区间较粗；speaker-disjoint 评估未被证明，因此不能把该结果表述为 speaker-disjoint 外部验证。所有结果均来自冻结模型和冻结阈值，没有根据 AVA-Speech 结果调整阈值或路由。

## 5. 讨论

### 5.1 机制发现的意义

距离扫描和机制归因共同指向一个有限、来源驱动的解释：长上下文并非主要提供远端时间顺序，而是提供录音来源、噪声类别、SNR 和环境统计等较慢变化的信息。这一解释与以下现象一致：

- 来源替换造成较大的 log-loss 增加，而顺序打乱接近零；
- 不同噪声类别和不同 SNR 的替换效应大于同类别不同录音来源；
- 同实例换时间的对照不明确；
- 3.00 s 附近来源替换效应接近零。

因此，长上下文的价值更适合被描述为“有限窗口内的环境/来源信息”，而不是“越长越好的时序建模”。

### 5.2 方法失败的原因分析

M2 的负结果与机制发现并不矛盾。即使长上下文在困难帧上具有价值，部署一个路由器还需要同时满足两个条件：可修复价值不能过于稀疏，且路由决策所需信息必须在推理时稳定可观测。E3 中 M2 与不确定性门控的差异跨零，M2 相对 Shallow GBDT 的点估计为负；VoxConverse 外部验证中，M2 相对不确定性门控的区间完全为负。这说明当前方法没有把机制价值稳定转化为可部署的决策优势。

AVA-Speech 的结果进一步说明，跨域泛化比内部测试更困难。来源驱动机制在方向上可以跨数据集保持一致，但 Long-RF 的难帧收益和不确定性门控优势没有同时复现。真实录音上的收益可能依赖训练分布、聚类结构和校准方式，不能仅由内部测试的机制结果外推。

### 5.3 对 VAD 研究的启示

长上下文研究应区分三类问题：

1. 模型是否能够从远端历史中获得信息；
2. 这些信息是否在当前帧决策上具有可修复价值；
3. 是否存在更便宜、更稳定的统计量来传递同样的信息。

本文的证据支持第一类问题的有限正向答案，但没有支持第二类问题的部署级正向答案，也没有证明第三类问题已经有跨域等价的解决方案。更合理的研究方向是显式建模来源/环境统计，而不是单纯扩大感受野或增加路由器复杂度。

### 5.4 局限性与未解决问题

- A-ANALYSIS-v1 的全距离确认性规则未通过。0.40 至 2.00 s 有效窗口是描述性结果，不是严格确认性门槛。
- LibriVAD 每个噪声类别只有一个噪声源文件，无法单独识别噪声实例身份。
- AVA-Speech 官方 test split 只有 2 个 video/source cluster，且未证明 speaker-disjoint。
- A-ANALYSIS-v2 预注册的真实录音复现实验因当时 AVA-Speech 和 VOiCES 不在数据树中而未运行；后续 AVA-Speech-v1 使用独立冻结协议提供外部分布验证。
- 长时统计量实验只覆盖 τ=0.50、1.00、2.00 s；1 至 4 s 全范围和 3、4 s 结果不可用。
- 多 seed 汇总中，Adaptive 没有与 Short 和 Long-RF 相同的 553,532 帧统一评估总体；Adaptive 在统一表中为 `not_available`。
- AVA-Speech 主表中，WebRTC、Silero、NeMo MarbleNet 和 Shallow GBDT 的 MACs 或 streaming cache 缺失，统一写为 `not_available`。
- 两个冻结报告中的 Long-RF 缓存口径分别为 106,496 bytes 和 103,936 bytes；本报告不合并这两个口径，引用时保留各自来源。
- VoxConverse 外部验证的主效用对比为负；该结果应作为条件性路由方法的边界证据，而不是可忽略的补充材料。

## 6. 结论

长上下文在有限窗口内具有可观测价值，主要来自来源和环境信息，而不是远端帧的时间顺序。LibriVAD 上 Long-RF 的收益集中在困难帧，512-byte 长时统计量可以部分缩小差距；但 AVA-Speech 没有复现难帧收益或门控优势，VoxConverse 也没有支持 M2 的跨域 superiority。因此，现有证据支持重新定位长上下文 VAD 的价值分析，但不支持把条件性路由方法写成稳定的主方法。

