# A-ANALYSIS-v1 Claim 审计

## 1. 审计范围

本审计逐条检查 A-ANALYSIS-v1 论文打包中的核心结论。证据只来自已冻结的实验报告和结果文件，不重新运行实验、不调整阈值或路由，也不访问最终独立 OOD 数据。状态使用“支持”“部分支持”“不支持”“未复现”“未完成”等标准表述。M2 首次出现时定义为“基于可修复性感知的路由器”，在本文中仅作为条件性分析对象，不是主方法。

## 2. 核心 Claim 审计表

| Claim | 证据来源 | 状态 | 备注 |
| --- | --- | --- | --- |
| 长上下文的有效窗口约为 0.40 至 2.00 s | `results/difficulty_adaptive_context/a_analysis_v2_distance/distance_sweep_fine_report.md` | 支持（描述性） | 冻结规则找到一个包含 8 个相邻网格点的窗口；全距离确认性规则未通过，因此不能写成严格确认性门槛 |
| 来源替换比顺序打乱影响更大 | `mechanism_attribution_report.md`；`ava_speech_mechanism_results.csv` | 支持 | LibriVAD 上来源替换的正向效应明显，顺序打乱接近零或为负；AVA-Speech 方向一致，但只有 2 个 source cluster，区间较粗 |
| 3.00 s 后来源替换效应接近零 | `distance_sweep_fine_report.md`；`ava_speech_mechanism_results.csv` | 支持 | LibriVAD 上来源替换为 0.000420，95% CI 跨零；AVA-Speech 上两个视频的范围为 0.0016 至 0.0028，方向接近零 |
| 不同噪声类别、不同 SNR、不同录音来源的效应依次减弱 | `mechanism_attribution_report.md` | 部分支持 | d=1.00 s 的平均 Δlog-loss 排序为 0.143752、0.092778、0.060033；同实例换时间不明确，且 LibriVAD 不能单独识别噪声实例身份 |
| Long-RF 收益集中在难帧（LibriVAD） | `long_rf_by_window_report.md` | 支持 | 整体 proper-loss 改善为 0.032077，最难 20% 帧改善为 0.076483；距离分层高来源组在部分距离点上差异明显，但冻结窗口聚合的高/低来源对比不显著 |
| Long-RF 收益集中在难帧（AVA-Speech） | `ava_speech_main_results.csv`；`ava_speech_report.md` | 未复现 | 最难 20% 帧上，Long-RF 相对 Short 的 ΔF1 为 -0.0764，Δproper loss 为 +0.1685 |
| 512-byte MFCC 因果统计量可以部分替代 Long-RF | `long_term_stats_report.md`；`long_term_stats_results.csv` | 部分支持 | τ=0.50 s 和 2.00 s 有正向点估计，但仍与 Long-RF 存在差距；τ=1.00 s 的难帧收益为负；1 至 4 s 全范围未覆盖 |
| AVA-Speech 上的轻量适配器形成可用替代路径 | `ava_speech_lightweight_results.csv` | 部分支持 | 3 个 seed 的总体 proper loss 均值为 0.4635 ± 0.0371，但适配器在 AVA-Speech 上重新训练，不能声称跨域 zero-shot superiority 或与 Long-RF 等价 |
| M2 优于不确定性门控 | `e3_baseline_comparison/baseline_comparison_report.md` | 不支持 | 效用差值为 +0.00051674，source-cluster 95% CI [-0.00018877, 0.00132076]，区间跨零 |
| M2 优于最强简单基线 | `e3_baseline_comparison/baseline_comparison_report.md` | 不支持 | M2 相对 Shallow GBDT 的效用差值为 -0.00067777，95% CI [-0.00170659, 0.00013932]；点估计为负且区间跨零 |
| M2 具有跨域 superiority | `a_ext_val_voxconverse_v1/report.md`；`bootstrap.csv` | 不支持 | VoxConverse 上 ΔU 为 -0.0010571，95% CI [-0.0014702, -0.0006669]，5 个 seed 中 0 个为正 |
| AVA-Speech 复现来源驱动机制 | `ava_speech_mechanism_results.csv` | 支持 | 顺序打乱约 1 s 后消失，来源替换到 1 至 2 s 仍有影响，3 s 接近零；区间因只有 2 个 source cluster 而较粗 |
| AVA-Speech 复现 Long-RF 难帧收益 | `ava_speech_main_results.csv`；`ava_speech_report.md` | 不支持 | Long-RF 在难帧上反而更差，ΔF1 为 -0.0764，Δproper loss 为 +0.1685 |
| AVA-Speech 复现不确定性门控优势 | `ava_speech_main_results.csv`；`ava_speech_report.md` | 不支持 | Adaptive 不确定性门控相对随机门控的 ΔF1 为 -0.0039，Δproper loss 为 +0.0061，激活率为 0.0933 |
| 真实录音 A-ANALYSIS-v2 复现实验已完成 | `real_recording_distance_report.md` | 未完成 | 当时 AVA-Speech 和 VOiCES 不在数据树中，VOiCES speaker-overlap 检查无法执行；没有替换数据集或补做未冻结分析 |
| AVA-Speech 可以支持 speaker-disjoint 结论 | `ava_speech_report.md` | 不支持 | 官方 test split 只有 2 个视频，speaker-disjoint 评估未证明，报告明确不主张该结论 |
| 多 seed 汇总覆盖 Short、Long-RF、Adaptive | `a_analysis_v1/multiseed_results.csv`；`multiseed_report.md` | 部分支持 | Short 和 Long-RF 覆盖 seed 17、23、41；Adaptive 只有不同评估总体的 seed 17 结果，统一表中为 `not_available` |
| 主表覆盖全部计划模型与成本字段 | `a_analysis_v1/main_table.csv`；`main_table_report.md` | 部分支持 | 检测指标覆盖 Short、Long-RF、Adaptive、AlwaysRefine、Shallow GBDT、WebRTC、Silero、NeMo MarbleNet；WebRTC、Silero、NeMo、GBDT 的部分 MACs 或缓存字段为 `not_available` |
| M2 的负结果应作为论文边界条件保留 | `e3_baseline_comparison/baseline_comparison_report.md`；`a_ext_val_voxconverse_v1/report.md` | 支持 | 内部比较未支持优势，VoxConverse 外部验证点估计为负；不能把 M2 写成稳定主方法 |

## 3. 证据强度分层

### 3.1 支持的主结论

- 长上下文在有限距离窗口内具有可观测价值。
- 远端来源/环境信息的替换效应大于远端顺序打乱效应。
- LibriVAD 上 Long-RF 的收益集中在最难帧。
- 长时统计量可以部分缩小 Long-RF 与 Short 的差距。
- M2 在当前冻结证据下没有形成稳定的部署优势。

### 3.2 需要条件化表述的结论

- 0.40 至 2.00 s 是描述性有效窗口，不是严格确认性门槛。
- 来源驱动机制在 AVA-Speech 上方向一致，但 source-level 不确定性很粗。
- 轻量适配器在 LibriVAD 上有部分收益，在 AVA-Speech 上需要重新训练，不能声称跨域等价。
- Long-RF 的难帧收益在 LibriVAD 上成立，在 AVA-Speech 上未复现。

### 3.3 不支持的结论

- M2 优于不确定性门控。
- M2 优于最强简单基线。
- M2 具有跨域 superiority。
- AVA-Speech 复现 Long-RF 难帧收益。
- AVA-Speech 复现不确定性门控优势。
- AVA-Speech 可以支持 speaker-disjoint 结论。

## 4. 术语与表述审计

正式报告和本审计表统一使用标准统计与学术表述：正向但未在最终独立 OOD 上确认的趋势、置信区间跨零的条件性正向效应、外部验证未支持 superiority 且点估计为负、决策改变型可修复性净比例（P(R) - P(H)）、价值可观测性受限、可修复价值稀缺。内部状态标签不再作为结论语言。M2 作为条件性分析对象的名称保留，但已在首次出现时给出定义和角色说明。

## 5. 未解决问题

- A-ANALYSIS-v1 的全距离确认性规则未通过，有效窗口只能作描述性主张。
- LibriVAD 每个噪声类别只有一个噪声源文件，噪声实例身份无法单独识别。
- AVA-Speech 官方 test split 只有 2 个 video/source cluster，且未证明 speaker-disjoint。
- A-ANALYSIS-v2 预注册真实录音复现实验因数据不可用而未完成。
- 长时统计量只覆盖 τ=0.50、1.00、2.00 s；3、4 s 以及 1 至 4 s 全范围为 `not_available`。
- 多 seed 统一表中 Adaptive 为 `not_available`。
- AVA-Speech 主表中 WebRTC、Silero、NeMo MarbleNet、Shallow GBDT 的部分 MACs 或缓存字段为 `not_available`。
- 两个冻结报告中的 Long-RF 缓存口径分别为 106,496 bytes 和 103,936 bytes，未在本打包中合并。
- VoxConverse 外部验证为负结果，必须与内部 E3 结果同时保留。
