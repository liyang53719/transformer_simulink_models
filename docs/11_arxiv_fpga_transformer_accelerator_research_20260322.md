# arXiv FPGA Transformer/LLM 加速器调研与对当前架构的借鉴

更新时间：2026-03-22

## 调研目标
- 补充阅读 arXiv 上与 Transformer/LLM 在 FPGA 上加速相关的公开论文，重点看 attention、prefill/decode、流水线切分、片上缓存、混合精度、长上下文和 edge 部署。
- 将公开论文中的共性方法映射到当前仓库的第一层 block Simulink 主线问题上，明确下一步优先修改哪些架构，而不是继续做无方向的局部试错。

## 当前问题背景
- 当前 canonical builder 在 [scripts/implement_stage1_rmsnorm_qkv.m](scripts/implement_stage1_rmsnorm_qkv.m) 中已经完成了一轮 attention 分母链路的最小修正，例如 `head_group_norm` 分数定点化、`score_tile_bias` 去掉 `tile_k`。
- 真实 tracked SLX 审计表明：`attn_out` 已经相对接近，但从 `ffn_gate_mul` 到 `residual_out` 误差仍显著放大。
- 当前 builder 顶层 prefill 主线仍是 `attention_u -> ffn_swiglu_u -> residual_u` 的 placeholder 串接，尚未显式实现 `attn_residual_out` 与 `post_attn_norm_out` 这两个真实 block stage。
- 因此接下来的判断标准不应只是“某个常数能否再降一点误差”，而是“当前 stage 边界是否表达了真实 block 的硬件友好语义”。

## 本轮补充阅读的论文

### 1. A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA
- arXiv: https://arxiv.org/abs/2603.05931
- 关键词：persistent state、decode memory-bound、五阶段流水、on-chip state
- 核心思路：把 decode 阶段的大状态常驻片上 BRAM，避免每 token 往返 HBM；通过五阶段 dataflow pipeline 把准备、计算、写回重叠起来。
- 对我们的借鉴：decode 和 prefill 不是同一种访存/算术问题，不能用一套完全相同的 stage 语义去解释两者；调度常量应服务于 dataflow，不应直接混入算术归一化。

### 2. FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill
- arXiv: https://arxiv.org/abs/2602.20515
- 关键词：prefill、dynamic sparse attention、memory-aware execution order、dual-tier cache
- 核心思路：prefill 在长上下文下会转为 memory-bound，需要 fused pipeline 和 memory-aware 执行顺序来减少 irregular access，并用双层缓存降低 KV 访问压力。
- 对我们的借鉴：prefill attention 的 stage 切分要围绕数据活性和缓存复用，而不是单纯围绕数学公式分段；同理，我们当前 block 里 residual add 和 post-attn norm 也应视为明确的 stage 边界。

### 3. FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design
- arXiv: https://arxiv.org/abs/2601.15710
- 关键词：stage-customized inference、prefill/decode hybrid、quantization suite、long-context plug-in
- 核心思路：prefill 与 decode 采用不同的 temporal reuse / spatial dataflow；通过可组合 HLS 库暴露这些自由度，而不是强行共享一个固定模板。
- 对我们的借鉴：builder 内部应允许 stageProfile 驱动真实的 stage 结构差异，但这种差异应落在 pipeline/dataflow/缓存边界，不应表现为 verification 期 patch 或把硬件 tile 常数写进数学分母。

### 4. PD-Swap: Prefill-Decode Logic Swapping for End-to-End LLM Inference on Edge FPGAs via Dynamic Partial Reconfiguration
- arXiv: https://arxiv.org/abs/2512.11550
- 关键词：DPR、prefill-decode asymmetry、phase-specialized attention
- 核心思路：prefill 是 compute-bound，decode 是 bandwidth-bound，因此用动态重配置切换两种注意力引擎，避免一套静态逻辑同时为两种 phase 过度妥协。
- 对我们的借鉴：虽然当前仓库不需要上 DPR，但需要承认 prefill path 和 decode path 的硬件语义不同。当前 builder 已有 `build_prefill_path`/`build_decode_path` 框架，后续应让两条路径在 stage 设计上真正分化，而不是仅仅连线不同。

### 5. TeLLMe / TeLLMe v2
- arXiv: https://arxiv.org/abs/2504.16266
- arXiv: https://arxiv.org/abs/2510.15926
- 关键词：ternary、prefill+decode、streaming dataflow、fused elementwise ops、reversed-reordered prefill attention
- 核心思路：低比特线性层之外，attention 里的 element-wise、norm、量化/反量化也要与主流水线融合，才能真正隐藏延迟；prefill attention 采用专门的 memory-efficient 顺序。
- 对我们的借鉴：当前 block 里 residual add、post-attn norm、gate norm 这类 element-wise/nonlinear 边界，不应被当作无足轻重的“附属连线”，而应作为显式微流水 stage 处理。

### 6. MEADOW: Memory-efficient Dataflow and Data Packing for Low Power Edge LLMs
- arXiv: https://arxiv.org/abs/2503.11663
- 关键词：TPHS dataflow、weight packing、低功耗 edge、减少中间 token 往返
- 核心思路：用 token-parallel head-sequential dataflow 与权重打包降低 off-chip 访问，避免把每一层都按通用 GEMM 模式反复搬运中间结果。
- 对我们的借鉴：我们当前最大的风险也是把 block 内部真实 stage 边界过度扁平化，导致 FFN 输入和 residual skip 路径表达失真。正确做法是围绕 token/head/stage 数据流重建边界，而不是只看最终算术结果。

### 7. QUARK / TATAA / mixed-precision 方向
- QUARK: arXiv 搜索结果中 2025 年关于 nonlinear operations circuit sharing 的工作
- TATAA: https://arxiv.org/abs/2411.03697
- 关键词：mixed precision、nonlinear ops、transformable arithmetic
- 核心思路：线性层和非线性层往往适合不同精度和不同算术模式，非线性、softmax、norm 一类运算不能简单沿用线性层的统一定点继承。
- 对我们的借鉴：`head_group_norm` 曾被截断为 0 已经说明 builder 里存在类型继承风险。后续需要继续关注 `score_norm`、`gate_norm`、未来的 `post_attn_norm` 是否也有相同问题。

### 8. FAMOUS / ProTEA / ADAPTOR
- FAMOUS: https://arxiv.org/abs/2409.14023
- ProTEA: https://arxiv.org/abs/2409.13975
- ADAPTOR: https://arxiv.org/abs/2411.18148
- 关键词：dense attention、runtime programmable、matrix tiling、fully quantized
- 核心思路：tile size、parallel heads、矩阵切分是资源调度问题，目标是提升 PE 与 on-chip memory 利用率。
- 对我们的借鉴：`tile_k`、`tile_out`、`psum_bank_count` 更像硬件调度参数，不应直接主导 attention 算术分母。当前 builder 的 `score_tile_bias`/`scorev_den` 后续需要继续按这个原则重构。

### 9. HG-PIPE / ME-ViT / SWAT
- HG-PIPE: https://arxiv.org/abs/2407.17879
- ME-ViT: https://arxiv.org/abs/2402.09709
- SWAT: https://arxiv.org/abs/2405.17025
- 关键词：hybrid-grained pipeline、single-load policy、kernel fusion、row-wise/input-stationary dataflow
- 核心思路：真正有效的流水不是把模块简单串接，而是按数据依赖和 buffer 成本切 stage，并通过 kernel fusion 和 stationary dataflow 消除 bubble。
- 对我们的借鉴：我们不能继续把“attention 输出直接进 FFN”当作稳定结构。block 里 residual add 和 post-attn norm 需要成为显式 stage，否则 FFN 入口语义会长期漂移。

### 10. FlightLLM / EdgeLLM / HLSTransform / On-Device Qwen2.5
- FlightLLM: https://arxiv.org/abs/2401.03868
- EdgeLLM: https://arxiv.org/abs/2407.21325
- HLSTransform: https://arxiv.org/abs/2405.00738
- On-Device Qwen2.5: https://arxiv.org/abs/2504.17376
- 关键词：always-on-chip decode、heterogeneous memory hierarchy、CPU-FPGA 协同、mixed precision、Qwen edge deployment
- 核心思路：真实 LLM 部署中，decode 更依赖片上缓存与分层内存，而不同算子往往用不同精度与不同后端协同执行。
- 对我们的借鉴：当前第一层 block 主线应先把 stage 语义做对，再谈更深的系统优化；但在类型和 buffer 设计上，现在就应避免把所有路径都压成同一种数值/时序处理方式。

## 论文共性总结

### 共性 1：prefill 和 decode 必须区别建模
- 几乎所有较新的 LLM FPGA 工作都明确区分 prefill 与 decode。
- prefill 更偏 compute-bound 或混合瓶颈；decode 更偏 memory-bound。
- 对我们而言，这意味着 canonical builder 里的 `build_prefill_path` 与 `build_decode_path` 应继续分化，但第一优先级仍是先把 prefill 第一层 block 的 stage 语义做对。

### 共性 2：流水线切分应按真实数据依赖，而不是按占位模块边界
- 优秀设计通常会把 residual add、norm、softmax、element-wise gate 视作关键 stage，而不是把它们附着在线性层边上。
- 当前仓库里最像根因的问题也正是这一点：`attn_out` 接近，但 `ffn_gate_mul` 和 `residual_out` 偏差大，说明 FFN 入口 stage 语义不对。

### 共性 3：硬件调度常量不应污染数学归一化语义
- tile、bank、burst、parallel head 数常用于决定 dataflow 与 memory mapping。
- 它们可以影响吞吐与 buffer 组织，但不应该直接变成 attention 数值分母中的主导常量。
- 当前 `score_tile_bias` 和 `scorev_den` 仍值得继续重构，但其优先级低于修正 post-attn stage 边界。

### 共性 4：非线性和归一化路径需要单独的数值策略
- mixed-precision 论文普遍不把 softmax/norm/gate 简化成与 GEMM 相同的数值链。
- 当前 builder 已证实存在 `head_group_norm` 被截断的问题，因此未来新增 `post_attn_norm` 时必须从一开始就固定好类型，而不是依赖自动继承。

### 共性 5：减少中间结果的无意义回写和重读
- 从 ME-ViT、MEADOW 到 decode 持久状态工作，都在强调减少中间 tensor 往返。
- 对当前 block 而言，这对应于明确 stage 边界和 buffer 语义，而不是把中间结果在 placeholder 路径上来回折返。

## 对当前仓库最有价值的架构修改建议

### 优先级 A：先补出真实的 post-attn 微流水 stage
建议在 canonical builder 中显式实现以下链路，而不是继续维持 `attention -> FFN` 的 placeholder 直连：

1. `attn_out`
2. `attn_residual_out = attn_out + in_residual_aligned`
3. `post_attn_norm_out = norm(attn_residual_out)`
4. `post_attn_norm_out -> ffn_swiglu_u`
5. `residual_u` 的 skip 输入改为与 `attn_residual_out` 对齐的真实 stage，而不是原始 `in_residual`

注意：
- 这不是重复之前“顶层直接插一个 attn_residual_sum”那种试验。
- 这次应按微流水 stage 做 valid、delay、类型、命名和 logging，而不是仅加一个求和块。

### 优先级 B：把 attention 里的“数学项”和“调度项”分开
- 后续应把 `head_group_norm`、真实 score scaling 这类数学项，与 `tile_k`、`tile_out`、`psum_bank_count` 这类调度项显式分开。
- builder 内部可以保留调度参数参与 PE/buffer 组织，但不要再让它们直接主导分母数值。
- 这一步应放在 post-attn stage 修复之后执行，否则很容易把真正的 stage 根因和数值微调混淆。

### 优先级 C：继续清查 norm/gate 路径的类型继承
- 关注对象：`score_norm`、`gate_norm`、未来 `post_attn_norm`。
- 验证方式：继续使用真实 tracked SLX 观测关键节点值，避免在 verification 中临时 patch。

### 优先级 D：只在 stage 语义稳定后，再进一步扩展 prefill/decode 差异化
- 当前研究已经足以证明 prefill/decode 需要差异化，但不应在第一层 block 语义还未收敛时同步大改两条路径。
- 当前仓库更适合先收敛 prefill block 的真实结构，再考虑 decode 持久状态或更复杂的 memory-first 优化。

## 明确不建议的方向
- 不建议继续只围绕 `tile_k`、`tile_out`、`scorev_den` 做局部试错。
- 不建议再做 verification-time patch 去强行模拟 post-attn stage。
- 不建议现在把主线目标切回更大的 memory-first 顶层系统规划，当前最紧要的是先把第一层 block 的真实结构语义收敛。

## 重规划后的执行顺序

### Phase 1：block 结构收敛
1. 在 [scripts/implement_stage1_rmsnorm_qkv.m](scripts/implement_stage1_rmsnorm_qkv.m) 中补出 `attn_residual_out` 与 `post_attn_norm_out` 微流水。
2. 在真实 tracked SLX 上记录这两个 stage 的信号与 valid。
3. 调整 FFN 输入与最终 residual skip 的来源，保证 builder 内部结构与 real trace stage 命名一致。

### Phase 2：real-first 审计升级
1. 在 reference audit / stage trace audit 中正式引入 `attn_residual_out` 与 `post_attn_norm_out` 的对比。
2. 以真实 tracked SLX 为准，验证新 stage 是否降低 `ffn_gate_mul`、`ffn_down_stage`、`residual_out` 的偏差。

### Phase 3：attention 数值链重构
1. 在 stage 边界稳定后，重新审视 `score_tile_bias` 与 `scorev_den`。
2. 拆分“数学归一化项”和“调度项”，再做受控 builder 实验。

### Phase 4：更大范围的 prefill/decode 差异化
1. 等 block 结构稳定后，再决定是否推进更激进的 memory/dataflow 优化。
2. 包括 decode 持久状态、prefill memory-aware schedule、异构精度等更大动作。

## 最终判断
- 这轮调研没有推翻当前 canonical builder 主线，但明确改变了下一步优先级。
- 目前最该修改的不是 attention 分母常数，而是 block 的 post-attn 结构边界。
- 只要 `attn_residual_out` 和 `post_attn_norm_out` 还没有在 builder 中成为显式 stage，后续 FFN 和 residual 的误差都很可能只是症状，不是根因。