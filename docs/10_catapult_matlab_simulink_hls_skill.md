# Catapult MATLAB/Simulink HLS 建议整理

更新时间：2026-03-17

## 调研范围

本次主要浏览了以下 Catapult 安装目录中的 MATLAB / Simulink 相关材料：

- `examples/methodology/matlab_integration/MatlabFlow.pdf`
- `examples/methodology/matlab_integration/run_fir.tcl`
- `examples/methodology/matlab2hls/ImplementingSimpleDOAEstimatorwithCatapult.pdf`
- `examples/methodology/matlab2hls/run_optimized_fixpt.tcl`
- `examples/matchlib/toolkit/examples/70_python_matlab_integration/README`

## Catapult 给出的核心建议

1. MATLAB/Simulink 更适合作为算法参考模型、激励环境和等价性校验环境，而不是直接综合输入。
2. 从 MATLAB 到 HLS 没有“自动一键变成可综合 C++”的主路径，推荐流程是先做人手控制的功能拆分，再把硬件候选部分改写成可综合 C++ / SystemC。
3. Catapult 强调先做模型分析和软硬件分区，再做 C++ 重写，再做定点化与性能优化，而不是先把整个浮点 MATLAB 模型直接硬塞进 HLS。
4. MATLAB 集成流的价值主要在于：通过 MEX 或 wrapper 方式，把手写的 HLS C++ 放回 MATLAB/Simulink testbench 里复用原始验证环境，持续比较功能等价性。
5. 优化阶段依赖明确的结构化约束：内存映射、bank/interleave、loop unroll、pipeline、设计目标在 Catapult 中是显式约束，不建议把这些硬件决策继续藏在高层脚本隐喻里。
6. Matchlib 的 Python/MATLAB integration 示例进一步说明：高层环境最好把 HLS 设计当作“可步进、可调用的外部部件”，而不是把高层仿真运行时和硬件线程模型混在一起。

## 对当前项目最有指导意义的点

### 1. Simulink 顶层可以继续保留，但职责要收紧

Catapult 的示例没有鼓励把复杂硬件调度、片外内存协议、流水时序细节长期留在 MATLAB/Simulink 中再期待后续自动转 HLS。更合理的分工是：

- MATLAB/Simulink 保留为系统级参考环境、接口编排环境、可视化联调环境。
- 真正面向 HLS/RTL 收敛的计算核心，逐步沉淀为更稳定的模块边界、固定点数据类型、显式 memory contract。

这和当前主线并不冲突，但意味着后续不要把 `qwen2_block_top` 无限做成“所有实现细节都只存在于 Simulink”的终态。

### 2. 当前 external weight / DDR wrapper 方向是对的

Catapult 反复强调内存接口、bank、吞吐和调度边界必须前置显式化。当前仓库把：

- `w_rd_req_bus / w_rd_rsp_bus`
- `kv_mem_rd_* / kv_mem_wr_*`
- wrapper 外部 responder

逐步从内部 stub 中抽出来，这一方向符合 HLS 收敛思路，应该继续。

### 3. 参数响应必须配置化，不能长期靠硬编码 stub

从 Catapult 的方法学看，地址、bank、stride、资源映射都应是结构参数，而不是散落的固定字面量。当前把 `weight_ref_u` 的页签名改成 cfg-driven，是必要的一步。后续还应继续推进：

- 从“地址 + tag”返回，走向“地址页表 + 参数布局描述”返回。
- 把 attention / FFN / RMSNorm 的参数页布局逐步从 smoke 语义升级到真实 layer 布局近似。

### 4. attention 内部应优先固化阶段边界，而不是先追求算术逼真度

Catapult 的优化对象是结构化硬件。对当前 attention 主线，优先级应是：

- 请求/响应 valid 链
- score / softmax / score·V 的阶段边界
- tile / bank / active sequence 长度对数据流的影响

这比一开始就追求更复杂的浮点近似更重要。当前把 stage valid 和调度参数进一步接入 attention，是符合这个方向的。

## 对架构设计与计划的结论

结论：主线方向不需要推翻，但需要更明确地区分“系统参考模型”和“可综合硬件内核收敛路径”。

建议保留的方向：

1. 保留 `qwen2_block_top` + wrapper 的 Simulink 联调主线。
2. 继续推进 external weight path、DDR path、KV cache path 的显式接口化。
3. 继续用 smoke + real regression 做每轮闭环。

建议调整的地方：

1. 后续新增能力时，优先增加 cfg、memory contract、valid pipeline、tile/bank 边界，不要优先增加更多固定常量 stub。
2. 对 attention、FFN、RMSNorm 等模块，逐步形成“可迁移到 HLS C++/SystemC 的结构说明”，避免实现语义只存在于 Simulink 连线中。
3. 文档和计划里应开始增加“未来 HLS kernel 边界”的描述，例如哪部分属于调度/接口壳，哪部分属于计算核心。

## 对当前阶段的具体行动建议

1. 短期：继续把 wrapper 外部参数响应从 cfg-driven tag 提升到更接近真实页布局的 responder。
2. 短期：继续细化 attention 的 staged valid、tile 和 bank 语义，并给 smoke 增加对应观测点。
3. 中期：为 `rmsnorm_u`、`qkv_proj_u`、`attention_u`、`ffn_swiglu_u` 各写一页“潜在 HLS kernel 边界说明”。
4. 中期：把当前 Simulink 中已经稳定的 memory contract 和配置口整理成统一 spec，减少后续迁移时的重复解释成本。

## 本次判断

- 不建议现在大改总体架构。
- 建议继续沿“外部内存接口显式化 + 内部阶段边界显式化 + cfg 化”这条主线推进。
- 建议从现在开始把 Simulink 视为主验证/主集成环境，而不是未来唯一实现载体。