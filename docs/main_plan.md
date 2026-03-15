## Plan: Qwen2 NPU 系统架构 V2（Memory-First）

本轮以“功能闭环优先”为原则，按你确认的方向并行兼顾 FPGA 可验证与 ASIC 可迁移：先在 Simulink/SoC Blockset 中把 prefill+decode+KV+DDR 跑通，再把同一套接口与时序约束抽象为 ASIC 版本。性能约束先按 1.5B 4bit、decode 10 tok/s、1GHz 目标倒推带宽与流水深度，FlashAttention 采用分阶段落地（先可跑通，再替换优化核）。

**Steps**
1. Phase A 需求冻结与预算建模（阻塞后续）
1.1 冻结双模式执行语义：prefill 为“批量写 KV、产出最后 token logits”，decode 为“单 token 增量读写 KV”。
1.2 形成首版周期预算模型：按目标 tok/s 推导每 token 周期预算、每层可用周期、每子模块时延上限。
1.3 形成首版带宽预算模型：拆分权重流、KV 读、KV 写、激活搬运四类流量，给出峰值与均值预算。
1.4 识别不可达项并定义退化策略：若预算超标，优先调整并行度、分块策略与缓存复用，不先改功能语义。

2. Phase B 顶层接口与时钟复位重构（依赖 1）
2.1 在现有 block 接口上补齐系统端口分层：
- 算法域端口（token/hidden/mode）。
- 存储域端口（AXI4 master 或等价抽象，含读写地址、长度、有效握手）。
- 控制域端口（start/done/busy/irq、错误码）。
2.2 明确 clk/rst_n 策略：
- Simulink 行为模型允许隐式步进时钟。
- HDL DUT 必须显式单时钟输入与同步复位策略；若后续多时钟，补 CDC 边界与异步 FIFO 约束。
2.3 明确 reset 行为：cold reset 清空状态机与片上缓存；warm reset 保留可配置状态（可选）。

3. Phase C KV Cache 与外存层级设计（依赖 1,2）
3.1 定义三级存储层级：寄存器/片上 SRAM（行缓冲）/外部 DDR（主 KV 存储）。
3.2 定义 KV 地址映射：按 layer-head-token-stride 线性化，确保 decode 连续读，prefill 连续写。
3.3 定义 KV 事务粒度：按 head 或 tile 组织 burst，避免随机小包访问。
3.4 增加 KV 一致性规则：prefill 完成后写入有效长度；decode 按当前长度读历史 KV 并追加写回。

4. Phase D Attention 内核路线（依赖 3）
4.1 V1（功能闭环核）：标准 scaled-dot + softmax + V，分块实现，确保与 MATLAB 参考数值一致。
4.2 V2（优化核）：替换为 FlashAttention 风格在线 softmax（tile 流水），减少外存往返与中间缓存压力。
4.3 保持接口不变：V1/V2 共用同一 attention 子系统端口，降低切换风险。
4.4 明确验收优先级：先 V1 跑通 prefill/decode+KV+DDR，再启用 V2 性能优化。

5. Phase E DDR/AXI 建模与争用分析（依赖 3,4）
5.1 在 SoC Blockset 中引入 Memory Controller + AXI4 Random Access Memory + Memory Traffic Generator。
5.2 先建单通道 DDR 基线，记录各 master 带宽；若不满足预算，再扩展双通道或多控制器分流。
5.3 为主要流量建立独立 master：权重读、KV 读、KV 写（必要时激活写回单独 master）。
5.4 运行争用仿真并输出指标：每 master 实际 MB/s、突发丢失/阻塞告警、队列深度。
5.5 基于结果迭代 burst 长度、仲裁权重、分块尺寸，直到满足功能闭环实时性。

6. Phase F 控制器/FSM 扩展到 prefill+decode 全链（依赖 2,3,4,5）
6.1 将现有 ctrl_fsm 从 block 内局部状态扩展为“任务级调度 FSM”：
- INIT -> PREFILL_LOOP -> PREFILL_COMMIT -> DECODE_LOOP -> STOP。
6.2 增加模式切换条件：prefill 完成标志、EOS/最大长度、外部停止请求。
6.3 增加异常处理状态：DDR 超时、协议错误、数值异常（可选告警寄存器）。

7. Phase G 验证闭环与门禁（依赖 6）
7.1 行为一致性：MATLAB reference vs Simulink 功能模型（prefill/decode 两模式）。
7.2 访存一致性：行为内存模型 vs 外部 DDR 模型输出一致。
7.3 RTL 一致性：HDL Coder 生成 RTL 后做 cosim/仿真比对。
7.4 性能一致性：报告 token 延迟分解（计算/访存/控制）与 DDR 带宽占用。
7.5 门禁通过条件：功能正确 + 无关键协议告警 + 达到阶段性 tok/s 下限。

8. Phase H ASIC 迁移抽象（可与 7 并行）
8.1 输出与 FPGA 无关的微架构文档：缓存层级、AXI 等价协议、时钟复位、异常语义。
8.2 标注 FPGA 专属构件替换点（AXI Manager、特定 IP）与 ASIC 对应替代。
8.3 形成 ASIC 预研输入：面积/带宽热点列表、潜在瓶颈与可选优化杠杆。


**Immediate Slice（下一轮执行包）**
1. S1-接口冻结补丁（文档）
- 在 docs/05 中新增任务级控制口定义（busy/irq/error_code）与 reset 语义（cold/warm）。
- 在 docs/07 中把 ctrl_fsm_u 扩展为任务级状态并明确 prefill/decode 进入与退出条件。
- 验收：接口文档与顶层占位清单无冲突，端口命名一一对应。

2. S2-预算文件落地（文档）
- 新增 docs/09_memory_compute_budget.md，写清 10 tok/s@1GHz 的周期预算与带宽预算口径。
- 内容最少包含：每 token 周期预算、每层预算、KV 读写带宽预算、权重流量预算、冗余裕量。
- 验收：预算表可被 regression 报告字段直接引用。

3. S3-自动化门禁扩展（脚本规范）
- 在 docs/08 增加 DDR 模型回归入口和带宽阈值门禁定义。
- 约定 verification 报告增加 memory_metrics 字段（各 master MB/s、阻塞告警计数）。
- 验收：最小回归说明中出现功能+访存双门禁。

4. S4-实现脚本改造设计说明（不立刻编码）
- 为 scripts/implement_stage1_rmsnorm_qkv.m 输出“改造清单”：从 stage1 占位转为可插拔 prefill/decode 路径，并预留 KV/DDR 接口挂点。
- 验收：改造清单明确输入输出、依赖顺序、回归影响面。

5. S5-执行顺序与并行性
- 顺序依赖：S1 -> S2 -> S3 -> S4。
- 可并行项：S1 与 S2 可并行起草；S3 依赖 S2 指标定义；S4 依赖 S1/S3。

**File-Level Blueprint（函数/字段级）**
1. docs/05_m1_block_interface_spec.md
- 新增小节：任务级控制口（busy、irq、error_code、stop_req）。
- 新增小节：reset 策略（cold reset/warm reset）及寄存器保留范围。
- 新增小节：KV 外存抽象接口（rd_addr/rd_len/rd_valid/rd_ready、wr_addr/wr_len/wr_valid/wr_ready）。
- 更新状态机描述：从 block 内流水状态扩展为任务级状态（INIT/PREFILL_LOOP/PREFILL_COMMIT/DECODE_LOOP/STOP/ERROR）。

2. docs/07_m1_simulink_top_placeholder.md
- 在“顶层模型结构”增加 memory/traffic 相关子系统占位：axi_master_rd_u、axi_master_wr_u、ddr_model_if_u。
- 在“连线规则”增加 prefill 与 decode 两条事务链（prefill: write-heavy；decode: read-modify-write）。
- 在“ctrl_fsm_u 最小状态”中加入异常状态与 stop 路径。

3. docs/09_memory_compute_budget.md（新增）
- 固定预算口径：1GHz、10 tok/s、1.5B 4bit。
- 表1：每 token 周期预算（总周期、每层预算、每阶段预算）。
- 表2：带宽预算（权重流、KV 读、KV 写、激活流，峰值/均值、裕量%）。
- 表3：DDR 配置候选（单通道/双通道）与是否满足预算。
- 明确阈值字段名供回归使用：memory_bw_min_mb_s、memory_stall_max_count。

4. docs/08_m1_automation_and_consistency.md
- 在执行顺序增加 DDR 模型回归步骤（位于 block regression 之后）。
- 在通过标准新增访存门禁：
  - 每个关键 master 的带宽下限。
  - 阻塞/丢突发告警上限。
- 规范报告结构：新增 memory_metrics.master_bw_mb_s、memory_metrics.stall_count、memory_metrics.dropped_burst_count。

5. scripts/check_block_interface_spec_consistency.m
- 扩展 requiredIn/requiredOut：增加 busy/irq/error_code/stop_req 及 AXI 抽象端口检查。
- 新增 requiredSubs：检查新增 memory 相关子系统占位是否存在。
- 结果结构扩展：missing_axi_ports、missing_ctrl_ports，便于 CI 精确报错。

6. verification/run_block_regression.m
- options 扩展：ModeSet（prefill/decode/both）、EnableMemoryMetrics（bool）。
- summary.details 扩展字段：mode、latency_cycle_est、memory_bw_mb_s、stall_count。
- baselineMode=real 路径保持兼容，新增 memory 指标为空时的降级行为（不阻塞功能回归）。

7. scripts/run_m1_minimal_regression.m
- 三步执行扩展为四步：bootstrap -> operator -> block -> memory-model-check。
- result 结构扩展 memory 字段，聚合 memory_metrics 与门禁结果。
- report json 保持向后兼容：旧字段不删，只追加。

8. scripts/run_m1_real_reference_regression.m
- 将 BaselineMode=real 保持默认，追加 options.EnableMemoryMetrics 透传。
- reportTag 增加 mode 信息，区分纯功能回归与含访存指标回归。

9. scripts/implement_stage1_rmsnorm_qkv.m（设计改造清单）
- 现状保留 stage1 占位生效逻辑；新增参数化入口 stageProfile（stage1/stage2_memory_ready）。
- 抽取连接构建为子函数：build_prefill_path、build_decode_path、build_kv_memory_stubs。
- 在不破坏现有 PASS 的前提下，先引入端口与占位子系统，再逐步替换内部算子。
- 明确回归影响：任何端口新增必须同步更新 check_block_interface_spec_consistency 与 docs/05。

**Doc Patch Template（逐段落写作模板）**
1. docs/05 追加段落模板
- 段落 A: 任务级控制接口定义表（字段、方向、位宽、时序语义）。
- 段落 B: reset 行为矩阵（cold/warm 对每类状态的影响）。
- 段落 C: KV/DDR 抽象事务定义（读写事务触发条件、握手、错误返回）。
- 段落 D: 任务级 FSM 图文字版（进入条件/退出条件/异常分支）。

2. docs/07 追加段落模板
- 段落 A: 子系统拓扑（新增 memory 三子系统与现有链路连接）。
- 段落 B: prefill 事务序列（起始、批量写 KV、提交）。
- 段落 C: decode 事务序列（读历史 KV、计算、追加写）。
- 段落 D: ctrl_fsm_u 状态与信号驱动表（每状态驱动哪些使能）。

3. docs/09 新建模板
- 章节 1: 目标与假设（10 tok/s@1GHz，4bit 量化，单 token decode 口径）。
- 章节 2: 周期预算推导（token->layer->stage 三级分解）。
- 章节 3: 带宽预算推导（权重/KV读/KV写/激活）。
- 章节 4: DDR 配置比较（单通道/双通道的预算满足度）。
- 章节 5: 回归阈值映射（预算字段到脚本门禁字段）。

4. docs/08 追加段落模板
- 段落 A: memory-model-check 执行入口与前置条件。
- 段落 B: memory_metrics JSON schema（字段名、类型、单位）。
- 段落 C: 门禁规则（带宽下限、stall 上限、drop 上限）。

**Implementation Order Map（脚本改造顺序图）**
1. P0 先改 scripts/check_block_interface_spec_consistency.m
- 目标：先允许新端口/子系统被验证，避免后续改造无检查护栏。
- 验收：缺失分类报错可区分 in/out/sub/axi/control。

2. P1 再改 scripts/implement_stage1_rmsnorm_qkv.m
- 目标：引入 stageProfile + 3 个 build_* 框架函数，但默认仍走 stage1。
- 验收：不改变当前最小回归结果（功能兼容）。

3. P2 改 verification/run_block_regression.m
- 目标：扩展 ModeSet 与 memory 指标字段，先支持空指标降级。
- 验收：baseline=real/stored 都可跑，summary 新字段存在。

4. P3 改 scripts/run_m1_minimal_regression.m
- 目标：加入 memory-model-check 第四步与 result.memory 聚合。
- 验收：报告 JSON 向后兼容并新增 memory_metrics。

5. P4 改 scripts/run_m1_real_reference_regression.m
- 目标：透传 EnableMemoryMetrics 与 reportTag 模式标识。
- 验收：awq/gptq/gguf 三入口报告标签可区分是否启用访存指标。

6. P5 回写文档 docs/05, docs/07, docs/08, docs/09
- 目标：实现与文档严格对齐，避免“代码先行、规格滞后”。
- 验收：文档字段与脚本字段逐项对应，无命名漂移。

**Gate-by-Gate Acceptance（分阶段验收）**
1. Gate-1（接口门）
- check_block_interface_spec_consistency 通过，且新增端口全部可见。
2. Gate-2（兼容门）
- run_m1_minimal_regression 在未启用 memory 指标时与历史结果一致。
3. Gate-3（访存门）
- 启用 memory-model-check 后生成 memory_metrics，满足阈值或给出可解释告警。
4. Gate-4（真实路径门）
- run_m1_real_reference_regression 三入口（AWQ/GPTQ/GGUF）全部产出结构化报告。

**Relevant files**
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/docs/08_m1_automation_and_consistency.md — 增加外存模型一致性与带宽分析门禁。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/docs/05_m1_block_interface_spec.md — 扩展顶层端口分层、clk/rst/reset 语义、prefill/decode 任务级状态。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/docs/07_m1_simulink_top_placeholder.md — 从 block 占位升级到 memory-first 结构与 DDR/AXI 连接。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/scripts/implement_stage1_rmsnorm_qkv.m — 为 prefill/decode 双路径与外存接口预留注入点。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/scripts/check_block_interface_spec_consistency.m — 增加系统端口与模式状态检查。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/verification/run_block_regression.m — 增加 prefill/decode 分模式回归与 DDR 模式回归。
- /home/yang/Documents/prj/matlab_DL_transfromer/transformer_simulink_models/transformer_simulink_models/simulink/models/qwen2_block_top.slx — 接入 KV/DDR 通路与任务级控制器。

**Execution Batches（首轮落地批次）**
1. Batch-A（文档先行）
- 修改 docs/05、docs/07，新增任务级接口与事务语义。
- 新建 docs/09，填入预算口径与阈值字段。
- 产出物：文档 PR + 字段对照表。

2. Batch-B（检查器与框架）
- 修改 check_block_interface_spec_consistency，支持新增端口分类检查。
- 修改 implement_stage1_rmsnorm_qkv，仅引入 stageProfile 与 build_* 框架，不改默认行为。
- 产出物：接口检查 PASS，最小回归与基线一致。

3. Batch-C（回归指标扩展）
- 修改 run_block_regression、run_m1_minimal_regression、run_m1_real_reference_regression。
- 加入 memory_metrics 结构与可选门禁，保持向后兼容。
- 产出物：JSON 报告含 memory_metrics；功能模式仍全通过。

4. Batch-D（收口）
- 修改 docs/08，对齐最终执行顺序、门禁阈值、报告字段。
- 复核文档字段与脚本字段一一对应。
- 产出物：最终一致性审计记录。

**Verification**
1. 预算验证：提交周期预算表和带宽预算表，确认目标 10 tok/s 下每层预算闭合。
2. 行为验证：prefill 与 decode 在 stored/real baseline 下均通过误差门限。
3. 存储验证：在 SoC 内存模型下无关键丢包/超时告警，关键 master 带宽满足预算下限。
4. RTL 验证：生成 RTL 并完成关键路径仿真，结果与 Simulink 对齐。
5. 回归验证：最小回归脚本输出结构化报告（功能+性能+访存三类指标）。

**Decisions**
- 目标平台：两者并行，先 FPGA 可验证再抽象 ASIC。
- 近期优先级：先保证 prefill+decode+KV+DDR 功能闭环，再做吞吐优化。
- 性能目标：1.5B 4bit、decode 10 tok/s、1GHz 时钟（作为预算起点，不作为首轮唯一门禁）。
- FlashAttention：不是第一阶段阻塞项，但接口必须从现在起按可替换优化核设计。

**Further Considerations**
1. 若单通道 DDR 无法闭合预算，优先选择“权重/KV 分通道”还是“同通道多 master 仲裁优化”。
2. 是否在本轮就引入多时钟域（compute 与 memory 分频），还是先单时钟闭环后再拆分。