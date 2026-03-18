# 验证目录说明

- `testvectors/`: 固定输入向量与期望输出

验证层次：
1. 算子级
2. Block 级
3. 端到端

首版要求：每次改动至少通过 1 组算子级和 1 组 Block 级回归。

附加 smoke（stage2 实装后建议执行）：
- `run_stage2_decode_internal_smoke`：验证 `mode_decode` 内部路径切换、`kv_addr_gen_u` 参数化地址生成结构与常量值，以及 `axi_master_rd_u` 的 `avalid` 保持/突发完成关键内部连线。
- `run_stage2_kv_cache_boundary_smoke`：验证 `kv_cache_if_u` 的边界端口集合是否完整，并检查它与 `qkv_proj_u`、`axi_master_rd_u`、`axi_master_wr_u` 以及顶层 `kv_mem_*` 边界之间的连通性。
- `run_stage2_top_kv_io_tb_smoke`：最小 TB 就绪性检查。当前它会先审计 `qwen2_block_top` 根输入类型；若存在 `sfix*` 等不适合直接用 workspace external input 驱动的端口，会返回 `tb_ready=false` 和阻塞项，而不是纳入 fast smoke。
- `run_stage2_wrapper_tb_smoke`：专用 wrapper TB smoke。它不再依赖 root external input，而是给 DUT 外挂 typed Constant source，并加一个 SoC 风格的 DDR responder 子系统去接 `kv_mem_*` 事务边界；当前 wrapper 同时显式放置 `weight_ref_u`，把 `w_rd_req_bus -> w_rd_rsp_bus` 的参数返回链路接回 DUT，用于验证 `qwen2_block_top` 在 wrapper 中是否能真正产生 KV 读写活动并消费外部参数响应。`weight_ref_u` 的地址到数据映射已改为由 `cfg_weight_rsp_tag_base/tag_stride` 这类 cfg-style 常量驱动，而不是固定页签名字面量。当前默认配置下已可 PASS。
- `run_stage2_kv_banking_pipeline_smoke`：KV banking 内部语义 smoke。它复用 wrapper TB，并直接记录 `kv_cache_if_u` 内部的 `seq_window_sum`、`bank_sum`、`bank_addr`、`bank_sel`、`kv_seq_gate` 以及对应输入调度信号，验证 `tile_seq/active_seq_len/tile_k/tile_out/x_bank_count/kv_bank_count/kv_phase_first` 到 banked 地址、bank selector、写使能的公式关系已经在模型内部成立。
- `run_stage2_attention_pipeline_smoke`：attention 内部阶段 smoke。它复用 wrapper TB，但会直接对 `attention_u` 内部关键线启用 signal logging，验证 `qk_pair_valid -> softmax_valid -> scorev_input_valid` 的 staged valid 传播顺序、各级 1-cycle delay，以及 `row_sum_accum` 这类 running-softmax 累积信号已经真实活动，用来把 attention 细节从“最终输出活动”推进到“内部阶段可回归”。
- `run_stage2_ffn_pipeline_smoke`：FFN 内部阶段 smoke。它复用 wrapper TB，直接记录 `ffn_swiglu_u` 内部 `gateup_pair_valid -> swiglu_valid -> down_valid` 的阶段链路，验证 gate/up 融合前级与 down 投影后级已经具备明确的串行时序语义，而不是仅靠一串近似算子直接组合。
- `run_stage2_qkv_pipeline_smoke`：QKV 内部阶段 smoke。它复用 wrapper TB，直接记录 `qkv_proj_u` 内部 `kv_pair_valid -> fused_qkv_valid` 的共享阶段链路，验证 K/V 已先形成共享请求对、Q 随后进入 fused issue 阶段，并检查 `QkvStreamBus.group_idx` 已由 `q_valid + 2 * kv_valid` 驱动，而不是固定占位常量。
- `run_stage2_attention_ddr_integration_smoke`：主线联调 smoke。复用同一个 wrapper TB，同时检查 attention 权重请求有效位是否发出、attention 请求地址与 `weight_ref_u` 返回的 cfg-driven 响应数据是否精确匹配、DDR 读响应是否返回、`out_hidden` 是否产生非零结果，以及 KV 写回是否落到外层 DDR 通路上，用来覆盖“attention 细节 + DDR 联调”这条主线。
- 可视化外层 TB 模型：`simulink/models/qwen2_block_top_wrapper_tb.slx`。这个模型把 `qwen2_block_top` 作为 DUT，并在外层显式放置 SoC 风格的 `ddr_ref_u` responder；`ddr_ref_u` 当前已拆成可见的 `Input Read Memory` / `Output Write Memory` 两段结构，且内部进一步细化为 `AXI4Master* BusCreator / Controller / BusSelector` 风格的分层组织，接近 `simulink/ref/soc_image_rotation_fpga.slx` 的“DUT + DDR 环境”布局。
- `run_stage2_axi_rd_functional_smoke`：以确定性向量检查 `axi_master_rd_u` 行为语义（`avalid` 保持到握手、突发计数完成后清除活动状态）。
- `run_stage2_axi_wr_functional_smoke`：以确定性向量检查 `axi_master_wr_u` 行为语义（`wr_valid` 发起保持到写完成、`request_next_line` 仅在写完成时脉冲）。
- `run_stage2_smoke_suite_fast`：快速套件，复用建模结果串行执行 decode、`kv_cache_if_u` 边界、prefill-attention、wrapper TB、KV banking pipeline、attention pipeline、FFN pipeline、QKV pipeline、attention+DDR 集成 smoke、rd/wr smoke，并附加一组非默认 `KvAddressConfig`（`rd_base/wr_base/stride_bytes/decode_burst_len`）覆盖。
