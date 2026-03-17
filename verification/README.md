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
- `run_stage2_wrapper_tb_smoke`：专用 wrapper TB smoke。它不再依赖 root external input，而是给 DUT 外挂 typed Constant source，并加一个 SoC 风格的 DDR responder 子系统去接 `kv_mem_*` 事务边界，用于验证 `qwen2_block_top` 在 wrapper 中是否能真正产生 KV 读写活动。
- `run_stage2_axi_rd_functional_smoke`：以确定性向量检查 `axi_master_rd_u` 行为语义（`avalid` 保持到握手、突发计数完成后清除活动状态）。
- `run_stage2_axi_wr_functional_smoke`：以确定性向量检查 `axi_master_wr_u` 行为语义（`wr_valid` 发起保持到写完成、`request_next_line` 仅在写完成时脉冲）。
- `run_stage2_smoke_suite_fast`：快速套件，复用建模结果串行执行 decode、`kv_cache_if_u` 边界、prefill-attention、rd/wr smoke，并附加一组非默认 `KvAddressConfig`（`rd_base/wr_base/stride_bytes/decode_burst_len`）覆盖。
