# 验证目录说明

- `testvectors/`: 固定输入向量与期望输出

验证层次：
1. 算子级
2. Block 级
3. 端到端

首版要求：每次改动至少通过 1 组算子级和 1 组 Block 级回归。

附加 smoke（stage2 实装后建议执行）：
- `run_stage2_decode_internal_smoke`：验证 `mode_decode` 内部路径切换、`kv_addr_gen_u` 参数化地址生成结构与常量值，以及 `axi_master_rd_u` 的 `avalid` 保持/突发完成关键内部连线。
- `run_stage2_axi_rd_functional_smoke`：以确定性向量检查 `axi_master_rd_u` 行为语义（`avalid` 保持到握手、突发计数完成后清除活动状态）。
- `run_stage2_axi_wr_functional_smoke`：以确定性向量检查 `axi_master_wr_u` 行为语义（`wr_valid` 发起保持到写完成、`request_next_line` 仅在写完成时脉冲）。
- `run_stage2_smoke_suite_fast`：快速套件，复用建模结果串行执行 decode/rd/wr smoke，并附加一组非默认 `KvAddressConfig`（`rd_base/wr_base/stride_bytes/decode_burst_len`）覆盖。
