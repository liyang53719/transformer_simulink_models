# M1 Simulink 顶层占位与连线清单

更新时间：2026-03-14
适用对象：`qwen2_block_top`

## 1. 目标
为 `simulink/models/` 中的顶层模型建立可落地的占位子系统和接口连线规则，降低后续建模歧义。

## 2. 顶层模型结构
建议文件名：`qwen2_block_top.slx`

顶层子系统顺序：
1. `rmsnorm_u`
2. `rope_u`
3. `qkv_proj_u`
4. `attention_u`
5. `ffn_swiglu_u`
6. `residual_u`
7. `kv_cache_if_u`
8. `ctrl_fsm_u`
9. `axi_master_rd_u`（V2 预留）
10. `axi_master_wr_u`（V2 预留）
11. `ddr_model_if_u`（V2 预留）

## 3. 连线规则
- 数据主通路：`in_hidden -> rmsnorm_u -> rope_u -> qkv_proj_u -> attention_u -> ffn_swiglu_u -> residual_u -> out_hidden`
- 控制通路：`ctrl_fsm_u` 输出各子模块 enable/phase
- KV 通路：
  - decode：`kv_cache_if_u` 提供读数据给 `attention_u`
  - prefill/decode：`qkv_proj_u` 输出的新 K/V 写入 `kv_cache_if_u`

## 3.1 事务链路（V2）
- prefill（write-heavy）：
  1. `ctrl_fsm_u` 发起 prefill 批次。
  2. `qkv_proj_u` 生成新 K/V。
  3. `axi_master_wr_u` 通过 `ddr_model_if_u` 批量写入 KV。
  4. `PREFILL_COMMIT` 更新有效长度并发出阶段完成信号。
- decode（read-modify-write）：
  1. `ctrl_fsm_u` 发起单 token decode。
  2. `axi_master_rd_u` 读取历史 KV。
  3. `attention_u` 消费历史 KV 并计算当前 token。
  4. `axi_master_wr_u` 写回当前 token K/V。

## 4. 接口信号集合（顶层）
- `clk, rst_n`
- `start, done`
- `mode_decode`
- `in_valid, in_ready`
- `out_valid, out_ready`
- `cfg_seq_len, cfg_token_pos, cfg_eps`
- `kv_cache_rd_data, kv_cache_rd_valid, kv_cache_wr_data, kv_cache_wr_en`
- `kv_mem_rd_addr, kv_mem_rd_len, kv_mem_rd_valid, kv_mem_rd_ready`（V2 预留）
- `kv_mem_wr_addr, kv_mem_wr_len, kv_mem_wr_valid, kv_mem_wr_ready`（V2 预留）
- `busy, irq, error_code, stop_req`（V2 预留）

## 5. 子系统占位要求
- 每个子系统先使用固定延迟占位（例如 N 周期延迟 + 透传）
- 每个子系统都保留 `valid/ready` 接口
- 禁止在占位阶段引入跨子系统的隐式全局信号

## 6. ctrl_fsm_u 最小状态
`IDLE -> LOAD -> RUN_NORM -> RUN_ROPE -> RUN_QKV -> RUN_ATTN -> RUN_FFN -> RUN_RES -> WRITEBACK -> DONE`

任务级扩展（V2）：
`INIT -> PREFILL_LOOP -> PREFILL_COMMIT -> DECODE_LOOP -> STOP -> DONE`

异常扩展（V2）：
`ANY_STATE -> ERROR`

额外约束：
- decode 模式下必须等待 KV read valid 后再进入 `RUN_ATTN`
- `done` 只在 `out_valid && out_ready` 成立后触发
- `stop_req` 到达后，完成当前事务再进入 `STOP`
- `ERROR` 状态必须输出 `error_code`，并触发 `irq`

## 7. HDL Coder 兼容注意事项
- 避免不可综合块（动态尺寸、可变长循环）
- 所有循环边界参数化但在生成前可静态确定
- 明确数据类型转换点，避免隐式双精度

## 8. 完成定义（本文件对应）
以下条件满足即视为“顶层占位完成”：
1. 顶层模型创建并包含全部子系统
2. 主通路与控制通路连线完成
3. 占位模型可执行一次空跑仿真
4. 接口名称与 `docs/05_m1_block_interface_spec.md` 一致
5. V2 预留端口在模型中可见（即使暂未驱动真实事务）
