# M1 Block 接口规格（冻结版）

更新时间：2026-03-14
适用阶段：M1 单 Block 闭环

## 1. 规格目标
本规格用于冻结 `qwen2_block_top` 的外部接口、时序协议和数据格式，确保参考模型、Simulink 模型、RTL 仿真三方接口一致。

## 2. 参数分层

### 2.1 已冻结参数（M1 必须）
- 数值基线：FP16
- 运行模式：prefill + decode
- 上下文目标：128 token（用于 M2 端到端）
- 采样策略：argmax（M1/M2）

### 2.2 待冻结参数（M1 可先参数化）
- `num_layers`
- `hidden_size`
- `num_heads`
- `head_dim`
- `ffn_intermediate_size`

说明：M1 单 Block 允许使用参数化输入，不阻塞模块联调；进入 M2 前必须冻结为具体 profile。

## 3. 顶层端口定义（Block 级）

### 3.1 时钟与复位
- `clk`：输入，主时钟
- `rst_n`：输入，低有效同步复位

### 3.2 控制端口
- `mode_decode`：输入，`0=prefill`，`1=decode`
- `start`：输入，启动一次 block 计算
- `done`：输出，本次 block 计算完成脉冲
- `eos_in`：输入，上游 EOS 标志
- `eos_out`：输出，透传或本层更新后的 EOS 标志

### 3.3 输入数据端口
- `in_valid`：输入
- `in_ready`：输出
- `in_hidden`：输入，形状 `[TOKENS_PER_CALL, hidden_size]`
- `in_residual`：输入，形状 `[TOKENS_PER_CALL, hidden_size]`
- `kv_cache_rd_data`：输入，decode 模式下读取历史 K/V
- `kv_cache_rd_valid`：输入

### 3.4 输出数据端口
- `out_valid`：输出
- `out_ready`：输入
- `out_hidden`：输出，形状 `[TOKENS_PER_CALL, hidden_size]`
- `kv_cache_wr_data`：输出，新 token 的 K/V
- `kv_cache_wr_en`：输出

### 3.5 参数与配置端口
- `cfg_seq_len`：输入，当前序列长度
- `cfg_token_pos`：输入，当前 token 位置（用于 RoPE）
- `cfg_eps`：输入，RMSNorm epsilon

## 4. 时序协议
- 采用 `valid/ready` 握手。
- 当 `in_valid=1 && in_ready=1` 时采样输入。
- 当 `out_valid=1 && out_ready=1` 时完成输出传输。
- `start` 在空闲态拉高 1 周期进入 `RUN`，`done` 在输出完成后拉高 1 周期。

## 5. 数据格式
- `in_hidden/out_hidden`：FP16（IEEE half）
- 中间累加可提升到 FP32（实现可选），但对外保持 FP16
- `cfg_*` 控制字段使用无符号定点或整数寄存器

## 6. 子模块边界（必须保持）
- `rmsnorm_u`
- `rope_u`
- `qkv_proj_u`
- `attention_u`
- `ffn_swiglu_u`
- `residual_u`

要求：子模块间接口同样使用 `valid/ready`，禁止隐式全局使能。

## 7. 状态机（Block 内）
`IDLE -> NORM -> ROPE -> QKV -> ATTN -> FFN -> RESIDUAL -> WRITEBACK -> DONE`

decode 模式下在 `ATTN` 阶段额外执行 KV cache 读取路径；prefill 模式侧重批 token 并行路径。

## 8. 版本管理规则
- 本文件每次修改必须同步更新 `docs/02_live_status.md`。
- 涉及端口增删改必须在提交说明中标注“接口变更”。
