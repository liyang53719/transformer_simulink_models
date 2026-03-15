# M1 Memory/Compute 预算口径（V2）

更新时间：2026-03-14
适用阶段：M1 -> M2 过渡（memory-first）

## 1. 目标与假设
- 目标模型：Qwen2 1.5B（4bit 量化权重）
- decode 目标：10 tok/s
- 主时钟假设：1 GHz
- 预算口径：单 token decode 为主，prefill 作为批量事务单独核算
- 说明：本文件用于门禁与建模口径统一，不代表最终硬件实现上限

## 2. 周期预算推导
### 2.1 每 token 总预算
- 目标吞吐：10 tok/s
- 每 token 时间预算：$0.1\ \text{s}$
- 每 token 周期预算：$0.1 \times 10^9 = 1.0 \times 10^8$ cycles

### 2.2 每层预算（示例口径）
设层数为 `L`，则每层预算：
$$
C_{layer} = \frac{10^8}{L}
$$

若 `L=28`（示例），则每层约 `3.57e6 cycles`。

### 2.3 每阶段预算（示例比例）
- NORM+ROPE+QKV：30%
- ATTN：40%
- FFN+RES：30%

对应每层阶段预算：
- `C_qkv = 0.30 * C_layer`
- `C_attn = 0.40 * C_layer`
- `C_ffn = 0.30 * C_layer`

## 3. 带宽预算口径
将流量拆成 4 类：
1. 权重流量 `BW_w`
2. KV 读流量 `BW_kv_rd`
3. KV 写流量 `BW_kv_wr`
4. 激活搬运 `BW_act`

总带宽预算：
$$
BW_{total} = BW_w + BW_{kv\_rd} + BW_{kv\_wr} + BW_{act}
$$

门禁建议保留 20% 裕量：
$$
BW_{required} = 1.2 \times BW_{total}
$$

## 4. DDR 配置候选（用于 SoC 建模）
### 4.1 单通道 DDR
- 优点：结构简单，验证路径短
- 风险：权重流与 KV 读写争用明显
- 建议：作为 baseline 必跑

### 4.2 双通道 DDR（或双控制器）
- 优点：可按流量分流（如权重/KV 分离）
- 风险：控制与地址映射复杂度增加
- 建议：单通道不达标时启用

## 5. 回归阈值映射
以下字段名供脚本和报告统一使用：
- `memory_bw_min_mb_s`
- `memory_stall_max_count`
- `memory_dropped_burst_max_count`

示例阈值（占位，后续按仿真实测更新）：
- `memory_bw_min_mb_s = 200`
- `memory_stall_max_count = 0`
- `memory_dropped_burst_max_count = 0`

## 6. 与自动化脚本的对齐要求
- `run_block_regression` 增加 `EnableMemoryMetrics` 时，应输出 `memory_metrics`。
- `run_m1_minimal_regression` 汇总 `memory_metrics` 并给出 `memory_gate` 判定。
- 文档字段名与 JSON 字段名必须完全一致，避免命名漂移。
