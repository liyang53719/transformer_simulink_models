# M1 自动化与一致性执行说明

更新时间：2026-03-14

## 1. 目标
将 M1 的“文档定义”变成“可执行入口”，减少手工步骤。

## 2. 已提供脚本
- `scripts/create_qwen2_block_top_placeholder.m`
- `scripts/check_simulink_placeholder_consistency.m`
- `scripts/bootstrap_m1_vectors.m`
- `scripts/run_m1_minimal_regression.m`
- `scripts/run_m1_real_reference_regression.m`
- `verification/run_operator_smoke_test.m`
- `verification/run_block_regression.m`

## 3. 一次性执行顺序
在 MATLAB 中进入仓库根目录后执行：
1. `addpath('scripts'); addpath('verification'); addpath('matlab_ref');`
2. `create_qwen2_block_top_placeholder;`
3. `check_simulink_placeholder_consistency;`
4. `implement_stage1_rmsnorm_qkv('', struct('StageProfile','stage2_memory_ready'));`
5. `check_block_interface_spec_consistency('', struct('Profile','v2_memory_first'));`
6. `run_m1_minimal_regression;`
7. `run_m1_minimal_regression(struct('RegressionOptions', struct('EnableMemoryMetrics', true)));`

真实参考路径建议使用 name-value 调用方式：
- `scripts.run_m1_real_reference_regression('module_awq', 'EnableMemoryMetrics', true)`
- `scripts.run_m1_real_reference_regression('module_gptq', 'EnableMemoryMetrics', true)`
- `scripts.run_m1_real_reference_regression('module_gguf', 'EnableMemoryMetrics', true)`

## 4. 通过标准
- 占位模型创建成功并通过一致性检查。
- 算子 smoke test PASS。
- block regression PASS。
- memory-model-check PASS（若启用 `EnableMemoryMetrics=true`）。

说明：当前版本已升级为硬门禁，启用 `EnableMemoryMetrics=true` 时必须提供实值指标：
- `memory_metrics.available = true`
- `memory_gate.pass = true`
- 若指标缺失，则 `memory_gate.pass = false`

## 4.1 访存门禁（V2）
- `memory_metrics.master_bw_mb_s.<master_name> >= memory_bw_min_mb_s`
- `memory_metrics.stall_count <= memory_stall_max_count`
- `memory_metrics.dropped_burst_count <= memory_dropped_burst_max_count`

若未启用内存建模，则允许 `memory_metrics` 为空，并仅执行功能门禁。

## 5. 后续替换策略
- 当真实 block 参考函数可用后，用真实实现替换 `qwen2_block_ref_placeholder`。
- 保留现有脚本入口与 case 命名，避免回归链路重写。

## 6. 真实参考适配路径（已支持）
- 通过 `run_m1_real_reference_regression(paramsFile)` 可优先尝试接入已有 `+qwen2` / `+qwen2_quant` 的 block 实现。
- 若 MATLAB path 或参数结构不满足要求，会自动报错提示并可退回 placeholder 模式。
- 支持 module 别名输入：`module_awq`、`module_gptq`、`module_gguf`，自动映射到 `matlab_ref/module` 下对应目录。

## 7. 报告字段约定（新增）
最小回归 JSON 报告在保持原字段兼容的前提下新增：
- `memory_metrics.master_bw_mb_s`：结构体，记录各 master 的 MB/s
- `memory_metrics.stall_count`：整数，事务阻塞计数
- `memory_metrics.dropped_burst_count`：整数，丢突发计数
- `memory_gate.pass`：布尔，访存门禁结果
- `memory_gate.reason`：字符串，失败原因或降级说明

其中 `run_memory_model_check` 当前采用“测试向量 + AXI burst 传输模型”生成实值指标，后续可无缝替换为 SoC Blockset DDR 实测采集。
