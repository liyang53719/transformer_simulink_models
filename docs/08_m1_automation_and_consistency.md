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
4. `run_m1_minimal_regression;`

## 4. 通过标准
- 占位模型创建成功并通过一致性检查。
- 算子 smoke test PASS。
- block regression PASS。

## 5. 后续替换策略
- 当真实 block 参考函数可用后，用真实实现替换 `qwen2_block_ref_placeholder`。
- 保留现有脚本入口与 case 命名，避免回归链路重写。

## 6. 真实参考适配路径（已支持）
- 通过 `run_m1_real_reference_regression(paramsFile)` 可优先尝试接入已有 `+qwen2` / `+qwen2_quant` 的 block 实现。
- 若 MATLAB path 或参数结构不满足要求，会自动报错提示并可退回 placeholder 模式。
- 支持 module 别名输入：`module_awq`、`module_gptq`、`module_gguf`，自动映射到 `matlab_ref/module` 下对应目录。
