# scripts 目录说明

## 最小执行入口
- 生成 M1 占位向量：
  - `bootstrap_m1_vectors`
- 运行 M1 最小回归（算子 + block）：
  - `run_m1_minimal_regression`
- 运行 M1 真实参考回归（接入 +qwen2 / +qwen2_quant）：
  - `run_m1_real_reference_regression(paramsFile)`
    - `run_m1_real_reference_regression('module_awq')`
    - `run_m1_real_reference_regression('module_awq','RunStage2FastSmoke',true)`
  - `run_m1_real_reference_regression('module_gptq')`
  - `run_m1_real_reference_regression('module_gguf')`
  - `run_m1_real_reference_regression('module_gguf','BaselineMode','real')`
  - `run_m1_real_reference_regression('module_awq','ReportDir','verification/reports')`
- 生成 Simulink 顶层占位模型：
  - `create_qwen2_block_top_placeholder`
- 实装 stage-1 子系统（rmsnorm_u + qkv_proj_u）：
  - `implement_stage1_rmsnorm_qkv`
- 检查 Simulink 占位一致性：
  - `check_simulink_placeholder_consistency`
- 检查顶层接口与冻结规格一致性：
  - `check_block_interface_spec_consistency`

## 推荐顺序
1. `create_qwen2_block_top_placeholder`
2. `implement_stage1_rmsnorm_qkv`
3. `check_simulink_placeholder_consistency`
4. `check_block_interface_spec_consistency`
5. `run_m1_minimal_regression`

若要接入已有推理实现进行真实 block 参考对比：
1. 将 `+qwen2` 或 `+qwen2_quant` 加入 MATLAB path
2. 准备参数文件（`.mat`）或配置 `matlab_ref/module` 软链接
3. 执行 `run_m1_real_reference_regression('<paramsFile>')` 或 `run_m1_real_reference_regression('module_awq')`

内置 module 别名（对应 `matlab_ref/module`）：
- `module_awq` -> `Qwen2.5-1.5B-Instruct-AWQ`
- `module_gptq` -> `Qwen2.5-1.5B-Instruct-GPTQ-Int4`
- `module_gguf` -> `qwen_gguf`

`BaselineMode` 参数：
- `real`：使用实时参考输出作为基线（适合真实模型联调）
- `stored`：使用 `verification/testvectors/*.mat` 中的 golden（默认流程）

报告输出：
- `run_m1_minimal_regression` 和 `run_m1_real_reference_regression` 会自动在 `verification/reports` 生成 JSON 报告。
- 可通过 `ReportDir` 或 `ReportPath` 覆盖输出位置。

## 参考说明
以上脚手架的字段命名和运行流程参考了 `transformer-models` 仓中 `+qwen2` 与 `+qwen2_quant` 的推理与测试习惯（如 prefill/decode、KV cache、RuntimeConfig 结构），但代码实现保持本仓独立维护。
