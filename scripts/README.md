# scripts 目录说明

## 最小执行入口
- 生成 M1 占位向量：
  - `bootstrap_m1_vectors`
- 运行 M1 最小回归（算子 + block）：
  - `run_m1_minimal_regression`
- 运行 M1 真实参考回归（接入 +qwen2 / +qwen2_quant）：
  - `run_m1_real_reference_regression(paramsFile)`
- 生成 Simulink 顶层占位模型：
  - `create_qwen2_block_top_placeholder`
- 检查 Simulink 占位一致性：
  - `check_simulink_placeholder_consistency`

## 推荐顺序
1. `create_qwen2_block_top_placeholder`
2. `check_simulink_placeholder_consistency`
3. `run_m1_minimal_regression`

若要接入已有推理实现进行真实 block 参考对比：
1. 将 `+qwen2` 或 `+qwen2_quant` 加入 MATLAB path
2. 准备参数文件（`.mat`）
3. 执行 `run_m1_real_reference_regression('<paramsFile>')`

## 参考说明
以上脚手架的字段命名和运行流程参考了 `transformer-models` 仓中 `+qwen2` 与 `+qwen2_quant` 的推理与测试习惯（如 prefill/decode、KV cache、RuntimeConfig 结构），但代码实现保持本仓独立维护。
