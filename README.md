# transformer_simulink_models

面向 Qwen2-1.5B 的 Transformer NPU 设计与 RTL 生成项目。

本仓库目标：
- 使用 Simulink HDL Coder 为主线生成可综合 Verilog。
- 建立从算法参考到 RTL 仿真的一致性验证链路。
- 逐步实现 prefill + decode 的硬件自主执行，直到 EOS 自动停机。

## 当前阶段
- 阶段：项目初始化（治理与流程基线）
- 下一里程碑：完成最小可运行架构定义并开始 Milestone 1（单 Block）

## 文档导航
- 长期主线任务：`docs/01_mainline_plan.md`
- 实时状态跟进：`docs/02_live_status.md`
- 版本提交与推送规则：`docs/03_git_workflow_rules.md`
- M1 执行计划：`docs/04_m1_execution_plan.md`
- M1 Block 接口规格：`docs/05_m1_block_interface_spec.md`
- M1 测试向量与误差预算：`docs/06_m1_testvector_error_budget.md`
- M1 Simulink 顶层占位清单：`docs/07_m1_simulink_top_placeholder.md`
- M1 自动化与一致性执行：`docs/08_m1_automation_and_consistency.md`

## 最小执行入口
- MATLAB 脚本入口说明：`scripts/README.md`

## 开发原则
- 先正确，再优化：先打通功能闭环，再推进吞吐优化。
- 先行为一致，再 RTL 调试：减少 Verilog 端功能性返工。
- 小步提交、可回溯：每轮可验证改动都应提交并尽快推送。

## 启动步骤
1. 阅读上述三份核心文档。
2. 在 `docs/02_live_status.md` 更新“今日目标”。
3. 按任务分支或主线小步提交开发内容。
4. 每轮改动完成后执行对应验证并更新状态记录。
