# 实时状态跟进（Live Status）

本文件用于记录“当前在做什么、做到哪里、下一步做什么”。

更新时间：2026-03-17

## 使用规则
- 每次开始工作前：更新“今日目标”。
- 每轮改动验证后：追加一条“进展日志”。
- 每次 push 前：更新“当前风险”和“下一步”。
- 任何阻塞项必须进入“阻塞列表”。

## 今日目标
- [x] 完成项目治理基线文档
- [x] 建立首批目录与命名规范
- [x] 准备 M1（单 Block）任务拆分

## 进展日志

### 2026-03-17
- wrapper 联调路径新增 attention+DDR 集成 smoke，已纳入 fast suite。
- `qwen2_block_top` 支持 stage2 wrapper 模式下的外部 `w_rd_rsp_bus` 参数响应入口，用于把权重返回路径从内部 `axi_weight_rd_u` stub 迁移到 wrapper 外层 responder。
- `build_stage2_wrapper_tb_model` 现在会在需要时自动重建为 external-weight-response 版本 DUT，并在 wrapper 中显式创建 `weight_ref_u`，把 `w_rd_req_bus -> weight_ref_u -> w_rd_rsp_bus` 接通。
- `weight_ref_u` 进一步从简单缩放 stub 推进为“按请求地址返回带页签名的数据”，attention 联调 smoke 也开始检查外部响应数据非零且彼此可区分。
- attention 联调 smoke 进一步收紧为：直接检查 attention 请求地址与外部响应数据之间的精确映射关系，而不是只看 valid/非零活动。
- `cfg_weight_num_heads` / `cfg_weight_page_base` / `cfg_weight_page_stride` 根类型改为整数定点，避免 wrapper 配置真实分页参数时发生溢出。
- 验证通过：`run_stage2_wrapper_tb_smoke`、`run_stage2_attention_ddr_integration_smoke`、`run_stage2_smoke_suite_fast`。
- 验证通过：`run_m1_real_reference_regression('module_awq','EnableMemoryMetrics',true,'RunStage2FastSmoke',true)`。
- 下一步：继续推进 attention/weight path 的真实数据流细节，逐步减少内部 weight stub 依赖，并让更多参数返回语义贴近真实页布局。

### 2026-03-14
- 初始化本仓库 git 管理与远端 origin。
- 创建主线任务、实时状态、提交流程三份核心文档。
- 推送首个基线提交到远端 main 分支。
- 新增 M1 执行计划文档与目录骨架（simulink/matlab_ref/rtl/verification/scripts）。
- 完成 M1 接口规格、测试向量模板、Simulink 顶层占位清单三份执行文档。
- 新增 scripts/verification 最小可执行脚手架，并补充 Simulink 占位模型创建与一致性检查脚本。
- 新增真实 qwen2 block 适配层与真实参考回归入口，支持接入外部 `+qwen2` / `+qwen2_quant` 路径验证。
- 确认 `matlab_ref/module` 下 AWQ/GPTQ/GGUF 三个软链接可见，并为真实回归入口增加 module 别名自动解析。
- 新增真实回归自动报告落盘能力（JSON），可按 module/baseline 维度归档每轮结果。
- 完成 Simulink stage-1：顶层接口扩展到冻结规格，并实装 `rmsnorm_u` 与 `qkv_proj_u` 可运行子系统。
- 下一步：在 MATLAB 中执行真实参考回归并记录首轮 PASS/FAIL 日志。

## 当前风险
- 尚未冻结 Qwen2-1.5B 精确模型配置（层数/维度/头数）到仓库配置文件。
- 真实参考回归依赖本地 MATLAB path 和参数文件格式，首次联调可能出现接口不匹配。
- 参数外部返回路径目前优先在 wrapper 联调模式启用，尚未推广到所有非 wrapper 验证入口。

## 阻塞列表
- 暂无。

## 下一步（按优先级）
1. 继续推进 attention/weight path 的真实数据流细节，逐步让 wrapper 外部参数路径承接更多模块。
2. 复查 `run_m1_real_reference_regression` 当前失败点，判断是否由新 wrapper 主线之外的旧入口引起。
3. 在 Simulink 顶层与 wrapper 联调模型基础上收敛第一版真实 block 接口信号。
