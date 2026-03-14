# 实时状态跟进（Live Status）

本文件用于记录“当前在做什么、做到哪里、下一步做什么”。

更新时间：2026-03-14

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

### 2026-03-14
- 初始化本仓库 git 管理与远端 origin。
- 创建主线任务、实时状态、提交流程三份核心文档。
- 推送首个基线提交到远端 main 分支。
- 新增 M1 执行计划文档与目录骨架（simulink/matlab_ref/rtl/verification/scripts）。
- 完成 M1 接口规格、测试向量模板、Simulink 顶层占位清单三份执行文档。
- 新增 scripts/verification 最小可执行脚手架，并补充 Simulink 占位模型创建与一致性检查脚本。
- 新增真实 qwen2 block 适配层与真实参考回归入口，支持接入外部 `+qwen2` / `+qwen2_quant` 路径验证。
- 确认 `matlab_ref/module` 下 AWQ/GPTQ/GGUF 三个软链接可见，并为真实回归入口增加 module 别名自动解析。
- 下一步：在 MATLAB 中执行真实参考回归并记录首轮 PASS/FAIL 日志。

## 当前风险
- 尚未冻结 Qwen2-1.5B 精确模型配置（层数/维度/头数）到仓库配置文件。
- 真实参考回归依赖本地 MATLAB path 和参数文件格式，首次联调可能出现接口不匹配。

## 阻塞列表
- 暂无。

## 下一步（按优先级）
1. 在 MATLAB 执行 `run_m1_real_reference_regression('<paramsFile>')` 完成真实路径联调。
2. 若真实路径失败，记录失败日志并回退 `run_m1_minimal_regression` 保持基线可用。
3. 在 Simulink 占位模型基础上对齐第一版真实 block 接口信号。
