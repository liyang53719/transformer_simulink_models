# M1 执行计划（单 Block 闭环）

更新时间：2026-03-14

## 目标
在 Simulink 主线下完成单个 Transformer Block 的功能闭环，并建立与参考模型的一致性验证。

## 任务拆分
1. 接口冻结
- 定义 block 输入输出张量形状（hidden, q, k, v, residual）。
- 定义数据类型（首版 FP16）和定点占位字段。
- 定义时序接口（valid/ready、帧边界）。

2. 参考链路准备
- 在 `matlab_ref/` 建立 block 参考函数入口。
- 固定首批测试向量并写入 `verification/testvectors/`。
- 定义误差阈值与一致率统计方式。

3. Simulink 模块骨架
- 在 `simulink/models/` 建立 block 顶层模型。
- 子模块：RMSNorm、RoPE、QKV、Attention、FFN、Residual。
- 预留 KV cache 接口信号。

4. HDL 生成预检查
- 检查可综合约束（不支持操作、循环边界、内存映射）。
- 首次生成 RTL 到 `rtl/`。
- 确认生成日志无阻塞错误。

5. 回归验证
- 运行算子级用例。
- 运行 block 级用例。
- 更新 `docs/02_live_status.md` 的日志、风险、下一步。

## M1 验收标准
- 单 Block 行为输出与参考模型在阈值内一致。
- RTL 成功生成且可进入仿真。
- 至少 1 组固定向量回归可复现通过。
