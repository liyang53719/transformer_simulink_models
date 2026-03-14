# 长期主线任务（Mainline Plan）

更新时间：2026-03-14
负责人：Yang + Copilot

## 0. 总目标
在 ASIC 预研导向下，构建可由 Simulink HDL Coder 生成 RTL 的 Transformer NPU，支持 Qwen2-1.5B 的 prefill + decode 推理闭环，并完成 Verilog 仿真可运行验证。

## 1. 约束与阶段策略
- 首版基线：FP16（先跑通）
- 首个端到端范围：上下文 128 token
- 首阶段验收优先级：tokens/s（在功能正确前提下）
- 运行时策略：硬件自调度优先，采用最小元数据，不上完整编译器

## 2. 分阶段里程碑

### M0: 项目治理与规范基线（已启动）
- 建立主线计划、实时状态、提交推送规则。
- 明确目录结构、命名规则、验证入口。

### M1: 单 Block 功能闭环
- 目标：单个 Transformer Block 跑通 prefill + decode。
- 覆盖：RMSNorm、RoPE、QKV、Attention、SwiGLU、Residual。
- 交付：
  - 行为级模型通过参考对齐。
  - HDL 生成成功且可综合。

### M2: 端到端 128 token 闭环
- 目标：完整层堆叠，完成 prefill + decode + EOS 停机。
- 覆盖：Embedding/LM Head/argmax 采样（首版）。
- 交付：
  - Verilog 仿真跑通固定 prompt。
  - 输出 token 序列可复现。

### M3: 吞吐优化
- 目标：提升 tokens/s。
- 手段：tile、双缓冲、流水线、KV cache 访问优化。
- 交付：
  - prefill/decode 分别给出吞吐统计。
  - 瓶颈定位报告（算力或带宽）。

### M4: 量化扩展
- 目标：从 FP16 过渡到低比特方案（优先 W8A16）。
- 交付：
  - 精度-性能-面积权衡报告。
  - 量化版本回归基线。

## 3. 模块分解（硬件视角）
- 主控制器 FSM（INIT/PREFILL/DECODE/EOS/STOP）
- Matmul 计算阵列
- Norm + RoPE 处理单元
- Attention 单元（含 mask/softmax）
- FFN（SwiGLU）单元
- KV Cache 管理器
- DMA 与双缓冲控制
- 片外接口（后续 ASIC 方向映射到总线协议）
- 采样与停机控制

## 4. 验证主线
- 参考链路：PyTorch/Qwen2 参考 -> MATLAB 参考 -> Simulink 行为 -> RTL 仿真
- 规则：先算子级，再 Block 级，再端到端。
- 每一里程碑必须有固定测试向量与回归记录。

## 5. 退出条件（项目完成定义）
- 在固定测试集上，端到端 Verilog 仿真可稳定跑完 128 token。
- 硬件可自主执行 prefill + decode 并基于 EOS 停机。
- 至少一轮吞吐优化完成并有可复现实验记录。
