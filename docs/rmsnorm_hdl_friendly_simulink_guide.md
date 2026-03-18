# RMSNorm HDL 友好 Simulink 建模指南

## 1. 目标

本文总结本次在 `transformer_simulink/layer` 下实现 RMSNorm 流式 DUT、推动 HDL Coder 生成 Verilog、并完成 VCS/Verdi/FSDB 验证过程中的经验。重点不是介绍 RMSNorm 数学本身，而是回答下面几个工程问题：

1. 如何把算法改写成 HDL Coder 更容易接受的 Simulink 结构。
2. 如何避免 MATLAB Function 中隐藏的 double 计算污染 HDL。
3. 如何在浮点数据通路下规避 delay balancing、sample time、mixed type、algebraic loop 等常见问题。
4. 如何从 Simulink 模型一路走到可仿真的 Verilog、filelist、testbench、FSDB 波形。


## 2. 本次设计的真实边界条件

本次 DUT 不是一个“矩阵整体输入、矩阵整体输出”的教学模型，而是一个更接近真实硬件的数据搬运与计算一体化模块：

1. `X` 的尺寸为 `64 x 1536`。
2. `g` 的尺寸为 `1 x 1536`。
3. DDR 接口每拍返回 `256 bit`，对应 `8 x single`。
4. DUT 先装载 gamma，再根据 `ddrReadAddr/ddrReadEn` 向外请求 token 数据。
5. DUT 内部缓存 token/gamma，按 beat 累加平方和，计算 `rsqrt`，再逐 beat 输出结果。
6. on-chip 不能容纳任意大输入，因此建模必须显式体现流式读取、局部缓存、状态机控制。

这类边界条件直接决定了建模方式：

1. 需要控制路径与算术路径分离。
2. 需要 beat 级、而不是 tensor 级接口。
3. 需要显式 state machine。
4. 需要显式时序延迟对齐。


## 3. 设计演进路径

本次工作实际经历了四个阶段：

### 3.1 纯算法参考

最开始只有 MATLAB 算法参考实现，适合作为 golden model，但不适合直接综合。

问题在于：

1. 张量级接口过大，不是真实 RTL 边界。
2. 隐式矩阵运算过多，不利于子模块拆解。
3. 没有显式握手与时序状态。

### 3.2 MATLAB 流式 DUT

随后实现了 `+transformer_impl/+layer/rmsNormalization.m`，把算法重写为流式状态机。这个版本的意义是：

1. 先确认接口语义。
2. 先确认控制流程。
3. 先确认输入输出拍序。
4. 为 Simulink 建模提供 executable spec。

这一层非常关键。没有一个能跑通的流式 MATLAB DUT，直接做 Simulink HDL 模型会让定位问题极其低效。

### 3.3 Simulink 流式 DUT

再往后把整个设计搬进 Simulink，并逐步拆成显式算术子模块。最终主要结构是：

1. `SquareBeat`
2. `BeatReduce`
3. `BeatAccumulator`
4. `ScalarRsqrt`
5. `InvRmsLatch`
6. `LaneMultiply`
7. `Controller`

这是本次 HDL 友好的核心结构。

### 3.4 HDL 生成与外部仿真

最后完成：

1. Simulink HDL 生成
2. Verilog `filelist.f`
3. FSDB testbench
4. VCS 编译仿真
5. Verdi 波形打开

这一步把“模型能跑”推进到“生成 RTL 可观察、可波形调试、可外部仿真”。


## 4. HDL 友好建模总原则

### 4.1 先固定接口，再搭数据通路

不要先从块图开始，而要先回答：

1. 输入什么时候有效。
2. 输出什么时候有效。
3. 中间状态什么时候清零。
4. 数据从哪里来。
5. 数据在哪一级被消费。

如果这些问题没回答清楚，后面的 Delay、Spec、Valid、Enable 都会变成补丁式修修补补。

### 4.2 控制与算术分层

本次实践证明，最稳定的结构是：

1. 控制路径单独在 `Controller`。
2. 算术路径拆成多个简单子模块。
3. 控制信号只驱动选择、使能、清零、捕获。
4. 算术子模块尽量只做数值计算，不掺状态机。

原因很直接：

1. HDL Coder 对短小、明确的算术图更友好。
2. 控制路径单独集中后，时序关系更容易检查。
3. 问题定位时能快速判断是状态问题还是算术问题。

### 4.3 不要依赖 HDL Coder 自动“理解意图”

在 Simulink 里能跑通，不等于 HDL Coder 能稳定生成高质量 RTL。生成器更偏好：

1. 类型明确
2. 延迟明确
3. 维度明确
4. 数据依赖明确
5. 反馈路径明确

“模型很聪明，工具会猜到我的意思”这种思路在 HDL 流程里通常会失败。


## 5. 数据类型经验总结

### 5.1 控制口尽量用 boolean

本次明确收敛后的结论是：

1. `reset_1`
2. `start`
3. `cfgGammaValid`
4. `ddrDataValid`

这些都应该是 `boolean`，而不是 `double`。

收益：

1. 顶层端口在生成 RTL 中自然成为 1-bit。
2. 避免控制逻辑中引入不必要的浮点辅助模块。
3. 时序语义更清晰。

### 5.2 数据口统一 single

本次数据通路的主线是 `single`。包括：

1. gamma beat
2. DDR data beat
3. beat sum
4. invRms
5. 输出 beat

如果控制口和数据口混在一起做乘法、加法、Switch 控制，工具容易报 mixed type 问题。

### 5.3 MATLAB Function 内避免隐式 double

这次一个关键问题是：顶层口已经改成 boolean/single 了，但生成的 `Controller.v` 仍然引入 `nfp_add_double`、`nfp_sub_double`、`nfp_gain_pow2_double` 等 helper。

根因不是顶层 Inport，而是 MATLAB Function 中存在类似下面的写法：

1. `double(gammaWriteBeat)`
2. `double(receiveBeatIndex)`
3. `double(outputBeatIndex)`
4. 循环里用 double 参与地址计算

经验结论：

1. 只要在 MATLAB Function 里出现 double 风格索引运算，HDL 里就可能偷偷长出 double helper。
2. 地址与索引计算应尽量使用 `uint8/uint16` 等整数类型。
3. lane 级赋值可以显式展开，不要依赖工具把复杂索引表达式自动优化掉。


## 6. 子模块拆分经验

### 6.1 `SquareBeat`

职责非常单一：

1. 输入一个 8-lane beat。
2. 每个 lane 单独平方。

这种模块最适合独立存在，因为：

1. 输入输出维度稳定。
2. 无状态。
3. 延迟容易估算。

### 6.2 `BeatReduce`

职责是把 8-lane 的平方结果加成一个标量 `beatSum`。

建议：

1. 结构上尽量固定为树形加法。
2. 不要和状态机混在一起。
3. 让输出只承担“当前 beat 的局部和”。

### 6.3 `BeatAccumulator`

这是本次最容易出问题、也最值得总结的模块。

最初的直接浮点累加写法在 Simulink 中功能可以对，但 HDL 生成阶段容易遇到：

1. delay balancing 失败
2. 分布式流水插不进去
3. 浮点反馈路径太难平衡

最终可行方案是 banked / interleaved accumulator。

为什么有效：

1. 把长反馈链拆成多个独立 bank。
2. 减轻单一路径的浮点反馈压力。
3. 给工具更多可实现的时序空间。

经验：

1. 浮点反馈累加器要谨慎。
2. 如果工具在反馈环上反复失败，优先考虑结构重写，不要一直调参数。
3. 通过 banking 消除延迟平衡瓶颈，通常比硬堆 Delay 更稳。

### 6.4 `ScalarRsqrt`

建议保持独立：

1. 输入一个 single 标量。
2. 输出一个 single 标量。

这样便于后续把 rsqrt 近似、LUT、CORDIC、牛顿迭代替换进去。

### 6.5 `InvRmsLatch`

作用是把某个计算周期得到的 `invRms` 明确捕获下来，供后续整个输出阶段使用。

这一层很重要，因为它把“结果生成时刻”和“结果使用阶段”明确切开了。

### 6.6 `LaneMultiply`

职责是：

1. `x * g * invRms`
2. 逐 lane 并行完成

建议不要把它和控制信号混在一个大 MATLAB Function 中，否则会造成数据与控制强耦合。


## 7. 控制器建模经验

### 7.1 MATLAB Function 可以保留，但必须克制

`Controller` 这次最终保留为 MATLAB Function，是可行的，但前提是：

1. 只做清晰的状态机逻辑。
2. 只做简单、强类型的索引与赋值。
3. 不掺复杂矩阵运算。
4. 不掺 double。

### 7.2 索引不要“写得太聪明”

曾经导致 HDL 里生成 double helper 的一个重要原因是索引表达式过于紧凑、过于 MATLAB 风格。

更稳妥的方法是：

1. 先算 `startIdx = gammaWriteBeat * uint16(8)`
2. 再做每个 lane 的显式赋值
3. 每一步保持整数类型

也就是说，在 HDL 目标下，展开和啰嗦常常比抽象和简洁更好。

### 7.3 输出有效信号与输出数据要成对设计

如果 `outValid` 与 `outBeat` 不来自同一时序参考，后期一定会靠一堆 delayMatch 修补。

经验是：

1. 明确谁是“主基准”。
2. 让其他信号对齐到它。
3. 不要两边各自插 Delay，最后靠试错找对齐。


## 8. Sample Time、Delay、对齐经验

### 8.1 全路径 SampleTime 明确设为 1

本次稳定下来的配置是所有相关路径使用离散单速率：

1. `FixedStepDiscrete`
2. `FixedStep = 1`
3. 关键模块 `SampleTime = 1`

如果有隐式继承、不同步率或不清晰的速率传播，HDL Coder 常常会把错误报在完全无关的位置。

### 8.2 Signal Specification 不要省

本次大量使用 `Signal Specification`，原因很实际：

1. 固定维度
2. 固定数据类型
3. 阻断工具错误推导

对于 beat 向量、标量累加结果、invRms 等节点，显式 Spec 非常有帮助。

### 8.3 Delay 不是越多越好，而是要围绕语义插

这次最终 `DUT.v` 中出现了大量 `delayMatch*`。这说明生成器确实做了很多对齐，但设计时不要依赖“多放几个 Delay 试试看”。

推荐原则：

1. 先定义数据在哪一级有效。
2. 再定义 valid 在哪一级有效。
3. 最后补最少的 Delay。

### 8.4 如果工具在 delay balancing 上反复失败，优先改结构

本次最有价值的经验之一就是：

1. delay balancing 失败通常不是“Delay 数量不够”。
2. 更常见的根因是结构不适合自动平衡。
3. 浮点反馈累加环尤其如此。

遇到这种问题，优先考虑：

1. 切断反馈环
2. 多 bank 化
3. 增加 pipeline stage
4. 把控制与数据路径进一步解耦


## 9. HDL Coder 调试经验

### 9.1 先追“第一性错误”

HDL Coder 报错往往是连锁的。常见误区是看到 10 条错误就试图同时修 10 个地方。

正确方法：

1. 找第一条真正的结构性错误。
2. 修完重新生成。
3. 观察剩余错误是否自然消失。

### 9.2 先看生成的 RTL，再看目录残留

本次一个实际问题是：HDL 目录里还残留旧的 `nfp_add_double.v`、`nfp_sub_double.v` 等文件。

这类文件不一定代表当前生成逻辑真的在使用它们。

所以排查时应优先：

1. 看最新 `Controller.v`
2. grep 当前是否仍引用相关 helper
3. 再判断目录中哪些是历史残留

### 9.3 生成成功不代表接口最优

生成成功只是第一步。还需要进一步看：

1. 端口类型是否合理
2. helper module 是否冗余
3. 延迟链是否过长
4. 总体吞吐是否合理


## 10. 验证流程经验

### 10.1 参考顺序

推荐验证顺序：

1. golden MATLAB 算法
2. 流式 MATLAB DUT
3. Simulink DUT
4. HDL 生成结果
5. VCS testbench
6. Verdi/FSDB 波形

每一层都只和上一层比较，不要跨层直接定位。

### 10.2 测试类路径问题

本次测试类有一个容易忽略的问题：

1. 在 `Constant` property 中保存 package function handle
2. MATLAB 在类加载阶段就尝试解析这些 handle
3. 这时 repo path 还没加好
4. 导致测试还没真正执行就失败

经验：

1. package function 解析尽量放到 runtime。
2. `TestClassSetup` 中先把路径加好。
3. 测试体中再直接调用 package 函数。

### 10.3 外部 testbench 的价值

本次补了 `filelist.f` 和 `tb_rmsNormalization_fsdb.sv`，收益很明确：

1. 能独立于 Simulink 观察 RTL 波形。
2. 能在 VCS 中确认 `done`、`ddrReadEn`、`outValid` 等真实时序。
3. 能作为后续替换接口、插入流水线后的回归骨架。


## 11. 本次已经验证过的关键结论

1. 控制端口可以安全收敛为 boolean，并在 RTL 中成为 1-bit。
2. 控制器中隐藏的 double 索引运算会污染 HDL，需要彻底清除。
3. banked accumulator 是解决浮点 delay balancing 问题的有效结构。
4. 使用 VCS + Verdi + FSDB 可以把生成 RTL 拉到外部仿真环境中验证。
5. 仅仅“生成通过”不够，还要确认 RTL 不再引用不期望的 helper 模块。


## 12. 常见踩坑清单

### 12.1 控制信号是 double

症状：

1. 顶层端口宽度异常
2. 控制路径长出浮点模块

处理：改成 boolean。

### 12.2 MATLAB Function 中 double 索引

症状：

1. `Controller.v` 引入 `nfp_add_double` 等 helper

处理：索引全改为整数类型，并显式展开。

### 12.3 浮点反馈累加器 delay balancing 失败

症状：

1. HDL 生成卡在 pipeline / balancing

处理：改结构，优先 bank 化。

### 12.4 valid 与 data 错位

症状：

1. 仿真输出数值对，但拍序错
2. RTL testbench 跑不通或 done 不到

处理：围绕语义重新定义主对齐点，不要盲目加 Delay。

### 12.5 目录中残留旧 helper 文件造成误判

症状：

1. 看到目录有 `nfp_*double*.v` 就以为当前控制器仍依赖 double

处理：以最新 RTL 是否引用为准。


## 13. 下一步 DUT 改进空间

下面两项是下一阶段最值得做的架构升级。

### 13.1 边界接口改成 256-bit packed 端口

当前 DUT 顶层接口仍是拆开的 8 个 `single` 端口，例如：

1. `cfgGammaBeat_0` 到 `cfgGammaBeat_7`
2. `ddrDataBeat_0` 到 `ddrDataBeat_7`
3. `outBeat_0` 到 `outBeat_7`

这在验证阶段是可接受的，但在真实硬件接口上并不理想。

建议下一步统一成 packed 端口：

1. `input [255:0] cfgGammaBeat`
2. `input [255:0] ddrDataBeat`
3. `output [255:0] outBeat`

收益：

1. 更符合 DDR/AXI/片上总线的真实 beat 语义。
2. 顶层端口数明显减少，接口更清晰。
3. 更容易继续封装为 AXI-Stream 或自定义 256-bit 数据通道。
4. testbench、filelist、后续 SoC 集成更简单。

实现建议：

1. DUT 边界先改 packed。
2. DUT 内部再用 Unpack/Bus Selector 或 MATLAB Function 小模块拆成 8 lane。
3. 算术子模块内部仍然保留 8-lane single 语义，不必一次性把内部全部改写成 packed 位向量操作。

也就是说，先改边界，不急着改内部算法表达。

### 13.2 从独占式执行改成读算重叠流水

当前 DUT 的工作方式仍然偏“独占式”：

1. 读完当前 token
2. 计算平方和
3. 得到 invRms
4. 输出整个 token
5. 最终结果出来前，不继续读下一份数据

这会导致 DDR 带宽利用率和计算资源利用率都偏低。

建议的下一步方向是阶段流水化：

1. Stage A: 读取 token N+1
2. Stage B: 对 token N 做平方和累加
3. Stage C: 对 token N-1 做归一化输出

这样可以形成读、算、写重叠。

#### 13.2.1 最直接的结构方案

可以把 token 处理拆成三个相互解耦的子流水段：

1. `Load Engine`
2. `Reduce Engine`
3. `Output Engine`

每段之间放 FIFO / queue / descriptor：

1. `token metadata`
2. `partial sum result`
3. `latched invRms`

#### 13.2.2 最小改动版本

如果不想一次性大改，可以先做“乒乓缓存”：

1. bank A 用于当前输出 token
2. bank B 用于下一 token 预取
3. 输出 token A 的同时，后台读取 token B
4. A/B 轮换

这样就能先把“读取”和“输出”重叠起来。

#### 13.2.3 更进一步的版本

当乒乓缓存稳定后，可以继续做：

1. 多 token in-flight
2. `sum` 与 `output` 两条通路并发
3. 状态机从单 token ownership 改成 descriptor 驱动

这会显著提升吞吐，但控制复杂度也会上升。

### 13.3 与当前生成 RTL 相关的具体观察

从当前 `DUT.v` 可以看到：

1. `outValid`、`ddrReadEn`、`done`、`busy` 等输出后面仍串有较长的 `delayMatch` 对齐链。
2. `cfgGammaBeat_*`、`ddrDataBeat_*` 也先进入长延迟寄存器阵列再喂给 `Controller`。
3. 这说明当前结构更偏向“先把时序对齐做稳”，而不是“吞吐最优”。

因此下一阶段的重点不应是继续做微调，而应是：

1. 重新组织边界接口
2. 重新规划流水阶段
3. 减少全局 delayMatch 依赖


## 14. 推荐的下一阶段实施顺序

建议按下面顺序推进，而不是同时大改所有内容：

1. 先把边界从 8 x single 改成 256-bit packed。
2. 保持内部 8-lane 语义不变，确保功能回归通过。
3. 再引入乒乓缓存，让读取与输出重叠。
4. 最后再考虑多 token in-flight 和更激进的流水化。

这样每一步的风险都可控，也容易在 VCS/Verdi 中做波形对比。


## 15. 对后续建模的最终建议

如果目标是“可生成 HDL”，那么：

1. 先把接口、类型、状态机定义清楚。
2. 再搭最小可行算术模块。
3. 对每个中间节点显式给类型和维度。
4. 遇到 HDL 生成器无法处理的结构，优先改结构，而不是堆参数。
5. 始终保留一个外部 RTL 仿真 testbench，避免只在 Simulink 内自洽。

如果目标进一步升级为“高吞吐、可综合、可集成”，那么：

1. 边界应尽早转向 packed bus。
2. 单 token 独占执行应尽早转向分阶段重叠流水。
3. 控制器应逐步从单大状态机演化为更清晰的 load / reduce / output 多阶段控制。


## 16. 本文适用范围

本文经验适用于以下类型的 Simulink HDL 项目：

1. 浮点数据通路
2. beat-based 流式输入输出
3. 带状态机的 transformer 层级算子
4. 需要从 Simulink 走到外部 RTL 仿真的场景

对于纯定点、纯组合、无状态的小模块，其中一些约束可以放宽；但对于 transformer 类流式算子，这些经验基本都是真正踩坑换来的。