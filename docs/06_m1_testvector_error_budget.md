# M1 测试向量与误差预算模板

更新时间：2026-03-14
适用阶段：M1 单 Block 闭环

## 1. 目标
定义统一测试向量格式与误差评估规则，保证 MATLAB 参考、Simulink、RTL 结果可直接比对。

## 2. 测试集分层

### 2.1 算子级
- RMSNorm
- RoPE
- Attention（含 causal mask）
- SwiGLU

### 2.2 Block 级
- 单次 prefill（小 token 批）
- 单次 decode（1 token，含 KV cache 读取）

## 3. 目录与命名
测试向量目录：`verification/testvectors/`

命名建议：
- `op_rmsnorm_case01.mat`
- `op_rope_case01.mat`
- `op_attention_case01.mat`
- `block_prefill_case01.mat`
- `block_decode_case01.mat`

## 4. 向量字段规范（MAT 文件或等价格式）
每个 case 至少包含：
- `meta.case_id`
- `meta.seed`
- `meta.mode`（prefill/decode）
- `meta.seq_len`
- `meta.token_pos`
- `input.hidden`
- `input.residual`
- `input.kv_cache`（decode 必填）
- `golden.output_hidden`
- `golden.output_kv`

## 5. 误差指标
- `max_abs_err = max(abs(y_ref - y_dut))`
- `mean_abs_err = mean(abs(y_ref - y_dut))`
- `rel_l2_err = ||y_ref - y_dut||2 / (||y_ref||2 + 1e-12)`
- `match_ratio = matched_elements / total_elements`

## 6. M1 默认阈值（FP16 基线）
- `max_abs_err <= 3e-2`
- `mean_abs_err <= 3e-3`
- `rel_l2_err <= 2e-2`
- `match_ratio >= 99.0%`

说明：阈值用于 M1 快速闭环，M2 前可按统计结果收紧。

## 7. 回归门禁
每次提交前至少通过：
1. 1 组算子级用例
2. 1 组 Block prefill 用例
3. 1 组 Block decode 用例

若任一失败：
- 在 `docs/02_live_status.md` 记录失败 case、指标和处理计划。
- 禁止将该轮标记为“功能完成”。

## 8. 报告模板（文本）
- Case: <case_id>
- Mode: <prefill/decode>
- max_abs_err: <value>
- mean_abs_err: <value>
- rel_l2_err: <value>
- match_ratio: <value>
- Result: PASS/FAIL
