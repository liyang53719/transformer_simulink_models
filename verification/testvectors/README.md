# testvectors 目录说明

本目录存放 M1 回归输入向量与黄金输出。

当前生成方式：
- 由 MATLAB 脚本 `scripts/bootstrap_m1_vectors.m` 生成：
  - `block_prefill_case01.mat`
  - `block_decode_case01.mat`

字段约定遵循：
- `meta`
- `input`
- `golden`

详见：`docs/06_m1_testvector_error_budget.md`
