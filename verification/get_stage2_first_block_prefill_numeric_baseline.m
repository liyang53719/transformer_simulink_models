function baseline = get_stage2_first_block_prefill_numeric_baseline()
%GET_STAGE2_FIRST_BLOCK_PREFILL_NUMERIC_BASELINE Baseline spec for the first-block Simulink prefill regression.

    time = (0:20)';

    baseline = struct();
    baseline.stop_time = double(time(end));
    baseline.stimulus = struct();
    baseline.stimulus.time = time;
    baseline.stimulus.mode_decode = zeros(size(time));
    baseline.stimulus.start = [1; zeros(numel(time) - 1, 1)];
    baseline.stimulus.eos_in = zeros(size(time));
    baseline.stimulus.in_valid = [ones(4, 1); zeros(numel(time) - 4, 1)];
    baseline.stimulus.out_ready = ones(size(time));
    baseline.stimulus.in_hidden = [0.25; -0.50; 0.75; 1.00; zeros(numel(time) - 4, 1)];
    baseline.stimulus.in_residual = [-0.125; 0.375; -0.625; 0.875; zeros(numel(time) - 4, 1)];
    baseline.stimulus.cfg_seq_len = 4;
    baseline.stimulus.cfg_token_pos = 1;
    baseline.stimulus.cfg_eps = 1e-5;
    baseline.stimulus.stop_req = 0;
    baseline.stimulus.cfg_weight_num_heads = 12;
    baseline.stimulus.cfg_weight_page_base = 64;
    baseline.stimulus.cfg_weight_page_stride = 8;
    baseline.stimulus.cfg_rope_theta_scale = 1;
    baseline.stimulus.cfg_rope_sin_mix_scale = 1;

    baseline.expected_out_valid = [0; 1; 0; 1; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    baseline.expected_out_hidden = [-0.125; -0.125; -0.125; -0.125; -0.125; 0.375; 0.375; 0.375; 202637968; 0.375; -0.625; -0.625; -0.625; -0.625; -0.625; 0.875; 0.875; 0.875; 0.875; 0.875; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
end