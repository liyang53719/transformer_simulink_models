function baseline = get_stage2_first_block_prefill_numeric_baseline(options)
%GET_STAGE2_FIRST_BLOCK_PREFILL_NUMERIC_BASELINE Baseline spec for the first-block Simulink prefill regression.

    if nargin < 1 || ~isstruct(options)
        options = struct();
    end

    numTokens = max(1, round(double(getFieldOr(options, 'NumTokens', 4))));
    tailCycles = max(16, round(double(getFieldOr(options, 'TailCycles', 16))));
    stopTime = double(getFieldOr(options, 'StopTime', numTokens + tailCycles));
    time = (0:stopTime)';

    tokenIndex = (0:numTokens-1)';
    baseHidden = [0.25; -0.50; 0.75; 1.00];
    baseResidual = [-0.125; 0.375; -0.625; 0.875];
    tokenHidden = baseHidden(mod(tokenIndex, numel(baseHidden)) + 1) + 0.03125 * floor(tokenIndex / numel(baseHidden));
    tokenResidual = baseResidual(mod(tokenIndex, numel(baseResidual)) + 1) - 0.015625 * floor(tokenIndex / numel(baseResidual));

    baseline = struct();
    baseline.stop_time = double(stopTime);
    baseline.stimulus = struct();
    baseline.stimulus.time = time;
    baseline.stimulus.mode_decode = zeros(size(time));
    baseline.stimulus.start = [1; zeros(numel(time) - 1, 1)];
    baseline.stimulus.eos_in = zeros(size(time));
    baseline.stimulus.in_valid = [ones(numTokens, 1); zeros(max(0, numel(time) - numTokens), 1)];
    baseline.stimulus.out_ready = ones(size(time));
    baseline.stimulus.in_hidden = [tokenHidden; zeros(max(0, numel(time) - numTokens), 1)];
    baseline.stimulus.in_residual = [tokenResidual; zeros(max(0, numel(time) - numTokens), 1)];
    baseline.stimulus.cfg_seq_len = numTokens;
    baseline.stimulus.cfg_token_pos = 1;
    baseline.stimulus.cfg_eps = 1e-5;
    baseline.stimulus.stop_req = 0;
    baseline.stimulus.cfg_weight_num_heads = 12;
    baseline.stimulus.cfg_weight_page_base = 64;
    baseline.stimulus.cfg_weight_page_stride = 8;
    baseline.stimulus.cfg_rope_theta_scale = 1;
    baseline.stimulus.cfg_rope_sin_mix_scale = 1;

    if numTokens == 4
        baseline.expected_out_valid = [0; 1; 0; 1; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
        baseline.expected_out_hidden = [-0.125; -0.125; -0.125; -0.125; -0.125; 0.375; 0.375; 0.375; 202637968; 0.375; -0.625; -0.625; -0.625; -0.625; -0.625; 0.875; 0.875; 0.875; 0.875; 0.875; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    else
        baseline.expected_out_valid = [];
        baseline.expected_out_hidden = [];
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end