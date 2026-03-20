function baseline = get_stage2_first_block_prefill_reference_baseline(rootDir, options)
%GET_STAGE2_FIRST_BLOCK_PREFILL_REFERENCE_BASELINE Build a real-reference baseline for first-block prefill.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));
    layerIndex = double(getFieldOr(options, 'LayerIndex', 1));
    stimulus = getFieldOr(options, 'Stimulus', []);
    if isempty(stimulus)
        numericBaseline = get_stage2_first_block_prefill_numeric_baseline(getFieldOr(options, 'BaselineOptions', struct()));
        stimulus = numericBaseline.stimulus;
    end

    [params, sourceInfo] = load_qwen_reference_params(paramsSource, rootDir);
    tokenMask = logical(stimulus.in_valid(:));
    tokenHidden = single(stimulus.in_hidden(tokenMask));
    tokenResidual = single(stimulus.in_residual(tokenMask));
    tokenCount = numel(tokenHidden);
    tokenSampleIndices = find(tokenMask);
    hiddenSize = double(params.Hyperparameters.HiddenSize);

    inHiddenMatrix = repmat(reshape(tokenHidden, 1, []), hiddenSize, 1);
    inResidualMatrix = repmat(reshape(tokenResidual, 1, []), hiddenSize, 1);
    ctx = struct('Parameters', params, 'LayerIndex', layerIndex, ...
        'TokenPos', double(getScalarOr(stimulus, 'cfg_token_pos', tokenSampleIndices(1), 1)));
    [outHiddenMatrix, outKV] = qwen2_block_ref_real_adapter(inHiddenMatrix, inResidualMatrix, single([]), ctx);

    contractOutputs = zeros(tokenCount, 1, 'single');
    for tokenIndex = 1:tokenCount
        contract = build_prefill_contract(stimulus, tokenHidden(tokenIndex), tokenResidual(tokenIndex), tokenIndex, tokenSampleIndices(tokenIndex));
        [summary, ~] = qwen2_block_ref_stage2_contract_adapter(contract, ctx);
        contractOutputs(tokenIndex) = single(summary.out_hidden);
    end

    baseline = struct();
    baseline.params_source = string(sourceInfo);
    baseline.layer_index = layerIndex;
    baseline.hidden_size = hiddenSize;
    baseline.token_count = tokenCount;
    baseline.token_indices = find(tokenMask);
    baseline.token_hidden = tokenHidden(:);
    baseline.token_residual = tokenResidual(:);
    baseline.reference_prefill_out_hidden_mean = single(mean(single(outHiddenMatrix), 1))';
    baseline.reference_prefill_out_hidden_abs_mean = single(mean(abs(single(outHiddenMatrix)), 1))';
    baseline.reference_scalar_contract_out_hidden = contractOutputs(:);
    baseline.reference_kv_present = isstruct(outKV) && isfield(outKV, 'keys') && ~isempty(outKV.keys);
end

function contract = build_prefill_contract(stimulus, tokenHidden, tokenResidual, tokenIndex, sampleIndex)
    contract = struct();
    contract.mode_decode = false;
    contract.start = tokenIndex == 1;
    contract.eos_in = logical(getScalarOr(stimulus, 'eos_in', sampleIndex, 0));
    contract.in_valid = true;
    contract.out_ready = logical(getScalarOr(stimulus, 'out_ready', sampleIndex, 1));
    contract.in_hidden = single(tokenHidden);
    contract.in_residual = single(tokenResidual);
    contract.kv_cache_rd_data = single(0);
    contract.kv_cache_rd_valid = false;
    contract.cfg_seq_len = double(getFieldOr(stimulus, 'cfg_seq_len', tokenIndex));
    contract.cfg_token_pos = double(getScalarOr(stimulus, 'cfg_token_pos', sampleIndex, tokenIndex));
    contract.cfg_eps = double(getFieldOr(stimulus, 'cfg_eps', 1e-5));
    contract.stop_req = logical(getFieldOr(stimulus, 'stop_req', false));
    contract.cfg_weight_num_heads = double(getFieldOr(stimulus, 'cfg_weight_num_heads', 12));
    contract.cfg_weight_page_base = double(getFieldOr(stimulus, 'cfg_weight_page_base', 64));
    contract.cfg_weight_page_stride = double(getFieldOr(stimulus, 'cfg_weight_page_stride', 8));
    contract.cfg_rope_theta_scale = double(getFieldOr(stimulus, 'cfg_rope_theta_scale', 1));
    contract.cfg_rope_sin_mix_scale = double(getFieldOr(stimulus, 'cfg_rope_sin_mix_scale', 1));
end

function value = getScalarOr(stimulus, name, index, defaultValue)
    raw = getFieldOr(stimulus, name, defaultValue);
    if isnumeric(raw) || islogical(raw)
        raw = raw(:);
        if isempty(raw)
            value = defaultValue;
        else
            value = raw(min(index, numel(raw)));
        end
    else
        value = defaultValue;
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end