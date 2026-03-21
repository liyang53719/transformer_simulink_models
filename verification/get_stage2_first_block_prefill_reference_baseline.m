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
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    enableStageTrace = logical(getFieldOr(options, 'EnableStageTrace', false));
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
    if enableStageTrace
        baseline.reference_stage_contract_trace = build_reference_stage_contract_trace(stimulus, tokenHidden, tokenResidual, tokenSampleIndices, ctx, weightRspCfg);
    else
        baseline.reference_stage_contract_trace = struct();
    end
end

function stageTrace = build_reference_stage_contract_trace(stimulus, tokenHidden, tokenResidual, tokenSampleIndices, ctx, weightRspCfg)
    stageTrace = struct();
    tokenCount = numel(tokenHidden);
    hiddenSize = double(ctx.Parameters.Hyperparameters.HiddenSize);
    for tokenIndex = 1:tokenCount
        traceCtx = ctx;
        traceCtx.RuntimeConfig = struct('TracePrecision', false, 'TraceTensors', true);
        traceCtx.TokenPos = double(getScalarOr(stimulus, 'cfg_token_pos', tokenSampleIndices(tokenIndex), tokenIndex));
        seqLen = max(1, round(double(getFieldOr(stimulus, 'cfg_seq_len', tokenIndex))));

        inHiddenMatrix = repmat(single(tokenHidden(tokenIndex)), hiddenSize, seqLen);
        inResidualMatrix = zeros(hiddenSize, seqLen, 'single');
        kvReadData = single([]);

        [outHiddenMatrix, ~, debugInfo] = qwen2_block_ref_real_adapter(inHiddenMatrix, inResidualMatrix, kvReadData, traceCtx);
        tokenTrace = reduce_reference_stage_trace(getFieldOr(debugInfo, 'TensorTrace', struct([])));
        placeholderTrace = build_placeholder_stage_trace(stimulus, tokenHidden(tokenIndex), tokenSampleIndices(tokenIndex), tokenIndex, tokenTrace, weightRspCfg);
        tokenTrace = merge_stage_trace(tokenTrace, placeholderTrace);
        tokenTrace.residual_out = single(mean(single(outHiddenMatrix(:, end)), 'all') + single(tokenResidual(tokenIndex)));
        names = fieldnames(tokenTrace);
        for i = 1:numel(names)
            name = names{i};
            if ~isfield(stageTrace, name)
                stageTrace.(name) = zeros(tokenCount, 1, 'single');
            end
            stageTrace.(name)(tokenIndex) = single(tokenTrace.(name));
        end
    end
end

function mergedTrace = merge_stage_trace(baseTrace, overrideTrace)
    mergedTrace = baseTrace;
    if ~isstruct(overrideTrace)
        return;
    end

    names = fieldnames(overrideTrace);
    for i = 1:numel(names)
        mergedTrace.(names{i}) = overrideTrace.(names{i});
    end
end

function stageTrace = build_placeholder_stage_trace(stimulus, tokenHidden, sampleIndex, tokenIndex, realTokenTrace, weightRspCfg)
    stageTrace = struct();
    sampleTables = getFieldOr(weightRspCfg, 'sample_tables', {});
    if ~(iscell(sampleTables) && numel(sampleTables) >= 10)
        return;
    end

    tokenPos = double(getScalarOr(stimulus, 'cfg_token_pos', sampleIndex, tokenIndex));
    numHeads = double(getScalarOr(stimulus, 'cfg_weight_num_heads', sampleIndex, 12));
    pageBase = double(getScalarOr(stimulus, 'cfg_weight_page_base', sampleIndex, 64));
    pageStride = double(getScalarOr(stimulus, 'cfg_weight_page_stride', sampleIndex, 8));
    thetaScale = double(getScalarOr(stimulus, 'cfg_rope_theta_scale', sampleIndex, 1));
    sinMixScale = double(getScalarOr(stimulus, 'cfg_rope_sin_mix_scale', sampleIndex, 1));
    epsValue = double(getScalarOr(stimulus, 'cfg_eps', sampleIndex, 1e-5));

    baseAddr = pageBase + (tokenPos * numHeads) * pageStride;
    ropePhase = tokenPos * thetaScale;
    ropeScale = cos(ropePhase) + sin(ropePhase) * sinMixScale;
    ropeOut = single(tokenHidden) * single(ropeScale);
    rmsDen = sqrt(single(ropeOut .* ropeOut) + single(epsValue));
    rmsNorm = single(ropeOut ./ rmsDen);

    gammaWeight = lookup_decoded_weight(sampleTables, 1, baseAddr + 0);
    qWeight = lookup_decoded_weight(sampleTables, 2, baseAddr + 1);
    kWeight = lookup_decoded_weight(sampleTables, 3, baseAddr + 2);
    vWeight = lookup_decoded_weight(sampleTables, 4, baseAddr + 3);
    upWeight = lookup_decoded_weight(sampleTables, 8, baseAddr + 7);
    gateWeight = lookup_decoded_weight(sampleTables, 9, baseAddr + 8);
    downWeight = lookup_decoded_weight(sampleTables, 10, baseAddr + 9);

    rmsOutPlaceholder = single(rmsNorm * gammaWeight);
    stageTrace.q_proj_out = single(rmsOutPlaceholder * qWeight);
    stageTrace.k_proj_out = single(rmsOutPlaceholder * kWeight);
    stageTrace.v_proj_out = single(rmsOutPlaceholder * vWeight);
    stageTrace.qkv_out = single(stageTrace.q_proj_out + stageTrace.k_proj_out + stageTrace.v_proj_out);

    attnOut = single(getFieldOr(realTokenTrace, 'attn_out', 0));
    stageTrace.ffn_up_mul = single(attnOut * upWeight);
    stageTrace.ffn_gate_mul = single(attnOut * gateWeight);
    stageTrace.ffn_gate_norm = single(stageTrace.ffn_gate_mul / (abs(stageTrace.ffn_gate_mul) + 1));
    stageTrace.ffn_gate_norm_gate = stageTrace.ffn_gate_norm;
    stageTrace.ffn_swiglu_mul = single(stageTrace.ffn_up_mul * stageTrace.ffn_gate_norm_gate);
    stageTrace.ffn_down_stage = single(stageTrace.ffn_swiglu_mul * downWeight);
    stageTrace.ffn_out = stageTrace.ffn_down_stage;
end

function stageTrace = reduce_reference_stage_trace(tensorTrace)
    stageTrace = struct();
    if ~isstruct(tensorTrace) || isempty(tensorTrace)
        return;
    end

    for i = 1:numel(tensorTrace)
        opName = string(getFieldOr(tensorTrace(i), 'Op', ''));
        value = getFieldOr(tensorTrace(i), 'Value', []);
        if strlength(opName) == 0 || isempty(value)
            continue;
        end
        stageName = map_reference_trace_op(opName);
        if strlength(stageName) == 0
            continue;
        end
        stageTrace.(stageName) = reduce_reference_trace_tensor(value);
    end

    if isfield(stageTrace, 'q_proj_out') && isfield(stageTrace, 'k_proj_out') && isfield(stageTrace, 'v_proj_out')
        stageTrace.qkv_out = stageTrace.q_proj_out + stageTrace.k_proj_out + stageTrace.v_proj_out;
    end
    if isfield(stageTrace, 'o_proj_out')
        stageTrace.attn_out = stageTrace.o_proj_out;
        stageTrace.attn_scorev_reduce = stageTrace.o_proj_out;
    end
    if isfield(stageTrace, 'swiglu_out')
        stageTrace.ffn_swiglu_mul = stageTrace.swiglu_out;
    end
    if isfield(stageTrace, 'down_proj_out')
        stageTrace.ffn_down_stage = stageTrace.down_proj_out;
    end
end

function stageName = map_reference_trace_op(opName)
    switch char(opName)
        case 'block.input_norm.output'
            stageName = "rms_out";
        case 'attn.q_proj.output'
            stageName = "q_proj_out";
        case 'attn.k_proj.output'
            stageName = "k_proj_out";
        case 'attn.v_proj.output'
            stageName = "v_proj_out";
        case 'attn.o_proj.output'
            stageName = "o_proj_out";
        case 'mlp.up_proj.output'
            stageName = "ffn_up_mul";
        case 'mlp.gate_proj.output'
            stageName = "ffn_gate_mul";
        case 'mlp.swiglu.output'
            stageName = "swiglu_out";
        case 'mlp.down_proj.output'
            stageName = "down_proj_out";
        otherwise
            stageName = "";
    end
end

function scalarValue = reduce_reference_trace_tensor(value)
    data = single(value);
    dims = size(data);
    if isempty(data)
        scalarValue = single(0);
        return;
    end
    if numel(dims) >= 2
        idx = repmat({':'}, 1, numel(dims));
        idx{2} = dims(2);
        data = single(data(idx{:}));
    end
    scalarValue = single(mean(data, 'all'));
end

function decoded = lookup_decoded_weight(sampleTables, laneIndex, addr)
    if laneIndex > numel(sampleTables) || isempty(sampleTables{laneIndex})
        decoded = single(0);
        return;
    end

    table = sampleTables{laneIndex};
    idx = floor(double(addr)) + 1;
    idx = max(1, min(idx, numel(table)));
    decoded = decode_weight_byte_local(uint8(table(idx)));
end

function decoded = decode_weight_byte_local(value)
    decoded = (single(value) - 128) / 128;
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