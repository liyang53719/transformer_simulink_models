function cfg = build_qwen2_first_block_weight_rsp_config(rootDir, options)
%BUILD_QWEN2_FIRST_BLOCK_WEIGHT_RSP_CONFIG Build real first-block weight sample tables.
%   This helper extracts deterministic scalar-weight sample tables from the cached
%   Qwen2-1.5B first-layer parameters and packages them into a
%   WeightRspConfig struct that build_stage2_wrapper_tb_model can feed into
%   weight_ref_u.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    paramsMatFile = getFieldOr(options, 'ParamsMatFile', '');
    weightTableMode = char(string(getFieldOr(options, 'WeightTableMode', 'effective_scalar')));
    layerIndex = double(getFieldOr(options, 'LayerIndex', 1));
    tableLengthOpt = double(getFieldOr(options, 'TableLength', 256));
    tokenPos = resolve_effective_token_pos(options);
    numHeads = double(getFieldOr(options, 'NumHeads', 12));
    pageBase = double(getFieldOr(options, 'PageBase', 64));
    pageStride = double(getFieldOr(options, 'PageStride', 8));
    tagBase = double(getFieldOr(options, 'TagBase', 0));
    tagStride = double(getFieldOr(options, 'TagStride', 8));

    matPath = resolve_params_mat_file(rootDir, paramsMatFile);
    layerPrefix = sprintf('layer_%d_', layerIndex - 1);

    laneSpecs = build_lane_specs(layerPrefix);
    raw = load(matPath, laneSpecs.load_names{:});
    requestAddrMax = pageBase + tokenPos * numHeads * pageStride + 9;
    tableLength = max(tableLengthOpt, double(requestAddrMax) + 1);
    empiricalTables = build_empirical_ffn_tables(rootDir, options, tableLength, layerIndex);

    vectorCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    tables = cell(1, numel(laneSpecs.entries));
    laneDecodeScales = ones(1, numel(laneSpecs.entries), 'single');
    for i = 1:numel(laneSpecs.entries)
        laneName = char(laneSpecs.entries{i}.lane_name);
        if isstruct(empiricalTables) && isfield(empiricalTables, laneName)
            laneProfile = get_stage1_scalar_weight_profile(laneName);
            laneDecodeScales(i) = laneProfile.decode_scale;
            tables{i} = encode_weight_bytes(single(empiricalTables.(laneName)), laneDecodeScales(i));
        else
            [tables{i}, laneDecodeScales(i)] = build_lane_table(raw, laneSpecs.entries{i}, tableLength, vectorCache, weightTableMode);
        end
    end

    laneExpectedAddrs = double(requestAddrMax - 9 + (0:9));
    laneSampleValues = zeros(1, 10);
    laneDecodedValues = zeros(1, 10, 'single');
    for i = 1:10
        laneSampleValues(i) = double(tables{i}(laneExpectedAddrs(i) + 1));
        laneDecodedValues(i) = decode_weight_byte(uint8(laneSampleValues(i)), laneDecodeScales(i));
    end

    cfg = struct();
    cfg.mode = 'sample_table';
    cfg.tag_base = tagBase;
    cfg.tag_stride = tagStride;
    cfg.sample_tables = tables;
    cfg.lane_decode_scales = laneDecodeScales;
    cfg.params_mat = string(matPath);
    cfg.layer_index = layerIndex;
    cfg.table_length = double(tableLength);
    cfg.request_addr_max = double(requestAddrMax);
    cfg.sample_values = laneSampleValues;
    cfg.decoded_sample_values = laneDecodedValues;
    cfg.sample_encoding = 'int8_affine_q7_scaled';
    cfg.weight_table_mode = string(weightTableMode);
    cfg.lane_expected_addrs = laneExpectedAddrs;
    cfg.lane_names = {'gamma', 'qkv_q', 'qkv_k', 'qkv_v', 'attn_q', 'attn_k', 'attn_v', 'ffn_up', 'ffn_gate', 'ffn_down'};
end

function tokenPos = resolve_effective_token_pos(options)
    tokenPos = double(getFieldOr(options, 'TokenPos', NaN));
    if ~isnan(tokenPos) && tokenPos > 0
        return;
    end

    stimulus = getFieldOr(options, 'Stimulus', []);
    if isstruct(stimulus) && isfield(stimulus, 'cfg_token_pos') && ~isempty(stimulus.cfg_token_pos)
        tokenPos = max(double(stimulus.cfg_token_pos(:)));
        if tokenPos > 0
            return;
        end
    end

    baselineOptions = getFieldOr(options, 'BaselineOptions', struct());
    if isstruct(baselineOptions) && logical(getFieldOr(baselineOptions, 'DriveTokenPosSequence', false))
        tokenPos = max(1, double(getFieldOr(baselineOptions, 'NumTokens', 1)));
        return;
    end

    tokenPos = 1;
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function matPath = resolve_params_mat_file(rootDir, explicitPath)
    if strlength(string(explicitPath)) > 0
        matPath = char(explicitPath);
        if exist(matPath, 'file') ~= 2
            error('build_qwen2_first_block_weight_rsp_config:MissingMatFile', ...
                'Params MAT file not found: %s', matPath);
        end
        return;
    end

    preferred = fullfile(rootDir, 'matlab_ref', 'cache', 'Qwen2.5-1_params.mat');
    if exist(preferred, 'file') == 2
        matPath = preferred;
        return;
    end

    listing = dir(fullfile(rootDir, 'matlab_ref', 'cache', 'Qwen2*.mat'));
    if ~isempty(listing)
        matPath = fullfile(listing(1).folder, listing(1).name);
        return;
    end

    error('build_qwen2_first_block_weight_rsp_config:MissingCache', ...
        'No cached Qwen2 parameter MAT file found under matlab_ref/cache.');
end

function specs = build_lane_specs(layerPrefix)
    specs = struct();
    specs.entries = {
        struct('lane_name', 'gamma', 'kind', 'float', 'field', [layerPrefix 'input_layernorm'], 'phase', 0), ...
        struct('lane_name', 'qkv_q', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_q_proj'], 'phase', 0), ...
        struct('lane_name', 'qkv_k', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_k_proj'], 'phase', 11), ...
        struct('lane_name', 'qkv_v', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_v_proj'], 'phase', 23), ...
        struct('lane_name', 'attn_q', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 0), ...
        struct('lane_name', 'attn_k', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 37), ...
        struct('lane_name', 'attn_v', 'kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 73), ...
        struct('lane_name', 'ffn_up', 'kind', 'quant', 'prefix', [layerPrefix 'mlp_up_proj'], 'phase', 0), ...
        struct('lane_name', 'ffn_gate', 'kind', 'quant', 'prefix', [layerPrefix 'mlp_gate_proj'], 'phase', 19), ...
        struct('lane_name', 'ffn_down', 'kind', 'quant', 'prefix', [layerPrefix 'mlp_down_proj'], 'phase', 41)};

    loadNames = {};
    for i = 1:numel(specs.entries)
        if strcmp(specs.entries{i}.kind, 'float')
            loadNames{end + 1} = specs.entries{i}.field; %#ok<AGROW>
        else
            prefix = specs.entries{i}.prefix;
            loadNames = [loadNames, quant_load_names(prefix)]; %#ok<AGROW>
        end
    end
    specs.load_names = unique(loadNames, 'stable');
end

function names = quant_load_names(prefix)
    names = {[prefix '_qweight'], [prefix '_qzeros'], [prefix '_scales'], ...
        [prefix '_bits'], [prefix '_group_size'], [prefix '_in_features'], [prefix '_out_features']};
    names{end + 1} = [prefix '_quant_type'];
    names{end + 1} = [prefix '_g_idx'];
end

function [table, laneDecodeScale] = build_lane_table(raw, spec, tableLength, vectorCache, weightTableMode)
    cacheKey = lane_cache_key(spec);
    if isKey(vectorCache, cacheKey)
        vec = vectorCache(cacheKey);
    else
        vec = build_lane_vector(raw, spec, weightTableMode);
        vectorCache(cacheKey) = vec;
    end

    if isempty(vec)
        vec = single(0);
    end
    laneProfile = get_stage1_scalar_weight_profile(spec.lane_name);
    if strcmp(weightTableMode, 'effective_scalar') || strcmp(weightTableMode, 'empirical_scalar')
        laneDecodeScale = laneProfile.decode_scale;
    else
        laneDecodeScale = single(1);
    end
    idx = mod((0:tableLength - 1) + double(spec.phase), numel(vec)) + 1;
    table = encode_weight_bytes(single(vec(idx)), laneDecodeScale);
end

function key = lane_cache_key(spec)
    if strcmp(spec.kind, 'float')
        key = spec.field;
    else
        key = spec.prefix;
    end
end

function vec = build_lane_vector(raw, spec, weightTableMode)
    laneProfile = get_stage1_scalar_weight_profile(spec.lane_name);
    if strcmp(spec.kind, 'float')
        if strcmp(weightTableMode, 'empirical_scalar') && ~isnan(laneProfile.constant_value)
            vec = laneProfile.constant_value;
        elseif strcmp(weightTableMode, 'effective_scalar') && ~isnan(laneProfile.constant_value)
            vec = laneProfile.constant_value;
        elseif strcmp(weightTableMode, 'effective_scalar') && laneProfile.reduction_mode == "mean"
            vec = single(mean(single(raw.(spec.field)(:)), 'all'));
        else
            vec = single(raw.(spec.field)(:));
        end
        return;
    end

    w = load_quant_linear(raw, spec.prefix);
    vec = dequantize_quant_linear(w);
    if strcmp(weightTableMode, 'empirical_scalar') && ~isnan(laneProfile.constant_value)
        vec = laneProfile.constant_value;
    elseif strcmp(weightTableMode, 'effective_scalar') && ~isnan(laneProfile.constant_value)
        vec = laneProfile.constant_value;
    elseif strcmp(weightTableMode, 'effective_scalar') && laneProfile.reduction_mode == "mean_colsum"
        vec = single(mean(sum(single(vec), 1), 'all'));
    else
        vec = single(vec(:));
    end
end

function tables = build_empirical_ffn_tables(rootDir, options, tableLength, layerIndex)
    tables = struct();

    weightTableMode = char(string(getFieldOr(options, 'WeightTableMode', 'effective_scalar')));
    if ~strcmp(weightTableMode, 'effective_scalar')
        return;
    end

    stimulus = getFieldOr(options, 'Stimulus', []);
    if ~(isstruct(stimulus) && isfield(stimulus, 'in_valid') && any(logical(stimulus.in_valid(:))))
        return;
    end

    addpath(fullfile(rootDir, 'matlab_ref'));

    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));
    [params, ~] = load_qwen_reference_params(paramsSource, rootDir);
    hiddenSize = double(params.Hyperparameters.HiddenSize);
    tokenMask = logical(stimulus.in_valid(:));
    tokenHidden = single(stimulus.in_hidden(tokenMask));
    tokenSampleIndices = find(tokenMask);
    pageBase = double(getFieldOr(options, 'PageBase', 64));
    pageStride = double(getFieldOr(options, 'PageStride', 8));
    numHeads = double(getFieldOr(options, 'NumHeads', 12));

    tables.ffn_up = single(get_stage1_scalar_weight_profile('ffn_up').constant_value) * ones(1, tableLength, 'single');
    tables.ffn_gate = single(get_stage1_scalar_weight_profile('ffn_gate').constant_value) * ones(1, tableLength, 'single');
    tables.ffn_down = single(get_stage1_scalar_weight_profile('ffn_down').constant_value) * ones(1, tableLength, 'single');

    for tokenIndex = 1:numel(tokenHidden)
        sampleIndex = tokenSampleIndices(tokenIndex);
        tokenPos = double(getScalarOr(stimulus, 'cfg_token_pos', sampleIndex, tokenIndex));
        seqLen = max(1, round(double(getFieldOr(stimulus, 'cfg_seq_len', tokenIndex))));

        traceCtx = struct('Parameters', params, 'LayerIndex', layerIndex, ...
            'RuntimeConfig', struct('TracePrecision', false, 'TraceTensors', true), ...
            'TokenPos', tokenPos);
        inHiddenMatrix = repmat(single(tokenHidden(tokenIndex)), hiddenSize, seqLen);
        inResidualMatrix = zeros(hiddenSize, seqLen, 'single');
        [~, ~, debugInfo] = qwen2_block_ref_real_adapter(inHiddenMatrix, inResidualMatrix, single([]), traceCtx);
        tokenTrace = reduce_reference_stage_trace_local(getFieldOr(debugInfo, 'TensorTrace', struct([])));

        postAttnNorm = single(getFieldOr(tokenTrace, 'post_attn_norm_out', 0));
        ffnUpMul = single(getFieldOr(tokenTrace, 'ffn_up_mul', 0));
        ffnGateMul = single(getFieldOr(tokenTrace, 'ffn_gate_mul', 0));
        ffnSwigluMul = single(getFieldOr(tokenTrace, 'ffn_swiglu_mul', 0));
        ffnDownStage = single(getFieldOr(tokenTrace, 'ffn_down_stage', 0));

        baseAddr = pageBase + tokenPos * numHeads * pageStride;
        upAddr = baseAddr + 7;
        gateAddr = baseAddr + 8;
        downAddr = baseAddr + 9;
        if upAddr + 1 <= tableLength
            tables.ffn_up(upAddr + 1) = safe_effective_div(ffnUpMul, postAttnNorm, tables.ffn_up(upAddr + 1));
        end
        if gateAddr + 1 <= tableLength
            tables.ffn_gate(gateAddr + 1) = safe_effective_div(ffnGateMul, postAttnNorm, tables.ffn_gate(gateAddr + 1));
        end
        if downAddr + 1 <= tableLength
            tables.ffn_down(downAddr + 1) = safe_effective_div(ffnDownStage, ffnSwigluMul, tables.ffn_down(downAddr + 1));
        end
    end
end

function value = safe_effective_div(numerator, denominator, fallback)
    if abs(double(denominator)) <= 1e-9
        value = single(fallback);
    else
        value = single(numerator ./ denominator);
    end
end

function stageTrace = reduce_reference_stage_trace_local(tensorTrace)
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
        stageName = map_reference_trace_op_local(opName);
        if strlength(stageName) == 0
            continue;
        end
        stageTrace.(stageName) = reduce_reference_trace_tensor_local(value);
    end

    if isfield(stageTrace, 'swiglu_out')
        stageTrace.ffn_swiglu_mul = stageTrace.swiglu_out;
    end
    if isfield(stageTrace, 'down_proj_out')
        stageTrace.ffn_down_stage = stageTrace.down_proj_out;
    end
end

function stageName = map_reference_trace_op_local(opName)
    switch char(opName)
        case 'block.post_attn_norm.output'
            stageName = "post_attn_norm_out";
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

function value = reduce_reference_trace_tensor_local(tensor)
    value = single(mean(single(tensor), 'all'));
end

function value = getScalarOr(s, name, index, defaultValue)
    value = getFieldOr(s, name, defaultValue);
    if ~isscalar(value)
        value = value(index);
    end
end

function w = load_quant_linear(raw, prefix)
    w = struct();
    w.qweight = int32(raw.([prefix '_qweight']));
    w.qzeros = int32(raw.([prefix '_qzeros']));
    w.scales = single(raw.([prefix '_scales']));
    w.bits = double(raw.([prefix '_bits']));
    w.group_size = double(raw.([prefix '_group_size']));
    w.in_features = double(raw.([prefix '_in_features']));
    w.out_features = double(raw.([prefix '_out_features']));
    if isfield(raw, [prefix '_g_idx'])
        w.g_idx = int32(raw.([prefix '_g_idx']));
    else
        w.g_idx = int32(floor((0:w.in_features - 1) ./ w.group_size));
    end
    if isfield(raw, [prefix '_quant_type'])
        qt = raw.([prefix '_quant_type']);
        if iscell(qt)
            w.QuantType = upper(string(qt{1}));
        else
            w.QuantType = upper(string(qt));
        end
    else
        w.QuantType = "GPTQ_INT4";
    end
end

function w_dequant = dequantize_quant_linear(w)
    if double(w.bits) ~= 4
        error('build_qwen2_first_block_weight_rsp_config:UnsupportedBits', ...
            'Only int4 quant weights are supported, got bits=%g.', double(w.bits));
    end

    useAwqOrder = strcmpi(string(w.QuantType), "AWQ_INT4");
    qweight_u4 = unpack_qweight_int4(w.qweight, w.in_features, w.out_features, useAwqOrder);
    qzeros_u4 = unpack_qzeros_int4(w.qzeros, w.out_features, useAwqOrder);
    g_idx = double(w.g_idx(:)) + 1;

    if numel(g_idx) ~= w.in_features
        error('build_qwen2_first_block_weight_rsp_config:InvalidGroupIndex', ...
            'Expected g_idx length %d, got %d.', w.in_features, numel(g_idx));
    end

    w_dequant = (single(qweight_u4) - single(qzeros_u4(g_idx, :))) .* w.scales(g_idx, :);
end

function q_u4 = unpack_qweight_int4(qweight_packed, in_features, out_features, useAwqOrder)
    packed = int32(qweight_packed);

    if isequal(size(packed), [in_features / 8, out_features])
        q_u4 = zeros(in_features, out_features, 'uint8');
        for p = 0:7
            rows = (p + 1):8:in_features;
            srcNib = map_unpack_nibble(p, useAwqOrder);
            q_u4(rows, :) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
        end
        return;
    end

    if useAwqOrder
        if isequal(size(packed), [in_features, out_features / 8])
            q_u4 = zeros(in_features, out_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:out_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4(:, cols) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
            end
            return;
        end

        if isequal(size(packed), [out_features, in_features / 8])
            q_u4_t = zeros(out_features, in_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:in_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4_t(:, cols) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
            end
            q_u4 = q_u4_t.';
            return;
        end
    else
        if isequal(size(packed), [out_features, in_features / 8])
            q_u4_t = zeros(out_features, in_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:in_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4_t(:, cols) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
            end
            q_u4 = q_u4_t.';
            return;
        end

        if isequal(size(packed), [in_features, out_features / 8])
            q_u4 = zeros(in_features, out_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:out_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4(:, cols) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
            end
            return;
        end
    end

    if isequal(size(packed), [out_features / 8, in_features])
        packed_t = packed.';
        q_u4 = zeros(in_features, out_features, 'uint8');
        for p = 0:7
            cols = (p + 1):8:out_features;
            srcNib = map_unpack_nibble(p, useAwqOrder);
            q_u4(:, cols) = uint8(bitand(bitshift(packed_t, -4 * srcNib), int32(15)));
        end
        return;
    end

    error('build_qwen2_first_block_weight_rsp_config:UnsupportedLayout', ...
        'Unsupported qweight layout [%d,%d] for in=%d out=%d.', ...
        size(packed, 1), size(packed, 2), in_features, out_features);
end

function z_u4 = unpack_qzeros_int4(qzeros_packed, out_features, useAwqOrder)
    num_groups = size(qzeros_packed, 1);
    z_u4 = zeros(num_groups, out_features, 'uint8');
    packed = int32(qzeros_packed);
    for p = 0:7
        cols = (p + 1):8:out_features;
        srcNib = map_unpack_nibble(p, useAwqOrder);
        z_u4(:, cols) = uint8(bitand(bitshift(packed, -4 * srcNib), int32(15)));
    end
end

function srcNib = map_unpack_nibble(dstPos, useAwqOrder)
    if ~useAwqOrder
        srcNib = dstPos;
        return;
    end
    invMap = [0, 4, 1, 5, 2, 6, 3, 7];
    srcNib = invMap(dstPos + 1);
end

function encoded = encode_weight_bytes(values, decodeScale)
    scaledValues = single(values) ./ max(single(decodeScale), single(1e-6));
    encoded = round(scaledValues * 128) + 128;
    encoded = max(min(encoded, 255), 0);
    encoded = uint8(encoded);
end

function decoded = decode_weight_byte(value, decodeScale)
    decoded = ((single(value) - 128) / 128) * single(decodeScale);
end