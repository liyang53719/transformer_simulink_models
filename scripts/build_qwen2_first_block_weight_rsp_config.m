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
    layerIndex = double(getFieldOr(options, 'LayerIndex', 1));
    tableLengthOpt = double(getFieldOr(options, 'TableLength', 256));
    tokenPos = double(getFieldOr(options, 'TokenPos', 1));
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

    vectorCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    tables = cell(1, numel(laneSpecs.entries));
    for i = 1:numel(laneSpecs.entries)
        tables{i} = build_lane_table(raw, laneSpecs.entries{i}, tableLength, vectorCache);
    end

    laneExpectedAddrs = double(requestAddrMax - 9 + (0:9));
    laneSampleValues = zeros(1, 10);
    laneDecodedValues = zeros(1, 10, 'single');
    for i = 1:10
        laneSampleValues(i) = double(tables{i}(laneExpectedAddrs(i) + 1));
        laneDecodedValues(i) = decode_weight_byte(uint8(laneSampleValues(i)));
    end

    cfg = struct();
    cfg.mode = 'sample_table';
    cfg.tag_base = tagBase;
    cfg.tag_stride = tagStride;
    cfg.sample_tables = tables;
    cfg.params_mat = string(matPath);
    cfg.layer_index = layerIndex;
    cfg.table_length = double(tableLength);
    cfg.request_addr_max = double(requestAddrMax);
    cfg.sample_values = laneSampleValues;
    cfg.decoded_sample_values = laneDecodedValues;
    cfg.sample_encoding = 'int8_affine_q7_128';
    cfg.lane_expected_addrs = laneExpectedAddrs;
    cfg.lane_names = {'gamma', 'qkv_q', 'qkv_k', 'qkv_v', 'attn_q', 'attn_k', 'attn_v', 'ffn_up', 'ffn_gate', 'ffn_down'};
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
        struct('kind', 'float', 'field', [layerPrefix 'input_layernorm'], 'phase', 0), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_q_proj'], 'phase', 0), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_k_proj'], 'phase', 11), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_v_proj'], 'phase', 23), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 0), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 37), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'self_attn_o_proj'], 'phase', 73), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'mlp_up_proj'], 'phase', 0), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'mlp_gate_proj'], 'phase', 19), ...
        struct('kind', 'quant', 'prefix', [layerPrefix 'mlp_down_proj'], 'phase', 41)};

    loadNames = {specs.entries{1}.field};
    for i = 2:numel(specs.entries)
        prefix = specs.entries{i}.prefix;
        loadNames = [loadNames, quant_load_names(prefix)]; %#ok<AGROW>
    end
    specs.load_names = unique(loadNames, 'stable');
end

function names = quant_load_names(prefix)
    names = {[prefix '_qweight'], [prefix '_qzeros'], [prefix '_scales'], ...
        [prefix '_bits'], [prefix '_group_size'], [prefix '_in_features'], [prefix '_out_features']};
    names{end + 1} = [prefix '_quant_type'];
    names{end + 1} = [prefix '_g_idx'];
end

function table = build_lane_table(raw, spec, tableLength, vectorCache)
    cacheKey = lane_cache_key(spec);
    if isKey(vectorCache, cacheKey)
        vec = vectorCache(cacheKey);
    else
        vec = build_lane_vector(raw, spec);
        vectorCache(cacheKey) = vec;
    end

    if isempty(vec)
        vec = single(0);
    end
    idx = mod((0:tableLength - 1) + double(spec.phase), numel(vec)) + 1;
    table = encode_weight_bytes(single(vec(idx)));
end

function key = lane_cache_key(spec)
    if strcmp(spec.kind, 'float')
        key = spec.field;
    else
        key = spec.prefix;
    end
end

function vec = build_lane_vector(raw, spec)
    if strcmp(spec.kind, 'float')
        vec = single(raw.(spec.field)(:));
        return;
    end

    w = load_quant_linear(raw, spec.prefix);
    vec = dequantize_quant_linear(w);
    vec = single(vec(:));
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

function encoded = encode_weight_bytes(values)
    encoded = round(single(values) * 128) + 128;
    encoded = max(min(encoded, 255), 0);
    encoded = uint8(encoded);
end

function decoded = decode_weight_byte(value)
    decoded = (single(value) - 128) / 128;
end