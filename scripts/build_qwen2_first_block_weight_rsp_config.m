function cfg = build_qwen2_first_block_weight_rsp_config(rootDir, options)
%BUILD_QWEN2_FIRST_BLOCK_WEIGHT_RSP_CONFIG Build real first-block weight sample tables.
%   This helper extracts deterministic uint8 sample tables from the cached
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

    gammaName = [layerPrefix 'input_layernorm'];
    qName = [layerPrefix 'self_attn_q_proj_qweight'];
    kName = [layerPrefix 'self_attn_k_proj_qweight'];
    vName = [layerPrefix 'self_attn_v_proj_qweight'];
    oName = [layerPrefix 'self_attn_o_proj_qweight'];
    upName = [layerPrefix 'mlp_up_proj_qweight'];
    gateName = [layerPrefix 'mlp_gate_proj_qweight'];
    downName = [layerPrefix 'mlp_down_proj_qweight'];

    raw = load(matPath, gammaName, qName, kName, vName, oName, upName, gateName, downName);
    requestAddrMax = pageBase + tokenPos * numHeads * pageStride + 9;
    tableLength = max(tableLengthOpt, double(requestAddrMax) + 1);

    tables = cell(1, 10);
    tables{1} = build_lane_table(raw.(gammaName), tableLength, 0, true);
    tables{2} = build_lane_table(raw.(qName), tableLength, 0, false);
    tables{3} = build_lane_table(raw.(kName), tableLength, 11, false);
    tables{4} = build_lane_table(raw.(vName), tableLength, 23, false);
    tables{5} = build_lane_table(raw.(oName), tableLength, 0, false);
    tables{6} = build_lane_table(raw.(oName), tableLength, 37, false);
    tables{7} = build_lane_table(raw.(oName), tableLength, 73, false);
    tables{8} = build_lane_table(raw.(upName), tableLength, 0, false);
    tables{9} = build_lane_table(raw.(gateName), tableLength, 19, false);
    tables{10} = build_lane_table(raw.(downName), tableLength, 41, false);

    laneExpectedAddrs = double(requestAddrMax - 9 + (0:9));
    laneSampleValues = zeros(1, 10);
    for i = 1:10
        laneSampleValues(i) = double(tables{i}(laneExpectedAddrs(i) + 1));
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

function table = build_lane_table(rawData, tableLength, phaseOffset, isFloatData)
    vec = double(rawData(:));
    if isempty(vec)
        vec = 0;
    end
    idx = mod((0:tableLength-1) + double(phaseOffset), numel(vec)) + 1;
    sampled = vec(idx);
    if isFloatData
        sampled = round(abs(sampled) * 127);
    else
        sampled = abs(sampled);
    end
    table = mod(sampled, 256);
end