function result = run_stage2_real_first_block_stage_delta_regression(rootDir, options)
%RUN_STAGE2_REAL_FIRST_BLOCK_STAGE_DELTA_REGRESSION Localize synthetic vs real-sample delta loss.
%   This regression runs the stage2 wrapper twice: once with the default
%   synthetic weight responder and once with real layer0 sample bytes
%   injected into weight_ref_u. It logs a focused set of internal SRAM and
%   stage accumulator signals so we can tell whether the real sample delta
%   reaches QKV/attention/FFN internals even when out_hidden stays the same.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    useDelayedRamReadAddr = getFieldOr(options, 'UseDelayedRamReadAddr', false);
    ffnGateValidExtraDelay = getFieldOr(options, 'FfnGateValidExtraDelay', 0);
    ffnSwigluValidExtraDelay = getFieldOr(options, 'FfnSwigluValidExtraDelay', 0);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    signalSpecs = build_signal_specs();

    baseline = simulate_stage_signals(rootDir, kvCfg, buildModel, [], signalSpecs, useDelayedRamReadAddr, ffnGateValidExtraDelay, ffnSwigluValidExtraDelay);
    realSample = simulate_stage_signals(rootDir, kvCfg, buildModel, double(weightRspCfg.sample_values), signalSpecs, useDelayedRamReadAddr, ffnGateValidExtraDelay, ffnSwigluValidExtraDelay);

    stageResults = repmat(struct( ...
        'name', '', ...
        'block', '', ...
        'sample_count', 0, ...
        'baseline_nonzero', false, ...
        'real_nonzero', false, ...
        'max_abs_diff', 0, ...
        'mean_abs_diff', 0, ...
        'changed', false), [1, numel(signalSpecs)]);

    changedNames = strings(0, 1);
    for i = 1:numel(signalSpecs)
        cmp = compare_signal_pair(baseline.logged.(signalSpecs{i}.name), realSample.logged.(signalSpecs{i}.name));
        stageResults(i).name = signalSpecs{i}.name;
        stageResults(i).block = signalSpecs{i}.block;
        stageResults(i).sample_count = cmp.sample_count;
        stageResults(i).baseline_nonzero = cmp.baseline_nonzero;
        stageResults(i).real_nonzero = cmp.real_nonzero;
        stageResults(i).max_abs_diff = cmp.max_abs_diff;
        stageResults(i).mean_abs_diff = cmp.mean_abs_diff;
        stageResults(i).changed = cmp.changed;
        if cmp.changed
            changedNames(end + 1, 1) = string(signalSpecs{i}.name); %#ok<AGROW>
        end
    end

    outHiddenCmp = compare_signal_pair(baseline.out_hidden, realSample.out_hidden);
    laneSummaries = build_lane_summaries(stageResults, realSample.logged, weightRspCfg);
    attentionOutputAlignment = build_attention_output_alignment_summary(baseline.logged, realSample.logged);
    ffnGateAlignment = build_ffn_gate_alignment_summary(baseline.logged, realSample.logged);

    result = struct();
    result.stage_results = stageResults;
    result.lane_summaries = laneSummaries;
    result.attention_output_alignment = attentionOutputAlignment;
    result.ffn_gate_alignment = ffnGateAlignment;
    result.changed_stage_names = cellstr(changedNames);
    result.changed_stage_count = numel(result.changed_stage_names);
    result.out_hidden = outHiddenCmp;
    result.internal_delta_seen = result.changed_stage_count > 0;
    result.output_delta_seen = outHiddenCmp.changed;
    result.delta_localized = result.internal_delta_seen && ~result.output_delta_seen;
    result.pass = result.internal_delta_seen;

    if result.pass
        fprintf('Stage2 real first-block stage delta regression PASS\n');
        fprintf('  internal_delta_seen=%d output_delta_seen=%d delta_localized=%d changed_stage_count=%d\n', ...
            result.internal_delta_seen, result.output_delta_seen, result.delta_localized, result.changed_stage_count);
        for i = 1:numel(stageResults)
            if stageResults(i).changed
                fprintf('  %s changed=%d max_abs_diff=%g mean_abs_diff=%g\n', ...
                    stageResults(i).name, stageResults(i).changed, ...
                    stageResults(i).max_abs_diff, stageResults(i).mean_abs_diff);
            end
        end
        print_lane_summaries(laneSummaries);
        print_attention_output_alignment(attentionOutputAlignment);
        print_ffn_gate_alignment(ffnGateAlignment);
        fprintf('  out_hidden changed=%d max_abs_diff=%g mean_abs_diff=%g\n', ...
            outHiddenCmp.changed, outHiddenCmp.max_abs_diff, outHiddenCmp.mean_abs_diff);
    else
        fprintf('Stage2 real first-block stage delta regression FAIL\n');
        fprintf('  No internal stage signal changed between synthetic and real samples\n');
        fprintf('  out_hidden changed=%d max_abs_diff=%g mean_abs_diff=%g\n', ...
            outHiddenCmp.changed, outHiddenCmp.max_abs_diff, outHiddenCmp.mean_abs_diff);
        error('run_stage2_real_first_block_stage_delta_regression:Failed', ...
            'Synthetic vs real-sample stage delta localization failed');
    end
end

function signalSpecs = build_signal_specs()
    signalSpecs = {
        struct('block', 'attention_u/row_sum_accum', 'port', 1, 'name', 'stage_attn_row_sum_accum'), ...
        struct('block', 'in_hidden', 'port', 1, 'name', 'flow_root_in_hidden'), ...
        struct('block', 'rope_u/apply_rot_scale', 'port', 1, 'name', 'flow_rope_apply_rot_scale'), ...
        struct('block', 'qkv_proj_u/x_in', 'port', 1, 'name', 'flow_qkv_x_in'), ...
        struct('block', 'rmsnorm_u/x_in', 'port', 1, 'name', 'flow_rms_x_in'), ...
        struct('block', 'rmsnorm_u/x_norm', 'port', 1, 'name', 'flow_rms_x_norm'), ...
        struct('block', 'qkv_proj_u/q_valid_alias', 'port', 1, 'name', 'flow_qkv_q_valid', 'setName', false), ...
        struct('block', 'qkv_proj_u/kv_valid_alias', 'port', 1, 'name', 'flow_qkv_kv_valid', 'setName', false), ...
        struct('block', 'kv_cache_if_u/attn_compose', 'port', 1, 'name', 'flow_kv_cache_attn_compose')};

    for i = 1:9
        signalSpecs = [signalSpecs, append_weight_ref_lane_specs(i)]; %#ok<AGROW>
    end

    signalSpecs = [signalSpecs, append_weight_lane_specs('rmsnorm_u', 'gamma')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('qkv_proj_u', 'q')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('qkv_proj_u', 'k')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('qkv_proj_u', 'v')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('attention_u', 'q')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('attention_u', 'k')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('attention_u', 'v')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('ffn_swiglu_u', 'up')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_weight_lane_specs('ffn_swiglu_u', 'gate')]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_flow_specs('qkv_proj_u', { ...
        'qk_sum', ...
        'qkv_sum'})]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_flow_specs('attention_u', { ...
        'head_group_norm', ...
        'qk_pair_valid', ...
        'qk_pair_valid_z', ...
        'score_mul', ...
        'score_stage_gate', ...
        'score_shift', ...
        'score_norm', ...
        'softmax_valid', ...
        'softmax_valid_z', ...
        'softmax_value_gate', ...
        'scorev_input_valid', ...
        'scorev_valid_z', ...
        'value_weight', ...
        'scorev_reduce', ...
        'output_valid_gate', ...
        'scorev_stage_z'})]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_flow_specs('ffn_swiglu_u', { ...
        'gateup_pair_valid', ...
        'gateup_pair_valid_z', ...
        'gate_norm', ...
        'gate_norm_gate', ...
        'swiglu_mul', ...
        'swiglu_valid_z', ...
        'swiglu_stage_gate', ...
        'down_proj', ...
        'down_valid_z', ...
        'down_stage_gate'})]; %#ok<AGROW>
    signalSpecs = [signalSpecs, append_flow_specs('residual_u', { ...
        'main_scale', ...
        'res_sum'})]; %#ok<AGROW>
end

function specs = append_weight_ref_lane_specs(laneIndex)
    baseName = ['weight_ref_lane_' num2str(laneIndex)];
    specs = { ...
    struct('scope', 'tb', 'block', ['weight_ref_u/addr_d2_' num2str(laneIndex)], 'port', 1, 'name', [baseName '_addr_d2'], 'setName', false), ...
    struct('scope', 'tb', 'block', ['weight_ref_u/val_d2_' num2str(laneIndex)], 'port', 1, 'name', [baseName '_val_d2'], 'setName', false), ...
    struct('scope', 'tb', 'block', ['weight_ref_u/data_u8_' num2str(laneIndex)], 'port', 1, 'name', [baseName '_data_u8'], 'setName', false)};
end

function specs = append_weight_lane_specs(moduleName, prefix)
    baseName = ['stage_' module_alias(moduleName) '_' prefix];
    specs = { ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_addr_alias'], 'port', 1, 'name', [baseName '_sram_addr_alias']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_din_alias'], 'port', 1, 'name', [baseName '_sram_din_alias']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_we_alias'], 'port', 1, 'name', [baseName '_sram_we_alias']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_dout_alias'], 'port', 1, 'name', [baseName '_sram_dout_alias']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_ddr_data_u8'], 'port', 1, 'name', [baseName '_ddr_data_u8']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_ddr_valid_bool'], 'port', 1, 'name', [baseName '_ddr_valid_bool']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_req_needed'], 'port', 1, 'name', [baseName '_req_needed']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram'], 'port', 1, 'name', [baseName '_sram']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_data_double'], 'port', 1, 'name', [baseName '_sram_data_double']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_data_valid_z'], 'port', 1, 'name', [baseName '_sram_data_valid_z']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_valid_or'], 'port', 1, 'name', [baseName '_valid_or']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_sram_data_sel'], 'port', 1, 'name', [baseName '_sram_data_sel']), ...
        struct('scope', 'mdl', 'block', [moduleName '/' prefix '_mul'], 'port', 1, 'name', [baseName '_mul'], 'setName', ~strcmp(moduleName, 'qkv_proj_u'))};
end

function specs = append_flow_specs(moduleName, blockNames)
    alias = module_alias(moduleName);
    specs = cell(1, numel(blockNames));
    for i = 1:numel(blockNames)
        specs{i} = struct( ...
            'scope', 'mdl', ...
            'block', [moduleName '/' blockNames{i}], ...
            'port', 1, ...
            'name', ['flow_' alias '_' blockNames{i}]);
    end
end

function alias = module_alias(moduleName)
    switch moduleName
        case 'rmsnorm_u'
            alias = 'rms';
        case 'qkv_proj_u'
            alias = 'qkv';
        case 'attention_u'
            alias = 'attn';
        case 'ffn_swiglu_u'
            alias = 'ffn';
        otherwise
            alias = moduleName;
    end
end

function simResult = simulate_stage_signals(rootDir, kvCfg, buildModel, sampleValues, signalSpecs, useDelayedRamReadAddr, ffnGateValidExtraDelay, ffnSwigluValidExtraDelay)
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    if useDelayedRamReadAddr
        patch_ram_read_addr_to_delay(mdlName);
    end
    if ffnGateValidExtraDelay > 0
        patch_ffn_gate_valid_delay(mdlName, ffnGateValidExtraDelay);
    end
    if ffnSwigluValidExtraDelay > 0
        patch_ffn_swiglu_valid_delay(mdlName, ffnSwigluValidExtraDelay);
    end

    enable_signal_logging(tbName, mdlName, signalSpecs);

    if isnumeric(sampleValues) && ~isempty(sampleValues)
        inject_sample_values_into_weight_ref(tbName, sampleValues);
    end

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');
    yout = simOut.get('yout');

    simResult = struct();
    simResult.logged = struct();
    for i = 1:numel(signalSpecs)
        simResult.logged.(signalSpecs{i}.name) = double(extract_logged_signal_optional(logsout, signalSpecs{i}.name, signalSpecs{i}.block));
    end
    simResult.out_hidden = double(extract_signal(yout, 'out_hidden'));
end

function patch_ffn_gate_valid_delay(mdlName, extraDelay)
    if extraDelay <= 0
        return;
    end

    subPath = [mdlName '/ffn_swiglu_u'];
    try
        delete_line(subPath, 'gateup_pair_valid_z/1', 'gate_norm_gate/2');
    catch
    end

    prevBlock = 'gateup_pair_valid_z';
    for i = 1:extraDelay
        delayName = sprintf('gateup_pair_valid_gate_z%d', i);
        delayPath = [subPath '/' delayName];
        if getSimulinkBlockHandle(delayPath) == -1
            add_block('simulink/Discrete/Unit Delay', delayPath, ...
                'InitialCondition', '0', 'Position', [320 + 40 * i, 20, 350 + 40 * i, 45]);
        end
        try
            delete_line(subPath, [prevBlock '/1'], [delayName '/1']);
        catch
        end
        add_line(subPath, [prevBlock '/1'], [delayName '/1'], 'autorouting', 'on');
        prevBlock = delayName;
    end

    try
        delete_line(subPath, [prevBlock '/1'], 'gate_norm_gate/2');
    catch
    end
    add_line(subPath, [prevBlock '/1'], 'gate_norm_gate/2', 'autorouting', 'on');
end

function patch_ffn_swiglu_valid_delay(mdlName, extraDelay)
    if extraDelay <= 0
        return;
    end

    subPath = [mdlName '/ffn_swiglu_u'];
    try
        delete_line(subPath, 'swiglu_valid_z/1', 'swiglu_stage_gate/2');
    catch
    end

    prevBlock = 'swiglu_valid_z';
    for i = 1:extraDelay
        delayName = sprintf('swiglu_valid_gate_z%d', i);
        delayPath = [subPath '/' delayName];
        if getSimulinkBlockHandle(delayPath) == -1
            add_block('simulink/Discrete/Unit Delay', delayPath, ...
                'InitialCondition', '0', 'Position', [540 + 40 * i, 20, 570 + 40 * i, 45]);
        end
        try
            delete_line(subPath, [prevBlock '/1'], [delayName '/1']);
        catch
        end
        add_line(subPath, [prevBlock '/1'], [delayName '/1'], 'autorouting', 'on');
        prevBlock = delayName;
    end

    try
        delete_line(subPath, [prevBlock '/1'], 'swiglu_stage_gate/2');
    catch
    end
    add_line(subPath, [prevBlock '/1'], 'swiglu_stage_gate/2', 'autorouting', 'on');
end

function patch_ram_read_addr_to_delay(mdlName)
    specs = {
        struct('path', [mdlName '/rmsnorm_u'], 'prefix', 'gamma'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'q'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'k'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'v'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'q'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'k'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'v'), ...
        struct('path', [mdlName '/ffn_swiglu_u'], 'prefix', 'up'), ...
        struct('path', [mdlName '/ffn_swiglu_u'], 'prefix', 'gate')};

    for i = 1:numel(specs)
        subPath = specs{i}.path;
        prefix = specs{i}.prefix;
        sramPath = [subPath '/' prefix '_sram'];
        reqAddrCastPath = [subPath '/' prefix '_req_addr_u8'];
        reqAddrDelayPath = [subPath '/' prefix '_req_addr_z'];
        reqAddrDelayU8Path = [subPath '/' prefix '_req_addr_z_u8_exp'];
        try
            delete_line(subPath, [prefix '_req_addr_u8/1'], [prefix '_sram/4']);
        catch
        end
        try
            delete_line(subPath, [prefix '_req_addr_z_u8_exp/1'], [prefix '_sram/4']);
        catch
        end
        if getSimulinkBlockHandle(sramPath) ~= -1 && ...
                getSimulinkBlockHandle(reqAddrCastPath) ~= -1 && ...
                getSimulinkBlockHandle(reqAddrDelayPath) ~= -1
            if getSimulinkBlockHandle(reqAddrDelayU8Path) == -1
                add_block('simulink/Signal Attributes/Data Type Conversion', reqAddrDelayU8Path, ...
                    'OutDataTypeStr', 'uint8', 'Position', [395, 18, 445, 42]);
                add_line(subPath, [prefix '_req_addr_z/1'], [prefix '_req_addr_z_u8_exp/1'], 'autorouting', 'on');
            end
            add_line(subPath, [prefix '_req_addr_z_u8_exp/1'], [prefix '_sram/4'], 'autorouting', 'on');
        end
    end
end

function enable_signal_logging(tbName, mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        rootName = mdlName;
        if isfield(spec, 'scope') && strcmp(spec.scope, 'tb')
            rootName = tbName;
        end
        blockPath = [rootName '/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_real_first_block_stage_delta_regression:MissingLine', ...
                'Cannot find signal line for %s', spec.block);
        end
        Simulink.sdi.markSignalForStreaming(blockHandle, spec.port, 'on');
        if ~isfield(spec, 'setName') || spec.setName
            set_param(lineHandle, 'Name', spec.name);
        end
    end
end

function cmp = compare_signal_pair(baselineData, realData)
    baselineData = double(baselineData(:));
    realData = double(realData(:));
    commonLen = min(numel(baselineData), numel(realData));
    baselineData = baselineData(1:commonLen);
    realData = realData(1:commonLen);
    delta = realData - baselineData;

    cmp = struct();
    cmp.sample_count = commonLen;
    cmp.baseline_nonzero = any(abs(baselineData) > 0);
    cmp.real_nonzero = any(abs(realData) > 0);
    cmp.max_abs_diff = max_or_zero(abs(delta));
    cmp.mean_abs_diff = mean_or_zero(abs(delta));
    cmp.changed = any(abs(delta) > 1e-9);
end

function laneSummaries = build_lane_summaries(stageResults, realLogged, weightRspCfg)
    laneTable = { ...
        'stage_rms_gamma', 1; ...
        'stage_qkv_q', 2; ...
        'stage_qkv_k', 3; ...
        'stage_qkv_v', 4; ...
        'stage_attn_q', 5; ...
        'stage_attn_k', 6; ...
        'stage_attn_v', 7; ...
        'stage_ffn_up', 8; ...
        'stage_ffn_gate', 9};

    laneSummaries = repmat(struct( ...
        'lane', '', ...
        'expected_addr', 0, ...
        'expected_data', 0, ...
        'weight_ref_data_changed', false, ...
        'weight_ref_valid_active', false, ...
        'weight_ref_addr_active', false, ...
        'weight_ref_addr_match', false, ...
        'weight_ref_data_match', false, ...
        'sram_addr_active', false, ...
        'sram_addr_match', false, ...
        'sram_din_changed', false, ...
        'sram_din_match', false, ...
        'sram_we_active', false, ...
        'sram_dout_changed', false, ...
        'sram_dout_match', false, ...
        'sram_dout_matches_din_d1', false, ...
        'ddr_data_changed', false, ...
        'ddr_valid_active', false, ...
        'req_needed_active', false, ...
        'valid_or_active', false, ...
        'sram_changed', false, ...
        'sram_valid_active', false, ...
        'sram_sel_changed', false), [1, size(laneTable, 1)]);

    for i = 1:size(laneTable, 1)
        key = laneTable{i, 1};
        laneIndex = laneTable{i, 2};
        expectedAddr = double(weightRspCfg.lane_expected_addrs(laneIndex));
        expectedData = double(weightRspCfg.sample_values(laneIndex));
        srcAddr = get_logged_or_empty(realLogged, ['weight_ref_lane_' num2str(laneIndex) '_addr_d2']);
        srcValid = get_logged_or_empty(realLogged, ['weight_ref_lane_' num2str(laneIndex) '_val_d2']);
        srcData = get_logged_or_empty(realLogged, ['weight_ref_lane_' num2str(laneIndex) '_data_u8']);
        ramAddr = get_logged_or_empty(realLogged, [key '_sram_addr_alias']);
        ramDin = get_logged_or_empty(realLogged, [key '_sram_din_alias']);
        ramWe = get_logged_or_empty(realLogged, [key '_sram_we_alias']);
        ramDout = get_logged_or_empty(realLogged, [key '_sram_dout_alias']);

        laneSummaries(i).lane = key;
        laneSummaries(i).expected_addr = expectedAddr;
        laneSummaries(i).expected_data = expectedData;
        laneSummaries(i).weight_ref_data_changed = result_field(stageResults, ['weight_ref_lane_' num2str(laneIndex) '_data_u8'], 'changed');
        laneSummaries(i).weight_ref_valid_active = any_nonzero(stageResults, ['weight_ref_lane_' num2str(laneIndex) '_val_d2']);
        laneSummaries(i).weight_ref_addr_active = any_nonzero(stageResults, ['weight_ref_lane_' num2str(laneIndex) '_addr_d2']);
        laneSummaries(i).weight_ref_addr_match = matches_expected_under_mask(srcAddr, srcValid, expectedAddr);
        laneSummaries(i).weight_ref_data_match = contains_value(srcData, expectedData);
        laneSummaries(i).sram_addr_active = any_nonzero(stageResults, [key '_sram_addr_alias']);
        laneSummaries(i).sram_addr_match = matches_expected_under_mask(ramAddr, ramWe, expectedAddr);
        laneSummaries(i).sram_din_changed = result_field(stageResults, [key '_sram_din_alias'], 'changed');
        laneSummaries(i).sram_din_match = matches_expected_under_mask(ramDin, ramWe, expectedData);
        laneSummaries(i).sram_we_active = any_nonzero(stageResults, [key '_sram_we_alias']);
        laneSummaries(i).sram_dout_changed = result_field(stageResults, [key '_sram_dout_alias'], 'changed');
        laneSummaries(i).sram_dout_match = contains_value(ramDout, expectedData);
        laneSummaries(i).sram_dout_matches_din_d1 = matches_shifted_write_data(ramDin, ramDout, ramWe, 1);
        laneSummaries(i).ddr_data_changed = result_field(stageResults, [key '_ddr_data_u8'], 'changed');
        laneSummaries(i).ddr_valid_active = any_nonzero(stageResults, [key '_ddr_valid_bool']);
        laneSummaries(i).req_needed_active = any_nonzero(stageResults, [key '_req_needed']);
        laneSummaries(i).valid_or_active = any_nonzero(stageResults, [key '_valid_or']);
        laneSummaries(i).sram_changed = result_field(stageResults, [key '_sram'], 'changed');
        laneSummaries(i).sram_valid_active = any_nonzero(stageResults, [key '_sram_data_valid_z']);
        laneSummaries(i).sram_sel_changed = result_field(stageResults, [key '_sram_data_sel'], 'changed');
    end
end

function print_lane_summaries(laneSummaries)
    for i = 1:numel(laneSummaries)
        fprintf(['  lane=%s exp_addr=%g exp_data=%g src_data_changed=%d src_valid_active=%d src_addr_active=%d ' ...
            'src_addr_match=%d src_data_match=%d ram_addr_active=%d ram_addr_match=%d ' ...
            'ram_din_changed=%d ram_din_match=%d ram_we_active=%d ram_dout_changed=%d ram_dout_match=%d ram_dout_d1=%d ' ...
            'ddr_data_changed=%d ddr_valid_active=%d ' ...
            'req_needed_active=%d valid_or_active=%d sram_changed=%d sram_valid_active=%d sram_sel_changed=%d\n'], ...
            laneSummaries(i).lane, ...
            laneSummaries(i).expected_addr, ...
            laneSummaries(i).expected_data, ...
            laneSummaries(i).weight_ref_data_changed, ...
            laneSummaries(i).weight_ref_valid_active, ...
            laneSummaries(i).weight_ref_addr_active, ...
            laneSummaries(i).weight_ref_addr_match, ...
            laneSummaries(i).weight_ref_data_match, ...
            laneSummaries(i).sram_addr_active, ...
            laneSummaries(i).sram_addr_match, ...
            laneSummaries(i).sram_din_changed, ...
            laneSummaries(i).sram_din_match, ...
            laneSummaries(i).sram_we_active, ...
            laneSummaries(i).sram_dout_changed, ...
            laneSummaries(i).sram_dout_match, ...
            laneSummaries(i).sram_dout_matches_din_d1, ...
            laneSummaries(i).ddr_data_changed, ...
            laneSummaries(i).ddr_valid_active, ...
            laneSummaries(i).req_needed_active, ...
            laneSummaries(i).valid_or_active, ...
            laneSummaries(i).sram_changed, ...
            laneSummaries(i).sram_valid_active, ...
            laneSummaries(i).sram_sel_changed);
    end
end

function summary = build_attention_output_alignment_summary(baselineLogged, realLogged)
    reduceBase = get_logged_or_empty(baselineLogged, 'flow_attn_scorev_reduce');
    reduceReal = get_logged_or_empty(realLogged, 'flow_attn_scorev_reduce');
    validReal = get_logged_or_empty(realLogged, 'flow_attn_scorev_valid_z');
    gateReal = get_logged_or_empty(realLogged, 'flow_attn_output_valid_gate');
    stageReal = get_logged_or_empty(realLogged, 'flow_attn_scorev_stage_z');

    commonLen = min([numel(reduceBase), numel(reduceReal), numel(validReal)]);
    reduceBase = double(reduceBase(1:commonLen));
    reduceReal = double(reduceReal(1:commonLen));
    validMask = double(validReal(1:commonLen)) > 0.5;
    deltaMask = abs(reduceReal - reduceBase) > 1e-9;

    lagRange = -3:3;
    lagHits = false(size(lagRange));
    for i = 1:numel(lagRange)
        lagHits(i) = masks_overlap_with_shift(deltaMask, validMask, lagRange(i));
    end

    summary = struct();
    summary.sample_count = commonLen;
    summary.scorev_reduce_changed = any(deltaMask);
    summary.scorev_valid_active = any(validMask);
    summary.changed_under_valid = any(deltaMask & validMask);
    summary.output_gate_nonzero = any(abs(double(gateReal(:))) > 1e-9);
    summary.output_stage_nonzero = any(abs(double(stageReal(:))) > 1e-9);
    summary.lag_range = lagRange;
    summary.lag_hits = lagHits;
end

function print_attention_output_alignment(summary)
    fprintf('  attn_output_alignment sample_count=%d reduce_changed=%d valid_active=%d changed_under_valid=%d gate_nonzero=%d stage_nonzero=%d\n', ...
        summary.sample_count, summary.scorev_reduce_changed, summary.scorev_valid_active, ...
        summary.changed_under_valid, summary.output_gate_nonzero, summary.output_stage_nonzero);
    for i = 1:numel(summary.lag_range)
        fprintf('    attn_output_alignment lag=%d overlap=%d\n', summary.lag_range(i), summary.lag_hits(i));
    end
end

function summary = build_ffn_gate_alignment_summary(baselineLogged, realLogged)
    gateNormBase = get_logged_or_empty(baselineLogged, 'flow_ffn_gate_norm');
    gateNormReal = get_logged_or_empty(realLogged, 'flow_ffn_gate_norm');
    validReal = get_logged_or_empty(realLogged, 'flow_ffn_gateup_pair_valid_z');
    gateGateReal = get_logged_or_empty(realLogged, 'flow_ffn_gate_norm_gate');
    swigluReal = get_logged_or_empty(realLogged, 'flow_ffn_swiglu_mul');

    commonLen = min([numel(gateNormBase), numel(gateNormReal), numel(validReal)]);
    gateNormBase = double(gateNormBase(1:commonLen));
    gateNormReal = double(gateNormReal(1:commonLen));
    validMask = double(validReal(1:commonLen)) > 0.5;
    deltaMask = abs(gateNormReal - gateNormBase) > 1e-9;

    lagRange = -3:3;
    lagHits = false(size(lagRange));
    for i = 1:numel(lagRange)
        lagHits(i) = masks_overlap_with_shift(deltaMask, validMask, lagRange(i));
    end

    summary = struct();
    summary.sample_count = commonLen;
    summary.gate_norm_changed = any(deltaMask);
    summary.valid_active = any(validMask);
    summary.changed_under_valid = any(deltaMask & validMask);
    summary.gated_output_nonzero = any(abs(double(gateGateReal(:))) > 1e-9);
    summary.swiglu_mul_nonzero = any(abs(double(swigluReal(:))) > 1e-9);
    summary.lag_range = lagRange;
    summary.lag_hits = lagHits;
end

function print_ffn_gate_alignment(summary)
    fprintf('  ffn_gate_alignment sample_count=%d gate_norm_changed=%d valid_active=%d changed_under_valid=%d gated_nonzero=%d swiglu_nonzero=%d\n', ...
        summary.sample_count, summary.gate_norm_changed, summary.valid_active, ...
        summary.changed_under_valid, summary.gated_output_nonzero, summary.swiglu_mul_nonzero);
    for i = 1:numel(summary.lag_range)
        fprintf('    ffn_gate_alignment lag=%d overlap=%d\n', summary.lag_range(i), summary.lag_hits(i));
    end
end

function yes = masks_overlap_with_shift(dataMask, validMask, lag)
    if isempty(dataMask) || isempty(validMask)
        yes = false;
        return;
    end

    commonLen = min(numel(dataMask), numel(validMask));
    dataMask = logical(dataMask(1:commonLen));
    validMask = logical(validMask(1:commonLen));

    if lag >= 0
        if commonLen <= lag
            yes = false;
            return;
        end
        yes = any(dataMask(1:end-lag) & validMask(1+lag:end));
    else
        lagAbs = -lag;
        if commonLen <= lagAbs
            yes = false;
            return;
        end
        yes = any(dataMask(1+lagAbs:end) & validMask(1:end-lagAbs));
    end
end

function yes = any_nonzero(stageResults, name)
    yes = result_field(stageResults, name, 'baseline_nonzero') || result_field(stageResults, name, 'real_nonzero');
end

function value = result_field(stageResults, name, fieldName)
    value = false;
    for i = 1:numel(stageResults)
        if strcmp(stageResults(i).name, name)
            value = stageResults(i).(fieldName);
            return;
        end
    end
end

function values = get_logged_or_empty(loggedStruct, name)
    if isfield(loggedStruct, name)
        values = double(loggedStruct.(name));
    else
        values = [];
    end
end

function yes = matches_expected_under_mask(values, maskValues, expected)
    values = double(values(:));
    maskValues = double(maskValues(:));
    commonLen = min(numel(values), numel(maskValues));
    values = values(1:commonLen);
    maskValues = maskValues(1:commonLen) > 0.5;
    yes = any(maskValues & abs(values - expected) < 1e-9);
end

function yes = contains_value(values, expected)
    values = double(values(:));
    yes = any(abs(values - expected) < 1e-9);
end

function yes = matches_shifted_write_data(writeData, readData, writeEnable, lag)
    writeData = double(writeData(:));
    readData = double(readData(:));
    writeEnable = double(writeEnable(:));
    commonLen = min([numel(writeData), numel(readData), numel(writeEnable)]);
    writeData = writeData(1:commonLen);
    readData = readData(1:commonLen);
    writeEnable = writeEnable(1:commonLen) > 0.5;
    if commonLen <= lag
        yes = false;
        return;
    end
    mask = writeEnable(1:end-lag);
    yes = any(mask) && all(abs(readData(1+lag:end) - writeData(1:end-lag)) < 1e-9 | ~mask);
end

function inject_sample_values_into_weight_ref(tbName, sampleValues)
    subPath = [tbName '/weight_ref_u'];
    for i = 1:min(9, numel(sampleValues))
        constName = ['sample_value_' num2str(i)];
        constPath = [subPath '/' constName];
        if isempty(find_system(subPath, 'SearchDepth', 1, 'Name', constName))
            add_block('simulink/Sources/Constant', constPath, ...
                'Position', [455, 25 + 30 * (i - 1) + 6, 520, 25 + 30 * (i - 1) + 26]);
        end
        set_param(constPath, 'Value', num2str(double(sampleValues(i))));

        try
            delete_line(subPath, ['data_page_tag_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        try
            delete_line(subPath, ['tag_lane_sum_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        try
            delete_line(subPath, ['sample_value_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        add_line(subPath, [constName '/1'], ['data_u8_' num2str(i) '/1'], 'autorouting', 'on');
    end
end

function lineHandle = get_src_line_handle(blockPath, portIndex)
    lineHandle = -1;
    if getSimulinkBlockHandle(blockPath) == -1
        return;
    end
    ph = get_param(blockPath, 'PortHandles');
    if numel(ph.Outport) < portIndex
        return;
    end
    lineHandle = get_param(ph.Outport(portIndex), 'Line');
end

function values = extract_logged_signal(logsout, name, blockSuffix)
    if nargin < 3
        blockSuffix = '';
    end
    for i = 1:logsout.numElements
        sig = logsout.get(i);
        sigName = string(sig.Name);
        blockPath = string('');
        try
            blockPath = string(sig.BlockPath.getBlock(1));
        catch
        end
        if sigName == string(name) || endsWith(blockPath, "/" + string(name)) || ...
                (strlength(string(blockSuffix)) > 0 && endsWith(blockPath, "/" + string(blockSuffix)))
            values = sig.Values.Data;
            return;
        end
    end
    error('run_stage2_real_first_block_stage_delta_regression:MissingLoggedSignal', ...
        'Logged signal not found: %s', name);
end

function values = extract_logged_signal_optional(logsout, name, blockSuffix)
    try
        values = extract_logged_signal(logsout, name, blockSuffix);
    catch ME
        if contains(ME.identifier, 'MissingLoggedSignal')
            values = [];
            return;
        end
        rethrow(ME);
    end
end

function values = extract_signal(yout, name)
    for i = 1:yout.numElements
        sig = yout.get(i);
        sigName = string(sig.Name);
        blockPath = string('');
        try
            blockPath = string(sig.BlockPath.getBlock(1));
        catch
        end
        if sigName == string(name) || endsWith(blockPath, "/" + string(name))
            values = sig.Values.Data;
            return;
        end
    end
    error('run_stage2_real_first_block_stage_delta_regression:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function value = max_or_zero(values)
    if isempty(values)
        value = 0;
    else
        value = max(values);
    end
end

function value = mean_or_zero(values)
    if isempty(values)
        value = 0;
    else
        value = mean(values);
    end
end

function safe_close_models(tbName, mdlName)
    if bdIsLoaded(tbName)
        close_system(tbName, 0);
    end
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
end