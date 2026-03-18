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
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    signalSpecs = build_signal_specs();

    baseline = simulate_stage_signals(rootDir, kvCfg, buildModel, [], signalSpecs);
    realSample = simulate_stage_signals(rootDir, kvCfg, false, double(weightRspCfg.sample_values), signalSpecs);

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

    result = struct();
    result.stage_results = stageResults;
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
            fprintf('  %s changed=%d max_abs_diff=%g mean_abs_diff=%g\n', ...
                stageResults(i).name, stageResults(i).changed, ...
                stageResults(i).max_abs_diff, stageResults(i).mean_abs_diff);
        end
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
        struct('block', 'rmsnorm_u/gamma_sram', 'port', 1, 'name', 'stage_gamma_sram'), ...
        struct('block', 'qkv_proj_u/q_sram', 'port', 1, 'name', 'stage_qkv_q_sram'), ...
        struct('block', 'attention_u/q_sram', 'port', 1, 'name', 'stage_attn_q_sram'), ...
        struct('block', 'attention_u/v_sram', 'port', 1, 'name', 'stage_attn_v_sram'), ...
        struct('block', 'attention_u/row_sum_accum', 'port', 1, 'name', 'stage_attn_row_sum_accum'), ...
        struct('block', 'ffn_swiglu_u/up_sram', 'port', 1, 'name', 'stage_ffn_up_sram'), ...
        struct('block', 'ffn_swiglu_u/gate_sram', 'port', 1, 'name', 'stage_ffn_gate_sram')};
end

function simResult = simulate_stage_signals(rootDir, kvCfg, buildModel, sampleValues, signalSpecs)
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    enable_signal_logging(mdlName, signalSpecs);

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
        simResult.logged.(signalSpecs{i}.name) = double(extract_logged_signal(logsout, signalSpecs{i}.name));
    end
    simResult.out_hidden = double(extract_signal(yout, 'out_hidden'));
end

function enable_signal_logging(mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockPath = [mdlName '/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_real_first_block_stage_delta_regression:MissingLine', ...
                'Cannot find signal line for %s', spec.block);
        end
        Simulink.sdi.markSignalForStreaming(blockHandle, spec.port, 'on');
        set_param(lineHandle, 'Name', spec.name);
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

function values = extract_logged_signal(logsout, name)
    for i = 1:logsout.numElements
        sig = logsout.get(i);
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
    error('run_stage2_real_first_block_stage_delta_regression:MissingLoggedSignal', ...
        'Logged signal not found: %s', name);
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