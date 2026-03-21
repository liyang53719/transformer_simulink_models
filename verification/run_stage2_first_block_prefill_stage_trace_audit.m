function result = run_stage2_first_block_prefill_stage_trace_audit(rootDir, options)
%RUN_STAGE2_FIRST_BLOCK_PREFILL_STAGE_TRACE_AUDIT Dump stage-aligned 64-token traces for first-block prefill.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = logical(getFieldOr(options, 'BuildModel', false));
    assert_stage2_manual_model_policy(buildModel, mfilename);

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));

    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    baselineOptions = getFieldOr(options, 'BaselineOptions', struct('NumTokens', 64, 'DriveTokenPosSequence', true));
    baseline = get_stage2_first_block_prefill_numeric_baseline(baselineOptions);
    signalSpecs = build_stage_signal_specs();

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    enable_signal_logging(tbName, mdlName, signalSpecs);
    configure_prefill_tb_sources(tbName, baseline.stimulus);
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', num2str(baseline.stop_time), 'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', 'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');

    sampleTimes = double(baseline.stimulus.time(:));
    reference = get_stage2_first_block_prefill_reference_baseline(rootDir, struct( ...
        'BaselineOptions', baselineOptions, ...
        'EnableStageTrace', true, ...
        'Stimulus', baseline.stimulus));

    result = struct();
    result.reference_out_hidden = double(reference.reference_scalar_contract_out_hidden(:));
    result.reference_stage_contract_trace = getFieldOr(reference, 'reference_stage_contract_trace', struct());
    result.stages = repmat(empty_stage_result(), 1, numel(signalSpecs));

    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        validSig = extract_logged_signal_object(logsout, spec.valid_name, spec.valid_block);
        dataSig = extract_logged_signal_object(logsout, spec.data_name, spec.data_block);
        phase = resolve_output_sample_phase(validSig.Values.Time, validSig.Values.Data, sampleTimes);
        alignedTimes = sampleTimes + phase;
        validValues = resample_numeric_values(validSig.Values.Time, validSig.Values.Data, alignedTimes);
        dataValues = resample_numeric_values(dataSig.Values.Time, dataSig.Values.Data, alignedTimes);
        validMask = validValues(:) > 0.5;
        tokenValues = dataValues(validMask);
        [~, maxIdx] = max(abs(tokenValues));
        firstExplodeIdx = find(abs(tokenValues) > getFieldOr(options, 'ExplosionThreshold', 1e3), 1, 'first');

        stageResult = empty_stage_result();
        stageResult.name = spec.name;
        stageResult.phase = phase;
        stageResult.valid_count = sum(validMask);
        stageResult.valid_indices = find(validMask);
        stageResult.values = tokenValues(:);
        stageResult.max_abs_value = max_or_zero(abs(tokenValues));
        stageResult.mean_abs_value = mean_or_zero(abs(tokenValues));
        stageResult.max_abs_token_index = conditional_index(maxIdx, tokenValues);
        stageResult.first_exploded_token_index = conditional_index(firstExplodeIdx, tokenValues);
        stageResult.head_values = tokenValues(1:min(8, numel(tokenValues)));
        stageResult.tail_values = tokenValues(max(1, numel(tokenValues) - 7):numel(tokenValues));
        if strcmp(spec.name, 'residual_out')
            [stageResult.reference_common_count, stageResult.reference_max_abs_diff] = compare_prefix(tokenValues, result.reference_out_hidden);
            stageResult.placeholder_common_count = 0;
            stageResult.placeholder_max_abs_diff = compare_residual_placeholder(stageResult, result.stages, baseline.stimulus);
        else
            [stageResult.reference_common_count, stageResult.reference_max_abs_diff] = compare_reference_stage(spec.name, tokenValues, result.reference_stage_contract_trace);
        end
        result.stages(i) = stageResult;
    end

    fprintf('Stage2 first-block prefill stage trace audit PASS\n');
    for i = 1:numel(result.stages)
        stageResult = result.stages(i);
        fprintf('  %s phase=%.1f valid_count=%d max_abs=%g ref_diff=%s placeholder_diff=%s first_exploded_token=%s head=%s tail=%s\n', ...
            stageResult.name, stageResult.phase, stageResult.valid_count, stageResult.max_abs_value, ...
            printable_scalar(stageResult.reference_max_abs_diff), printable_scalar(stageResult.placeholder_max_abs_diff), ...
            printable_scalar(stageResult.first_exploded_token_index), ...
            mat2str(stageResult.head_values', 6), mat2str(stageResult.tail_values', 6));
    end
end

function specs = build_stage_signal_specs()
    specs = { ...
        struct('name', 'rms_out', 'data_name', 'prefill_rms_out', 'data_block', 'rmsnorm_u', 'data_port', 1, 'valid_name', 'prefill_in_valid', 'valid_block', 'src_in_valid', 'valid_port', 1, 'valid_scope', 'tb'), ...
        struct('name', 'qkv_out', 'data_name', 'prefill_qkv_out', 'data_block', 'qkv_proj_u', 'data_port', 1, 'valid_name', 'prefill_qkv_valid', 'valid_block', 'qkv_proj_u', 'valid_port', 4), ...
        struct('name', 'attn_out', 'data_name', 'prefill_attn_out', 'data_block', 'attention_u', 'data_port', 1, 'valid_name', 'prefill_attn_valid', 'valid_block', 'attention_u', 'valid_port', 3), ...
        struct('name', 'ffn_out', 'data_name', 'prefill_ffn_out', 'data_block', 'ffn_swiglu_u', 'data_port', 1, 'valid_name', 'prefill_ffn_valid', 'valid_block', 'ffn_swiglu_u', 'valid_port', 3), ...
        struct('name', 'residual_out', 'data_name', 'prefill_residual_out', 'data_block', 'residual_u', 'data_port', 1, 'valid_name', 'prefill_residual_valid', 'valid_block', 'residual_u', 'valid_port', 2), ...
        struct('name', 'attn_score_norm', 'data_name', 'prefill_attn_score_norm', 'data_block', 'attention_u/score_norm', 'data_port', 1, 'valid_name', 'prefill_attn_softmax_valid', 'valid_block', 'attention_u/softmax_valid_z', 'valid_port', 1), ...
        struct('name', 'attn_scorev_reduce', 'data_name', 'prefill_attn_scorev_reduce', 'data_block', 'attention_u/scorev_reduce', 'data_port', 1, 'valid_name', 'prefill_attn_scorev_valid', 'valid_block', 'attention_u/scorev_valid_out_z', 'valid_port', 1), ...
        struct('name', 'ffn_up_mul', 'data_name', 'prefill_ffn_up_mul', 'data_block', 'ffn_swiglu_u/up_mul', 'data_port', 1, 'valid_name', 'prefill_ffn_gateup_valid', 'valid_block', 'ffn_swiglu_u/gateup_pair_valid_z', 'valid_port', 1), ...
        struct('name', 'ffn_gate_mul', 'data_name', 'prefill_ffn_gate_mul', 'data_block', 'ffn_swiglu_u/gate_mul', 'data_port', 1, 'valid_name', 'prefill_ffn_gateup_valid', 'valid_block', 'ffn_swiglu_u/gateup_pair_valid_z', 'valid_port', 1), ...
        struct('name', 'ffn_gate_norm', 'data_name', 'prefill_ffn_gate_norm', 'data_block', 'ffn_swiglu_u/gate_norm', 'data_port', 1, 'valid_name', 'prefill_ffn_gateup_valid', 'valid_block', 'ffn_swiglu_u/gateup_pair_valid_z', 'valid_port', 1), ...
        struct('name', 'ffn_gate_norm_gate', 'data_name', 'prefill_ffn_gate_norm_gate', 'data_block', 'ffn_swiglu_u/gate_norm_gate', 'data_port', 1, 'valid_name', 'prefill_ffn_gateup_valid', 'valid_block', 'ffn_swiglu_u/gateup_pair_valid_z', 'valid_port', 1), ...
        struct('name', 'ffn_swiglu_mul', 'data_name', 'prefill_ffn_swiglu_mul', 'data_block', 'ffn_swiglu_u/swiglu_mul', 'data_port', 1, 'valid_name', 'prefill_ffn_swiglu_valid', 'valid_block', 'ffn_swiglu_u/swiglu_valid_gate_z2', 'valid_port', 1), ...
        struct('name', 'ffn_down_stage', 'data_name', 'prefill_ffn_down_stage', 'data_block', 'ffn_swiglu_u/down_stage_gate', 'data_port', 1, 'valid_name', 'prefill_ffn_down_valid', 'valid_block', 'ffn_swiglu_u/down_pair_valid_z2', 'valid_port', 1) ...
    };
end

function stageResult = empty_stage_result()
    stageResult = struct( ...
        'name', '', ...
        'phase', 0, ...
        'valid_count', 0, ...
        'valid_indices', [], ...
        'values', [], ...
        'max_abs_value', 0, ...
        'mean_abs_value', 0, ...
        'max_abs_token_index', [], ...
        'first_exploded_token_index', [], ...
        'head_values', [], ...
        'tail_values', [], ...
        'reference_common_count', 0, ...
        'reference_max_abs_diff', NaN, ...
        'placeholder_common_count', 0, ...
        'placeholder_max_abs_diff', NaN);
end

function maxAbsDiff = compare_residual_placeholder(stageResult, stageResults, stimulus)
    ffnIndex = find(strcmp({stageResults.name}, 'ffn_out'), 1, 'last');
    if isempty(ffnIndex)
        maxAbsDiff = NaN;
        return;
    end

    ffnStage = stageResults(ffnIndex);
    commonCount = min([numel(stageResult.values), numel(ffnStage.values), numel(stageResult.valid_indices)]);
    if commonCount == 0
        maxAbsDiff = NaN;
        return;
    end

    residualSeries = double(stimulus.in_residual(:));
    sampleIndices = stageResult.valid_indices(1:commonCount);
    sampleIndices = sampleIndices(sampleIndices >= 1 & sampleIndices <= numel(residualSeries));
    commonCount = min(commonCount, numel(sampleIndices));
    if commonCount == 0
        maxAbsDiff = NaN;
        return;
    end

    expected = double(ffnStage.values(1:commonCount)) + residualSeries(sampleIndices(:));
    maxAbsDiff = max(abs(double(stageResult.values(1:commonCount)) - expected(:)));
end

function [commonCount, maxAbsDiff] = compare_reference_stage(stageName, actual, referenceStageTrace)
    if ~isstruct(referenceStageTrace) || ~isfield(referenceStageTrace, stageName)
        commonCount = 0;
        maxAbsDiff = NaN;
        return;
    end
    [commonCount, maxAbsDiff] = compare_prefix(actual, referenceStageTrace.(stageName));
end

function enable_signal_logging(tbName, mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        enable_block_output_logging(mdlName, signalSpecs{i}.data_block, signalSpecs{i}.data_name, signalSpecs{i}.data_port);
        validRoot = mdlName;
        if isfield(signalSpecs{i}, 'valid_scope') && strcmp(signalSpecs{i}.valid_scope, 'tb')
            validRoot = tbName;
        end
        enable_block_output_logging(validRoot, signalSpecs{i}.valid_block, signalSpecs{i}.valid_name, signalSpecs{i}.valid_port);
    end
end

function enable_block_output_logging(rootName, relativeBlock, signalName, portIndex)
    fullBlock = [rootName '/' relativeBlock];
    blockHandle = getSimulinkBlockHandle(fullBlock);
    lineHandle = get_src_line_handle(fullBlock, portIndex);
    if blockHandle == -1 || lineHandle == -1
        error('run_stage2_first_block_prefill_stage_trace_audit:MissingBlock', 'Missing block: %s', fullBlock);
    end
    Simulink.sdi.markSignalForStreaming(blockHandle, portIndex, 'on');
    try
        set_param(lineHandle, 'Name', signalName);
    catch
    end
end

function lineHandle = get_src_line_handle(blockPath, portIndex)
    lineHandle = -1;
    if getSimulinkBlockHandle(blockPath) == -1
        return;
    end
    portHandles = get_param(blockPath, 'PortHandles');
    if numel(portHandles.Outport) < portIndex
        return;
    end
    lineHandle = get_param(portHandles.Outport(portIndex), 'Line');
end

function phase = resolve_output_sample_phase(rawTimes, rawValues, baseSampleTimes)
    solverStep = infer_signal_step(rawTimes);
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';
    bestPhase = 0;
    bestCount = -1;
    for i = 1:numel(candidatePhases)
        sampledValues = resample_numeric_values(rawTimes, rawValues, double(baseSampleTimes(:)) + candidatePhases(i));
        count = sum(sampledValues(:) > 0.5);
        if count > bestCount
            bestCount = count;
            bestPhase = candidatePhases(i);
        end
    end
    phase = double(bestPhase);
end

function values = resample_numeric_values(rawTimes, rawValues, queryTimes)
    rawTimes = double(rawTimes(:));
    rawValues = double(rawValues(:));
    queryTimes = double(queryTimes(:));
    inRangeMask = queryTimes >= rawTimes(1) & queryTimes <= rawTimes(end);
    sampled = nan(size(queryTimes));
    sampledTs = resample(timeseries(rawValues, rawTimes), queryTimes(inRangeMask));
    sampled(inRangeMask) = double(sampledTs.Data(:));
    values = sampled;
end

function step = infer_signal_step(times)
    diffs = diff(double(times(:)));
    diffs = diffs(diffs > 1e-9);
    if isempty(diffs)
        step = 0.2;
    else
        step = min(diffs);
    end
end

function [commonCount, maxAbsDiff] = compare_prefix(actual, expected)
    actual = double(actual(:));
    expected = double(expected(:));
    commonCount = min(numel(actual), numel(expected));
    if commonCount == 0
        maxAbsDiff = NaN;
        return;
    end
    maxAbsDiff = max(abs(actual(1:commonCount) - expected(1:commonCount)));
end

function value = conditional_index(indexValue, values)
    if isempty(indexValue) || isempty(values)
        value = [];
    else
        value = indexValue;
    end
end

function text = printable_scalar(value)
    if isempty(value)
        text = '[]';
    else
        text = num2str(value);
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

function sig = extract_logged_signal_object(logsout, signalName, blockPathHint)
    sig = [];
    for i = 1:logsout.numElements
        candidate = logsout.get(i);
        candidateName = string(candidate.Name);
        candidatePath = string('');
        try
            candidatePath = string(candidate.BlockPath.getBlock(1));
        catch
        end
        if candidateName == string(signalName) || endsWith(candidatePath, "/" + string(signalName)) || ...
                endsWith(candidatePath, "/" + string(blockPathHint))
            sig = candidate;
            return;
        end
    end
    if isempty(sig)
        error('run_stage2_first_block_prefill_stage_trace_audit:MissingLoggedSignal', ...
            'Missing logged signal: %s (%s)', signalName, blockPathHint);
    end
end

function configure_prefill_tb_sources(tbName, stimulus)
    set_constant_value(tbName, 'mode_decode', stimulus.mode_decode(1));
    set_constant_value(tbName, 'cfg_seq_len', stimulus.cfg_seq_len);
    set_constant_value(tbName, 'cfg_eps', stimulus.cfg_eps);
    set_constant_value(tbName, 'stop_req', stimulus.stop_req);
    set_constant_value(tbName, 'cfg_weight_num_heads', stimulus.cfg_weight_num_heads);
    set_constant_value(tbName, 'cfg_weight_page_base', stimulus.cfg_weight_page_base);
    set_constant_value(tbName, 'cfg_weight_page_stride', stimulus.cfg_weight_page_stride);
    set_constant_value(tbName, 'cfg_rope_theta_scale', stimulus.cfg_rope_theta_scale);
    set_constant_value(tbName, 'cfg_rope_sin_mix_scale', stimulus.cfg_rope_sin_mix_scale);
    set_constant_value(tbName, 'eos_in', stimulus.eos_in(1));
    set_constant_value(tbName, 'out_ready', stimulus.out_ready(1));

    install_workspace_source(tbName, 'start', 'tb_prefill_start_seq', timeseries(logical(stimulus.start(:)), double(stimulus.time(:))));
    install_workspace_source(tbName, 'in_valid', 'tb_prefill_in_valid_seq', timeseries(logical(stimulus.in_valid(:)), double(stimulus.time(:))));
    install_numeric_source_if_needed(tbName, 'cfg_token_pos', 'tb_prefill_cfg_token_pos_seq', stimulus.cfg_token_pos, stimulus.time);
    install_workspace_source(tbName, 'in_hidden', 'tb_prefill_in_hidden_seq', make_sfix64_en30_timeseries(stimulus.in_hidden(:), stimulus.time(:)));
    install_workspace_source(tbName, 'in_residual', 'tb_prefill_in_residual_seq', make_sfix64_en30_timeseries(stimulus.in_residual(:), stimulus.time(:)));
end

function install_numeric_source_if_needed(tbName, signalName, variableName, values, time)
    values = double(values(:));
    if numel(values) <= 1
        set_constant_value(tbName, signalName, values(1));
        return;
    end
    install_workspace_source(tbName, signalName, variableName, make_ufix17_timeseries(values, time));
end

function ts = make_ufix17_timeseries(values, time)
    ts = timeseries(fi(double(values(:)), false, 17, 0), double(time(:)));
end

function ts = make_sfix64_en30_timeseries(values, time)
    ts = timeseries(fi(double(values(:)), true, 64, 30), double(time(:)));
end

function set_constant_value(tbName, signalName, value)
    blk = [tbName '/src_' signalName];
    if strcmp(get_param(blk, 'BlockType'), 'FromWorkspace')
        return;
    end
    set_param(blk, 'Value', num2str(double(value), '%.17g'));
end

function install_workspace_source(tbName, signalName, variableName, variableValue)
    blk = [tbName '/src_' signalName];
    assignin('base', variableName, variableValue);
    if strcmp(get_param(blk, 'BlockType'), 'FromWorkspace')
        set_param(blk, 'VariableName', variableName);
        configure_from_workspace_block(blk);
        return;
    end

    pos = get_param(blk, 'Position');
    ph = get_param(blk, 'PortHandles');
    lineHandle = get_param(ph.Outport(1), 'Line');
    dstPorts = [];
    if lineHandle ~= -1
        dstPorts = get_param(lineHandle, 'DstPortHandle');
        dstPorts = dstPorts(dstPorts ~= -1);
        delete_line(lineHandle);
    end
    delete_block(blk);
    add_block('simulink/Sources/From Workspace', blk, 'VariableName', variableName, 'Position', pos);
    configure_from_workspace_block(blk);

    for i = 1:numel(dstPorts)
        dstBlk = get_param(dstPorts(i), 'Parent');
        dstPort = get_param(dstPorts(i), 'PortNumber');
        dstName = erase(string(dstBlk), string(tbName) + "/");
        add_line(tbName, ['src_' signalName '/1'], char(dstName) + "/" + num2str(dstPort), 'autorouting', 'on');
    end
end

function configure_from_workspace_block(blockPath)
    try
        set_param(blockPath, 'Interpolate', 'off');
    catch
    end
    try
        set_param(blockPath, 'OutputAfterFinalValue', 'Holding final value');
    catch
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
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
