function result = debug_stage2_prefill_residual_phase_logs(rootDir, options)
%DEBUG_STAGE2_PREFILL_RESIDUAL_PHASE_LOGS Probe residual-boundary stages across candidate sample phases.

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
    baselineOptions = getFieldOr(options, 'BaselineOptions', struct('NumTokens', 64, 'DriveTokenPosSequence', true));
    baseline = get_stage2_first_block_prefill_numeric_baseline(baselineOptions);
    weightRspOptions = options;
    weightRspOptions.BaselineOptions = baselineOptions;
    weightRspOptions.Stimulus = baseline.stimulus;
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, weightRspOptions);
    refBaseline = get_stage2_first_block_prefill_reference_baseline(rootDir, struct( ...
        'BaselineOptions', baselineOptions, ...
        'EnableStageTrace', true, ...
        'Stimulus', baseline.stimulus));

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    signalSpecs = { ...
        struct('name', 'attn_out', 'data_name', 'probe_attn_out', 'data_block', 'attention_u', 'data_port', 1, 'valid_name', 'probe_attn_valid', 'valid_block', 'attention_u', 'valid_port', 3), ...
        struct('name', 'attn_residual_out', 'data_name', 'probe_attn_residual_out', 'data_block', 'attn_residual_u', 'data_port', 1, 'valid_name', 'probe_attn_residual_valid', 'valid_block', 'attn_residual_u', 'valid_port', 2), ...
        struct('name', 'attn_residual_skip_live', 'data_name', 'probe_attn_residual_skip_live', 'data_block', 'attn_residual_u/x_skip', 'data_port', 1, 'valid_name', 'probe_in_valid', 'valid_block', 'src_in_valid', 'valid_port', 1, 'valid_scope', 'tb'), ...
        struct('name', 'attn_residual_main_delayed', 'data_name', 'probe_attn_residual_main_delayed', 'data_block', 'attn_residual_u/main_delay', 'data_port', 1, 'valid_name', 'probe_attn_residual_valid_delayed', 'valid_block', 'attn_residual_u/valid_delay', 'valid_port', 1), ...
        struct('name', 'post_attn_norm_out', 'data_name', 'probe_post_attn_norm_out', 'data_block', 'post_attn_norm_u', 'data_port', 1, 'valid_name', 'probe_post_attn_norm_valid', 'valid_block', 'post_attn_norm_u', 'valid_port', 2), ...
        struct('name', 'ffn_out', 'data_name', 'probe_ffn_out', 'data_block', 'ffn_swiglu_u', 'data_port', 1, 'valid_name', 'probe_ffn_valid', 'valid_block', 'ffn_swiglu_u', 'valid_port', 3), ...
        struct('name', 'residual_out', 'data_name', 'probe_residual_out', 'data_block', 'residual_u', 'data_port', 1, 'valid_name', 'probe_residual_valid', 'valid_block', 'residual_u', 'valid_port', 2)};

    enable_signal_logging(tbName, mdlName, signalSpecs);
    configure_prefill_tb_sources(tbName, baseline.stimulus);
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', num2str(baseline.stop_time), 'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', 'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');
    sampleTimes = double(baseline.stimulus.time(:));

    result = struct();
    result.stages = cell(1, numel(signalSpecs));
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        stageResult = phase_scan_stage(logsout, spec, sampleTimes, refBaseline);
        result.stages{i} = stageResult;
        print_stage_result(stageResult);
    end
    result.residual_relation = analyze_residual_relation(result.stages, baseline.stimulus);
    print_residual_relation(result.residual_relation);
    result.residual_live_skip = analyze_residual_live_skip(logsout, result.stages, sampleTimes, baseline.stimulus);
    print_residual_live_skip(result.residual_live_skip);
end

function stageResult = phase_scan_stage(logsout, spec, sampleTimes, refBaseline)
    validSig = extract_logged_signal_object(logsout, spec.valid_name, spec.valid_block);
    dataSig = extract_logged_signal_object(logsout, spec.data_name, spec.data_block);
    solverStep = min(infer_signal_step(validSig.Values.Time), infer_signal_step(dataSig.Values.Time));
    if ~(solverStep > 0)
        solverStep = max(infer_signal_step(validSig.Values.Time), infer_signal_step(dataSig.Values.Time));
    end
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';

    contractTrace = getFieldOr(refBaseline, 'reference_stage_contract_trace', struct());
    realTrace = getFieldOr(refBaseline, 'reference_stage_contract_trace_real', struct());
    placeholderTrace = getFieldOr(refBaseline, 'reference_stage_contract_trace_placeholder', struct());
    contractTarget = get_reference_vector(contractTrace, spec.name);
    realTarget = get_reference_vector(realTrace, spec.name);
    placeholderTarget = get_reference_vector(placeholderTrace, spec.name);

    scans = repmat(struct( ...
        'phase', 0, ...
        'valid_count', 0, ...
        'contract_diff', NaN, ...
        'real_diff', NaN, ...
        'placeholder_diff', NaN, ...
        'values', zeros(0, 1), ...
        'head_values', zeros(0, 1)), 1, numel(candidatePhases));
    bestContractIdx = 1;
    bestContractDiff = Inf;
    for i = 1:numel(candidatePhases)
        alignedTimes = sampleTimes + candidatePhases(i);
        validSampled = resample_numeric_values(validSig.Values.Time, validSig.Values.Data, alignedTimes);
        dataSampled = resample_numeric_values(dataSig.Values.Time, dataSig.Values.Data, alignedTimes);
        values = double(dataSampled(validSampled(:) > 0.5));
        scans(i).phase = double(candidatePhases(i));
        scans(i).valid_count = numel(values);
        scans(i).contract_diff = compare_prefix(values, contractTarget);
        scans(i).real_diff = compare_prefix(values, realTarget);
        scans(i).placeholder_diff = compare_prefix(values, placeholderTarget);
        scans(i).values = values;
        scans(i).head_values = values(1:min(8, numel(values)));
        if isfinite(scans(i).contract_diff) && scans(i).contract_diff < bestContractDiff
            bestContractDiff = scans(i).contract_diff;
            bestContractIdx = i;
        end
    end

    stageResult = struct();
    stageResult.name = spec.name;
    stageResult.best_contract_phase = scans(bestContractIdx).phase;
    stageResult.best_contract_diff = scans(bestContractIdx).contract_diff;
    stageResult.best_real_diff = scans(bestContractIdx).real_diff;
    stageResult.best_placeholder_diff = scans(bestContractIdx).placeholder_diff;
    stageResult.best_valid_count = scans(bestContractIdx).valid_count;
    stageResult.best_values = scans(bestContractIdx).values;
    stageResult.best_head_values = scans(bestContractIdx).head_values;
    [stageResult.best_contract_shift, stageResult.best_shifted_contract_diff] = compare_shifted_prefix(scans(bestContractIdx).values, contractTarget, 8);
    stageResult.scans = scans;
end

function print_stage_result(stageResult)
    fprintf('Residual phase scan %s\n', stageResult.name);
    fprintf('  best_contract_phase=%.1f valid=%d contract_diff=%g real_diff=%g placeholder_diff=%g best_shift=%d shifted_diff=%g head=%s\n', ...
        stageResult.best_contract_phase, stageResult.best_valid_count, stageResult.best_contract_diff, ...
        stageResult.best_real_diff, stageResult.best_placeholder_diff, stageResult.best_contract_shift, ...
        stageResult.best_shifted_contract_diff, mat2str(stageResult.best_head_values', 6));
    for i = 1:numel(stageResult.scans)
        scan = stageResult.scans(i);
        fprintf('    phase=%.1f valid=%d contract_diff=%g real_diff=%g placeholder_diff=%g head=%s\n', ...
            scan.phase, scan.valid_count, scan.contract_diff, scan.real_diff, scan.placeholder_diff, mat2str(scan.head_values', 6));
    end
end

function relation = analyze_residual_relation(stageResults, stimulus)
    relation = struct();
    relation.available = false;
    relation.best_residual_phase = NaN;
    relation.best_attn_phase = NaN;
    relation.best_valid_count = 0;
    relation.residual_component_diff = NaN;
    relation.residual_component_shift = NaN;
    relation.residual_component_shifted_diff = NaN;
    relation.attn_residual_vs_input_diff = NaN;
    relation.attn_residual_vs_input_shift = NaN;
    relation.attn_residual_vs_input_shifted_diff = NaN;
    relation.delta_head_values = zeros(0, 1);
    relation.residual_head_values = zeros(0, 1);
    relation.best_pair_index = [NaN NaN];

    stageNames = cellfun(@(s) string(s.name), stageResults, 'UniformOutput', false);
    residualIdx = find(strcmp([stageNames{:}], "attn_residual_out"), 1, 'first');
    attnIdx = find(strcmp([stageNames{:}], "attn_out"), 1, 'first');
    if isempty(residualIdx) || isempty(attnIdx)
        return;
    end

    tokenMask = logical(stimulus.in_valid(:));
    residualTarget = double(stimulus.in_residual(tokenMask));
    residualStage = stageResults{residualIdx};
    attnStage = stageResults{attnIdx};
    if isempty(residualTarget) || ~isfield(residualStage, 'scans') || ~isfield(attnStage, 'scans')
        return;
    end

    bestScore = Inf;
    for residualScanIdx = 1:numel(residualStage.scans)
        residualScan = residualStage.scans(residualScanIdx);
        for attnScanIdx = 1:numel(attnStage.scans)
            attnScan = attnStage.scans(attnScanIdx);
            commonCount = min(numel(residualScan.values), numel(attnScan.values));
            if commonCount == 0
                continue;
            end

            deltaValues = double(residualScan.values(1:commonCount)) - double(attnScan.values(1:commonCount));
            componentDiff = compare_prefix(deltaValues, residualTarget);
            [componentShift, componentShiftedDiff] = compare_shifted_prefix(deltaValues, residualTarget, 8);
            directResidualDiff = compare_prefix(double(residualScan.values(1:commonCount)), residualTarget);
            [directResidualShift, directResidualShiftedDiff] = compare_shifted_prefix(double(residualScan.values(1:commonCount)), residualTarget, 8);

            if isfinite(componentShiftedDiff) && componentShiftedDiff < bestScore
                bestScore = componentShiftedDiff;
                relation.available = true;
                relation.best_residual_phase = residualScan.phase;
                relation.best_attn_phase = attnScan.phase;
                relation.best_valid_count = commonCount;
                relation.residual_component_diff = componentDiff;
                relation.residual_component_shift = componentShift;
                relation.residual_component_shifted_diff = componentShiftedDiff;
                relation.attn_residual_vs_input_diff = directResidualDiff;
                relation.attn_residual_vs_input_shift = directResidualShift;
                relation.attn_residual_vs_input_shifted_diff = directResidualShiftedDiff;
                relation.delta_head_values = deltaValues(1:min(8, commonCount));
                relation.residual_head_values = residualScan.values(1:min(8, commonCount));
                relation.best_pair_index = [residualScanIdx, attnScanIdx];
            end
        end
    end
end

function print_residual_relation(relation)
    fprintf('Residual relation attn_residual_out - attn_out vs input residual\n');
    if ~relation.available
        fprintf('  unavailable\n');
        return;
    end

    fprintf(['  residual_phase=%.1f attn_phase=%.1f valid=%d ' ...
        'delta_diff=%g delta_shift=%d delta_shifted_diff=%g ' ...
        'residual_diff=%g residual_shift=%d residual_shifted_diff=%g ' ...
        'delta_head=%s residual_head=%s\n'], ...
        relation.best_residual_phase, relation.best_attn_phase, relation.best_valid_count, ...
        relation.residual_component_diff, relation.residual_component_shift, relation.residual_component_shifted_diff, ...
        relation.attn_residual_vs_input_diff, relation.attn_residual_vs_input_shift, relation.attn_residual_vs_input_shifted_diff, ...
        mat2str(relation.delta_head_values', 6), mat2str(relation.residual_head_values', 6));
end

function relation = analyze_residual_live_skip(logsout, stageResults, sampleTimes, stimulus)
    relation = struct();
    relation.available = false;
    relation.output_phase = NaN;
    relation.output_valid_count = 0;
    relation.skip_diff = NaN;
    relation.skip_shift = NaN;
    relation.skip_shifted_diff = NaN;
    relation.skip_head_values = zeros(0, 1);
    relation.output_head_values = zeros(0, 1);

    names = cellfun(@(s) string(s.name), stageResults, 'UniformOutput', false);
    residualIdx = find(strcmp([names{:}], "attn_residual_out"), 1, 'first');
    skipIdx = find(strcmp([names{:}], "attn_residual_skip_live"), 1, 'first');
    if isempty(residualIdx) || isempty(skipIdx)
        return;
    end

    residualStage = stageResults{residualIdx};
    outputPhase = residualStage.best_contract_phase;
    residualValidSig = extract_logged_signal_object(logsout, 'probe_attn_residual_valid', 'attn_residual_u');
    skipSig = extract_logged_signal_object(logsout, 'probe_attn_residual_skip_live', 'attn_residual_u/x_skip');
    if isempty(residualValidSig) || isempty(skipSig)
        return;
    end

    alignedTimes = sampleTimes + outputPhase;
    outputValid = resample_numeric_values(residualValidSig.Values.Time, residualValidSig.Values.Data, alignedTimes);
    skipData = resample_numeric_values(skipSig.Values.Time, skipSig.Values.Data, alignedTimes);
    outputMask = outputValid(:) > 0.5;
    skipValues = double(skipData(outputMask));
    outputValues = double(residualStage.best_values(:));

    tokenMask = logical(stimulus.in_valid(:));
    residualTarget = double(stimulus.in_residual(tokenMask));
    if isempty(skipValues)
        return;
    end

    relation.available = true;
    relation.output_phase = outputPhase;
    relation.output_valid_count = sum(outputMask);
    relation.skip_diff = compare_prefix(skipValues, residualTarget);
    [relation.skip_shift, relation.skip_shifted_diff] = compare_shifted_prefix(skipValues, residualTarget, 8);
    relation.skip_head_values = skipValues(1:min(8, numel(skipValues)));
    relation.output_head_values = outputValues(1:min(8, numel(outputValues)));
end

function print_residual_live_skip(relation)
    fprintf('Residual live skip sampled at residual output phase\n');
    if ~relation.available
        fprintf('  unavailable\n');
        return;
    end

    fprintf(['  phase=%.1f valid=%d skip_diff=%g skip_shift=%d skip_shifted_diff=%g ' ...
        'skip_head=%s residual_out_head=%s\n'], ...
        relation.output_phase, relation.output_valid_count, relation.skip_diff, relation.skip_shift, relation.skip_shifted_diff, ...
        mat2str(relation.skip_head_values', 6), mat2str(relation.output_head_values', 6));
end

function scan = find_scan_by_phase(scans, phase)
    scan = [];
    for i = 1:numel(scans)
        if abs(double(scans(i).phase) - double(phase)) < 1e-9
            scan = scans(i);
            return;
        end
    end
end

function values = get_reference_vector(traceStruct, stageName)
    if isstruct(traceStruct) && isfield(traceStruct, stageName)
        values = double(traceStruct.(stageName)(:));
    else
        values = zeros(0, 1);
    end
end

function maxAbsDiff = compare_prefix(actual, expected)
    actual = double(actual(:));
    expected = double(expected(:));
    commonCount = min(numel(actual), numel(expected));
    if commonCount == 0
        maxAbsDiff = NaN;
        return;
    end
    maxAbsDiff = max(abs(actual(1:commonCount) - expected(1:commonCount)));
end

function [bestShift, bestDiff] = compare_shifted_prefix(actual, expected, maxShift)
    actual = double(actual(:));
    expected = double(expected(:));
    bestShift = 0;
    bestDiff = Inf;
    for shift = 0:maxShift
        shiftedExpected = expected;
        if shift > 0
            shiftedExpected = shiftedExpected(1 + shift:end);
        end
        commonCount = min(numel(actual), numel(shiftedExpected));
        if commonCount == 0
            continue;
        end
        diffValue = max(abs(actual(1:commonCount) - shiftedExpected(1:commonCount)));
        if diffValue < bestDiff
            bestDiff = diffValue;
            bestShift = shift;
        end
    end
    if ~isfinite(bestDiff)
        bestDiff = NaN;
    end
end

function enable_signal_logging(tbName, mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        enable_block_output_logging(mdlName, spec.data_block, spec.data_name, spec.data_port);
        validRoot = mdlName;
        if isfield(spec, 'valid_scope') && strcmp(spec.valid_scope, 'tb')
            validRoot = tbName;
        end
        enable_block_output_logging(validRoot, spec.valid_block, spec.valid_name, spec.valid_port);
    end
end

function enable_block_output_logging(rootName, relativeBlock, signalName, portIndex)
    fullBlock = [rootName '/' relativeBlock];
    blockHandle = getSimulinkBlockHandle(fullBlock);
    lineHandle = get_src_line_handle(fullBlock, portIndex);
    if blockHandle == -1 || lineHandle == -1
        error('debug_stage2_prefill_residual_phase_logs:MissingBlock', 'Missing block: %s', fullBlock);
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
    error('debug_stage2_prefill_residual_phase_logs:MissingLoggedSignal', ...
        'Missing logged signal: %s (%s)', signalName, blockPathHint);
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

function safe_close_models(varargin)
    for i = 1:numel(varargin)
        modelName = char(string(varargin{i}));
        if bdIsLoaded(modelName)
            try
                close_system(modelName, 0);
            catch
            end
        end
    end
end