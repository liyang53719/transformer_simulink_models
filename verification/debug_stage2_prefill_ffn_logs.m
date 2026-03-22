function debug_stage2_prefill_ffn_logs(rootDir, options)
%DEBUG_STAGE2_PREFILL_FFN_LOGS Probe FFN-related logs used by the prefill stage trace audit.

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

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    signalSpecs = { ...
        struct('block', 'ffn_swiglu_u', 'port', 1, 'name', 'prefill_ffn_out'), ...
        struct('block', 'ffn_swiglu_u', 'port', 3, 'name', 'prefill_ffn_valid'), ...
        struct('block', 'ffn_swiglu_u/down_stage_gate', 'port', 1, 'name', 'prefill_ffn_down_stage'), ...
        struct('block', 'ffn_swiglu_u/down_pair_valid_z2', 'port', 1, 'name', 'prefill_ffn_down_valid'), ...
        struct('block', 'ffn_swiglu_u/up_weight_scale', 'port', 1, 'name', 'probe_up_weight_scale'), ...
        struct('block', 'ffn_swiglu_u/up_valid_or', 'port', 1, 'name', 'probe_up_data_valid'), ...
        struct('block', 'ffn_swiglu_u/gate_weight_scale', 'port', 1, 'name', 'probe_gate_weight_scale'), ...
        struct('block', 'ffn_swiglu_u/gate_valid_or', 'port', 1, 'name', 'probe_gate_data_valid'), ...
        struct('block', 'ffn_swiglu_u/down_mul', 'port', 1, 'name', 'probe_down_mul'), ...
        struct('block', 'ffn_swiglu_u/down_mul_z2', 'port', 1, 'name', 'probe_down_mul_z2'), ...
        struct('block', 'residual_u', 'port', 1, 'name', 'prefill_residual_out'), ...
        struct('block', 'residual_u', 'port', 2, 'name', 'prefill_residual_valid')};

    enable_signal_logging(mdlName, signalSpecs);
    configure_prefill_tb_sources(tbName, baseline.stimulus);
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', num2str(baseline.stop_time), 'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', 'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');
    yout = simOut.get('yout');

    fprintf('FFN logsout probe:\n');
    for i = 1:logsout.numElements
        sig = logsout.get(i);
        sigName = char(string(sig.Name));
        blockPath = '';
        try
            blockPath = char(sig.BlockPath.getBlock(1));
        catch
        end
        if contains(sigName, 'ffn', 'IgnoreCase', true) || contains(blockPath, 'ffn_swiglu_u')
            data = double(sig.Values.Data(:));
            nz = find(abs(data) > 1e-9, 1, 'first');
            if isempty(nz)
                nz = 0;
            end
            fprintf('  name=%s | block=%s | first=%d | nnz=%d\n', sigName, blockPath, nz, nnz(abs(data) > 1e-9));
        end
    end

    sampleTimes = double(baseline.stimulus.time(:));
    for i = 1:numel(signalSpecs)
        sig = extract_logged_signal_object(logsout, signalSpecs{i}.name, signalSpecs{i}.block);
        phase = resolve_output_sample_phase(sig.Values.Time, sig.Values.Data, sampleTimes);
        alignedTimes = sampleTimes + phase;
        sampled = resample_numeric_values(sig.Values.Time, sig.Values.Data, alignedTimes);
        nz = find(abs(sampled) > 1e-9, 1, 'first');
        if isempty(nz)
            nz = 0;
        end
        fprintf('  sampled name=%s | phase=%.1f | first=%d | nnz=%d\n', signalSpecs{i}.name, phase, nz, nnz(abs(sampled) > 1e-9));
    end

    ffnOutSig = extract_logged_signal_object(logsout, 'prefill_ffn_out', 'ffn_swiglu_u');
    solverStep = infer_signal_step(ffnOutSig.Values.Time);
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';
    fprintf('FFN phase candidates:\n');
    for i = 1:numel(candidatePhases)
        sampled = resample_numeric_values(ffnOutSig.Values.Time, ffnOutSig.Values.Data, sampleTimes + candidatePhases(i));
        nz = find(abs(sampled) > 1e-9);
        head = [];
        if ~isempty(nz)
            head = sampled(nz(1:min(end, 6)));
        end
        fprintf('  phase=%.1f | nnz=%d | head=%s\n', candidatePhases(i), nnz(abs(sampled) > 1e-9), mat2str(head', 6));
    end

    print_probe_phase_candidates(logsout, sampleTimes, 'probe_up_weight_scale', 'ffn_swiglu_u/up_weight_scale');
    print_probe_phase_candidates(logsout, sampleTimes, 'probe_gate_weight_scale', 'ffn_swiglu_u/gate_weight_scale');
    print_probe_phase_candidates(logsout, sampleTimes, 'probe_gate_data_valid', 'ffn_swiglu_u/gate_valid_or');

    residualSig = extract_logged_signal_object(logsout, 'prefill_residual_out', 'residual_u');
    solverStep = infer_signal_step(residualSig.Values.Time);
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';
    fprintf('Residual phase candidates:\n');
    for i = 1:numel(candidatePhases)
        sampled = resample_numeric_values(residualSig.Values.Time, residualSig.Values.Data, sampleTimes + candidatePhases(i));
        nz = find(abs(sampled) > 1e-9);
        head = [];
        if ~isempty(nz)
            head = sampled(nz(1:min(end, 6)));
        end
        fprintf('  phase=%.1f | nnz=%d | head=%s\n', candidatePhases(i), nnz(abs(sampled) > 1e-9), mat2str(head', 6));
    end

    print_wrapper_ffn_observers(yout);
end

function print_wrapper_ffn_observers(yout)
    signalNames = { ...
        'tb_ffn_up_req_valid', 'tb_ffn_gate_req_valid', 'tb_ffn_down_req_valid', ...
        'tb_ffn_up_rsp_valid', 'tb_ffn_gate_rsp_valid', 'tb_ffn_down_rsp_valid', ...
        'tb_ffn_up_rsp_data', 'tb_ffn_gate_rsp_data', 'tb_ffn_down_rsp_data'};

    fprintf('Wrapper FFN observers:\n');
    for i = 1:numel(signalNames)
        element = try_get_dataset_signal(yout, signalNames{i});
        if ~isempty(element)
            values = double(element.Values.Data(:));
            nz = find(abs(values) > 1e-9);
            firstIdx = 0;
            lastIdx = 0;
            if ~isempty(nz)
                firstIdx = nz(1);
                lastIdx = nz(end);
            end
            fprintf('  %s | nnz=%d | first=%d | last=%d | unique=%s\n', ...
                signalNames{i}, numel(nz), firstIdx, lastIdx, summarize_unique_values(values));
        end
    end
end

function print_probe_phase_candidates(logsout, sampleTimes, signalName, blockPathHint)
    sig = extract_logged_signal_object(logsout, signalName, blockPathHint);
    solverStep = infer_signal_step(sig.Values.Time);
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';
    fprintf('%s phase candidates:\n', signalName);
    for i = 1:numel(candidatePhases)
        sampled = resample_numeric_values(sig.Values.Time, sig.Values.Data, sampleTimes + candidatePhases(i));
        nz = find(abs(sampled) > 1e-9);
        head = [];
        if ~isempty(nz)
            head = sampled(nz(1:min(end, 6)));
        end
        fprintf('  phase=%.1f | nnz=%d | head=%s\n', candidatePhases(i), nnz(abs(sampled) > 1e-9), mat2str(head', 6));
    end
end

function element = try_get_dataset_signal(yout, signalName)
    element = [];
    try
        candidate = yout.get(signalName);
    catch
        return;
    end
    if isa(candidate, 'Simulink.SimulationData.Signal')
        element = candidate;
    end
end

function text = summarize_unique_values(values)
    uniqueVals = unique(double(values(:)));
    if isempty(uniqueVals)
        text = '[]';
        return;
    end
    limit = min(numel(uniqueVals), 8);
    text = mat2str(uniqueVals(1:limit)', 6);
    if numel(uniqueVals) > limit
        text = [text ' ...'];
    end
end

function enable_signal_logging(mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockPath = [mdlName '/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('debug_stage2_prefill_ffn_logs:MissingLine', ...
                'Cannot find signal line for %s', spec.block);
        end
        Simulink.sdi.markSignalForStreaming(blockHandle, spec.port, 'on');
        set_param(lineHandle, 'Name', spec.name);
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
    error('debug_stage2_prefill_ffn_logs:MissingLoggedSignal', ...
        'Missing logged signal: %s (%s)', signalName, blockPathHint);
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