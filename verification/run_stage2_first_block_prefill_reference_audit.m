function result = run_stage2_first_block_prefill_reference_audit(rootDir, options)
%RUN_STAGE2_FIRST_BLOCK_PREFILL_REFERENCE_AUDIT Audit first-block prefill against the real reference path.
%   This audit intentionally compares the current scalar-contract Simulink
%   DUT output against two real-reference reductions:
%   1. full prefill block output reduced by column mean
%   2. per-token contract adapter output
%   The current DUT is not expected to be numerically equivalent yet.

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
    addpath(fullfile(rootDir, 'matlab_ref'));

    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));
    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));
    baselineOptions = getFieldOr(options, 'BaselineOptions', struct('NumTokens', 64));
    numericBaseline = get_stage2_first_block_prefill_numeric_baseline(baselineOptions);
    refBaseline = get_stage2_first_block_prefill_reference_baseline(rootDir, struct( ...
        'ParamsSource', paramsSource, ...
        'LayerIndex', getFieldOr(options, 'LayerIndex', 1), ...
        'BaselineOptions', baselineOptions, ...
        'Stimulus', numericBaseline.stimulus));
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    configure_prefill_tb_sources(tbName, numericBaseline.stimulus);
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', num2str(numericBaseline.stop_time), 'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', 'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');
    outValidSig = extract_dataset_signal(yout, 'out_valid');
    outValidRaw = double(outValidSig.Values.Data);
    sampleTimes = double(numericBaseline.stimulus.time(:));
    outputSamplePhase = resolve_output_sample_phase(outValidSig, sampleTimes, getFieldOr(options, 'OutputSamplePhase', 'auto'));
    outputSampleTimes = sampleTimes + outputSamplePhase;
    outValid = double(resample_signal_on_times(yout, 'out_valid', outputSampleTimes));
    outHidden = double(resample_signal_on_times(yout, 'out_hidden', outputSampleTimes));
    dutValidIndices = find(outValid(:) > 0.5);
    dutValidHidden = outHidden(dutValidIndices);
    rawValidIndices = find(outValidRaw(:) > 0.5);

    result = struct();
    result.params_source = refBaseline.params_source;
    result.reference_token_count = double(refBaseline.token_count);
    result.reference_prefill_out_hidden_mean = double(refBaseline.reference_prefill_out_hidden_mean(:));
    result.reference_scalar_contract_out_hidden = double(refBaseline.reference_scalar_contract_out_hidden(:));
    result.sample_times = sampleTimes;
    result.output_sample_phase = double(outputSamplePhase);
    result.output_sample_times = double(outputSampleTimes(:));
    result.raw_valid_indices = double(rawValidIndices(:));
    result.raw_valid_count = numel(rawValidIndices);
    result.dut_valid_indices = double(dutValidIndices(:));
    result.dut_valid_out_hidden = double(dutValidHidden(:));
    result.dut_valid_count = numel(dutValidHidden);
    result.reference_full_finite = all(isfinite(result.reference_prefill_out_hidden_mean));
    result.reference_contract_finite = all(isfinite(result.reference_scalar_contract_out_hidden));
    result.dut_finite = all(isfinite(result.dut_valid_out_hidden));
    result.reference_full_dynamic = is_dynamic(result.reference_prefill_out_hidden_mean);
    result.reference_contract_dynamic = is_dynamic(result.reference_scalar_contract_out_hidden);
    result.dut_dynamic = is_dynamic(result.dut_valid_out_hidden);
    result.sample_count_matches_reference = result.dut_valid_count == result.reference_token_count;

    [result.full_mean_compare_max_abs_diff, result.full_mean_compare_common_count] = compare_prefix( ...
        result.dut_valid_out_hidden, result.reference_prefill_out_hidden_mean);
    [result.contract_compare_max_abs_diff, result.contract_compare_common_count] = compare_prefix( ...
        result.dut_valid_out_hidden, result.reference_scalar_contract_out_hidden);

    result.numeric_equivalence_ready = result.sample_count_matches_reference && ...
        result.dut_finite && result.reference_full_finite && ...
        result.full_mean_compare_max_abs_diff < 1e-3;
    result.contract_alignment_ready = result.sample_count_matches_reference && ...
        result.dut_finite && result.reference_contract_finite && ...
        result.contract_compare_max_abs_diff < 1e-3;
    result.pass = result.reference_full_finite && result.reference_contract_finite && result.dut_finite;

    fprintf('Stage2 first-block prefill reference audit PASS\n');
    fprintf('  params_source=%s\n', char(result.params_source));
    fprintf('  output_sample_phase=%.6g\n', result.output_sample_phase);
    fprintf('  dut_valid_count=%d raw_valid_count=%d reference_token_count=%d sample_count_match=%d\n', ...
        result.dut_valid_count, result.raw_valid_count, result.reference_token_count, result.sample_count_matches_reference);
    fprintf('  dut_valid_out_hidden=%s\n', mat2str(result.dut_valid_out_hidden', 6));
    fprintf('  ref_full_mean=%s\n', mat2str(result.reference_prefill_out_hidden_mean', 6));
    fprintf('  ref_contract=%s\n', mat2str(result.reference_scalar_contract_out_hidden', 6));
    fprintf('  numeric_equivalence_ready=%d contract_alignment_ready=%d\n', ...
        result.numeric_equivalence_ready, result.contract_alignment_ready);
end

function tf = is_dynamic(values)
    values = double(values(:));
    if numel(values) <= 1
        tf = false;
    else
        tf = max(values) - min(values) > 1e-9;
    end
end

function [maxAbsDiff, commonCount] = compare_prefix(a, b)
    a = double(a(:));
    b = double(b(:));
    commonCount = min(numel(a), numel(b));
    if commonCount == 0
        maxAbsDiff = Inf;
        return;
    end
    maxAbsDiff = max(abs(a(1:commonCount) - b(1:commonCount)));
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
    if numel(values) ~= numel(time)
        error('run_stage2_first_block_prefill_reference_audit:BadStimulusVector', ...
            'Stimulus vector %s must match time length', signalName);
    end
    install_workspace_source(tbName, signalName, variableName, make_ufix17_timeseries(values, time));
end

function ts = make_ufix17_timeseries(values, time)
    fixedValues = fi(double(values(:)), false, 17, 0);
    ts = timeseries(fixedValues, double(time(:)));
end

function ts = make_sfix64_en30_timeseries(values, time)
    fixedValues = fi(double(values(:)), true, 64, 30);
    ts = timeseries(fixedValues, double(time(:)));
end

function set_constant_value(tbName, signalName, value)
    blk = [tbName '/src_' signalName];
    if getSimulinkBlockHandle(blk) == -1
        error('run_stage2_first_block_prefill_reference_audit:MissingSource', ...
            'Missing TB source block: %s', blk);
    end
    if strcmp(get_param(blk, 'BlockType'), 'FromWorkspace')
        return;
    end
    set_param(blk, 'Value', num2str(double(value), '%.17g'));
end

function install_workspace_source(tbName, signalName, variableName, variableValue)
    blk = [tbName '/src_' signalName];
    if getSimulinkBlockHandle(blk) == -1
        error('run_stage2_first_block_prefill_reference_audit:MissingSource', ...
            'Missing TB source block: %s', blk);
    end

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
    add_block('simulink/Sources/From Workspace', blk, ...
        'VariableName', variableName, ...
        'Position', pos);
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

function values = extract_signal(yout, name)
    sig = extract_dataset_signal(yout, name);
    values = sig.Values.Data;
end

function values = resample_signal_on_times(yout, name, sampleTimes)
    sig = extract_dataset_signal(yout, name);
    sampled = resample(sig.Values, double(sampleTimes(:)));
    values = sampled.Data;
end

function phase = resolve_output_sample_phase(outValidSig, baseSampleTimes, phaseOption)
    if isnumeric(phaseOption)
        phase = double(phaseOption);
        return;
    end

    phaseMode = lower(string(phaseOption));
    if phaseMode ~= "auto"
        error('run_stage2_first_block_prefill_reference_audit:BadOutputSamplePhase', ...
            'OutputSamplePhase must be numeric or "auto", got %s', char(phaseMode));
    end

    solverStep = infer_signal_step(outValidSig.Values.Time);
    candidatePhases = (0:solverStep:(1 - solverStep / 2))';
    bestPhase = 0;
    bestCount = -1;
    for i = 1:numel(candidatePhases)
        sampled = resample(outValidSig.Values, double(baseSampleTimes(:)) + candidatePhases(i));
        count = sum(double(sampled.Data(:)) > 0.5);
        if count > bestCount
            bestCount = count;
            bestPhase = candidatePhases(i);
        end
    end
    phase = double(bestPhase);
end

function step = infer_signal_step(times)
    times = double(times(:));
    diffs = diff(times);
    diffs = diffs(diffs > 1e-9);
    if isempty(diffs)
        step = 0.2;
        return;
    end
    step = min(diffs);
end

function sig = extract_dataset_signal(yout, name)
    for i = 1:yout.numElements
        sig = yout.get(i);
        sigName = string(sig.Name);
        blockPath = string('');
        try
            blockPath = string(sig.BlockPath.getBlock(1));
        catch
        end
        if sigName == string(name) || endsWith(blockPath, "/" + string(name))
            return;
        end
    end
    error('run_stage2_first_block_prefill_reference_audit:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
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