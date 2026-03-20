function result = run_stage2_first_block_prefill_numeric_regression(rootDir, options)
%RUN_STAGE2_FIRST_BLOCK_PREFILL_NUMERIC_REGRESSION First-block prefill numeric regression driven by Simulink.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    dumpOnly = getFieldOr(options, 'DumpOnly', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));

    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    baseline = get_stage2_first_block_prefill_numeric_baseline();

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    configure_prefill_tb_sources(tbName, baseline.stimulus);
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', num2str(baseline.stop_time), 'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', 'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    outValid = double(extract_signal(yout, 'out_valid'));
    outHidden = double(extract_signal(yout, 'out_hidden'));

    result = struct();
    result.out_valid = outValid(:);
    result.out_hidden = outHidden(:);
    result.sample_count = numel(result.out_hidden);

    if dumpOnly || isempty(baseline.expected_out_valid) || isempty(baseline.expected_out_hidden)
        fprintf('Stage2 first-block prefill numeric regression DUMP\n');
        fprintf('  out_valid = %s\n', mat2str(result.out_valid', 17));
        fprintf('  out_hidden = %s\n', mat2str(result.out_hidden', 17));
        result.pass = false;
        return;
    end

    validCmp = compare_trace(result.out_valid, baseline.expected_out_valid(:));
    hiddenCmp = compare_trace(result.out_hidden, baseline.expected_out_hidden(:));
    result.valid_trace = validCmp;
    result.hidden_trace = hiddenCmp;
    result.pass = validCmp.match && hiddenCmp.match;

    if result.pass
        fprintf('Stage2 first-block prefill numeric regression PASS\n');
        fprintf('  sample_count=%d hidden_max_abs_diff=%g hidden_mean_abs_diff=%g\n', ...
            hiddenCmp.sample_count, hiddenCmp.max_abs_diff, hiddenCmp.mean_abs_diff);
    else
        fprintf('Stage2 first-block prefill numeric regression FAIL\n');
        fprintf('  valid_match=%d hidden_match=%d sample_count=%d hidden_max_abs_diff=%g hidden_mean_abs_diff=%g\n', ...
            validCmp.match, hiddenCmp.match, hiddenCmp.sample_count, hiddenCmp.max_abs_diff, hiddenCmp.mean_abs_diff);
        error('run_stage2_first_block_prefill_numeric_regression:Failed', ...
            'First-block prefill numeric regression deviated from stored Simulink baseline');
    end
end

function configure_prefill_tb_sources(tbName, stimulus)
    set_constant_value(tbName, 'mode_decode', stimulus.mode_decode(1));
    set_constant_value(tbName, 'cfg_seq_len', stimulus.cfg_seq_len);
    set_constant_value(tbName, 'cfg_token_pos', stimulus.cfg_token_pos);
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
    install_workspace_source(tbName, 'in_hidden', 'tb_prefill_in_hidden_seq', make_sfix64_en30_timeseries(stimulus.in_hidden(:), stimulus.time(:)));
    install_workspace_source(tbName, 'in_residual', 'tb_prefill_in_residual_seq', make_sfix64_en30_timeseries(stimulus.in_residual(:), stimulus.time(:)));
end

function ts = make_sfix64_en30_timeseries(values, time)
    fixedValues = fi(double(values(:)), true, 64, 30);
    ts = timeseries(fixedValues, double(time(:)));
end

function set_constant_value(tbName, signalName, value)
    blk = [tbName '/src_' signalName];
    if getSimulinkBlockHandle(blk) == -1
        error('run_stage2_first_block_prefill_numeric_regression:MissingSource', ...
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
        error('run_stage2_first_block_prefill_numeric_regression:MissingSource', ...
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

function cmp = compare_trace(actual, expected)
    actual = double(actual(:));
    expected = double(expected(:));
    commonLen = min(numel(actual), numel(expected));
    actual = actual(1:commonLen);
    expected = expected(1:commonLen);
    delta = actual - expected;

    cmp = struct();
    cmp.sample_count = commonLen;
    cmp.max_abs_diff = max_or_zero(abs(delta));
    cmp.mean_abs_diff = mean_or_zero(abs(delta));
    cmp.match = numel(actual) == numel(expected) && all(abs(delta) < 1e-9);
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
    error('run_stage2_first_block_prefill_numeric_regression:MissingSignal', ...
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