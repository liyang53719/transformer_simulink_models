function result = run_stage2_ffn_pipeline_smoke(rootDir, options)
%RUN_STAGE2_FFN_PIPELINE_SMOKE Validate staged gate/up and down-proj flow inside ffn_swiglu_u.
%   This smoke reuses the wrapper TB, logs FFN internal signals, and checks
%   that gate/up response validity forms a paired stage before the down
%   projection consumes the SwiGLU result.

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
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct('BuildModel', buildModel, 'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    signalSpecs = {
        struct('block', 'gateup_pair_valid', 'port', 1, 'name', 'ffn_gateup_pair_valid'), ...
        struct('block', 'gateup_pair_valid_z', 'port', 1, 'name', 'ffn_gateup_pair_valid_z'), ...
        struct('block', 'swiglu_valid_z', 'port', 1, 'name', 'ffn_swiglu_valid_z'), ...
        struct('block', 'down_valid_z', 'port', 1, 'name', 'ffn_down_valid_z'), ...
        struct('block', 'gate_norm_gate', 'port', 1, 'name', 'ffn_gate_norm_gate'), ...
        struct('block', 'swiglu_stage_gate', 'port', 1, 'name', 'ffn_swiglu_stage_gate'), ...
        struct('block', 'down_stage_gate', 'port', 1, 'name', 'ffn_down_stage_gate')};

    enable_ffn_signal_logging(mdlName, signalSpecs);

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');

    pairValid = extract_logged_signal(logsout, 'ffn_gateup_pair_valid');
    pairValidZ = extract_logged_signal(logsout, 'ffn_gateup_pair_valid_z');
    swigluValidZ = extract_logged_signal(logsout, 'ffn_swiglu_valid_z');
    downValidZ = extract_logged_signal(logsout, 'ffn_down_valid_z');
    result = struct();
    result.pair_valid_seen = any(pairValid > 0.5);
    result.pair_valid_delay_ok = has_single_cycle_delay(pairValid, pairValidZ);
    result.swiglu_valid_delay_ok = has_single_cycle_delay(pairValidZ, swigluValidZ);
    result.down_valid_delay_ok = has_single_cycle_delay(swigluValidZ, downValidZ);
    result.valid_chain_order_ok = respects_chain_order(pairValid, pairValidZ, swigluValidZ, downValidZ);
    result.pass = result.pair_valid_seen && result.pair_valid_delay_ok && ...
        result.swiglu_valid_delay_ok && result.down_valid_delay_ok && ...
        result.valid_chain_order_ok;

    if result.pass
        fprintf('Stage2 FFN pipeline smoke PASS\n');
    else
        fprintf('Stage2 FFN pipeline smoke FAIL\n');
        fprintf(['  pair_seen=%d pair_d1=%d swiglu_d1=%d down_d1=%d ' ...
            'chain_order=%d\n'], ...
            result.pair_valid_seen, result.pair_valid_delay_ok, ...
            result.swiglu_valid_delay_ok, result.down_valid_delay_ok, ...
            result.valid_chain_order_ok);
        error('run_stage2_ffn_pipeline_smoke:Failed', ...
            'FFN pipeline stage checks failed');
    end
end

function enable_ffn_signal_logging(mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockPath = [mdlName '/ffn_swiglu_u/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_ffn_pipeline_smoke:MissingLine', ...
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
    error('run_stage2_ffn_pipeline_smoke:MissingLoggedSignal', ...
        'Logged signal not found: %s', name);
end

function yes = has_single_cycle_delay(src, delayed)
    src = double(src(:));
    delayed = double(delayed(:));
    if numel(src) < 2 || numel(delayed) < 2
        yes = false;
        return;
    end
    yes = all(abs(delayed(2:end) - src(1:end-1)) < 1e-9);
end

function yes = respects_chain_order(pairValid, pairValidZ, swigluValidZ, downValidZ)
    idx0 = first_active_index(pairValid);
    idx1 = first_active_index(pairValidZ);
    idx2 = first_active_index(swigluValidZ);
    idx3 = first_active_index(downValidZ);
    yes = ~isnan(idx0) && ~isnan(idx1) && ~isnan(idx2) && ~isnan(idx3) && ...
        idx0 <= idx1 && idx1 <= idx2 && idx2 <= idx3;
end

function idx = first_active_index(values)
    hits = find(double(values(:)) > 0.5, 1, 'first');
    if isempty(hits)
        idx = NaN;
    else
        idx = hits;
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