function result = run_stage2_qkv_pipeline_smoke(rootDir, options)
%RUN_STAGE2_QKV_PIPELINE_SMOKE Validate fused QKV stage propagation inside qkv_proj_u.
%   This smoke reuses the wrapper TB, logs internal QKV stage signals, and
%   checks that K/V requests form a shared pair stage before Q joins the
%   fused QKV issue stage. It also checks that the emitted group index
%   reflects the fused Q+KV pool semantics instead of a fixed constant.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct('BuildModel', buildModel, 'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    signalSpecs = {
        struct('block', 'kv_pair_valid', 'port', 1, 'name', 'qkv_kv_pair_valid'), ...
        struct('block', 'kv_pair_valid_z', 'port', 1, 'name', 'qkv_kv_pair_valid_z'), ...
        struct('block', 'fused_qkv_valid', 'port', 1, 'name', 'qkv_fused_valid'), ...
        struct('block', 'fused_qkv_valid_z', 'port', 1, 'name', 'qkv_fused_valid_z'), ...
        struct('block', 'q_valid_alias', 'port', 1, 'name', 'q_valid'), ...
        struct('block', 'kv_valid_alias', 'port', 1, 'name', 'kv_valid'), ...
        struct('block', 'group_idx_sum', 'port', 1, 'name', 'group_idx')};

    enable_qkv_signal_logging(mdlName, signalSpecs);

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');

    kvPair = extract_logged_signal(logsout, 'qkv_kv_pair_valid');
    kvPairZ = extract_logged_signal(logsout, 'qkv_kv_pair_valid_z');
    fusedValid = extract_logged_signal(logsout, 'qkv_fused_valid');
    fusedValidZ = extract_logged_signal(logsout, 'qkv_fused_valid_z');
    qValid = extract_logged_signal(logsout, 'q_valid');
    kvValid = extract_logged_signal(logsout, 'kv_valid');
    groupIdx = extract_logged_signal(logsout, 'group_idx');

    result = struct();
    result.kv_pair_seen = any(kvPair > 0.5);
    result.kv_pair_delay_ok = has_single_cycle_delay(kvPair, kvPairZ);
    result.fused_valid_seen = any(fusedValid > 0.5);
    result.fused_valid_delay_ok = has_single_cycle_delay(fusedValid, fusedValidZ);
    result.q_valid_matches = signals_match(fusedValidZ, qValid);
    result.kv_valid_matches = signals_match(kvPairZ, kvValid);
    result.valid_chain_order_ok = respects_chain_order(kvPair, kvPairZ, fusedValid, fusedValidZ);
    result.group_idx_fused_ok = has_expected_group_idx(groupIdx, qValid, kvValid);
    result.pass = result.kv_pair_seen && result.kv_pair_delay_ok && ...
        result.fused_valid_seen && result.fused_valid_delay_ok && ...
        result.q_valid_matches && result.kv_valid_matches && ...
        result.valid_chain_order_ok && result.group_idx_fused_ok;

    if result.pass
        fprintf('Stage2 QKV pipeline smoke PASS\n');
    else
        fprintf('Stage2 QKV pipeline smoke FAIL\n');
        fprintf(['  kv_seen=%d kv_d1=%d fused_seen=%d fused_d1=%d ' ...
            'q_match=%d kv_match=%d chain_order=%d group_idx=%d\n'], ...
            result.kv_pair_seen, result.kv_pair_delay_ok, ...
            result.fused_valid_seen, result.fused_valid_delay_ok, ...
            result.q_valid_matches, result.kv_valid_matches, ...
            result.valid_chain_order_ok, result.group_idx_fused_ok);
        error('run_stage2_qkv_pipeline_smoke:Failed', ...
            'QKV pipeline stage checks failed');
    end
end

function enable_qkv_signal_logging(mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockPath = [mdlName '/qkv_proj_u/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_qkv_pipeline_smoke:MissingLine', ...
                'Cannot find signal line for %s', spec.block);
        end
        try
            Simulink.sdi.markSignalForStreaming(blockHandle, spec.port, 'on');
            if isfield(spec, 'name') && strlength(string(spec.name)) > 0
                set_param(lineHandle, 'Name', spec.name);
            end
        catch ME
            error('run_stage2_qkv_pipeline_smoke:LoggingSetupFailed', ...
                'Failed to enable logging for %s: %s', spec.block, ME.message);
        end
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
    error('run_stage2_qkv_pipeline_smoke:MissingLoggedSignal', ...
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

function yes = signals_match(a, b)
    a = double(a(:));
    b = double(b(:));
    yes = numel(a) == numel(b) && all(abs(a - b) < 1e-9);
end

function yes = respects_chain_order(kvPair, kvPairZ, fusedValid, fusedValidZ)
    idx0 = first_active_index(kvPair);
    idx1 = first_active_index(kvPairZ);
    idx2 = first_active_index(fusedValid);
    idx3 = first_active_index(fusedValidZ);
    yes = ~isnan(idx0) && ~isnan(idx1) && ~isnan(idx2) && ~isnan(idx3) && ...
        idx0 <= idx1 && idx1 <= idx2 && idx2 <= idx3;
end

function yes = has_expected_group_idx(groupIdx, qValid, kvValid)
    groupIdx = double(groupIdx(:));
    qValid = double(qValid(:));
    kvValid = double(kvValid(:));
    expected = qValid + 2 .* kvValid;
    yes = numel(groupIdx) == numel(expected) && all(abs(groupIdx - expected) < 1e-9) && any(groupIdx >= 3);
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