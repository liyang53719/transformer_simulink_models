function result = run_stage2_kv_banking_pipeline_smoke(rootDir, options)
%RUN_STAGE2_KV_BANKING_PIPELINE_SMOKE Validate internal KV banking formulas.
%   This smoke logs scheduler outputs and kv_cache_if internal banking lines
%   to ensure the stage2 prefill/decode path preserves the expected banked
%   address, selector, and gated write-enable relationships.

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
        struct('path', [mdlName '/kv_cache_if_u/tile_seq'], 'port', 1, 'name', 'sched_tile_seq'), ...
        struct('path', [mdlName '/kv_cache_if_u/active_seq_len'], 'port', 1, 'name', 'sched_active_seq_len'), ...
        struct('path', [mdlName '/kv_cache_if_u/tile_k'], 'port', 1, 'name', 'sched_tile_k'), ...
        struct('path', [mdlName '/kv_cache_if_u/tile_out'], 'port', 1, 'name', 'sched_tile_out'), ...
        struct('path', [mdlName '/kv_cache_if_u/x_bank_count'], 'port', 1, 'name', 'sched_x_bank_count'), ...
        struct('path', [mdlName '/kv_cache_if_u/kv_bank_count'], 'port', 1, 'name', 'sched_kv_bank_count'), ...
        struct('path', [mdlName '/kv_cache_if_u/kv_phase_first'], 'port', 1, 'name', 'sched_kv_phase_first'), ...
        struct('path', [mdlName '/kv_cache_if_u/seq_window_sum'], 'port', 1, 'name', 'kv_seq_window_sum'), ...
        struct('path', [mdlName '/kv_cache_if_u/bank_sum'], 'port', 1, 'name', 'kv_bank_sum'), ...
        struct('path', [mdlName '/kv_cache_if_u/bank_addr'], 'port', 1, 'name', 'kv_bank_addr'), ...
        struct('path', [mdlName '/kv_cache_if_u/bank_sel'], 'port', 1, 'name', 'kv_bank_sel'), ...
        struct('path', [mdlName '/kv_cache_if_u/kv_seq_gate'], 'port', 1, 'name', 'kv_bank_wr_en')};

    enable_signal_logging(signalSpecs);

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');

    tileSeq = extract_logged_signal(logsout, 'sched_tile_seq');
    activeSeqLen = extract_logged_signal(logsout, 'sched_active_seq_len');
    tileK = extract_logged_signal(logsout, 'sched_tile_k');
    tileOut = extract_logged_signal(logsout, 'sched_tile_out');
    xBanks = extract_logged_signal(logsout, 'sched_x_bank_count');
    kvBanks = extract_logged_signal(logsout, 'sched_kv_bank_count');
    kvPhaseFirst = extract_logged_signal(logsout, 'sched_kv_phase_first');
    seqWindowSum = extract_logged_signal(logsout, 'kv_seq_window_sum');
    bankSum = extract_logged_signal(logsout, 'kv_bank_sum');
    bankAddr = extract_logged_signal(logsout, 'kv_bank_addr');
    bankSel = extract_logged_signal(logsout, 'kv_bank_sel');
    bankWrEn = extract_logged_signal(logsout, 'kv_bank_wr_en');

    expectedSeqWindow = double(tileSeq(:)) + double(activeSeqLen(:));
    expectedBankSum = double(xBanks(:)) + double(kvBanks(:));
    expectedBankAddr = expectedSeqWindow .* double(tileK(:)) + double(tileOut(:));
    expectedBankSel = expectedBankSum .* double(tileOut(:));
    expectedWrEn = expectedSeqWindow .* double(kvPhaseFirst(:));

    result = struct();
    result.seq_window_matches = signals_equal(seqWindowSum, expectedSeqWindow);
    result.bank_sum_matches = signals_equal(bankSum, expectedBankSum);
    result.bank_addr_matches = signals_equal(bankAddr, expectedBankAddr);
    result.bank_sel_matches = signals_equal(bankSel, expectedBankSel);
    result.bank_wr_en_matches = signals_equal(bankWrEn, expectedWrEn);
    result.bank_addr_active = any(double(bankAddr(:)) > 0);
    result.bank_sel_active = any(double(bankSel(:)) > 0);
    result.pass = result.seq_window_matches && result.bank_sum_matches && ...
        result.bank_addr_matches && result.bank_sel_matches && ...
        result.bank_wr_en_matches && ...
        result.bank_addr_active && result.bank_sel_active;

    if result.pass
        fprintf('Stage2 KV banking pipeline smoke PASS\n');
    else
        fprintf('Stage2 KV banking pipeline smoke FAIL\n');
        fprintf(['  seq_window=%d bank_sum=%d bank_addr=%d bank_sel=%d ' ...
            'bank_wr_en=%d addr_active=%d sel_active=%d\n'], ...
            result.seq_window_matches, result.bank_sum_matches, ...
            result.bank_addr_matches, result.bank_sel_matches, ...
            result.bank_wr_en_matches, result.bank_addr_active, ...
            result.bank_sel_active);
        error('run_stage2_kv_banking_pipeline_smoke:Failed', ...
            'KV banking pipeline checks failed');
    end
end

function enable_signal_logging(signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockHandle = getSimulinkBlockHandle(spec.path);
        lineHandle = get_src_line_handle(spec.path, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_kv_banking_pipeline_smoke:MissingLine', ...
                'Cannot find signal line for %s', spec.path);
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
    error('run_stage2_kv_banking_pipeline_smoke:MissingLoggedSignal', ...
        'Logged signal not found: %s', name);
end

function yes = signals_equal(actual, expected)
    actual = double(actual(:));
    expected = double(expected(:));
    yes = numel(actual) == numel(expected) && all(abs(actual - expected) < 1e-9);
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