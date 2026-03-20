function result = run_stage2_kv_banking_pipeline_smoke(rootDir, options)
%RUN_STAGE2_KV_BANKING_PIPELINE_SMOKE Validate internal KV banking formulas.
%   This smoke captures kv_cache_if internal banking lines one at a time and
%   checks that the wrapper-visible default schedule produces the expected
%   banked address, selector, and gated write-enable relationships.

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

    schedCfg = prefill_attention_schedule_32x32();
    cfgSeqLen = str2double(get_param([tbName '/src_cfg_seq_len'], 'Value'));
    cfgTokenPos = str2double(get_param([tbName '/src_cfg_token_pos'], 'Value'));
    modeDecode = str2double(get_param([tbName '/src_mode_decode'], 'Value'));
    activeSeqLenScalar = cfgSeqLen;
    if activeSeqLenScalar > schedCfg.tile_seq
        activeSeqLenScalar = schedCfg.tile_seq;
    end
    if cfgTokenPos < 0 || modeDecode < 0
        activeSeqLenScalar = 0;
    end

    seqWindowSum = capture_internal_signal(tbName, [mdlName '/kv_cache_if_u/seq_window_sum'], 1);
    bankSum = capture_internal_signal(tbName, [mdlName '/kv_cache_if_u/bank_sum'], 1);
    bankAddr = capture_internal_signal(tbName, [mdlName '/kv_cache_if_u/bank_addr'], 1);
    bankSel = capture_internal_signal(tbName, [mdlName '/kv_cache_if_u/bank_sel'], 1);
    bankWrEn = capture_internal_signal(tbName, [mdlName '/kv_cache_if_u/kv_seq_gate'], 1);

    expectedSeqWindow = constant_trace(schedCfg.tile_seq, seqWindowSum);
    expectedBankSum = constant_trace(schedCfg.x_bank_count + schedCfg.kv_bank_count, bankSum);
    expectedBankAddr = constant_trace(schedCfg.tile_seq * schedCfg.tile_k + schedCfg.tile_out, bankAddr);
    expectedBankSel = constant_trace((schedCfg.x_bank_count + schedCfg.kv_bank_count) * schedCfg.tile_out, bankSel);
    expectedWrEn = constant_trace(schedCfg.tile_seq * schedCfg.kv_phase_first, bankWrEn);

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

function values = capture_internal_signal(tbName, signalPath, portIndex)
    blockHandle = getSimulinkBlockHandle(signalPath);
    lineHandle = get_src_line_handle(signalPath, portIndex);
    if blockHandle == -1 || lineHandle == -1
        error('run_stage2_kv_banking_pipeline_smoke:MissingLine', ...
            'Cannot find signal line for %s', signalPath);
    end

    Simulink.sdi.markSignalForStreaming(blockHandle, portIndex, 'on');
    cleanup = onCleanup(@()Simulink.sdi.markSignalForStreaming(blockHandle, portIndex, 'off')); %#ok<NASGU>
    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');
    if logsout.numElements < 1
        error('run_stage2_kv_banking_pipeline_smoke:MissingLoggedSignal', ...
            'No streamed signal captured for %s', signalPath);
    end
    values = double(logsout.get(logsout.numElements).Values.Data(:));
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

function trace = constant_trace(value, reference)
    trace = repmat(double(value), numel(reference), 1);
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