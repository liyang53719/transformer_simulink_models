function result = run_stage2_attention_pipeline_smoke(rootDir, options)
%RUN_STAGE2_ATTENTION_PIPELINE_SMOKE Validate staged valid propagation inside attention_u.
%   This smoke reuses the wrapper TB, enables signal logging on internal
%   attention pipeline lines, and checks that q/k response validity flows
%   through qk_pair_valid -> softmax_valid -> scorev_valid with the expected
%   one-cycle stage delays while the gated score and output stages remain
%   active.

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
        struct('block', 'qk_pair_valid', 'port', 1, 'name', 'attn_qk_pair_valid'), ...
        struct('block', 'qk_pair_valid_z', 'port', 1, 'name', 'attn_qk_pair_valid_z'), ...
        struct('block', 'softmax_valid', 'port', 1, 'name', 'attn_softmax_valid'), ...
        struct('block', 'softmax_valid_z', 'port', 1, 'name', 'attn_softmax_valid_z'), ...
        struct('block', 'scorev_input_valid', 'port', 1, 'name', 'attn_scorev_input_valid'), ...
        struct('block', 'scorev_valid_z', 'port', 1, 'name', 'attn_scorev_valid_z'), ...
        struct('block', 'row_sum_accum', 'port', 1, 'name', 'attn_row_sum_accum')};

    enable_attention_signal_logging(mdlName, signalSpecs);

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on', ...
        'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    logsout = simOut.get('logsout');

    qkPair = extract_logged_signal(logsout, 'attn_qk_pair_valid');
    qkPairZ = extract_logged_signal(logsout, 'attn_qk_pair_valid_z');
    softmaxValid = extract_logged_signal(logsout, 'attn_softmax_valid');
    softmaxValidZ = extract_logged_signal(logsout, 'attn_softmax_valid_z');
    scorevInputValid = extract_logged_signal(logsout, 'attn_scorev_input_valid');
    scorevValidZ = extract_logged_signal(logsout, 'attn_scorev_valid_z');
    rowSumAccum = extract_logged_signal(logsout, 'attn_row_sum_accum');

    result = struct();
    result.qk_pair_seen = any(qkPair > 0.5);
    result.qk_pair_delay_ok = has_single_cycle_delay(qkPair, qkPairZ);
    result.softmax_valid_seen = any(softmaxValid > 0.5);
    result.softmax_valid_delay_ok = has_single_cycle_delay(softmaxValid, softmaxValidZ);
    result.scorev_input_seen = any(scorevInputValid > 0.5);
    result.scorev_valid_delay_ok = has_single_cycle_delay(scorevInputValid, scorevValidZ);
    result.row_sum_active = any(abs(rowSumAccum) > 0);
    result.valid_chain_order_ok = respects_chain_order(qkPair, softmaxValid, scorevInputValid);
    result.pass = result.qk_pair_seen && result.qk_pair_delay_ok && ...
        result.softmax_valid_seen && result.softmax_valid_delay_ok && ...
        result.scorev_input_seen && result.scorev_valid_delay_ok && ...
        result.row_sum_active && result.valid_chain_order_ok;

    if result.pass
        fprintf('Stage2 attention pipeline smoke PASS\n');
    else
        fprintf('Stage2 attention pipeline smoke FAIL\n');
        fprintf(['  qk_seen=%d qk_d1=%d softmax_seen=%d softmax_d1=%d ' ...
            'scorev_seen=%d scorev_d1=%d row_sum_active=%d chain_order=%d\n'], ...
            result.qk_pair_seen, result.qk_pair_delay_ok, ...
            result.softmax_valid_seen, result.softmax_valid_delay_ok, ...
            result.scorev_input_seen, result.scorev_valid_delay_ok, ...
            result.row_sum_active, result.valid_chain_order_ok);
        error('run_stage2_attention_pipeline_smoke:Failed', ...
            'attention pipeline stage checks failed');
    end
end

function enable_attention_signal_logging(mdlName, signalSpecs)
    for i = 1:numel(signalSpecs)
        spec = signalSpecs{i};
        blockPath = [mdlName '/attention_u/' spec.block];
        blockHandle = getSimulinkBlockHandle(blockPath);
        lineHandle = get_src_line_handle(blockPath, spec.port);
        if blockHandle == -1 || lineHandle == -1
            error('run_stage2_attention_pipeline_smoke:MissingLine', ...
                'Cannot find signal line for %s', spec.block);
        end
        try
            Simulink.sdi.markSignalForStreaming(blockHandle, spec.port, 'on');
            set_param(lineHandle, 'Name', spec.name);
        catch ME
            error('run_stage2_attention_pipeline_smoke:LoggingSetupFailed', ...
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
    error('run_stage2_attention_pipeline_smoke:MissingLoggedSignal', ...
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

function yes = respects_chain_order(qkPair, softmaxValid, scorevInputValid)
    qkIdx = first_active_index(qkPair);
    softmaxIdx = first_active_index(softmaxValid);
    scorevIdx = first_active_index(scorevInputValid);
    yes = ~isnan(qkIdx) && ~isnan(softmaxIdx) && ~isnan(scorevIdx) && ...
        qkIdx <= softmaxIdx && softmaxIdx <= scorevIdx;
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