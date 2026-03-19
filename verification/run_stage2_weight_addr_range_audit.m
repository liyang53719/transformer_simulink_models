function result = run_stage2_weight_addr_range_audit(rootDir, options)
%RUN_STAGE2_WEIGHT_ADDR_RANGE_AUDIT Audit stage2 weight-request address envelope and boundary behavior.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = logical(getFieldOr(options, 'BuildModel', false));
    assert_stage2_manual_model_policy(buildModel, mfilename);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));

    safeCase = default_contract();
    safeEnvelope = compute_stage2_weight_addr_range(safeCase);

    boundaryCase = safeCase;
    boundaryCase.cfg_token_pos = safeEnvelope.max_safe_token_pos;
    boundaryEnvelope = compute_stage2_weight_addr_range(boundaryCase);

    overflowCase = safeCase;
    overflowCase.cfg_token_pos = safeEnvelope.max_safe_token_pos + 1;
    overflowEnvelope = compute_stage2_weight_addr_range(overflowCase);

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    ensure_weight_observers(tbName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    observedSafe = simulate_weight_req_addresses(tbName, safeCase);
    observedBoundary = simulate_weight_req_addresses(tbName, boundaryCase);

    result = struct();
    result.pass = true;
    result.safe_envelope = safeEnvelope;
    result.boundary_envelope = boundaryEnvelope;
    result.overflow_envelope = overflowEnvelope;
    result.safe_matches_expected = isequal(observedSafe, safeEnvelope.lane_addrs);
    result.boundary_matches_expected = isequal(observedBoundary, boundaryEnvelope.lane_addrs);
    result.safe_in_range = safeEnvelope.in_uint8_range;
    result.boundary_in_range = boundaryEnvelope.in_uint8_range;
    result.overflow_detected = overflowEnvelope.will_wrap;
    result.pass = result.safe_matches_expected && result.boundary_matches_expected && ...
        result.safe_in_range && result.boundary_in_range && result.overflow_detected;

    if result.pass
        fprintf('Stage2 weight address range audit PASS\n');
        fprintf('  safe_addr_range=[%g,%g] boundary_addr_range=[%g,%g] overflow_token_pos=%g\n', ...
            safeEnvelope.min_addr, safeEnvelope.max_addr, ...
            boundaryEnvelope.min_addr, boundaryEnvelope.max_addr, ...
            overflowCase.cfg_token_pos);
    else
        fprintf('Stage2 weight address range audit FAIL\n');
        fprintf('  safe_matches_expected=%d boundary_matches_expected=%d\n', ...
            result.safe_matches_expected, result.boundary_matches_expected);
        fprintf('  safe_in_range=%d boundary_in_range=%d overflow_detected=%d\n', ...
            result.safe_in_range, result.boundary_in_range, result.overflow_detected);
        fprintf('  observed_safe=%s expected_safe=%s\n', ...
            stringify_numeric(observedSafe), stringify_numeric(safeEnvelope.lane_addrs));
        fprintf('  observed_boundary=%s expected_boundary=%s\n', ...
            stringify_numeric(observedBoundary), stringify_numeric(boundaryEnvelope.lane_addrs));
        error('run_stage2_weight_addr_range_audit:Failed', ...
            'Stage2 weight address range audit failed.');
    end
end

function observed = simulate_weight_req_addresses(tbName, contract)
    apply_contract_constants(tbName, contract);
    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    signalNames = {'tb_gamma_req_addr', 'tb_qkv_q_req_addr', 'tb_qkv_k_req_addr', 'tb_qkv_v_req_addr', ...
        'tb_attn_q_req_addr', 'tb_attn_k_req_addr', 'tb_attn_v_req_addr', 'tb_ffn_up_req_addr', 'tb_ffn_gate_req_addr', 'tb_ffn_down_req_addr'};
    validNames = {'tb_gamma_req_valid', 'tb_qkv_q_req_valid', 'tb_qkv_k_req_valid', 'tb_qkv_v_req_valid', ...
        'tb_attn_q_req_valid', 'tb_attn_k_req_valid', 'tb_attn_v_req_valid', 'tb_ffn_up_req_valid', 'tb_ffn_gate_req_valid', 'tb_ffn_down_req_valid'};

    observed = zeros(1, numel(signalNames));
    for i = 1:numel(signalNames)
        addr = double(extract_signal(yout, signalNames{i}));
        valid = double(extract_signal(yout, validNames{i}));
        addr = addr(:);
        valid = valid(:);
        commonLen = min(numel(addr), numel(valid));
        addr = addr(1:commonLen);
        valid = valid(1:commonLen);
        mask = valid > 0.5;
        if any(mask)
            observed(i) = max(addr(mask));
        else
            observed(i) = NaN;
        end
    end
end

function apply_contract_constants(tbName, contract)
    fieldNames = fieldnames(contract);
    for i = 1:numel(fieldNames)
        name = fieldNames{i};
        blk = [tbName '/src_' name];
        if getSimulinkBlockHandle(blk) == -1
            continue;
        end
        value = contract.(name);
        if islogical(value)
            token = ternary(value, '1', '0');
        else
            token = num2str(double(value), '%.17g');
        end
        set_param(blk, 'Value', token);
    end
end

function contract = default_contract()
    contract = struct();
    contract.mode_decode = true;
    contract.start = true;
    contract.eos_in = false;
    contract.in_valid = true;
    contract.out_ready = true;
    contract.in_hidden = single(2);
    contract.in_residual = single(1);
    contract.kv_cache_rd_data = single(0);
    contract.kv_cache_rd_valid = false;
    contract.cfg_seq_len = 1;
    contract.cfg_token_pos = 1;
    contract.cfg_eps = single(1e-5);
    contract.stop_req = false;
    contract.cfg_weight_num_heads = 12;
    contract.cfg_weight_page_base = 64;
    contract.cfg_weight_page_stride = 8;
    contract.cfg_rope_theta_scale = 1;
    contract.cfg_rope_sin_mix_scale = 1;
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
    error('run_stage2_weight_addr_range_audit:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end

function text = stringify_numeric(values)
    text = strtrim(sprintf('%g ', values));
end

function out = ternary(cond, whenTrue, whenFalse)
    if cond
        out = whenTrue;
    else
        out = whenFalse;
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

function ensure_weight_observers(tbName)
    if ~isempty(find_system(tbName, 'SearchDepth', 1, 'Name', 'tb_qkv_q_rsp_data'))
        if ~isempty(find_system(tbName, 'SearchDepth', 1, 'Name', 'tb_gamma_rsp_data'))
            return;
        end
    end

    reqSrc = get_existing_source_endpoint([tbName '/tb_w_req_sel']);

    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_gamma_sel'], ...
        'OutputSignals', 'gamma_valid,gamma_data', ...
        'Position', [1080, 595, 1145, 645]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_gamma_rsp_valid'], 'Position', [1240, 120, 1270, 134]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_gamma_rsp_data'], 'Position', [1240, 160, 1270, 174]);

    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_qkv_sel'], ...
        'OutputSignals', 'qkv_q_valid,qkv_k_valid,qkv_v_valid', ...
        'Position', [1080, 655, 1145, 735]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_qkv_data_sel'], ...
        'OutputSignals', 'qkv_q_data,qkv_k_data,qkv_v_data', ...
        'Position', [1080, 745, 1145, 825]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_ffn_sel'], ...
        'OutputSignals', 'ffn_up_valid,ffn_gate_valid,ffn_down_valid', ...
        'Position', [1080, 835, 1145, 895]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_ffn_data_sel'], ...
        'OutputSignals', 'ffn_up_data,ffn_gate_data,ffn_down_data', ...
        'Position', [1080, 905, 1145, 965]);

    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_q_rsp_valid'], 'Position', [1240, 320, 1270, 334]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_k_rsp_valid'], 'Position', [1240, 360, 1270, 374]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_v_rsp_valid'], 'Position', [1240, 400, 1270, 414]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_q_rsp_data'], 'Position', [1240, 440, 1270, 454]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_k_rsp_data'], 'Position', [1240, 480, 1270, 494]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_v_rsp_data'], 'Position', [1240, 520, 1270, 534]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_up_rsp_valid'], 'Position', [1240, 560, 1270, 574]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_rsp_valid'], 'Position', [1240, 600, 1270, 614]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_up_rsp_data'], 'Position', [1240, 640, 1270, 654]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_rsp_data'], 'Position', [1240, 680, 1270, 694]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_down_rsp_valid'], 'Position', [1240, 720, 1270, 734]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_down_rsp_data'], 'Position', [1240, 760, 1270, 774]);

    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_gamma_sel/1', 'autorouting', 'on');
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_qkv_sel/1', 'autorouting', 'on');
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_qkv_data_sel/1', 'autorouting', 'on');
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_ffn_sel/1', 'autorouting', 'on');
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_ffn_data_sel/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_gamma_sel/1', 'tb_gamma_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_gamma_sel/2', 'tb_gamma_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_sel/1', 'tb_qkv_q_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_sel/2', 'tb_qkv_k_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_sel/3', 'tb_qkv_v_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_data_sel/1', 'tb_qkv_q_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_data_sel/2', 'tb_qkv_k_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_qkv_data_sel/3', 'tb_qkv_v_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_sel/1', 'tb_ffn_up_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_sel/2', 'tb_ffn_gate_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_sel/3', 'tb_ffn_down_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_data_sel/1', 'tb_ffn_up_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_data_sel/2', 'tb_ffn_gate_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_data_sel/3', 'tb_ffn_down_rsp_data/1', 'autorouting', 'on');

    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_gamma_sel'], ...
        'OutputSignals', 'gamma_addr,gamma_valid', ...
        'Position', [1080, 505, 1145, 555]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_qkv_sel'], ...
        'OutputSignals', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid', ...
        'Position', [1080, 560, 1145, 700]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_ffn_sel'], ...
        'OutputSignals', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid,ffn_down_addr,ffn_down_valid', ...
        'Position', [1080, 710, 1145, 810]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_gamma_req_addr'], 'Position', [1240, 40, 1270, 54]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_gamma_req_valid'], 'Position', [1240, 80, 1270, 94]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_q_req_addr'], 'Position', [1240, 560, 1270, 574]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_q_req_valid'], 'Position', [1240, 200, 1270, 214]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_k_req_addr'], 'Position', [1240, 600, 1270, 614]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_k_req_valid'], 'Position', [1240, 240, 1270, 254]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_v_req_addr'], 'Position', [1240, 640, 1270, 654]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_qkv_v_req_valid'], 'Position', [1240, 280, 1270, 294]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_up_req_addr'], 'Position', [1240, 720, 1270, 734]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_up_req_valid'], 'Position', [1240, 760, 1270, 774]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_req_addr'], 'Position', [1240, 800, 1270, 814]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_req_valid'], 'Position', [1240, 840, 1270, 854]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_down_req_addr'], 'Position', [1240, 880, 1270, 894]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_ffn_down_req_valid'], 'Position', [1240, 920, 1270, 934]);

    add_line(tbName, reqSrc, 'tb_w_req_gamma_sel/1', 'autorouting', 'on');
    add_line(tbName, reqSrc, 'tb_w_req_qkv_sel/1', 'autorouting', 'on');
    add_line(tbName, reqSrc, 'tb_w_req_ffn_sel/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_gamma_sel/1', 'tb_gamma_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_gamma_sel/2', 'tb_gamma_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/1', 'tb_qkv_q_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/2', 'tb_qkv_q_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/3', 'tb_qkv_k_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/4', 'tb_qkv_k_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/5', 'tb_qkv_v_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_qkv_sel/6', 'tb_qkv_v_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/1', 'tb_ffn_up_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/2', 'tb_ffn_up_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/3', 'tb_ffn_gate_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/4', 'tb_ffn_gate_req_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/5', 'tb_ffn_down_req_addr/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_req_ffn_sel/6', 'tb_ffn_down_req_valid/1', 'autorouting', 'on');
end

function srcEndpoint = get_existing_source_endpoint(blockPath)
    modelName = bdroot(blockPath);
    ph = get_param(blockPath, 'PortHandles');
    lineHandle = get_param(ph.Inport(1), 'Line');
    srcPortHandle = get_param(lineHandle, 'SrcPortHandle');
    srcBlockHandle = get_param(srcPortHandle, 'Parent');
    srcBlock = getfullname(srcBlockHandle);
    srcPort = get_param(srcPortHandle, 'PortNumber');
    prefix = [modelName '/'];
    if startsWith(srcBlock, prefix)
        srcBlock = extractAfter(srcBlock, strlength(prefix));
    end
    srcEndpoint = sprintf('%s/%d', srcBlock, srcPort);
end