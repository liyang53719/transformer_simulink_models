function result = run_stage2_real_first_block_direct_bus_regression(rootDir, options)
%RUN_STAGE2_REAL_FIRST_BLOCK_DIRECT_BUS_REGRESSION Inject real first-block samples via direct WeightRspBus.
%   This regression bypasses the request-driven wrapper responder and feeds
%   sampled Qwen2 first-layer parameter bytes directly into w_rd_rsp_bus,
%   then checks multi-lane response observability and non-trivial block
%   output activity.

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
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    meta = struct();
    meta.sample_values = double(weightRspCfg.sample_values);
    meta.expected_addrs = double(weightRspCfg.lane_expected_addrs);
    meta.lane_names = weightRspCfg.lane_names;
    meta.source = weightRspCfg.params_mat;
    meta.layer_index = weightRspCfg.layer_index;
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    ensure_weight_observers(tbName);
    inject_sample_values_into_weight_ref(tbName, meta.sample_values);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    result = struct();
    result.gamma_rsp_ok = matches_constant_response(yout, 'tb_gamma_rsp_data', 'tb_gamma_rsp_valid', meta.sample_values(1));
    result.qkv_q_rsp_ok = matches_constant_response(yout, 'tb_qkv_q_rsp_data', 'tb_qkv_q_rsp_valid', meta.sample_values(2));
    result.qkv_k_rsp_ok = matches_constant_response(yout, 'tb_qkv_k_rsp_data', 'tb_qkv_k_rsp_valid', meta.sample_values(3));
    result.qkv_v_rsp_ok = matches_constant_response(yout, 'tb_qkv_v_rsp_data', 'tb_qkv_v_rsp_valid', meta.sample_values(4));
    result.attn_q_rsp_ok = matches_constant_response(yout, 'tb_attn_q_rsp_data', 'tb_attn_q_rsp_valid', meta.sample_values(5));
    result.attn_k_rsp_ok = matches_constant_response(yout, 'tb_attn_k_rsp_data', 'tb_attn_k_rsp_valid', meta.sample_values(6));
    result.attn_v_rsp_ok = matches_constant_response(yout, 'tb_attn_v_rsp_data', 'tb_attn_v_rsp_valid', meta.sample_values(7));
    result.ffn_up_rsp_ok = matches_constant_response(yout, 'tb_ffn_up_rsp_data', 'tb_ffn_up_rsp_valid', meta.sample_values(8));
    result.ffn_gate_rsp_ok = matches_constant_response(yout, 'tb_ffn_gate_rsp_data', 'tb_ffn_gate_rsp_valid', meta.sample_values(9));
    result.gamma_addr_ok = matches_request_address(yout, 'tb_gamma_req_addr', 'tb_gamma_req_valid', meta.expected_addrs(1));
    result.qkv_q_addr_ok = matches_request_address(yout, 'tb_qkv_q_req_addr', 'tb_qkv_q_req_valid', meta.expected_addrs(2));
    result.qkv_k_addr_ok = matches_request_address(yout, 'tb_qkv_k_req_addr', 'tb_qkv_k_req_valid', meta.expected_addrs(3));
    result.qkv_v_addr_ok = matches_request_address(yout, 'tb_qkv_v_req_addr', 'tb_qkv_v_req_valid', meta.expected_addrs(4));
    result.attn_q_addr_ok = matches_request_address(yout, 'tb_attn_q_req_addr', 'tb_attn_q_req_valid', meta.expected_addrs(5));
    result.attn_k_addr_ok = matches_request_address(yout, 'tb_attn_k_req_addr', 'tb_attn_k_req_valid', meta.expected_addrs(6));
    result.attn_v_addr_ok = matches_request_address(yout, 'tb_attn_v_req_addr', 'tb_attn_v_req_valid', meta.expected_addrs(7));
    result.ffn_up_addr_ok = matches_request_address(yout, 'tb_ffn_up_req_addr', 'tb_ffn_up_req_valid', meta.expected_addrs(8));
    result.ffn_gate_addr_ok = matches_request_address(yout, 'tb_ffn_gate_req_addr', 'tb_ffn_gate_req_valid', meta.expected_addrs(9));
    outHidden = double(extract_signal(yout, 'out_hidden'));
    result.out_hidden_nonzero = any(abs(outHidden) > 0);
    result.out_hidden_dynamic = numel(unique(outHidden(:))) > 1;
    result.pass = result.gamma_rsp_ok && ...
        result.qkv_q_rsp_ok && result.qkv_k_rsp_ok && result.qkv_v_rsp_ok && ...
        result.attn_q_rsp_ok && result.attn_k_rsp_ok && result.attn_v_rsp_ok && ...
        result.ffn_up_rsp_ok && result.ffn_gate_rsp_ok && ...
        result.gamma_addr_ok && ...
        result.qkv_q_addr_ok && result.qkv_k_addr_ok && result.qkv_v_addr_ok && ...
        result.attn_q_addr_ok && result.attn_k_addr_ok && result.attn_v_addr_ok && ...
        result.ffn_up_addr_ok && result.ffn_gate_addr_ok && ...
        result.out_hidden_nonzero;

    if result.pass
        fprintf('Stage2 real first-block direct bus regression PASS\n');
    else
        fprintf('Stage2 real first-block direct bus regression FAIL\n');
        fprintf(['  gamma_rsp=%d qkv_q_rsp=%d qkv_k_rsp=%d qkv_v_rsp=%d ' ...
            'attn_q_rsp=%d attn_k_rsp=%d attn_v_rsp=%d ffn_up_rsp=%d ffn_gate_rsp=%d\n'], ...
            result.gamma_rsp_ok, result.qkv_q_rsp_ok, result.qkv_k_rsp_ok, result.qkv_v_rsp_ok, ...
            result.attn_q_rsp_ok, result.attn_k_rsp_ok, result.attn_v_rsp_ok, ...
            result.ffn_up_rsp_ok, result.ffn_gate_rsp_ok);
        fprintf(['  gamma_addr=%d qkv_q_addr=%d qkv_k_addr=%d qkv_v_addr=%d ' ...
            'attn_q_addr=%d attn_k_addr=%d attn_v_addr=%d ffn_up_addr=%d ffn_gate_addr=%d ' ...
            'out_hidden_nonzero=%d out_hidden_dynamic=%d\n'], ...
            result.gamma_addr_ok, result.qkv_q_addr_ok, result.qkv_k_addr_ok, result.qkv_v_addr_ok, ...
            result.attn_q_addr_ok, result.attn_k_addr_ok, result.attn_v_addr_ok, ...
            result.ffn_up_addr_ok, result.ffn_gate_addr_ok, ...
            result.out_hidden_nonzero, result.out_hidden_dynamic);
        fprintf('  tb_gamma_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_gamma_rsp_data'));
        fprintf('  tb_qkv_q_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_qkv_q_rsp_data'));
        fprintf('  tb_qkv_k_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_qkv_k_rsp_data'));
        fprintf('  tb_qkv_v_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_qkv_v_rsp_data'));
        fprintf('  tb_attn_q_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_q_rsp_data'));
        fprintf('  tb_attn_k_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_k_rsp_data'));
        fprintf('  tb_attn_v_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_v_rsp_data'));
        fprintf('  tb_ffn_up_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_ffn_up_rsp_data'));
        fprintf('  tb_ffn_gate_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_ffn_gate_rsp_data'));
        fprintf('  tb_gamma_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_gamma_rsp_valid'));
        fprintf('  tb_qkv_q_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_qkv_q_rsp_valid'));
        fprintf('  tb_qkv_k_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_qkv_k_rsp_valid'));
        fprintf('  tb_qkv_v_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_qkv_v_rsp_valid'));
        fprintf('  tb_attn_q_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_q_rsp_valid'));
        fprintf('  tb_attn_k_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_k_rsp_valid'));
        fprintf('  tb_attn_v_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_v_rsp_valid'));
        fprintf('  tb_ffn_up_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_ffn_up_rsp_valid'));
        fprintf('  tb_ffn_gate_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_ffn_gate_rsp_valid'));
        error('run_stage2_real_first_block_direct_bus_regression:Failed', ...
            'Direct real first-block bus injection checks failed');
    end
end

function yes = matches_constant_response(yout, dataName, validName, expectedValue)
    data = double(extract_signal(yout, dataName));
    valid = double(extract_signal(yout, validName));
    data = data(:);
    valid = valid(:);
    yes = any(valid > 0.5) && any(abs(data - double(expectedValue)) < 1e-9);
end

function yes = matches_request_address(yout, addrName, validName, expectedAddr)
    addr = double(extract_signal(yout, addrName));
    valid = double(extract_signal(yout, validName));
    addr = addr(:);
    valid = valid(:);
    commonLen = min(numel(addr), numel(valid));
    addr = addr(1:commonLen);
    valid = valid(1:commonLen);
    mask = valid > 0.5;
    yes = any(mask & abs(addr - double(expectedAddr)) < 1e-9);
end

function inject_sample_values_into_weight_ref(tbName, sampleValues)
    subPath = [tbName '/weight_ref_u'];
    for i = 1:min(9, numel(sampleValues))
        constName = ['sample_value_' num2str(i)];
        constPath = [subPath '/' constName];
        if isempty(find_system(subPath, 'SearchDepth', 1, 'Name', constName))
            add_block('simulink/Sources/Constant', constPath, ...
                'Position', [455, 25 + 30 * (i - 1) + 6, 520, 25 + 30 * (i - 1) + 26]);
        end
        set_param(constPath, 'Value', num2str(double(sampleValues(i))));

        try
            delete_line(subPath, ['data_page_tag_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        try
            delete_line(subPath, ['tag_lane_sum_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        try
            delete_line(subPath, ['sample_value_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        catch
        end
        add_line(subPath, [constName '/1'], ['data_u8_' num2str(i) '/1'], 'autorouting', 'on');
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
        'OutputSignals', 'ffn_up_valid,ffn_gate_valid', ...
        'Position', [1080, 835, 1145, 895]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_ffn_data_sel'], ...
        'OutputSignals', 'ffn_up_data,ffn_gate_data', ...
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
    add_line(tbName, 'tb_w_rsp_ffn_data_sel/1', 'tb_ffn_up_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_ffn_data_sel/2', 'tb_ffn_gate_rsp_data/1', 'autorouting', 'on');

    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_gamma_sel'], ...
        'OutputSignals', 'gamma_addr,gamma_valid', ...
        'Position', [1080, 505, 1145, 555]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_qkv_sel'], ...
        'OutputSignals', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid', ...
        'Position', [1080, 560, 1145, 700]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_ffn_sel'], ...
        'OutputSignals', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid', ...
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
    error('run_stage2_real_first_block_direct_bus_regression:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end

function summary = summarize_unique_values(yout, name)
    values = double(extract_signal(yout, name));
    uniqueValues = unique(values(:));
    summary = strtrim(sprintf('%g ', uniqueValues));
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