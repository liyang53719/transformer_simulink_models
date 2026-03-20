function result = run_stage2_real_first_block_weight_regression(rootDir, options)
%RUN_STAGE2_REAL_FIRST_BLOCK_WEIGHT_REGRESSION Replay real first-block parameter samples through weight_ref_u.
%   This regression feeds weight_ref_u with request-driven sample tables
%   extracted from the cached real Qwen2 first-layer parameters and checks
%   exact lane-by-lane address -> response data mapping inside the wrapper TB.

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
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, struct( ...
        'LayerIndex', getFieldOr(options, 'LayerIndex', 1), ...
        'TokenPos', getFieldOr(options, 'TokenPos', 1), ...
        'NumHeads', getFieldOr(options, 'NumHeads', 12), ...
        'PageBase', getFieldOr(options, 'PageBase', 64), ...
        'PageStride', getFieldOr(options, 'PageStride', 8), ...
        'TableLength', getFieldOr(options, 'TableLength', 256)));

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg, ...
        'WeightRspConfig', weightRspCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    patch_attention_score_norm_guard(mdlName);
    retarget_weight_ref_to_sample_tables(tbName, weightRspCfg);
    ensure_weight_observers(tbName);
    ensure_weight_lane_workspace_logs(tbName);
    clear_weight_lane_workspace_logs();
    set_param(tbName, 'SimulationCommand', 'update');

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    laneNames = weightRspCfg.lane_names;
    expectedAddrs = double(weightRspCfg.lane_expected_addrs);
    expectedValues = double(weightRspCfg.sample_values);
    result = struct();
    result.lane_results = repmat(struct( ...
        'lane', '', 'valid_seen', false, 'addr_seen', false, 'data_match', false, 'match_count', 0, ...
        'observed_first_valid_data', NaN, 'observed_peak_data', NaN), [1, numel(laneNames)]);
    result.out_hidden_nonzero = any(abs(extract_signal(yout, 'out_hidden')) > 0);

    allPass = true;
    for i = 1:numel(laneNames)
        prefix = laneNames{i};
        rspValid = double(get_sim_output_data(simOut, ['lane_rsp_valid_' num2str(i)], extract_signal(yout, ['tb_' prefix '_rsp_valid'])));
        rspData = double(get_sim_output_data(simOut, ['lane_rsp_data_' num2str(i)], extract_signal(yout, ['tb_' prefix '_rsp_data'])));
        reqValid = double(extract_signal(yout, ['tb_' prefix '_req_valid']));
        reqAddr = double(extract_signal(yout, ['tb_' prefix '_req_addr']));
        rspData = rspData(:);
        rspValid = rspValid(:);
        reqValid = reqValid(:);
        reqAddr = reqAddr(:);
        commonRspLen = min(numel(rspData), numel(rspValid));
        rspData = rspData(1:commonRspLen);
        rspValid = rspValid(1:commonRspLen);
        commonReqLen = min(numel(reqAddr), numel(reqValid));
        reqAddr = reqAddr(1:commonReqLen);
        reqValid = reqValid(1:commonReqLen);
        dataMatchMask = (rspValid > 0.5) & (abs(rspData - expectedValues(i)) < 1e-9);
        addrMask = (reqValid > 0.5) & (round(reqAddr) == expectedAddrs(i));
        lanePass = any(dataMatchMask) && any(addrMask);
        firstValidIdx = find(rspValid > 0.5, 1, 'first');
        result.lane_results(i).lane = laneNames{i};
        result.lane_results(i).valid_seen = any(rspValid > 0.5);
        result.lane_results(i).addr_seen = any(addrMask);
        result.lane_results(i).data_match = any(dataMatchMask);
        result.lane_results(i).match_count = sum(dataMatchMask);
        if ~isempty(firstValidIdx)
            result.lane_results(i).observed_first_valid_data = rspData(firstValidIdx);
        end
        result.lane_results(i).observed_peak_data = max(rspData(:));
        allPass = allPass && lanePass;
    end

    result.pass = allPass && result.out_hidden_nonzero;

    if result.pass
        fprintf('Stage2 real first-block weight regression PASS\n');
    else
        fprintf('Stage2 real first-block weight regression FAIL\n');
        for i = 1:numel(result.lane_results)
            fprintf(['  lane=%s valid_seen=%d addr_seen=%d data_match=%d match_count=%d ' ...
                'expected_addr=%g expected_data=%g observed_first_valid=%g observed_peak=%g\n'], ...
                result.lane_results(i).lane, ...
                result.lane_results(i).valid_seen, ...
                result.lane_results(i).addr_seen, ...
                result.lane_results(i).data_match, ...
                result.lane_results(i).match_count, ...
                expectedAddrs(i), expectedValues(i), ...
                result.lane_results(i).observed_first_valid_data, ...
                result.lane_results(i).observed_peak_data);
        end
        fprintf('  out_hidden_nonzero=%d\n', result.out_hidden_nonzero);
        error('run_stage2_real_first_block_weight_regression:Failed', ...
            'Real first-block parameter replay checks failed');
    end
end

function ensure_weight_observers(tbName)
    reqSrc = get_existing_source_endpoint([tbName '/tb_w_req_sel']);

    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_gamma_sel'], ...
        'OutputSignals', 'gamma_valid,gamma_data', ...
        'Position', [1080, 595, 1145, 645]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_gamma_rsp_valid'], 'Position', [1240, 120, 1270, 134]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_gamma_rsp_data'], 'Position', [1240, 160, 1270, 174]);

    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_qkv_sel'], ...
        'OutputSignals', 'qkv_q_valid,qkv_k_valid,qkv_v_valid,attn_q_valid,attn_k_valid,attn_v_valid', ...
        'Position', [1080, 655, 1145, 815]);
    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_qkv_data_sel'], ...
        'OutputSignals', 'qkv_q_data,qkv_k_data,qkv_v_data,attn_q_data,attn_k_data,attn_v_data', ...
        'Position', [1080, 825, 1145, 985]);
    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_ffn_sel'], ...
        'OutputSignals', 'ffn_up_valid,ffn_gate_valid,ffn_down_valid', ...
        'Position', [1080, 995, 1145, 1055]);
    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_ffn_data_sel'], ...
        'OutputSignals', 'ffn_up_data,ffn_gate_data,ffn_down_data', ...
        'Position', [1080, 1065, 1145, 1125]);

    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_q_rsp_valid'], 'Position', [1240, 320, 1270, 334]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_k_rsp_valid'], 'Position', [1240, 360, 1270, 374]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_v_rsp_valid'], 'Position', [1240, 400, 1270, 414]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_q_rsp_valid'], 'Position', [1240, 440, 1270, 454]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_k_rsp_valid'], 'Position', [1240, 480, 1270, 494]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_v_rsp_valid'], 'Position', [1240, 520, 1270, 534]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_q_rsp_data'], 'Position', [1240, 560, 1270, 574]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_k_rsp_data'], 'Position', [1240, 600, 1270, 614]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_v_rsp_data'], 'Position', [1240, 640, 1270, 654]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_q_rsp_data'], 'Position', [1240, 680, 1270, 694]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_k_rsp_data'], 'Position', [1240, 720, 1270, 734]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_v_rsp_data'], 'Position', [1240, 760, 1270, 774]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_up_rsp_valid'], 'Position', [1240, 800, 1270, 814]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_rsp_valid'], 'Position', [1240, 840, 1270, 854]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_up_rsp_data'], 'Position', [1240, 880, 1270, 894]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_rsp_data'], 'Position', [1240, 920, 1270, 934]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_down_rsp_valid'], 'Position', [1240, 960, 1270, 974]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_down_rsp_data'], 'Position', [1240, 1000, 1270, 1014]);

    add_line_if_missing(tbName, 'weight_ref_u/1', 'tb_w_rsp_gamma_sel/1');
    add_line_if_missing(tbName, 'weight_ref_u/1', 'tb_w_rsp_qkv_sel/1');
    add_line_if_missing(tbName, 'weight_ref_u/1', 'tb_w_rsp_qkv_data_sel/1');
    add_line_if_missing(tbName, 'weight_ref_u/1', 'tb_w_rsp_ffn_sel/1');
    add_line_if_missing(tbName, 'weight_ref_u/1', 'tb_w_rsp_ffn_data_sel/1');
    add_line_if_missing(tbName, 'tb_w_rsp_gamma_sel/1', 'tb_gamma_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_gamma_sel/2', 'tb_gamma_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/1', 'tb_qkv_q_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/2', 'tb_qkv_k_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/3', 'tb_qkv_v_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/4', 'tb_attn_q_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/5', 'tb_attn_k_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_sel/6', 'tb_attn_v_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/1', 'tb_qkv_q_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/2', 'tb_qkv_k_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/3', 'tb_qkv_v_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/4', 'tb_attn_q_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/5', 'tb_attn_k_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_qkv_data_sel/6', 'tb_attn_v_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_sel/1', 'tb_ffn_up_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_sel/2', 'tb_ffn_gate_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_sel/3', 'tb_ffn_down_rsp_valid/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_data_sel/1', 'tb_ffn_up_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_data_sel/2', 'tb_ffn_gate_rsp_data/1');
    add_line_if_missing(tbName, 'tb_w_rsp_ffn_data_sel/3', 'tb_ffn_down_rsp_data/1');

    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_gamma_sel'], ...
        'OutputSignals', 'gamma_addr,gamma_valid', ...
        'Position', [1080, 505, 1145, 555]);
    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_qkv_sel'], ...
        'OutputSignals', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid,attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid', ...
        'Position', [1080, 560, 1145, 820]);
    add_block_if_missing('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_ffn_sel'], ...
        'OutputSignals', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid,ffn_down_addr,ffn_down_valid', ...
        'Position', [1080, 830, 1145, 930]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_gamma_req_addr'], 'Position', [1240, 40, 1270, 54]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_gamma_req_valid'], 'Position', [1240, 80, 1270, 94]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_q_req_addr'], 'Position', [1240, 960, 1270, 974]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_q_req_valid'], 'Position', [1240, 1000, 1270, 1014]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_k_req_addr'], 'Position', [1240, 1040, 1270, 1054]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_k_req_valid'], 'Position', [1240, 1080, 1270, 1094]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_v_req_addr'], 'Position', [1240, 1120, 1270, 1134]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_qkv_v_req_valid'], 'Position', [1240, 1160, 1270, 1174]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_q_req_addr'], 'Position', [1240, 1200, 1270, 1214]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_q_req_valid'], 'Position', [1240, 1240, 1270, 1254]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_k_req_addr'], 'Position', [1240, 1280, 1270, 1294]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_k_req_valid'], 'Position', [1240, 1320, 1270, 1334]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_v_req_addr'], 'Position', [1240, 1360, 1270, 1374]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_attn_v_req_valid'], 'Position', [1240, 1400, 1270, 1414]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_up_req_addr'], 'Position', [1240, 1440, 1270, 1454]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_up_req_valid'], 'Position', [1240, 1480, 1270, 1494]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_req_addr'], 'Position', [1240, 1520, 1270, 1534]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_gate_req_valid'], 'Position', [1240, 1560, 1270, 1574]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_down_req_addr'], 'Position', [1240, 1600, 1270, 1614]);
    add_block_if_missing('simulink/Sinks/Out1', [tbName '/tb_ffn_down_req_valid'], 'Position', [1240, 1640, 1270, 1654]);

    add_line_if_missing(tbName, reqSrc, 'tb_w_req_gamma_sel/1');
    add_line_if_missing(tbName, reqSrc, 'tb_w_req_qkv_sel/1');
    add_line_if_missing(tbName, reqSrc, 'tb_w_req_ffn_sel/1');
    add_line_if_missing(tbName, 'tb_w_req_gamma_sel/1', 'tb_gamma_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_gamma_sel/2', 'tb_gamma_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/1', 'tb_qkv_q_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/2', 'tb_qkv_q_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/3', 'tb_qkv_k_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/4', 'tb_qkv_k_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/5', 'tb_qkv_v_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/6', 'tb_qkv_v_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/7', 'tb_attn_q_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/8', 'tb_attn_q_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/9', 'tb_attn_k_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/10', 'tb_attn_k_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/11', 'tb_attn_v_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_qkv_sel/12', 'tb_attn_v_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/1', 'tb_ffn_up_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/2', 'tb_ffn_up_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/3', 'tb_ffn_gate_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/4', 'tb_ffn_gate_req_valid/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/5', 'tb_ffn_down_req_addr/1');
    add_line_if_missing(tbName, 'tb_w_req_ffn_sel/6', 'tb_ffn_down_req_valid/1');
end

function ensure_weight_lane_workspace_logs(tbName)
    subPath = [tbName '/weight_ref_u'];
    for laneIndex = 1:10
        dataLogPath = [subPath '/lane_rsp_data_log_' num2str(laneIndex)];
        validLogPath = [subPath '/lane_rsp_valid_log_' num2str(laneIndex)];
        if getSimulinkBlockHandle(dataLogPath) == -1
            add_block('simulink/Sinks/To Workspace', dataLogPath, ...
                'VariableName', ['lane_rsp_data_' num2str(laneIndex)], ...
                'SaveFormat', 'Structure With Time', ...
                'Position', [700, 10 + 25 * laneIndex, 790, 28 + 25 * laneIndex]);
        end
        if getSimulinkBlockHandle(validLogPath) == -1
            add_block('simulink/Sinks/To Workspace', validLogPath, ...
                'VariableName', ['lane_rsp_valid_' num2str(laneIndex)], ...
                'SaveFormat', 'Structure With Time', ...
                'Position', [820, 10 + 25 * laneIndex, 910, 28 + 25 * laneIndex]);
        end
        add_line_if_missing(subPath, ['data_u8_' num2str(laneIndex) '/1'], ['lane_rsp_data_log_' num2str(laneIndex) '/1']);
        add_line_if_missing(subPath, ['val_d2_' num2str(laneIndex) '/1'], ['lane_rsp_valid_log_' num2str(laneIndex) '/1']);
    end
end

function clear_weight_lane_workspace_logs()
    for laneIndex = 1:10
        clear_base_var(['lane_rsp_data_' num2str(laneIndex)]);
        clear_base_var(['lane_rsp_valid_' num2str(laneIndex)]);
    end
end

function values = get_sim_output_data(simOut, variableName, fallback)
    values = fallback;
    if evalin('base', sprintf('exist(''%s'', ''var'')', variableName))
        logged = evalin('base', variableName);
        extracted = extract_workspace_values(logged);
        if ~isempty(extracted)
            values = extracted;
            return;
        end
    end
    try
        logged = simOut.get(variableName);
        extracted = extract_workspace_values(logged);
        if ~isempty(extracted)
            values = extracted;
        end
    catch
    end
end

function values = extract_workspace_values(logged)
    values = [];
    if isempty(logged)
        return;
    end
    if isstruct(logged) && isfield(logged, 'signals') && isfield(logged.signals, 'values')
        values = logged.signals.values;
    else
        values = logged;
    end
end

function clear_base_var(variableName)
    if evalin('base', sprintf('exist(''%s'', ''var'')', variableName))
        evalin('base', sprintf('clear(''%s'')', variableName));
    end
end

function add_block_if_missing(blockType, blockPath, varargin)
    if getSimulinkBlockHandle(blockPath) == -1
        add_block(blockType, blockPath, varargin{:});
    end
end

function add_line_if_missing(systemName, src, dst)
    try
        add_line(systemName, src, dst, 'autorouting', 'on');
    catch
    end
end

function srcEndpoint = get_existing_source_endpoint(blockPath)
    ph = get_param(blockPath, 'PortHandles');
    lineHandle = get_param(ph.Inport(1), 'Line');
    srcPortHandle = get_param(lineHandle, 'SrcPortHandle');
    srcBlockHandle = get_param(srcPortHandle, 'Parent');
    srcBlock = getfullname(srcBlockHandle);
    srcBlockName = extractAfter(string(srcBlock), string([bdroot(blockPath) '/']));
    srcPortNum = get_param(srcPortHandle, 'PortNumber');
    srcEndpoint = sprintf('%s/%d', char(srcBlockName), srcPortNum);
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
    error('run_stage2_real_first_block_weight_regression:MissingSignal', ...
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