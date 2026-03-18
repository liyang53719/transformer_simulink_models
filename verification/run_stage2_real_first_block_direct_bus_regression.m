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
    inject_sample_values_into_weight_ref(tbName, meta.sample_values);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    result = struct();
    result.attn_q_rsp_ok = matches_constant_response(yout, 'tb_attn_q_rsp_data', 'tb_attn_q_rsp_valid', meta.sample_values(5));
    result.attn_k_rsp_ok = matches_constant_response(yout, 'tb_attn_k_rsp_data', 'tb_attn_k_rsp_valid', meta.sample_values(6));
    result.attn_v_rsp_ok = matches_constant_response(yout, 'tb_attn_v_rsp_data', 'tb_attn_v_rsp_valid', meta.sample_values(7));
    result.attn_q_addr_ok = matches_request_address(yout, 'tb_attn_q_req_addr', 'tb_attn_q_req_valid', meta.expected_addrs(5));
    result.attn_k_addr_ok = matches_request_address(yout, 'tb_attn_k_req_addr', 'tb_attn_k_req_valid', meta.expected_addrs(6));
    result.attn_v_addr_ok = matches_request_address(yout, 'tb_attn_v_req_addr', 'tb_attn_v_req_valid', meta.expected_addrs(7));
    outHidden = double(extract_signal(yout, 'out_hidden'));
    result.out_hidden_nonzero = any(abs(outHidden) > 0);
    result.out_hidden_dynamic = numel(unique(outHidden(:))) > 1;
    result.pass = result.attn_q_rsp_ok && result.attn_k_rsp_ok && result.attn_v_rsp_ok && ...
        result.attn_q_addr_ok && result.attn_k_addr_ok && result.attn_v_addr_ok && ...
        result.out_hidden_nonzero;

    if result.pass
        fprintf('Stage2 real first-block direct bus regression PASS\n');
    else
        fprintf('Stage2 real first-block direct bus regression FAIL\n');
        fprintf(['  attn_q_rsp=%d attn_k_rsp=%d attn_v_rsp=%d ' ...
            'attn_q_addr=%d attn_k_addr=%d attn_v_addr=%d out_hidden_nonzero=%d out_hidden_dynamic=%d\n'], ...
            result.attn_q_rsp_ok, result.attn_k_rsp_ok, result.attn_v_rsp_ok, ...
            result.attn_q_addr_ok, result.attn_k_addr_ok, result.attn_v_addr_ok, ...
            result.out_hidden_nonzero, result.out_hidden_dynamic);
        fprintf('  tb_attn_q_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_q_rsp_data'));
        fprintf('  tb_attn_k_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_k_rsp_data'));
        fprintf('  tb_attn_v_rsp_data unique: %s\n', summarize_unique_values(yout, 'tb_attn_v_rsp_data'));
        fprintf('  tb_attn_q_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_q_rsp_valid'));
        fprintf('  tb_attn_k_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_k_rsp_valid'));
        fprintf('  tb_attn_v_rsp_valid unique: %s\n', summarize_unique_values(yout, 'tb_attn_v_rsp_valid'));
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