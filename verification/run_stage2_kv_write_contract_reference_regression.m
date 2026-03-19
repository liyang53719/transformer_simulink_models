function result = run_stage2_kv_write_contract_reference_regression(rootDir, options)
%RUN_STAGE2_KV_WRITE_CONTRACT_REFERENCE_REGRESSION Compare KV-write summary behavior between DUT and reference adapter.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));
    buildModel = logical(getFieldOr(options, 'BuildModel', true));
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    [params, sourceInfo] = load_qwen_reference_params(paramsSource, rootDir);
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    cases = build_contract_cases();
    refCtx = struct('Parameters', params, 'LayerIndex', 1);

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>
    inject_sample_values_into_weight_ref(tbName, double(weightRspCfg.sample_values));

    refSummaries = repmat(struct('kv_write_en', false, 'kv_write_data', 0, 'kv_write_abs_mean', 0, 'kv_write_finite', true), [1, numel(cases)]);
    dutSummaries = repmat(struct('kv_write_en_seen', false, 'kv_write_data_peak', 0, 'kv_write_abs_mean', 0, 'kv_write_finite', true), [1, numel(cases)]);
    for i = 1:numel(cases)
        [refSummary, ~] = qwen2_block_ref_stage2_contract_adapter(cases(i).contract, refCtx);
        refSummaries(i) = pick_ref_kv_summary(refSummary);
        dutSummaries(i) = simulate_dut_kv_summary(tbName, cases(i).contract);
    end

    result = struct();
    result.pass = true;
    result.params_source = sourceInfo;
    result.case_names = {cases.name};
    result.reference_kv_write_en = [refSummaries.kv_write_en];
    result.dut_kv_write_en = [dutSummaries.kv_write_en_seen];
    result.reference_kv_write_data = double([refSummaries.kv_write_data]);
    result.dut_kv_write_data = double([dutSummaries.kv_write_data_peak]);
    result.reference_kv_finite = all([refSummaries.kv_write_finite]);
    result.dut_kv_finite = all([dutSummaries.kv_write_finite]);
    result.reference_kv_write_present = any(result.reference_kv_write_en);
    result.dut_kv_write_present = any(result.dut_kv_write_en);
    result.reference_activation_sensitive = abs(result.reference_kv_write_data(2) - result.reference_kv_write_data(1)) > 1e-9;
    result.dut_activation_sensitive = abs(result.dut_kv_write_data(2) - result.dut_kv_write_data(1)) > 1e-9;
    result.reference_kv_read_sensitive = abs(result.reference_kv_write_data(3) - result.reference_kv_write_data(1)) > 1e-9;
    result.dut_kv_read_sensitive = abs(result.dut_kv_write_data(3) - result.dut_kv_write_data(1)) > 1e-9;
    result.shared_kv_read_sensitivity = result.reference_kv_read_sensitive && result.dut_kv_read_sensitive;
    result.pass = result.reference_kv_finite && result.dut_kv_finite && ...
        result.reference_kv_write_present && result.dut_kv_write_present && ...
        result.reference_activation_sensitive && result.dut_activation_sensitive;

    if result.pass
        fprintf('Stage2 KV write contract reference regression PASS\n');
        fprintf('  ref_kv_data=%s\n', strjoin(string(result.reference_kv_write_data), ', '));
        fprintf('  dut_kv_data=%s\n', strjoin(string(result.dut_kv_write_data), ', '));
        fprintf('  shared_kv_read_sensitivity=%d\n', result.shared_kv_read_sensitivity);
    else
        fprintf('Stage2 KV write contract reference regression FAIL\n');
        fprintf('  ref_finite=%d dut_finite=%d ref_present=%d dut_present=%d\n', ...
            result.reference_kv_finite, result.dut_kv_finite, ...
            result.reference_kv_write_present, result.dut_kv_write_present);
        fprintf('  ref_activation_sensitive=%d dut_activation_sensitive=%d\n', ...
            result.reference_activation_sensitive, result.dut_activation_sensitive);
        fprintf('  ref_kv_read_sensitive=%d dut_kv_read_sensitive=%d shared=%d\n', ...
            result.reference_kv_read_sensitive, result.dut_kv_read_sensitive, result.shared_kv_read_sensitivity);
        error('run_stage2_kv_write_contract_reference_regression:Failed', ...
            'KV write contract reference regression did not show expected sensitivity.');
    end
end

function cases = build_contract_cases()
    base = default_contract();
    actBump = base;
    actBump.in_hidden = single(3);
    kvReadBump = base;
    kvReadBump.kv_cache_rd_data = single(7);
    kvReadBump.kv_cache_rd_valid = true;
    kvReadBump.cfg_token_pos = 2;
    kvReadBump.cfg_weight_page_stride = 4;

    cases = repmat(struct('name', '', 'contract', struct()), [1, 3]);
    cases(1) = struct('name', 'base', 'contract', base);
    cases(2) = struct('name', 'activation_bump', 'contract', actBump);
    cases(3) = struct('name', 'kv_read_bump', 'contract', kvReadBump);
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

function summary = pick_ref_kv_summary(refSummary)
    summary = struct();
    summary.kv_write_en = logical(getFieldOr(refSummary, 'kv_write_en', false));
    summary.kv_write_data = double(getFieldOr(refSummary, 'kv_write_data', 0));
    summary.kv_write_abs_mean = double(getFieldOr(refSummary, 'kv_write_abs_mean', 0));
    summary.kv_write_finite = logical(getFieldOr(refSummary, 'kv_write_finite', false));
end

function summary = simulate_dut_kv_summary(tbName, contract)
    apply_contract_constants(tbName, contract);
    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');
    wrData = double(extract_signal(yout, 'kv_cache_wr_data'));
    wrEn = double(extract_signal(yout, 'kv_cache_wr_en'));
    wrData = wrData(:);
    wrEn = wrEn(:);
    commonLen = min(numel(wrData), numel(wrEn));
    wrData = wrData(1:commonLen);
    wrEn = wrEn(1:commonLen);
    mask = wrEn > 0.5;

    summary = struct();
    summary.kv_write_en_seen = any(mask);
    if any(mask)
        summary.kv_write_data_peak = max(abs(wrData(mask)));
        summary.kv_write_abs_mean = mean(abs(wrData(mask)));
    else
        summary.kv_write_data_peak = 0;
        summary.kv_write_abs_mean = 0;
    end
    summary.kv_write_finite = all(isfinite(wrData));
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
    error('run_stage2_kv_write_contract_reference_regression:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
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