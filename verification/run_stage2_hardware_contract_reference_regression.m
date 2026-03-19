function result = run_stage2_hardware_contract_reference_regression(rootDir, options)
%RUN_STAGE2_HARDWARE_CONTRACT_REFERENCE_REGRESSION Compare contract-level response sensitivity between DUT and reference adapter.

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

    refOutputs = zeros(1, numel(cases));
    dutObservations = repmat(struct('out_hidden', [], 'kv_mem_rd_addr', []), [1, numel(cases)]);
    for i = 1:numel(cases)
        [refSummary, ~] = qwen2_block_ref_stage2_contract_adapter(cases(i).contract, refCtx);
        refOutputs(i) = double(refSummary.out_hidden);
        dutObservations(i) = simulate_dut_contract_case(tbName, cases(i).contract);
    end

    result = struct();
    result.pass = true;
    result.params_source = sourceInfo;
    result.case_names = {cases.name};
    result.reference_outputs = refOutputs;
    result.dut_outputs = [summarize_last_finite(dutObservations(1).out_hidden), ...
        summarize_last_finite(dutObservations(2).out_hidden), ...
        summarize_last_finite(dutObservations(3).out_hidden)];
    result.reference_finite = all(isfinite(refOutputs));
    result.dut_finite = all(isfinite(result.dut_outputs));
    result.reference_activation_delta = abs(refOutputs(2) - refOutputs(1));
    result.reference_token_pos_delta = abs(refOutputs(3) - refOutputs(1));
    result.dut_activation_delta = waveform_delta(dutObservations(2).out_hidden, dutObservations(1).out_hidden);
    result.dut_token_pos_delta = waveform_delta(dutObservations(3).kv_mem_rd_addr, dutObservations(1).kv_mem_rd_addr);
    result.reference_activation_sensitive = result.reference_activation_delta > 1e-9;
    result.reference_token_pos_sensitive = result.reference_token_pos_delta > 1e-9;
    result.dut_activation_sensitive = result.dut_activation_delta > 1e-9;
    result.dut_token_pos_sensitive = result.dut_token_pos_delta > 1e-9;
    result.pass = result.reference_finite && result.dut_finite && ...
        result.reference_activation_sensitive && result.reference_token_pos_sensitive && ...
        result.dut_activation_sensitive && result.dut_token_pos_sensitive;

    if result.pass
        fprintf('Stage2 hardware contract reference regression PASS\n');
        fprintf('  ref_outputs=%s\n', strjoin(string(result.reference_outputs), ', '));
        fprintf('  dut_outputs=%s\n', strjoin(string(result.dut_outputs), ', '));
    else
        fprintf('Stage2 hardware contract reference regression FAIL\n');
        fprintf('  reference_finite=%d dut_finite=%d\n', result.reference_finite, result.dut_finite);
        fprintf('  ref_activation_sensitive=%d ref_token_pos_sensitive=%d\n', ...
            result.reference_activation_sensitive, result.reference_token_pos_sensitive);
        fprintf('  dut_activation_sensitive=%d dut_token_pos_sensitive=%d\n', ...
            result.dut_activation_sensitive, result.dut_token_pos_sensitive);
        error('run_stage2_hardware_contract_reference_regression:Failed', ...
            'Hardware-contract reference regression did not show expected sensitivity.');
    end
end

function cases = build_contract_cases()
    base = default_contract();
    actBump = base;
    actBump.in_hidden = single(3);
    tokenBump = base;
    tokenBump.kv_cache_rd_data = single(7);
    tokenBump.kv_cache_rd_valid = true;
    tokenBump.cfg_token_pos = 2;

    cases = repmat(struct('name', '', 'contract', struct()), [1, 3]);
    cases(1) = struct('name', 'base', 'contract', base);
    cases(2) = struct('name', 'activation_bump', 'contract', actBump);
    cases(3) = struct('name', 'token_pos_bump', 'contract', tokenBump);
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

function observation = simulate_dut_contract_case(tbName, contract)
    apply_contract_constants(tbName, contract);
    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');
    observation = struct();
    observation.out_hidden = double(extract_signal(yout, 'out_hidden'));
    observation.kv_mem_rd_addr = double(extract_signal(yout, 'kv_mem_rd_addr'));
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
    error('run_stage2_hardware_contract_reference_regression:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end

function delta = waveform_delta(a, b)
    a = double(a(:));
    b = double(b(:));
    commonLen = min(numel(a), numel(b));
    if commonLen == 0
        delta = 0;
        return;
    end
    delta = max(abs(a(1:commonLen) - b(1:commonLen)));
end

function value = summarize_last_finite(samples)
    samples = double(samples(:));
    finiteMask = isfinite(samples);
    if any(finiteMask)
        value = samples(find(finiteMask, 1, 'last'));
    else
        value = NaN;
    end
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