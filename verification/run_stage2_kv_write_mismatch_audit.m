function result = run_stage2_kv_write_mismatch_audit(rootDir, options)
%RUN_STAGE2_KV_WRITE_MISMATCH_AUDIT Diagnose kv_read-bump KV-write semantics.
%   This audit classifies whether kv_read perturbations create a real
%   shared mismatch between DUT and reference summaries, or whether both
%   sides are consistently insensitive after integer control typing fixes.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));
    buildModel = logical(getFieldOr(options, 'BuildModel', false));
    assert_stage2_manual_model_policy(buildModel, mfilename);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    [params, sourceInfo] = load_qwen_reference_params(paramsSource, rootDir);
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);
    refCtx = struct('Parameters', params, 'LayerIndex', 1);
    cases = build_cases();

    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>
    inject_sample_values_into_weight_ref(tbName, double(weightRspCfg.sample_values));

    refObservations = repmat(struct( ...
        'summary', struct(), ...
        'kv_last_dim', 0, ...
        'kv_last_slice_abs_mean', 0, ...
        'kv_last_slice_shape', [], ...
        'out_hidden_abs_mean', 0), [1, numel(cases)]);
    dutObservations = repmat(struct( ...
        'kv_write_en_seen', false, ...
        'kv_write_data_peak', 0, ...
        'kv_write_abs_mean', 0, ...
        'kv_write_finite', true), [1, numel(cases)]);

    for i = 1:numel(cases)
        [refSummary, refDetail] = qwen2_block_ref_stage2_contract_adapter(cases(i).contract, refCtx);
        refObservations(i) = capture_reference_observation(refSummary, refDetail);
        dutObservations(i) = simulate_dut_kv_summary(tbName, cases(i).contract);
    end

    baseRef = refObservations(1);
    bumpRef = refObservations(2);
    baseDut = dutObservations(1);
    bumpDut = dutObservations(2);

    result = struct();
    result.params_source = sourceInfo;
    result.case_names = {cases.name};
    result.reference_out_hidden_sensitive = abs(bumpRef.out_hidden_abs_mean - baseRef.out_hidden_abs_mean) > 1e-9;
    result.reference_kv_write_sensitive = abs(bumpRef.summary.kv_write_data - baseRef.summary.kv_write_data) > 1e-9;
    result.dut_kv_write_sensitive = abs(bumpDut.kv_write_data_peak - baseDut.kv_write_data_peak) > 1e-9;
    result.reference_kv_write_present = baseRef.summary.kv_write_en && bumpRef.summary.kv_write_en;
    result.dut_kv_write_present = baseDut.kv_write_en_seen && bumpDut.kv_write_en_seen;
    result.reference_kv_last_dim = [baseRef.kv_last_dim, bumpRef.kv_last_dim];
    result.reference_kv_last_slice_abs_mean = [baseRef.kv_last_slice_abs_mean, bumpRef.kv_last_slice_abs_mean];
    result.reference_kv_last_slice_shape = {mat2str(baseRef.kv_last_slice_shape), mat2str(bumpRef.kv_last_slice_shape)};
    result.reference_kv_last_slice_sensitive = abs(bumpRef.kv_last_slice_abs_mean - baseRef.kv_last_slice_abs_mean) > 1e-9;
    result.reference_kv_length_expands_with_read = bumpRef.kv_last_dim > baseRef.kv_last_dim;
    result.reference_kv_finite = logical(baseRef.summary.kv_write_finite && bumpRef.summary.kv_write_finite);
    result.dut_kv_finite = logical(baseDut.kv_write_finite && bumpDut.kv_write_finite);
    result.classification = classify_mismatch(result);
    result.pass = result.reference_kv_finite && result.dut_kv_finite && ...
        result.reference_kv_write_present && result.dut_kv_write_present && ...
        result.reference_out_hidden_sensitive && ...
        result.classification ~= "unclassified";

    if result.pass
        fprintf('Stage2 KV write mismatch audit PASS\n');
        fprintf('  classification=%s\n', result.classification);
        fprintf('  ref_out_hidden_sensitive=%d ref_kv_write_sensitive=%d dut_kv_write_sensitive=%d\n', ...
            result.reference_out_hidden_sensitive, result.reference_kv_write_sensitive, result.dut_kv_write_sensitive);
        fprintf('  ref_kv_last_dim=%s ref_kv_last_slice_sensitive=%d\n', ...
            mat2str(result.reference_kv_last_dim), result.reference_kv_last_slice_sensitive);
    else
        fprintf('Stage2 KV write mismatch audit FAIL\n');
        fprintf('  ref_finite=%d dut_finite=%d ref_present=%d dut_present=%d\n', ...
            result.reference_kv_finite, result.dut_kv_finite, ...
            result.reference_kv_write_present, result.dut_kv_write_present);
        fprintf('  ref_out_hidden_sensitive=%d ref_kv_write_sensitive=%d dut_kv_write_sensitive=%d\n', ...
            result.reference_out_hidden_sensitive, result.reference_kv_write_sensitive, result.dut_kv_write_sensitive);
        fprintf('  ref_kv_last_dim=%s ref_kv_last_slice_sensitive=%d classification=%s\n', ...
            mat2str(result.reference_kv_last_dim), result.reference_kv_last_slice_sensitive, result.classification);
        error('run_stage2_kv_write_mismatch_audit:Failed', ...
            'KV write mismatch audit could not classify the observed mismatch.');
    end
end

function cases = build_cases()
    base = default_contract();
    kvReadBump = base;
    kvReadBump.kv_cache_rd_data = single(7);
    kvReadBump.kv_cache_rd_valid = true;
    kvReadBump.cfg_token_pos = 2;
    kvReadBump.cfg_weight_page_stride = 4;

    cases = repmat(struct('name', '', 'contract', struct()), [1, 2]);
    cases(1) = struct('name', 'base', 'contract', base);
    cases(2) = struct('name', 'kv_read_bump', 'contract', kvReadBump);
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

function observation = capture_reference_observation(refSummary, refDetail)
    observation = struct();
    observation.summary = pick_ref_kv_summary(refSummary);
    observation.out_hidden_abs_mean = double(getFieldOr(refSummary, 'out_hidden_abs_mean', 0));

    valueTensor = pick_value_tensor(getFieldOr(refDetail, 'out_kv', struct()));
    if isempty(valueTensor)
        observation.kv_last_dim = 0;
        observation.kv_last_slice_abs_mean = 0;
        observation.kv_last_slice_shape = [0 0];
        return;
    end

    dims = size(valueTensor);
    if numel(dims) >= 3
        observation.kv_last_dim = dims(3);
    else
        observation.kv_last_dim = 1;
    end
    lastSlice = extract_last_slice(valueTensor);
    observation.kv_last_slice_abs_mean = double(mean(abs(single(lastSlice)), 'all'));
    observation.kv_last_slice_shape = size(lastSlice);
end

function valueTensor = pick_value_tensor(outKV)
    valueTensor = single([]);
    if ~isstruct(outKV)
        return;
    end
    if isfield(outKV, 'values') && ~isempty(outKV.values)
        valueTensor = single(outKV.values);
    elseif isfield(outKV, 'keys') && ~isempty(outKV.keys)
        valueTensor = single(outKV.keys);
    end
end

function slice = extract_last_slice(valueTensor)
    dims = size(valueTensor);
    if numel(dims) >= 3
        idx = repmat({':'}, 1, numel(dims));
        idx{3} = dims(3);
        slice = single(valueTensor(idx{:}));
    else
        slice = single(valueTensor);
    end
end

function classification = classify_mismatch(result)
    if result.reference_out_hidden_sensitive && ~result.reference_kv_write_sensitive && ...
            ~result.dut_kv_write_sensitive && result.reference_kv_length_expands_with_read && ...
            ~result.reference_kv_last_slice_sensitive
        classification = "shared_kv_write_insensitive_after_integer_control_typing";
    elseif result.reference_out_hidden_sensitive && ~result.reference_kv_write_sensitive && ...
            result.dut_kv_write_sensitive && result.reference_kv_length_expands_with_read && ...
            ~result.reference_kv_last_slice_sensitive
        classification = "reference_present_history_expands_but_last_write_slice_is_not_past_sensitive";
    elseif result.reference_out_hidden_sensitive && ~result.reference_kv_write_sensitive && ...
            result.dut_kv_write_sensitive
        classification = "reference_kv_write_summary_is_non_shared_with_hardware_write_path";
    else
        classification = "unclassified";
    end
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
    for i = 1:min(10, numel(sampleValues))
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
    error('run_stage2_kv_write_mismatch_audit:MissingSignal', ...
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