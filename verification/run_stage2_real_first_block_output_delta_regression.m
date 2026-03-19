function result = run_stage2_real_first_block_output_delta_regression(rootDir, options)
%RUN_STAGE2_REAL_FIRST_BLOCK_OUTPUT_DELTA_REGRESSION Compare synthetic vs real-sample wrapper outputs.
%   This regression establishes a block-level numerical delta baseline by
%   running the wrapper twice: once with the default synthetic responder and
%   once with real layer0 samples injected into weight_ref_u. The check is
%   intentionally weaker than a full golden comparison; it verifies that the
%   real sample injection changes out_hidden numerically while keeping the
%   output non-trivial.

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
    weightRspCfg = build_qwen2_first_block_weight_rsp_config(rootDir, options);

    baseline = simulate_wrapper_out_hidden(rootDir, kvCfg, buildModel, []);
    realSample = simulate_wrapper_out_hidden(rootDir, kvCfg, buildModel, double(weightRspCfg.sample_values));

    baselineData = baseline(:);
    realData = realSample(:);
    commonLen = min(numel(baselineData), numel(realData));
    baselineData = baselineData(1:commonLen);
    realData = realData(1:commonLen);
    delta = realData - baselineData;

    result = struct();
    result.sample_count = commonLen;
    result.baseline_nonzero = any(abs(baselineData) > 0);
    result.real_nonzero = any(abs(realData) > 0);
    result.max_abs_diff = max(abs(delta));
    result.mean_abs_diff = mean(abs(delta));
    result.rel_l2_diff = norm(delta) / max(norm(realData), eps);
    result.changed = any(abs(delta) > 1e-9);
    result.pass = result.baseline_nonzero && result.real_nonzero && result.changed;

    if result.pass
        fprintf('Stage2 real first-block output delta regression PASS\n');
        fprintf(['  scope=output activity delta only; ' ...
            'not a golden numeric match through top-level DDR\n']);
        fprintf('  sample_count=%d max_abs_diff=%g mean_abs_diff=%g rel_l2_diff=%g\n', ...
            result.sample_count, result.max_abs_diff, result.mean_abs_diff, result.rel_l2_diff);
    else
        fprintf('Stage2 real first-block output delta regression FAIL\n');
        fprintf('  baseline_nonzero=%d real_nonzero=%d changed=%d sample_count=%d\n', ...
            result.baseline_nonzero, result.real_nonzero, result.changed, result.sample_count);
        fprintf('  max_abs_diff=%g mean_abs_diff=%g rel_l2_diff=%g\n', ...
            result.max_abs_diff, result.mean_abs_diff, result.rel_l2_diff);
        error('run_stage2_real_first_block_output_delta_regression:Failed', ...
            'Synthetic vs real-sample output delta check failed');
    end
end

function outHidden = simulate_wrapper_out_hidden(rootDir, kvCfg, buildModel, sampleValues)
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct( ...
        'BuildModel', buildModel, ...
        'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    if isnumeric(sampleValues) && ~isempty(sampleValues)
        inject_sample_values_into_weight_ref(tbName, sampleValues);
    end

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');
    outHidden = double(extract_signal(yout, 'out_hidden'));
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
    error('run_stage2_real_first_block_output_delta_regression:MissingSignal', ...
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