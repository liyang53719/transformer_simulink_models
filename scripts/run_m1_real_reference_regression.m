function result = run_m1_real_reference_regression(paramsFileOrModule, options)
%RUN_M1_REAL_REFERENCE_REGRESSION Run M1 regression using real qwen2 block adapter.
%
% Example:
%   run_m1_real_reference_regression('path/to/qwen_params.mat')
%   run_m1_real_reference_regression('module_awq')
%   run_m1_real_reference_regression('all')
%   run_m1_real_reference_regression({'module_awq','module_gptq'})

    arguments
        paramsFileOrModule = 'module_awq'
        options.LayerIndex (1,1) double = 1
        options.PreferDequantizeNow (1,1) logical = false
        options.BaselineMode (1,:) char = 'real'
        options.EnableMemoryMetrics (1,1) logical = false
        options.RunStage2FastSmoke (1,1) logical = false
        options.RunStage2ReferenceReadinessAudit (1,1) logical = false
        options.RunStage2HardwareContractRegression (1,1) logical = false
        options.RunStage2KvWriteContractRegression (1,1) logical = false
        options.RunStage2KvWriteMismatchAudit (1,1) logical = false
        options.ReportDir (1,:) char = ''
        options.Modules = {}
        options.ContinueOnError (1,1) logical = false
        options.WriteBatchSummary (1,1) logical = true
    end

    rootDir = fileparts(fileparts(mfilename('fullpath')));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));
    ensure_local_transformer_models_on_path(rootDir);

    targets = resolve_requested_targets(paramsFileOrModule, options, rootDir);
    if numel(targets) == 1
        result = run_single_target_regression(char(targets(1)), options, rootDir);
        return;
    end

    result = run_batch_target_regression(targets, options, rootDir);
end

function result = run_batch_target_regression(targets, options, rootDir)
    reportDir = resolve_report_dir(rootDir, options);
    stamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    summaryTag = build_batch_report_tag(options);
    summaryJsonPath = fullfile(reportDir, summaryTag + "_" + stamp + ".json");
    summaryMdPath = fullfile(reportDir, summaryTag + "_" + stamp + ".md");

    fprintf('Running M1 real reference batch for %d targets\n', numel(targets));
    run_shared_batch_prechecks(rootDir, options);

    singleOptions = options;
    singleOptions.RunStage2FastSmoke = false;

    entries = repmat(empty_batch_entry(), 1, numel(targets));
    failureMessages = strings(0, 1);

    for i = 1:numel(targets)
        target = string(targets(i));
        fprintf('Batch target %d/%d: %s\n', i, numel(targets), target);
        try
            singleResult = run_single_target_regression(char(target), singleOptions, rootDir);
            entries(i) = summarize_batch_entry(singleResult, target);
        catch err
            entries(i) = summarize_batch_error(err, target);
            failureMessages(end + 1, 1) = string(target) + ": " + string(err.message); %#ok<AGROW>
            if ~options.ContinueOnError
                partial = build_batch_result(entries(1:i), targets(1:i), options, rootDir, summaryJsonPath, summaryMdPath);
                partial.failure_messages = failureMessages;
                if options.WriteBatchSummary
                    write_json_report(char(summaryJsonPath), partial);
                    write_markdown_report(char(summaryMdPath), build_batch_markdown(partial));
                end
                rethrow(err);
            end
        end
    end

    result = build_batch_result(entries, targets, options, rootDir, summaryJsonPath, summaryMdPath);
    result.failure_messages = failureMessages;

    if options.WriteBatchSummary
        write_json_report(char(summaryJsonPath), result);
        write_markdown_report(char(summaryMdPath), build_batch_markdown(result));
        fprintf('Saved batch summary JSON: %s\n', summaryJsonPath);
        fprintf('Saved batch summary Markdown: %s\n', summaryMdPath);
    end

    if ~result.pass
        error('run_m1_real_reference_regression:BatchFailed', ...
            'M1 real reference batch failed for %d/%d targets.', result.fail_count, result.target_count);
    end

    fprintf('M1 real reference batch PASS\n');
end

function result = run_single_target_regression(paramsFileOrModule, options, rootDir)
    [params, sourceInfo] = load_qwen_parameters_adapter(paramsFileOrModule, rootDir, options);
    fprintf('Resolved real reference source: %s\n', sourceInfo);

    if ~isfield(params, 'Hyperparameters') || ~isfield(params.Hyperparameters, 'HiddenSize')
        error('run_m1_real_reference_regression:MissingHiddenSize', ...
            'Resolved parameters do not contain Hyperparameters.HiddenSize.');
    end

    hp = params.Hyperparameters;

    vecOpt = struct('HiddenSize', double(hp.HiddenSize), 'TokensPrefill', 4, 'DecodeKvLen', 3);

    refCtx = struct();
    refCtx.Parameters = params;
    refCtx.LayerIndex = options.LayerIndex;

    regOpt = struct();
    regOpt.ReferenceMode = "real_auto";
    regOpt.ReferenceContext = refCtx;
    regOpt.BaselineMode = string(options.BaselineMode);
    regOpt.EnableMemoryMetrics = options.EnableMemoryMetrics;

    reportDir = resolve_report_dir(rootDir, options);
    safeToken = regexprep(string(paramsFileOrModule), '[^a-zA-Z0-9_\-]+', '_');
    modeTag = ternary(options.EnableMemoryMetrics, "mem_on", "mem_off");
    reportTag = "m1_real_" + safeToken + "_" + lower(string(options.BaselineMode)) + "_" + modeTag;
    reportPath = fullfile(reportDir, reportTag + "_" + string(datetime('now', 'Format', 'yyyyMMdd_HHmmss')) + ".json");

    if options.RunStage2FastSmoke
        fprintf('Precheck: stage2 fast smoke suite\n');
        run_stage2_smoke_suite_fast(rootDir);
    end

    if options.RunStage2ReferenceReadinessAudit
        fprintf('Precheck: stage2 reference readiness audit\n');
        run_stage2_reference_readiness_audit(rootDir, struct('ParamsSource', paramsFileOrModule));
    end

    if options.RunStage2HardwareContractRegression
        fprintf('Precheck: stage2 hardware contract reference regression\n');
        run_stage2_hardware_contract_reference_regression(rootDir, struct('ParamsSource', paramsFileOrModule));
    end

    if options.RunStage2KvWriteContractRegression
        fprintf('Precheck: stage2 KV write contract reference regression\n');
        run_stage2_kv_write_contract_reference_regression(rootDir, struct('ParamsSource', paramsFileOrModule));
    end

    if options.RunStage2KvWriteMismatchAudit
        fprintf('Precheck: stage2 KV write mismatch audit\n');
        run_stage2_kv_write_mismatch_audit(rootDir, struct('ParamsSource', paramsFileOrModule));
    end

    minimalResult = run_m1_minimal_regression(struct( ...
        'VectorOptions', vecOpt, ...
        'RegressionOptions', regOpt, ...
        'ReportPath', reportPath, ...
        'ReportDir', reportDir, ...
        'ReportTag', reportTag));

    result = minimalResult;
    result.target = string(paramsFileOrModule);
    result.source_info = string(sourceInfo);
    result.report_path = string(reportPath);
    result.report_tag = string(reportTag);
    result.baseline_mode = string(options.BaselineMode);
    result.enable_memory_metrics = options.EnableMemoryMetrics;
end

function run_shared_batch_prechecks(rootDir, options)
    if options.RunStage2FastSmoke
        fprintf('Batch precheck: stage2 fast smoke suite\n');
        run_stage2_smoke_suite_fast(rootDir);
    end
end

function targets = resolve_requested_targets(paramsFileOrModule, options, rootDir)
    if ~isempty(options.Modules)
        rawTargets = normalize_target_list(options.Modules);
    else
        rawTargets = normalize_target_list(paramsFileOrModule);
    end

    if isempty(rawTargets)
        error('run_m1_real_reference_regression:NoTargets', 'No params/module targets were provided.');
    end

    expanded = strings(0, 1);
    moduleNames = string(keys(localModuleMap(rootDir)));
    moduleNames = sort(moduleNames);
    for i = 1:numel(rawTargets)
        token = string(rawTargets(i));
        lowerToken = lower(strtrim(token));
        if any(lowerToken == ["all", "all_modules", "modules_all"])
            expanded = [expanded; reshape(moduleNames, [], 1)]; %#ok<AGROW>
        else
            expanded(end + 1, 1) = token; %#ok<AGROW>
        end
    end

    targets = unique(expanded, 'stable');
end

function targets = normalize_target_list(raw)
    if isstring(raw)
        targets = reshape(raw, [], 1);
        return;
    end

    if ischar(raw)
        targets = string({raw});
        return;
    end

    if iscell(raw)
        targets = strings(0, 1);
        for i = 1:numel(raw)
            item = raw{i};
            if ischar(item) || (isstring(item) && isscalar(item))
                targets(end + 1, 1) = string(item); %#ok<AGROW>
            else
                error('run_m1_real_reference_regression:BadTargetType', ...
                    'Unsupported target type in cell array at index %d.', i);
            end
        end
        return;
    end

    error('run_m1_real_reference_regression:BadTargets', ...
        'paramsFileOrModule must be char, string, string array, or cell array of chars/strings.');
end

function reportDir = resolve_report_dir(rootDir, options)
    reportDir = string(options.ReportDir);
    if strlength(reportDir) == 0
        reportDir = fullfile(rootDir, 'verification', 'reports');
    end
end

function tag = build_batch_report_tag(options)
    modeTag = ternary(options.EnableMemoryMetrics, "mem_on", "mem_off");
    tag = "m1_real_batch_" + lower(string(options.BaselineMode)) + "_" + modeTag;
end

function result = build_batch_result(entries, targets, options, rootDir, summaryJsonPath, summaryMdPath)
    passMask = arrayfun(@(entry) logical(entry.pass), entries);
    failedTargets = string.empty(0, 1);
    if any(~passMask)
        failedTargets = reshape(string({entries(~passMask).target}), [], 1);
    end

    result = struct();
    result.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    result.rootDir = string(rootDir);
    result.targets = reshape(string(targets), [], 1);
    result.target_count = numel(entries);
    result.pass_count = nnz(passMask);
    result.fail_count = numel(entries) - nnz(passMask);
    result.pass = all(passMask);
    result.baseline_mode = string(options.BaselineMode);
    result.enable_memory_metrics = options.EnableMemoryMetrics;
    result.entries = entries;
    result.failed_targets = failedTargets;
    result.summary_json_path = string(summaryJsonPath);
    result.summary_markdown_path = string(summaryMdPath);
end

function entry = empty_batch_entry()
    entry = struct( ...
        'target', "", ...
        'pass', false, ...
        'report_path', "", ...
        'source_info', "", ...
        'memory_gate_pass', false, ...
        'block_fail', NaN, ...
        'error_id', "", ...
        'error_message', "");
end

function entry = summarize_batch_entry(result, target)
    entry = empty_batch_entry();
    entry.target = string(target);
    entry.pass = logical(getFieldOr(result, 'pass', false));
    entry.report_path = string(getFieldOr(result, 'report_path', ""));
    entry.source_info = string(getFieldOr(result, 'source_info', ""));
    entry.memory_gate_pass = logical(getFieldOr(getFieldOr(result, 'memory_gate', struct('pass', false)), 'pass', false));
    entry.block_fail = double(getFieldOr(getFieldOr(result, 'block', struct('fail', NaN)), 'fail', NaN));
end

function entry = summarize_batch_error(err, target)
    entry = empty_batch_entry();
    entry.target = string(target);
    entry.pass = false;
    entry.error_id = string(err.identifier);
    entry.error_message = string(err.message);
end

function markdown = build_batch_markdown(result)
    lines = strings(0, 1);
    lines(end + 1, 1) = "# M1 真实参考批量回归汇总"; %#ok<AGROW>
    lines(end + 1, 1) = ""; %#ok<AGROW>
    lines(end + 1, 1) = "- 时间：" + result.timestamp; %#ok<AGROW>
    lines(end + 1, 1) = "- 基线模式：" + result.baseline_mode; %#ok<AGROW>
    lines(end + 1, 1) = "- 访存门禁：" + ternary(result.enable_memory_metrics, "启用", "未启用"); %#ok<AGROW>
    lines(end + 1, 1) = "- 总结果：" + ternary(result.pass, "PASS", "FAIL"); %#ok<AGROW>
    lines(end + 1, 1) = "- 通过/失败：" + string(result.pass_count) + "/" + string(result.fail_count); %#ok<AGROW>
    lines(end + 1, 1) = ""; %#ok<AGROW>
    lines(end + 1, 1) = "| 目标 | 结果 | Block Fail | Memory Gate | 报告 |"; %#ok<AGROW>
    lines(end + 1, 1) = "| --- | --- | --- | --- | --- |"; %#ok<AGROW>
    for i = 1:numel(result.entries)
        entry = result.entries(i);
        reportText = entry.report_path;
        if strlength(reportText) == 0
            reportText = "-";
        end
        lines(end + 1, 1) = "| " + entry.target + " | " + ternary(entry.pass, "PASS", "FAIL") + ...
            " | " + local_numeric_string(entry.block_fail) + " | " + ternary(entry.memory_gate_pass, "PASS", "FAIL") + ...
            " | " + reportText + " |"; %#ok<AGROW>
        if strlength(entry.error_message) > 0
            lines(end + 1, 1) = ""; %#ok<AGROW>
            lines(end + 1, 1) = "失败说明（" + entry.target + "）：" + entry.error_message; %#ok<AGROW>
        end
    end

    markdown = strjoin(cellstr(lines), newline);
end

function out = local_numeric_string(value)
    if isnan(value)
        out = "-";
    else
        out = string(value);
    end
end

function out = ternary(cond, whenTrue, whenFalse)
    if cond
        out = whenTrue;
    else
        out = whenFalse;
    end
end

function [params, sourceInfo] = load_qwen_parameters_adapter(paramsFileOrModule, rootDir, options)
    token = string(paramsFileOrModule);

    moduleMap = localModuleMap(rootDir);
    if isKey(moduleMap, lower(token))
        token = string(moduleMap(lower(token)));
    end

    if endsWith(lower(token), ".mat") && exist(token, 'file') == 2
        [params, sourceInfo] = load_from_mat_file(char(token));
        return;
    end

    if exist(token, 'dir') == 7
        [params, sourceInfo] = load_from_model_dir(char(token), rootDir, options);
        return;
    end

    error('run_m1_real_reference_regression:InputNotFound', ...
        'Input not found as .mat file or directory: %s', char(token));
end

function [params, sourceInfo] = load_from_mat_file(paramsFile)
    raw = load(paramsFile);

    if isfield(raw, 'Hyperparameters') && isfield(raw, 'Weights')
        params = raw;
        sourceInfo = sprintf('hierarchical mat: %s', paramsFile);
        return;
    end

    if symbol_exists('qwen2.load')
        params = qwen2.load(paramsFile);
        sourceInfo = sprintf('qwen2.load from mat: %s', paramsFile);
        return;
    end

    if symbol_exists('qwen2_quant.load_hf_quant_matlab')
        params = qwen2_quant.load_hf_quant_matlab(paramsFile);
        sourceInfo = sprintf('qwen2_quant.load_hf_quant_matlab from mat: %s', paramsFile);
        return;
    end

    error('run_m1_real_reference_regression:UnsupportedParams', ...
        ['Cannot parse parameters file. Provide hierarchical struct or add +qwen2/+qwen2_quant ' ...
         'to MATLAB path.']);
end

function [params, sourceInfo] = load_from_model_dir(modelDir, rootDir, options)
    ensure_external_dependency_paths(modelDir);

    nameLower = lower(string(modelDir));

    if contains(nameLower, 'awq') || contains(nameLower, 'gptq')
        [params, sourceInfo] = load_from_hf_quant_dir(modelDir, rootDir);
        return;
    end

    if symbol_exists('qwen2_quant.load_gguf')
        ggufPath = pick_first_gguf(modelDir);
        if strlength(ggufPath) > 0
            params = qwen2_quant.load_gguf(ggufPath, 'DequantizeNow', options.PreferDequantizeNow, 'Verbose', false);
            sourceInfo = sprintf('qwen2_quant.load_gguf from: %s', ggufPath);
            return;
        end
    end

    error('run_m1_real_reference_regression:UnsupportedModelDir', ...
        ['Unsupported model dir or missing loader for: %s. ' ...
         'Expected AWQ/GPTQ folder or GGUF folder with *.gguf files.'], modelDir);
end

function [params, sourceInfo] = load_from_hf_quant_dir(modelDir, rootDir)
    if ~symbol_exists('qwen2_quant.load_hf_quant_matlab')
        error('run_m1_real_reference_regression:MissingLoader', ...
            'qwen2_quant.load_hf_quant_matlab is required for AWQ/GPTQ model dirs.');
    end

    cacheDir = fullfile(rootDir, 'matlab_ref', 'cache');
    if ~exist(cacheDir, 'dir')
        mkdir(cacheDir);
    end

    [~, leaf] = fileparts(modelDir);
    matOut = fullfile(cacheDir, [leaf '_params.mat']);

    if exist(matOut, 'file') ~= 2
        if symbol_exists('qwen2_quant.prepare_hf_quant_matlab')
            ensure_transformer_models_python_shim(rootDir);
            qwen2_quant.prepare_hf_quant_matlab(string(modelDir), string(matOut), ...
                'LocalFilesOnly', true, 'TrustRemoteCode', true, 'HFEndpoint', "https://hf-mirror.com");
        else
            error('run_m1_real_reference_regression:MissingExporter', ...
                'Missing qwen2_quant.prepare_hf_quant_matlab and no cached mat found: %s', matOut);
        end
    end

    params = qwen2_quant.load_hf_quant_matlab(matOut);
    sourceInfo = sprintf('hf quant model dir: %s (mat cache: %s)', modelDir, matOut);
end

function ggufPath = pick_first_gguf(modelDir)
    ggufPath = "";
    listing = dir(fullfile(modelDir, '*.gguf'));
    if ~isempty(listing)
        ggufPath = string(fullfile(modelDir, listing(1).name));
    end
end

function m = localModuleMap(rootDir)
    moduleRoot = fullfile(rootDir, 'matlab_ref', 'module');
    m = containers.Map('KeyType', 'char', 'ValueType', 'char');

    m('module_awq') = fullfile(moduleRoot, 'Qwen2.5-1.5B-Instruct-AWQ');
    m('module_gptq') = fullfile(moduleRoot, 'Qwen2.5-1.5B-Instruct-GPTQ-Int4');
    m('module_gguf') = fullfile(moduleRoot, 'qwen_gguf');
end

function ensure_external_dependency_paths(modelDir)
    repoRoot = detect_dependency_repo_root(modelDir);
    if strlength(repoRoot) > 0
        addpath(char(repoRoot));
    end
end

function ensure_local_transformer_models_on_path(rootDir)
    localRepo = fullfile(rootDir, 'matlab_ref', 'transformer-models');
    if exist(fullfile(localRepo, '+qwen2_quant'), 'dir') == 7
        addpath(localRepo);
    end
end

function tf = symbol_exists(name)
    tf = ~isempty(which(char(name)));
end

function ensure_transformer_models_python_shim(rootDir)
    repoRoot = fullfile(rootDir, 'matlab_ref', 'transformer-models');
    venvPy = fullfile(repoRoot, '.venv', 'bin', 'python');
    if exist(venvPy, 'file') == 2
        return;
    end

    [statusPy3, py3Path] = system('command -v python3');
    if statusPy3 ~= 0
        [statusPy, pyPath] = system('command -v python');
        if statusPy ~= 0
            return;
        end
        resolved = strtrim(pyPath);
    else
        resolved = strtrim(py3Path);
    end

    binDir = fullfile(repoRoot, '.venv', 'bin');
    if ~exist(binDir, 'dir')
        mkdir(binDir);
    end

    cmd = sprintf('ln -sf "%s" "%s"', resolved, venvPy);
    system(cmd);
end

function repoRoot = detect_dependency_repo_root(modelDir)
    repoRoot = "";

    canonical = string(modelDir);
    try
        canonical = string(char(java.io.File(modelDir).getCanonicalPath()));
    catch
        % Keep input path if canonical path is unavailable.
    end

    current = canonical;
    for i = 1:8
        if exist(fullfile(current, '+qwen2'), 'dir') == 7 || ...
           exist(fullfile(current, '+qwen2_quant'), 'dir') == 7 || ...
           exist(fullfile(current, '+transformer'), 'dir') == 7
            repoRoot = current;
            return;
        end

        parent = string(fileparts(char(current)));
        if strlength(parent) == 0 || parent == current
            break;
        end
        current = parent;
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function write_json_report(path, data)
    outDir = fileparts(path);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    jsonText = jsonencode(data, 'PrettyPrint', true);
    fid = fopen(path, 'w');
    assert(fid > 0, 'Failed to open report file for write: %s', path);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonText);
end

function write_markdown_report(path, text)
    outDir = fileparts(path);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    fid = fopen(path, 'w');
    assert(fid > 0, 'Failed to open markdown report for write: %s', path);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', text);
end
