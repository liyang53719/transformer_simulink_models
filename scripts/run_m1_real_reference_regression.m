function run_m1_real_reference_regression(paramsFileOrModule, options)
%RUN_M1_REAL_REFERENCE_REGRESSION Run M1 regression using real qwen2 block adapter.
%
% Example:
%   run_m1_real_reference_regression('path/to/qwen_params.mat')
%   run_m1_real_reference_regression('module_awq')

    arguments
        paramsFileOrModule (1,:) char
        options.LayerIndex (1,1) double = 1
        options.PreferDequantizeNow (1,1) logical = false
        options.BaselineMode (1,:) char = 'real'
        options.EnableMemoryMetrics (1,1) logical = false
        options.RunStage2FastSmoke (1,1) logical = false
        options.RunStage2ReferenceReadinessAudit (1,1) logical = false
        options.RunStage2HardwareContractRegression (1,1) logical = false
        options.RunStage2KvWriteContractRegression (1,1) logical = false
        options.ReportDir (1,:) char = ''
    end

    rootDir = fileparts(fileparts(mfilename('fullpath')));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));
    ensure_local_transformer_models_on_path(rootDir);

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

    reportDir = string(options.ReportDir);
    if strlength(reportDir) == 0
        reportDir = fullfile(rootDir, 'verification', 'reports');
    end

    safeToken = regexprep(string(paramsFileOrModule), '[^a-zA-Z0-9_\-]+', '_');
    modeTag = ternary(options.EnableMemoryMetrics, "mem_on", "mem_off");
    reportTag = "m1_real_" + safeToken + "_" + lower(string(options.BaselineMode)) + "_" + modeTag;

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

    run_m1_minimal_regression(struct( ...
        'VectorOptions', vecOpt, ...
        'RegressionOptions', regOpt, ...
        'ReportDir', reportDir, ...
        'ReportTag', reportTag));
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

function out = ternary(cond, whenTrue, whenFalse)
    if cond
        out = whenTrue;
    else
        out = whenFalse;
    end
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
