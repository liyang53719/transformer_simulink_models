function run_m1_real_reference_regression(paramsFile, options)
%RUN_M1_REAL_REFERENCE_REGRESSION Run M1 regression using real qwen2 block adapter.
%
% Example:
%   run_m1_real_reference_regression('path/to/qwen_params.mat')

    arguments
        paramsFile (1,:) char
        options.LayerIndex (1,1) double = 1
    end

    rootDir = fileparts(fileparts(mfilename('fullpath')));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    params = load_qwen_parameters_adapter(paramsFile);
    hp = params.Hyperparameters;

    vecOpt = struct('HiddenSize', double(hp.HiddenSize), 'TokensPrefill', 4, 'DecodeKvLen', 3);

    refCtx = struct();
    refCtx.Parameters = params;
    refCtx.LayerIndex = options.LayerIndex;

    regOpt = struct();
    regOpt.ReferenceMode = "real_auto";
    regOpt.ReferenceContext = refCtx;

    run_m1_minimal_regression(struct('VectorOptions', vecOpt, 'RegressionOptions', regOpt));
end

function params = load_qwen_parameters_adapter(paramsFile)
    raw = load(paramsFile);

    if isfield(raw, 'Hyperparameters') && isfield(raw, 'Weights')
        params = raw;
        return;
    end

    if exist('qwen2.load', 'file') == 2
        params = qwen2.load(paramsFile);
        return;
    end

    if exist('qwen2_quant.load_hf_quant_matlab', 'file') == 2
        params = qwen2_quant.load_hf_quant_matlab(paramsFile);
        return;
    end

    error('run_m1_real_reference_regression:UnsupportedParams', ...
        ['Cannot parse parameters file. Provide hierarchical struct or add +qwen2/+qwen2_quant ' ...
         'to MATLAB path.']);
end
