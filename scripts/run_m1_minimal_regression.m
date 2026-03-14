function run_m1_minimal_regression(options)
%RUN_M1_MINIMAL_REGRESSION End-to-end minimal M1 verification entry.

    if nargin < 1 || ~isstruct(options)
        options = struct();
    end

    rootDir = fileparts(fileparts(mfilename('fullpath')));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    fprintf('Step 1/3: bootstrap vectors\n');
    vecOpt = getFieldOr(options, 'VectorOptions', struct());
    bootstrap_m1_vectors(rootDir, vecOpt);

    fprintf('Step 2/3: operator smoke\n');
    run_operator_smoke_test();

    fprintf('Step 3/3: block regression\n');
    regOpt = getFieldOr(options, 'RegressionOptions', struct());
    run_block_regression(rootDir, regOpt);

    fprintf('M1 minimal regression PASS\n');
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
