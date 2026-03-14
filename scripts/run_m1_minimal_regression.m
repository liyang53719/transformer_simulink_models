function result = run_m1_minimal_regression(options)
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
    opReport = run_operator_smoke_test();

    fprintf('Step 3/3: block regression\n');
    regOpt = getFieldOr(options, 'RegressionOptions', struct());
    blockSummary = run_block_regression(rootDir, regOpt);

    result = struct();
    result.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    result.rootDir = string(rootDir);
    result.operator = opReport;
    result.block = blockSummary;
    result.pass = blockSummary.fail == 0;

    reportPath = resolveReportPath(rootDir, options);
    if strlength(reportPath) > 0
        write_json_report(char(reportPath), result);
        fprintf('Saved regression report: %s\n', reportPath);
    end

    fprintf('M1 minimal regression PASS\n');
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function reportPath = resolveReportPath(rootDir, options)
    reportPath = string(getFieldOr(options, 'ReportPath', ""));
    if strlength(reportPath) > 0
        return;
    end

    reportDir = string(getFieldOr(options, 'ReportDir', fullfile(rootDir, 'verification', 'reports')));
    if strlength(reportDir) == 0
        return;
    end

    reportTag = string(getFieldOr(options, 'ReportTag', "m1_minimal"));
    stamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    reportPath = fullfile(reportDir, reportTag + "_" + stamp + ".json");
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
