function result = run_m1_minimal_regression(options)
%RUN_M1_MINIMAL_REGRESSION End-to-end minimal M1 verification entry.

    if nargin < 1 || ~isstruct(options)
        options = struct();
    end

    rootDir = fileparts(fileparts(mfilename('fullpath')));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    regOpt = getFieldOr(options, 'RegressionOptions', struct());
    enableMemoryMetrics = logical(getFieldOr(regOpt, 'EnableMemoryMetrics', false));

    totalSteps = 3 + double(enableMemoryMetrics);

    fprintf('Step 1/%d: bootstrap vectors\n', totalSteps);
    vecOpt = getFieldOr(options, 'VectorOptions', struct());
    bootstrap_m1_vectors(rootDir, vecOpt);

    fprintf('Step 2/%d: operator smoke\n', totalSteps);
    opReport = run_operator_smoke_test();

    fprintf('Step 3/%d: block regression\n', totalSteps);
    blockSummary = run_block_regression(rootDir, regOpt);

    if enableMemoryMetrics
        memOpt = getFieldOr(options, 'MemoryModelOptions', struct());
        memoryMetrics = run_memory_model_check(rootDir, memOpt);
    else
        memoryMetrics = getFieldOr(blockSummary, 'memory_metrics', struct());
    end

    memoryGate = evaluate_memory_gate(memoryMetrics, options, enableMemoryMetrics);

    if enableMemoryMetrics
        fprintf('Step 4/%d: memory-model-check\n', totalSteps);
        fprintf('Memory gate: %s (%s)\n', ternary(memoryGate.pass, 'PASS', 'FAIL'), memoryGate.reason);
    end

    result = struct();
    result.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    result.rootDir = string(rootDir);
    result.operator = opReport;
    result.block = blockSummary;
    result.memory_metrics = memoryMetrics;
    result.memory_gate = memoryGate;
    result.pass = blockSummary.fail == 0 && memoryGate.pass;

    reportPath = resolveReportPath(rootDir, options);
    if strlength(reportPath) > 0
        write_json_report(char(reportPath), result);
        fprintf('Saved regression report: %s\n', reportPath);
    end

    fprintf('M1 minimal regression PASS\n');
end

function gate = evaluate_memory_gate(memoryMetrics, options, enableMemoryMetrics)
    gate = struct();
    gate.enabled = enableMemoryMetrics;
    gate.pass = true;
    gate.reason = 'memory gate not enabled';
    gate.thresholds = resolve_memory_thresholds(options);

    if ~enableMemoryMetrics
        return;
    end

    if ~isstruct(memoryMetrics) || isempty(fieldnames(memoryMetrics)) || ...
       ~isfield(memoryMetrics, 'available') || ~memoryMetrics.available
        gate.pass = false;
        gate.reason = 'memory metrics unavailable while memory gate is enabled';
        return;
    end

    stallCount = getFieldOr(memoryMetrics, 'stall_count', NaN);
    droppedBurstCount = getFieldOr(memoryMetrics, 'dropped_burst_count', NaN);
    masterBw = getFieldOr(memoryMetrics, 'master_bw_mb_s', struct());
    minBw = gate.thresholds.memory_bw_min_mb_s;

    if isstruct(masterBw)
        names = fieldnames(masterBw);
        for i = 1:numel(names)
            if masterBw.(names{i}) < minBw
                gate.pass = false;
                gate.reason = sprintf('master %s bandwidth %.3f < min %.3f MB/s', ...
                    names{i}, masterBw.(names{i}), minBw);
                return;
            end
        end
    end

    if ~isnan(stallCount) && stallCount > gate.thresholds.memory_stall_max_count
        gate.pass = false;
        gate.reason = sprintf('stall_count %g > max %g', stallCount, gate.thresholds.memory_stall_max_count);
        return;
    end

    if ~isnan(droppedBurstCount) && droppedBurstCount > gate.thresholds.memory_dropped_burst_max_count
        gate.pass = false;
        gate.reason = sprintf('dropped_burst_count %g > max %g', ...
            droppedBurstCount, gate.thresholds.memory_dropped_burst_max_count);
        return;
    end

    gate.reason = 'memory metrics satisfy thresholds';
end

function thresholds = resolve_memory_thresholds(options)
    thresholds = struct( ...
        'memory_bw_min_mb_s', 200, ...
        'memory_stall_max_count', 0, ...
        'memory_dropped_burst_max_count', 0);

    custom = getFieldOr(options, 'MemoryGate', struct());
    if ~isstruct(custom)
        return;
    end

    fields = fieldnames(custom);
    for i = 1:numel(fields)
        thresholds.(fields{i}) = custom.(fields{i});
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
