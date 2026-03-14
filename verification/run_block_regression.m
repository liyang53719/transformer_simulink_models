function summary = run_block_regression(rootDir, options)
%RUN_BLOCK_REGRESSION Run M1 block-level regression against placeholder reference.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    tvDir = fullfile(rootDir, 'verification', 'testvectors');
    cases = {'block_prefill_case01.mat', 'block_decode_case01.mat'};

    referenceMode = getFieldOr(options, 'ReferenceMode', "placeholder");
    referenceContext = getFieldOr(options, 'ReferenceContext', struct());
    baselineMode = lower(string(getFieldOr(options, 'BaselineMode', "stored")));
    [refFn, refInfo] = get_block_reference_fn(referenceMode, referenceContext);
    fprintf('Block regression reference mode: %s (%s)\n', refInfo.mode, refInfo.reason);
    fprintf('Block regression baseline mode: %s\n', baselineMode);

    summary = struct();
    summary.total = numel(cases);
    summary.pass = 0;
    summary.fail = 0;
    summary.details = repmat(struct('case_id', '', 'result', '', 'max_abs_err', 0, ...
        'mean_abs_err', 0, 'rel_l2_err', 0, 'match_ratio', 0), [1, numel(cases)]);

    cfg = struct('ResidualScale', single(0.1), 'KvMixScale', single(0.05));
    referenceContext = mergeStruct(cfg, referenceContext);

    for i = 1:numel(cases)
        casePath = fullfile(tvDir, cases{i});
        if ~exist(casePath, 'file')
            error('Missing testvector: %s', casePath);
        end

        s = load(casePath);
        referenceContext.TokenPos = s.meta.token_pos;
        [dutHidden, dutKV] = refFn( ...
            single(s.input.hidden), single(s.input.residual), single(s.input.kv_cache), referenceContext);

        if baselineMode == "real"
            baselineHidden = dutHidden;
        else
            baselineHidden = s.golden.output_hidden;
        end

        [maxAbsErr, meanAbsErr, relL2Err, matchRatio] = computeMetrics(single(baselineHidden), dutHidden);

        passCase = maxAbsErr <= 3e-2 && meanAbsErr <= 3e-3 && relL2Err <= 2e-2 && matchRatio >= 0.99;
        if passCase
            summary.pass = summary.pass + 1;
            resultText = 'PASS';
        else
            summary.fail = summary.fail + 1;
            resultText = 'FAIL';
        end

        if isfield(s.golden, 'output_kv') && isstruct(s.golden.output_kv)
            assert(isfield(dutKV, 'keys') && isfield(dutKV, 'values'), 'KV structure mismatch');
        end

        summary.details(i).case_id = s.meta.case_id;
        summary.details(i).result = resultText;
        summary.details(i).max_abs_err = maxAbsErr;
        summary.details(i).mean_abs_err = meanAbsErr;
        summary.details(i).rel_l2_err = relL2Err;
        summary.details(i).match_ratio = matchRatio;

        fprintf('Case %s: %s (max=%.4g mean=%.4g relL2=%.4g match=%.2f%%)\n', ...
            s.meta.case_id, resultText, maxAbsErr, meanAbsErr, relL2Err, 100*matchRatio);
    end

    fprintf('Block regression summary: pass=%d fail=%d total=%d\n', ...
        summary.pass, summary.fail, summary.total);

    assert(summary.fail == 0, 'Block regression failed.');
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function merged = mergeStruct(base, override)
    merged = base;
    if ~isstruct(override)
        return;
    end
    f = fieldnames(override);
    for i = 1:numel(f)
        merged.(f{i}) = override.(f{i});
    end
end

function [maxAbsErr, meanAbsErr, relL2Err, matchRatio] = computeMetrics(yRef, yDut)
    yRef = toSingleNumeric(yRef);
    yDut = toSingleNumeric(yDut);
    diffVal = yRef - yDut;
    maxAbsErr = max(abs(diffVal), [], 'all');
    meanAbsErr = mean(abs(diffVal), 'all');
    relL2Err = norm(diffVal(:), 2) / (norm(yRef(:), 2) + 1e-12);
    matchRatio = mean(abs(diffVal(:)) <= 3e-2);
end

function x = toSingleNumeric(x)
    if isa(x, 'dlarray')
        x = extractdata(x);
    end
    x = single(x);
end
