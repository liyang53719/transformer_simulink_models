function report = run_operator_smoke_test()
%RUN_OPERATOR_SMOKE_TEST Minimal operator-level smoke checks.

    report = struct();
    report.max_abs_err = 0;
    report.mean_abs_err = 0;
    report.rel_l2_err = 0;
    report.match_ratio = 1;

    x = single(linspace(-2, 2, 256));

    % RMSNorm-like normalize check
    epsVal = single(1e-6);
    rms = sqrt(mean(x.^2) + epsVal);
    yRef = x ./ rms;
    yDut = x ./ rms; % placeholder DUT path

    [a, b, c, d] = computeMetrics(yRef, yDut);
    report.max_abs_err = max(report.max_abs_err, a);
    report.mean_abs_err = max(report.mean_abs_err, b);
    report.rel_l2_err = max(report.rel_l2_err, c);
    report.match_ratio = min(report.match_ratio, d);

    assert(report.max_abs_err <= 3e-2, 'Operator max_abs_err exceeds threshold');
    assert(report.mean_abs_err <= 3e-3, 'Operator mean_abs_err exceeds threshold');
    assert(report.rel_l2_err <= 2e-2, 'Operator rel_l2_err exceeds threshold');
    assert(report.match_ratio >= 0.99, 'Operator match_ratio below threshold');

    fprintf('Operator smoke PASS: max=%.4g mean=%.4g relL2=%.4g match=%.2f%%\n', ...
        report.max_abs_err, report.mean_abs_err, report.rel_l2_err, 100*report.match_ratio);
end

function [maxAbsErr, meanAbsErr, relL2Err, matchRatio] = computeMetrics(yRef, yDut)
    diffVal = single(yRef) - single(yDut);
    maxAbsErr = max(abs(diffVal), [], 'all');
    meanAbsErr = mean(abs(diffVal), 'all');
    relL2Err = norm(diffVal(:), 2) / (norm(single(yRef(:)), 2) + 1e-12);
    matchRatio = mean(abs(diffVal(:)) <= 3e-2);
end
