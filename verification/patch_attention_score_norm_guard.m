function patch_attention_score_norm_guard(mdlName, epsilon)
%PATCH_ATTENTION_SCORE_NORM_GUARD Add a small denominator guard before attention score_norm.

    if nargin < 2 || isempty(epsilon)
        epsilon = 1e-6;
    end

    subPath = [char(mdlName) '/attention_u'];
    scoreNormPath = [subPath '/score_norm'];
    rowSumPath = [subPath '/row_sum_accum'];
    guardPath = [subPath '/row_sum_guard'];
    epsPath = [subPath '/row_sum_guard_eps'];

    if getSimulinkBlockHandle(scoreNormPath) == -1 || getSimulinkBlockHandle(rowSumPath) == -1
        error('patch_attention_score_norm_guard:MissingBlocks', ...
            'Required attention blocks not found under %s', subPath);
    end

    if getSimulinkBlockHandle(guardPath) == -1
        add_block('simulink/Math Operations/Add', guardPath, ...
            'Inputs', '++', 'Position', [540, 25, 575, 55]);
    end
    if getSimulinkBlockHandle(epsPath) == -1
        add_block('simulink/Sources/Constant', epsPath, ...
            'Value', num2str(double(epsilon), '%.17g'), ...
            'Position', [500, 5, 530, 25]);
    else
        set_param(epsPath, 'Value', num2str(double(epsilon), '%.17g'));
    end

    try
        delete_line(subPath, 'row_sum_accum/1', 'score_norm/2');
    catch
    end
    try
        delete_line(subPath, 'row_sum_accum/1', 'row_sum_guard/1');
    catch
    end
    try
        delete_line(subPath, 'row_sum_guard_eps/1', 'row_sum_guard/2');
    catch
    end
    try
        delete_line(subPath, 'row_sum_guard/1', 'score_norm/2');
    catch
    end

    add_line(subPath, 'row_sum_accum/1', 'row_sum_guard/1', 'autorouting', 'on');
    add_line(subPath, 'row_sum_guard_eps/1', 'row_sum_guard/2', 'autorouting', 'on');
    add_line(subPath, 'row_sum_guard/1', 'score_norm/2', 'autorouting', 'on');
end