function implement_stage1_rmsnorm_qkv(rootDir)
%IMPLEMENT_STAGE1_RMSNORM_QKV Build runnable stage-1 internals for rmsnorm_u and qkv_proj_u.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s. Run create_qwen2_block_top_placeholder first.', mdlPath);
    end

    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    configure_rmsnorm([mdlName '/rmsnorm_u']);
    configure_qkv_proj([mdlName '/qkv_proj_u']);

    % Rebuild stage-1 top data/control pass-through lines (safe add).
    safe_add_line(mdlName, 'in_hidden/1', 'rmsnorm_u/1');
    safe_add_line(mdlName, 'cfg_eps/1', 'rmsnorm_u/2');
    safe_add_line(mdlName, 'rmsnorm_u/1', 'qkv_proj_u/1');
    safe_add_line(mdlName, 'qkv_proj_u/1', 'out_hidden/1');

    safe_add_line(mdlName, 'in_valid/1', 'out_valid/1');
    safe_add_line(mdlName, 'out_ready/1', 'in_ready/1');
    safe_add_line(mdlName, 'kv_cache_rd_data/1', 'kv_cache_wr_data/1');
    safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'kv_cache_wr_en/1');
    safe_add_line(mdlName, 'eos_in/1', 'eos_out/1');

    save_system(mdlName, mdlPath);
    close_system(mdlName, 0);

    fprintf('Implemented stage-1 runnable internals for rmsnorm_u and qkv_proj_u in %s\n', mdlPath);
end

function configure_rmsnorm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 40, 60, 54]);
    add_block('simulink/Sources/In1', [subPath '/eps_in'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Math Operations/Gain', [subPath '/x_scale'], ...
        'Gain', '1.0', 'Position', [110, 35, 170, 60]);
    add_block('simulink/Math Operations/Gain', [subPath '/eps_scale'], ...
        'Gain', '0.0', 'Position', [110, 105, 170, 130]);
    add_block('simulink/Math Operations/Sum', [subPath '/sum_out'], ...
        'Inputs', '++', 'Position', [220, 55, 250, 105]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [300, 75, 330, 89]);

    safe_add_line(subPath, 'x_in/1', 'x_scale/1');
    safe_add_line(subPath, 'eps_in/1', 'eps_scale/1');
    safe_add_line(subPath, 'x_scale/1', 'sum_out/1');
    safe_add_line(subPath, 'eps_scale/1', 'sum_out/2');
    safe_add_line(subPath, 'sum_out/1', 'y_out/1');
end

function configure_qkv_proj(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Math Operations/Gain', [subPath '/q_gain'], ...
        'Gain', '0.6', 'Position', [110, 35, 170, 60]);
    add_block('simulink/Math Operations/Gain', [subPath '/k_gain'], ...
        'Gain', '0.4', 'Position', [110, 105, 170, 130]);
    add_block('simulink/Math Operations/Sum', [subPath '/mix_sum'], ...
        'Inputs', '++', 'Position', [220, 55, 250, 105]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [300, 75, 330, 89]);

    safe_add_line(subPath, 'x_in/1', 'q_gain/1');
    safe_add_line(subPath, 'x_in/1', 'k_gain/1');
    safe_add_line(subPath, 'q_gain/1', 'mix_sum/1');
    safe_add_line(subPath, 'k_gain/1', 'mix_sum/2');
    safe_add_line(subPath, 'mix_sum/1', 'y_out/1');
end

function clear_subsystem_contents(subPath)
    lines = find_system(subPath, 'FindAll', 'on', 'Type', 'line');
    for i = 1:numel(lines)
        delete_line(lines(i));
    end

    blocks = find_system(subPath, 'SearchDepth', 1, 'Type', 'Block');
    for i = 1:numel(blocks)
        if strcmp(blocks{i}, subPath)
            continue;
        end
        delete_block(blocks{i});
    end
end

function safe_add_line(sys, src, dst)
    try
        add_line(sys, src, dst, 'autorouting', 'on');
    catch
        % Ignore duplicate or auto-route conflicts during incremental rebuild.
    end
end
