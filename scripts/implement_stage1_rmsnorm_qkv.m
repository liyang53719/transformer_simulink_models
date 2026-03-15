function implement_stage1_rmsnorm_qkv(rootDir, options)
%IMPLEMENT_STAGE1_RMSNORM_QKV Build staged internals for qwen2_block_top.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    stageProfile = lower(string(getFieldOr(options, 'StageProfile', 'stage1')));

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s. Run create_qwen2_block_top_placeholder first.', mdlPath);
    end

    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    configure_rmsnorm([mdlName '/rmsnorm_u']);
    configure_qkv_proj([mdlName '/qkv_proj_u']);

    if stageProfile == "stage2_memory_ready"
        ensure_stage2_ports(mdlName);
        ensure_memory_subsystems(mdlName);
        configure_axi_master_rd([mdlName '/axi_master_rd_u']);
        configure_axi_master_wr([mdlName '/axi_master_wr_u']);
        configure_ddr_model_if([mdlName '/ddr_model_if_u']);
    end

    build_prefill_path(mdlName);
    build_decode_path(mdlName, stageProfile);
    build_kv_memory_stubs(mdlName, stageProfile);

    save_system(mdlName, mdlPath);
    close_system(mdlName, 0);

    fprintf('Implemented %s internals for rmsnorm_u and qkv_proj_u in %s\n', stageProfile, mdlPath);
end

function build_prefill_path(mdlName)
    safe_add_line(mdlName, 'in_hidden/1', 'rmsnorm_u/1');
    safe_add_line(mdlName, 'cfg_eps/1', 'rmsnorm_u/2');
    safe_add_line(mdlName, 'rmsnorm_u/1', 'qkv_proj_u/1');
    safe_add_line(mdlName, 'qkv_proj_u/1', 'out_hidden/1');

    safe_add_line(mdlName, 'in_valid/1', 'out_valid/1');
    safe_add_line(mdlName, 'out_ready/1', 'in_ready/1');
end

function build_decode_path(mdlName, stageProfile)
    if stageProfile == "stage2_memory_ready"
        safe_add_line(mdlName, 'mode_decode/1', 'ctrl_fsm_u/1');
        safe_add_line(mdlName, 'start/1', 'ctrl_fsm_u/2');

        % Read path hooks (adapted from soc_image_rotation AXI4MasterReadController).
        safe_add_line(mdlName, 'kv_cache_rd_data/1', 'axi_master_rd_u/1');
        safe_add_line(mdlName, 'kv_mem_rd_ready/1', 'axi_master_rd_u/2');
        safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'axi_master_rd_u/3');
        safe_add_line(mdlName, 'start/1', 'axi_master_rd_u/4');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'axi_master_rd_u/5');
        safe_add_line(mdlName, 'axi_master_rd_u/3', 'kv_mem_rd_addr/1');
        safe_add_line(mdlName, 'axi_master_rd_u/4', 'kv_mem_rd_len/1');
        safe_add_line(mdlName, 'axi_master_rd_u/5', 'kv_mem_rd_valid/1');
    end
end

function build_kv_memory_stubs(mdlName, stageProfile)
    safe_add_line(mdlName, 'kv_cache_rd_data/1', 'kv_cache_wr_data/1');
    safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'kv_cache_wr_en/1');
    safe_add_line(mdlName, 'eos_in/1', 'eos_out/1');

    if stageProfile == "stage2_memory_ready"
        % Write path hooks (adapted from soc_image_rotation AXI4MasterWriteController).
        safe_add_line(mdlName, 'kv_cache_wr_data/1', 'axi_master_wr_u/1');
        safe_add_line(mdlName, 'kv_cache_wr_en/1', 'axi_master_wr_u/2');
        safe_add_line(mdlName, 'done/1', 'axi_master_wr_u/3');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'axi_master_wr_u/4');
        safe_add_line(mdlName, 'axi_master_wr_u/2', 'kv_mem_wr_addr/1');
        safe_add_line(mdlName, 'axi_master_wr_u/3', 'kv_mem_wr_len/1');
        safe_add_line(mdlName, 'axi_master_wr_u/4', 'kv_mem_wr_valid/1');

        % Minimal task-level control placeholders for V2 ports.
        safe_add_line(mdlName, 'in_valid/1', 'busy/1');
        safe_add_line(mdlName, 'done/1', 'irq/1');
        safe_add_line(mdlName, 'cfg_token_pos/1', 'error_code/1');
        safe_add_line(mdlName, 'stop_req/1', 'ctrl_fsm_u/3');

        % Feed DDR model interface counter block (3rd fcn-style subsystem).
        safe_add_line(mdlName, 'kv_mem_rd_valid/1', 'ddr_model_if_u/1');
        safe_add_line(mdlName, 'kv_mem_wr_valid/1', 'ddr_model_if_u/2');
        safe_add_line(mdlName, 'kv_mem_rd_ready/1', 'ddr_model_if_u/3');
        safe_add_line(mdlName, 'kv_mem_wr_ready/1', 'ddr_model_if_u/4');
    end
end

function ensure_stage2_ports(mdlName)
    ensure_inport(mdlName, 'stop_req', [20, 520, 50, 534]);
    ensure_inport(mdlName, 'kv_mem_rd_ready', [20, 560, 50, 574]);
    ensure_inport(mdlName, 'kv_mem_wr_ready', [20, 600, 50, 614]);

    ensure_outport(mdlName, 'busy', [1650, 520, 1680, 534]);
    ensure_outport(mdlName, 'irq', [1650, 560, 1680, 574]);
    ensure_outport(mdlName, 'error_code', [1650, 600, 1680, 614]);
    ensure_outport(mdlName, 'kv_mem_rd_addr', [1650, 640, 1680, 654]);
    ensure_outport(mdlName, 'kv_mem_rd_len', [1650, 680, 1680, 694]);
    ensure_outport(mdlName, 'kv_mem_rd_valid', [1650, 720, 1680, 734]);
    ensure_outport(mdlName, 'kv_mem_wr_addr', [1650, 760, 1680, 774]);
    ensure_outport(mdlName, 'kv_mem_wr_len', [1650, 800, 1680, 814]);
    ensure_outport(mdlName, 'kv_mem_wr_valid', [1650, 840, 1680, 854]);
end

function ensure_memory_subsystems(mdlName)
    ensure_subsystem(mdlName, 'axi_master_rd_u', [980, 430, 1220, 510]);
    ensure_subsystem(mdlName, 'axi_master_wr_u', [980, 540, 1220, 620]);
    ensure_subsystem(mdlName, 'ddr_model_if_u', [980, 650, 1220, 730]);
end

function ensure_subsystem(mdlName, name, pos)
    path = [mdlName '/' name];
    if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Ports & Subsystems/Subsystem', path, 'Position', pos);
    else
        set_param(path, 'Position', pos);
    end
end

function ensure_inport(mdlName, name, pos)
    if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Sources/In1', [mdlName '/' name], 'Position', pos);
    end
end

function ensure_outport(mdlName, name, pos)
    if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Sinks/Out1', [mdlName '/' name], 'Position', pos);
    end
end

function configure_axi_master_rd(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/rd_data'], 'Position', [20, 30, 50, 44]);
    add_block('simulink/Sources/In1', [subPath '/rd_aready'], 'Position', [20, 70, 50, 84]);
    add_block('simulink/Sources/In1', [subPath '/rd_dvalid'], 'Position', [20, 110, 50, 124]);
    add_block('simulink/Sources/In1', [subPath '/start'], 'Position', [20, 150, 50, 164]);
    add_block('simulink/Sources/In1', [subPath '/imageWidth'], 'Position', [20, 190, 50, 204]);

    add_block('simulink/Signal Routing/Goto', [subPath '/_placeholder_comment'], ...
        'Position', [140, 20, 230, 35], 'GotoTag', 'AXI4MasterReadControllerStub');
    add_block('simulink/Math Operations/Gain', [subPath '/addr_gain'], ...
        'Gain', '2.0', 'Position', [120, 170, 180, 195]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/avalid_logic'], ...
        'Operator', 'AND', 'Position', [120, 100, 150, 130]);
    add_block('simulink/Sources/Constant', [subPath '/dready_const'], ...
        'Value', '1', 'Position', [120, 220, 160, 240]);

    add_block('simulink/Sinks/Out1', [subPath '/rd_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dvalid_out'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_avalid'], 'Position', [370, 190, 400, 204]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dready'], 'Position', [370, 230, 400, 244]);

    safe_add_line(subPath, 'rd_data/1', 'rd_data_out/1');
    safe_add_line(subPath, 'rd_dvalid/1', 'rd_dvalid_out/1');
    safe_add_line(subPath, 'imageWidth/1', 'addr_gain/1');
    safe_add_line(subPath, 'addr_gain/1', 'rd_addr/1');
    safe_add_line(subPath, 'imageWidth/1', 'rd_len/1');
    safe_add_line(subPath, 'start/1', 'avalid_logic/1');
    safe_add_line(subPath, 'rd_aready/1', 'avalid_logic/2');
    safe_add_line(subPath, 'avalid_logic/1', 'rd_avalid/1');
    safe_add_line(subPath, 'dready_const/1', 'rd_dready/1');
end

function configure_axi_master_wr(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/wr_data'], 'Position', [20, 30, 50, 44]);
    add_block('simulink/Sources/In1', [subPath '/wr_dvalid'], 'Position', [20, 70, 50, 84]);
    add_block('simulink/Sources/In1', [subPath '/wr_complete'], 'Position', [20, 110, 50, 124]);
    add_block('simulink/Sources/In1', [subPath '/imageWidth'], 'Position', [20, 150, 50, 164]);

    add_block('simulink/Signal Routing/Goto', [subPath '/_placeholder_comment'], ...
        'Position', [140, 20, 230, 35], 'GotoTag', 'AXI4MasterWriteControllerStub');
    add_block('simulink/Math Operations/Gain', [subPath '/addr_gain'], ...
        'Gain', '2.0', 'Position', [120, 120, 180, 145]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/next_line_logic'], ...
        'Operator', 'AND', 'Position', [120, 170, 150, 200]);

    add_block('simulink/Sinks/Out1', [subPath '/wr_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_valid'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/request_next_line'], 'Position', [370, 190, 400, 204]);

    safe_add_line(subPath, 'wr_data/1', 'wr_data_out/1');
    safe_add_line(subPath, 'imageWidth/1', 'addr_gain/1');
    safe_add_line(subPath, 'addr_gain/1', 'wr_addr/1');
    safe_add_line(subPath, 'imageWidth/1', 'wr_len/1');
    safe_add_line(subPath, 'wr_dvalid/1', 'wr_valid/1');
    safe_add_line(subPath, 'wr_complete/1', 'next_line_logic/1');
    safe_add_line(subPath, 'wr_dvalid/1', 'next_line_logic/2');
    safe_add_line(subPath, 'next_line_logic/1', 'request_next_line/1');
end

function configure_ddr_model_if(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/rd_valid'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/wr_valid'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/rd_ready'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/Sources/In1', [subPath '/wr_ready'], 'Position', [20, 160, 50, 174]);

    add_block('simulink/Math Operations/Add', [subPath '/stall_sum'], ...
        'Position', [120, 70, 160, 130]);
    add_block('simulink/Sources/Constant', [subPath '/drop_const'], ...
        'Value', '0', 'Position', [120, 140, 160, 160]);
    add_block('simulink/Math Operations/Add', [subPath '/accepted_sum'], ...
        'Position', [120, 170, 160, 230]);

    add_block('simulink/Sinks/Out1', [subPath '/stall_count'], 'Position', [370, 60, 400, 74]);
    add_block('simulink/Sinks/Out1', [subPath '/dropped_burst_count'], 'Position', [370, 100, 400, 114]);
    add_block('simulink/Sinks/Out1', [subPath '/accepted_beats'], 'Position', [370, 140, 400, 154]);

    safe_add_line(subPath, 'rd_valid/1', 'stall_sum/1');
    safe_add_line(subPath, 'wr_valid/1', 'stall_sum/2');
    safe_add_line(subPath, 'stall_sum/1', 'stall_count/1');
    safe_add_line(subPath, 'drop_const/1', 'dropped_burst_count/1');
    safe_add_line(subPath, 'rd_ready/1', 'accepted_sum/1');
    safe_add_line(subPath, 'wr_ready/1', 'accepted_sum/2');
    safe_add_line(subPath, 'accepted_sum/1', 'accepted_beats/1');
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
        try
            delete_line(lines(i));
        catch
            % Ignore stale/invalid line handles during rebuild.
        end
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

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
