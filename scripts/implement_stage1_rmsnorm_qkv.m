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

    [~, mdlName] = fileparts(mdlPath);
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
    load_system(mdlPath);

    configure_rmsnorm([mdlName '/rmsnorm_u']);
    configure_qkv_proj([mdlName '/qkv_proj_u']);
    configure_attention([mdlName '/attention_u']);
    configure_ffn_swiglu([mdlName '/ffn_swiglu_u']);
    configure_residual([mdlName '/residual_u']);
    configure_rope([mdlName '/rope_u']);

    if stageProfile == "stage2_memory_ready"
        addpath(fullfile(rootDir, 'simulink', 'fsm'));
        kvCfg = resolve_kv_addr_config(options);
        ensure_stage2_ports(mdlName);
        ensure_memory_subsystems(mdlName);
        configure_ctrl_fsm([mdlName '/ctrl_fsm_u']);
        configure_kv_cache_if([mdlName '/kv_cache_if_u']);
        configure_kv_addr_gen([mdlName '/kv_addr_gen_u'], kvCfg);
        configure_axi_master_rd([mdlName '/axi_master_rd_u']);
        configure_axi_master_wr([mdlName '/axi_master_wr_u']);
        configure_ddr_model_if([mdlName '/ddr_model_if_u']);
        configure_weight_addr_map([mdlName '/weight_addr_map_u']);
        configure_axi_weight_rd([mdlName '/axi_weight_rd_u']);
    end

    build_prefill_path(mdlName, stageProfile);
    build_decode_path(mdlName, stageProfile);
    build_kv_memory_stubs(mdlName, stageProfile);

    save_system(mdlName, mdlPath, 'OverwriteIfChangedOnDisk', true);
    close_system(mdlName, 0);

    fprintf('Implemented %s internals for rmsnorm_u and qkv_proj_u in %s\n', stageProfile, mdlPath);
end

function cfg = resolve_kv_addr_config(options)
    cfg = struct();
    if isfield(options, 'KvAddressConfig') && isstruct(options.KvAddressConfig)
        userCfg = options.KvAddressConfig;
    else
        userCfg = struct();
    end

    cfg.rd_base = getFieldOr(userCfg, 'rd_base', 0);
    cfg.wr_base = getFieldOr(userCfg, 'wr_base', 0);
    cfg.stride_bytes = getFieldOr(userCfg, 'stride_bytes', 2);
    cfg.decode_burst_len = getFieldOr(userCfg, 'decode_burst_len', 1);
end

function build_prefill_path(mdlName, stageProfile)
    safe_add_line(mdlName, 'in_hidden/1', 'rope_u/1');
    safe_add_line(mdlName, 'cfg_token_pos/1', 'rope_u/2');
    force_add_line(mdlName, 'rope_u/1', 'rmsnorm_u/1');
    safe_add_line(mdlName, 'cfg_eps/1', 'rmsnorm_u/2');
    safe_add_line(mdlName, 'rmsnorm_u/1', 'qkv_proj_u/1');

    if stageProfile == "stage2_memory_ready"
        safe_add_line(mdlName, 'cfg_token_pos/1', 'weight_addr_map_u/1');
        safe_add_line(mdlName, 'cfg_weight_num_heads/1', 'weight_addr_map_u/2');
        safe_add_line(mdlName, 'cfg_weight_page_base/1', 'weight_addr_map_u/3');
        safe_add_line(mdlName, 'cfg_weight_page_stride/1', 'weight_addr_map_u/4');

        safe_add_line(mdlName, 'weight_addr_map_u/1', 'rmsnorm_u/5');
        safe_add_line(mdlName, 'weight_addr_map_u/2', 'qkv_proj_u/8');
        safe_add_line(mdlName, 'weight_addr_map_u/3', 'qkv_proj_u/9');
        safe_add_line(mdlName, 'weight_addr_map_u/4', 'qkv_proj_u/10');
        safe_add_line(mdlName, 'weight_addr_map_u/5', 'attention_u/8');
        safe_add_line(mdlName, 'weight_addr_map_u/6', 'attention_u/9');
        safe_add_line(mdlName, 'weight_addr_map_u/7', 'attention_u/10');
        safe_add_line(mdlName, 'weight_addr_map_u/8', 'ffn_swiglu_u/6');
        safe_add_line(mdlName, 'weight_addr_map_u/9', 'ffn_swiglu_u/7');

        add_or_reset_mux(mdlName, 'w_req_mux', 18, [1320, 900, 1360, 1120]);
        for k = 1:9
            add_or_reset_dtc(mdlName, ['w_req_v' num2str(k) '_u8'], 'uint8', ...
                [1270, 900 + 20 * k, 1305, 915 + 20 * k]);
        end
        force_add_line(mdlName, 'rmsnorm_u/2', 'w_req_mux/1');
        force_add_line(mdlName, 'rmsnorm_u/3', 'w_req_v1_u8/1');
        force_add_line(mdlName, 'w_req_v1_u8/1', 'w_req_mux/2');
        force_add_line(mdlName, 'qkv_proj_u/2', 'w_req_mux/3');
        force_add_line(mdlName, 'qkv_proj_u/3', 'w_req_v2_u8/1');
        force_add_line(mdlName, 'w_req_v2_u8/1', 'w_req_mux/4');
        force_add_line(mdlName, 'qkv_proj_u/4', 'w_req_mux/5');
        force_add_line(mdlName, 'qkv_proj_u/5', 'w_req_v3_u8/1');
        force_add_line(mdlName, 'w_req_v3_u8/1', 'w_req_mux/6');
        force_add_line(mdlName, 'qkv_proj_u/6', 'w_req_mux/7');
        force_add_line(mdlName, 'qkv_proj_u/7', 'w_req_v4_u8/1');
        force_add_line(mdlName, 'w_req_v4_u8/1', 'w_req_mux/8');
        force_add_line(mdlName, 'attention_u/2', 'w_req_mux/9');
        force_add_line(mdlName, 'attention_u/3', 'w_req_v5_u8/1');
        force_add_line(mdlName, 'w_req_v5_u8/1', 'w_req_mux/10');
        force_add_line(mdlName, 'attention_u/4', 'w_req_mux/11');
        force_add_line(mdlName, 'attention_u/5', 'w_req_v6_u8/1');
        force_add_line(mdlName, 'w_req_v6_u8/1', 'w_req_mux/12');
        force_add_line(mdlName, 'attention_u/6', 'w_req_mux/13');
        force_add_line(mdlName, 'attention_u/7', 'w_req_v7_u8/1');
        force_add_line(mdlName, 'w_req_v7_u8/1', 'w_req_mux/14');
        force_add_line(mdlName, 'ffn_swiglu_u/2', 'w_req_mux/15');
        force_add_line(mdlName, 'ffn_swiglu_u/3', 'w_req_v8_u8/1');
        force_add_line(mdlName, 'w_req_v8_u8/1', 'w_req_mux/16');
        force_add_line(mdlName, 'ffn_swiglu_u/4', 'w_req_mux/17');
        force_add_line(mdlName, 'ffn_swiglu_u/5', 'w_req_v9_u8/1');
        force_add_line(mdlName, 'w_req_v9_u8/1', 'w_req_mux/18');
        force_add_line(mdlName, 'w_req_mux/1', 'axi_weight_rd_u/1');
        force_add_line(mdlName, 'w_req_mux/1', 'w_rd_req_bus/1');

        add_or_reset_demux(mdlName, 'w_rsp_demux', 18, [1220, 900, 1260, 1120]);
        force_add_line(mdlName, 'axi_weight_rd_u/1', 'w_rsp_demux/1');
        force_add_line(mdlName, 'w_rsp_demux/1', 'rmsnorm_u/3');
        force_add_line(mdlName, 'w_rsp_demux/2', 'rmsnorm_u/4');
        force_add_line(mdlName, 'w_rsp_demux/3', 'qkv_proj_u/2');
        force_add_line(mdlName, 'w_rsp_demux/4', 'qkv_proj_u/3');
        force_add_line(mdlName, 'w_rsp_demux/5', 'qkv_proj_u/4');
        force_add_line(mdlName, 'w_rsp_demux/6', 'qkv_proj_u/5');
        force_add_line(mdlName, 'w_rsp_demux/7', 'qkv_proj_u/6');
        force_add_line(mdlName, 'w_rsp_demux/8', 'qkv_proj_u/7');
        force_add_line(mdlName, 'w_rsp_demux/9', 'attention_u/2');
        force_add_line(mdlName, 'w_rsp_demux/10', 'attention_u/3');
        force_add_line(mdlName, 'w_rsp_demux/11', 'attention_u/4');
        force_add_line(mdlName, 'w_rsp_demux/12', 'attention_u/5');
        force_add_line(mdlName, 'w_rsp_demux/13', 'attention_u/6');
        force_add_line(mdlName, 'w_rsp_demux/14', 'attention_u/7');
        force_add_line(mdlName, 'w_rsp_demux/15', 'ffn_swiglu_u/2');
        force_add_line(mdlName, 'w_rsp_demux/16', 'ffn_swiglu_u/3');
        force_add_line(mdlName, 'w_rsp_demux/17', 'ffn_swiglu_u/4');
        force_add_line(mdlName, 'w_rsp_demux/18', 'ffn_swiglu_u/5');

        safe_add_line(mdlName, 'cfg_rope_theta_scale/1', 'rope_u/3');
        safe_add_line(mdlName, 'cfg_rope_sin_mix_scale/1', 'rope_u/4');

        safe_add_line(mdlName, 'qkv_proj_u/1', 'kv_cache_if_u/1');
        safe_add_line(mdlName, 'mode_decode/1', 'kv_cache_if_u/3');
        force_add_line(mdlName, 'kv_cache_if_u/1', 'attention_u/1');
        force_add_line(mdlName, 'attention_u/1', 'ffn_swiglu_u/1');
        force_add_line(mdlName, 'ffn_swiglu_u/1', 'residual_u/1');
        force_add_line(mdlName, 'in_residual/1', 'residual_u/2');
        force_add_line(mdlName, 'residual_u/1', 'out_hidden/1');
    else
        safe_add_line(mdlName, 'qkv_proj_u/1', 'out_hidden/1');
    end

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
        safe_add_line(mdlName, 'kv_addr_gen_u/1', 'axi_master_rd_u/5');
        safe_add_line(mdlName, 'kv_addr_gen_u/2', 'axi_master_rd_u/6');
          force_add_line(mdlName, 'axi_master_rd_u/3', 'kv_mem_rd_addr/1');
          force_add_line(mdlName, 'axi_master_rd_u/4', 'kv_mem_rd_len/1');
          force_add_line(mdlName, 'axi_master_rd_u/5', 'kv_mem_rd_valid/1');

        % Internal decode closure: historical KV flows into kv_cache_if_u.
          force_add_line(mdlName, 'axi_master_rd_u/1', 'kv_cache_if_u/2');
    end
end

function build_kv_memory_stubs(mdlName, stageProfile)
    safe_add_line(mdlName, 'kv_cache_rd_data/1', 'kv_cache_wr_data/1');
    safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'kv_cache_wr_en/1');
    safe_add_line(mdlName, 'eos_in/1', 'eos_out/1');

    if stageProfile == "stage2_memory_ready"
        safe_add_line(mdlName, 'cfg_token_pos/1', 'kv_addr_gen_u/1');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'kv_addr_gen_u/2');
        safe_add_line(mdlName, 'mode_decode/1', 'kv_addr_gen_u/3');

        % Write path hooks (adapted from soc_image_rotation AXI4MasterWriteController).
          safe_add_line(mdlName, 'qkv_proj_u/1', 'axi_master_wr_u/1');
          safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'axi_master_wr_u/2');
          safe_add_line(mdlName, 'ctrl_fsm_u/1', 'axi_master_wr_u/3');
        safe_add_line(mdlName, 'kv_addr_gen_u/3', 'axi_master_wr_u/4');
        safe_add_line(mdlName, 'kv_addr_gen_u/4', 'axi_master_wr_u/5');
          force_add_line(mdlName, 'axi_master_wr_u/2', 'kv_mem_wr_addr/1');
          force_add_line(mdlName, 'axi_master_wr_u/3', 'kv_mem_wr_len/1');
          force_add_line(mdlName, 'axi_master_wr_u/4', 'kv_mem_wr_valid/1');

        % Minimal task-level control placeholders for V2 ports.
          force_add_line(mdlName, 'ctrl_fsm_u/2', 'busy/1');
          force_add_line(mdlName, 'ctrl_fsm_u/1', 'done/1');
          force_add_line(mdlName, 'ctrl_fsm_u/1', 'irq/1');
          force_add_line(mdlName, 'cfg_token_pos/1', 'error_code/1');
        safe_add_line(mdlName, 'stop_req/1', 'ctrl_fsm_u/3');

        % Feed DDR model interface counter block (3rd fcn-style subsystem).
          safe_add_line(mdlName, 'axi_master_rd_u/5', 'ddr_model_if_u/1');
          safe_add_line(mdlName, 'axi_master_wr_u/4', 'ddr_model_if_u/2');
        safe_add_line(mdlName, 'kv_mem_rd_ready/1', 'ddr_model_if_u/3');
        safe_add_line(mdlName, 'kv_mem_wr_ready/1', 'ddr_model_if_u/4');
    end
end

function ensure_stage2_ports(mdlName)
    remove_legacy_weight_ports(mdlName);

    ensure_inport(mdlName, 'stop_req', [20, 520, 50, 534]);
    ensure_inport(mdlName, 'kv_mem_rd_ready', [20, 560, 50, 574]);
    ensure_inport(mdlName, 'kv_mem_wr_ready', [20, 600, 50, 614]);
    ensure_inport(mdlName, 'cfg_weight_num_heads', [20, 900, 50, 914]);
    ensure_inport(mdlName, 'cfg_weight_page_base', [20, 940, 50, 954]);
    ensure_inport(mdlName, 'cfg_weight_page_stride', [20, 980, 50, 994]);
    ensure_inport(mdlName, 'cfg_rope_theta_scale', [20, 1020, 50, 1034]);
    ensure_inport(mdlName, 'cfg_rope_sin_mix_scale', [20, 1060, 50, 1074]);

    ensure_outport(mdlName, 'busy', [1650, 520, 1680, 534]);
    ensure_outport(mdlName, 'irq', [1650, 560, 1680, 574]);
    ensure_outport(mdlName, 'error_code', [1650, 600, 1680, 614]);
    ensure_outport(mdlName, 'kv_mem_rd_addr', [1650, 640, 1680, 654]);
    ensure_outport(mdlName, 'kv_mem_rd_len', [1650, 680, 1680, 694]);
    ensure_outport(mdlName, 'kv_mem_rd_valid', [1650, 720, 1680, 734]);
    ensure_outport(mdlName, 'kv_mem_wr_addr', [1650, 760, 1680, 774]);
    ensure_outport(mdlName, 'kv_mem_wr_len', [1650, 800, 1680, 814]);
    ensure_outport(mdlName, 'kv_mem_wr_valid', [1650, 840, 1680, 854]);
    ensure_outport(mdlName, 'w_rd_req_bus', [1650, 900, 1680, 914]);
end

function ensure_memory_subsystems(mdlName)
    ensure_subsystem(mdlName, 'axi_master_rd_u', [980, 430, 1220, 510]);
    ensure_subsystem(mdlName, 'axi_master_wr_u', [980, 540, 1220, 620]);
    ensure_subsystem(mdlName, 'ddr_model_if_u', [980, 650, 1220, 730]);
    ensure_subsystem(mdlName, 'kv_addr_gen_u', [700, 600, 920, 680]);
    ensure_subsystem(mdlName, 'weight_addr_map_u', [700, 720, 980, 820]);
    ensure_subsystem(mdlName, 'axi_weight_rd_u', [980, 820, 1220, 900]);
end

function remove_legacy_weight_ports(mdlName)
    legacyIn = {
        'w_gamma_ddr_data', 'w_gamma_ddr_valid', ...
        'w_qkv_q_ddr_data', 'w_qkv_q_ddr_valid', 'w_qkv_k_ddr_data', 'w_qkv_k_ddr_valid', 'w_qkv_v_ddr_data', 'w_qkv_v_ddr_valid', ...
        'w_attn_q_ddr_data', 'w_attn_q_ddr_valid', 'w_attn_k_ddr_data', 'w_attn_k_ddr_valid', 'w_attn_v_ddr_data', 'w_attn_v_ddr_valid', ...
        'w_ffn_up_ddr_data', 'w_ffn_up_ddr_valid', 'w_ffn_gate_ddr_data', 'w_ffn_gate_ddr_valid'};
    legacyOut = {
        'w_gamma_req_addr', 'w_gamma_req_valid', ...
        'w_qkv_q_req_addr', 'w_qkv_q_req_valid', 'w_qkv_k_req_addr', 'w_qkv_k_req_valid', 'w_qkv_v_req_addr', 'w_qkv_v_req_valid', ...
        'w_attn_q_req_addr', 'w_attn_q_req_valid', 'w_attn_k_req_addr', 'w_attn_k_req_valid', 'w_attn_v_req_addr', 'w_attn_v_req_valid', ...
        'w_ffn_up_req_addr', 'w_ffn_up_req_valid', 'w_ffn_gate_req_addr', 'w_ffn_gate_req_valid'};
    for i = 1:numel(legacyIn)
        remove_top_block_if_exists(mdlName, legacyIn{i});
    end
    for i = 1:numel(legacyOut)
        remove_top_block_if_exists(mdlName, legacyOut{i});
    end
end

function remove_top_block_if_exists(mdlName, name)
    p = [mdlName '/' name];
    if ~isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', name))
        try
            ph = get_param(p, 'PortHandles');
            hs = [ph.Inport(:); ph.Outport(:)];
            for k = 1:numel(hs)
                ln = get_param(hs(k), 'Line');
                if ln ~= -1
                    delete_line(ln);
                end
            end
        catch
        end
        try
            delete_block(p);
        catch
        end
    end
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
    add_block('simulink/Sources/In1', [subPath '/addr_base'], 'Position', [20, 190, 50, 204]);
    add_block('simulink/Sources/In1', [subPath '/burst_len'], 'Position', [20, 230, 50, 244]);

    % Address channel request-hold: keep avalid asserted until arready handshake.
    add_block('simulink/Discrete/Unit Delay', [subPath '/avalid_state_z'], ...
        'InitialCondition', '0', 'Position', [120, 110, 150, 140]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/start_or_hold'], ...
        'Operator', 'OR', 'Position', [180, 145, 210, 175]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/addr_hs_logic'], ...
        'Operator', 'AND', 'Position', [240, 145, 270, 175]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/not_addr_hs'], ...
        'Operator', 'NOT', 'Position', [300, 145, 330, 175]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/avalid_next_logic'], ...
        'Operator', 'AND', 'Position', [360, 145, 390, 175]);

    % Burst in-flight and completion: start on address handshake, clear at burst done.
    add_block('simulink/Discrete/Unit Delay', [subPath '/burst_active_z'], ...
        'InitialCondition', '0', 'Position', [180, 30, 210, 60]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/burst_count_z'], ...
        'InitialCondition', '0', 'Position', [180, 70, 210, 100]);
    add_block('simulink/Sources/Constant', [subPath '/zero_const'], ...
        'Value', '0', 'Position', [120, 280, 160, 300]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [180, 280, 220, 300]);
    add_block('simulink/Math Operations/Add', [subPath '/count_inc'], ...
        'Inputs', '++', 'Position', [240, 70, 270, 100]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/beat_fire_logic'], ...
        'Operator', 'AND', 'Inputs', '3', 'Position', [300, 55, 330, 115]);
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subPath '/count_done_cmp'], ...
        'Operator', '>=', 'Position', [360, 70, 390, 100]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/burst_done_logic'], ...
        'Operator', 'AND', 'Position', [420, 70, 450, 100]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/not_burst_done'], ...
        'Operator', 'NOT', 'Position', [420, 30, 450, 60]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/burst_hold_logic'], ...
        'Operator', 'AND', 'Position', [480, 30, 510, 60]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/burst_next_logic'], ...
        'Operator', 'OR', 'Position', [540, 30, 570, 60]);
    add_block('simulink/Signal Routing/Switch', [subPath '/count_on_beat_sw'], ...
        'Threshold', '0.5', 'Position', [480, 70, 530, 120]);
    add_block('simulink/Signal Routing/Switch', [subPath '/count_on_start_sw'], ...
        'Threshold', '0.5', 'Position', [560, 70, 610, 120]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/rd_valid_gate'], ...
        'Operator', 'AND', 'Position', [300, 30, 330, 60]);

    add_block('simulink/Sources/Constant', [subPath '/dready_const'], ...
        'Value', '1', 'Position', [120, 250, 160, 270]);

    add_block('simulink/Sinks/Out1', [subPath '/rd_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dvalid_out'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_avalid'], 'Position', [370, 190, 400, 204]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dready'], 'Position', [370, 230, 400, 244]);

    safe_add_line(subPath, 'rd_data/1', 'rd_data_out/1');
    safe_add_line(subPath, 'addr_base/1', 'rd_addr/1');
    safe_add_line(subPath, 'burst_len/1', 'rd_len/1');

    safe_add_line(subPath, 'start/1', 'start_or_hold/1');
    safe_add_line(subPath, 'avalid_state_z/1', 'start_or_hold/2');
    safe_add_line(subPath, 'start_or_hold/1', 'addr_hs_logic/1');
    safe_add_line(subPath, 'rd_aready/1', 'addr_hs_logic/2');
    safe_add_line(subPath, 'addr_hs_logic/1', 'not_addr_hs/1');
    safe_add_line(subPath, 'start_or_hold/1', 'avalid_next_logic/1');
    safe_add_line(subPath, 'not_addr_hs/1', 'avalid_next_logic/2');
    safe_add_line(subPath, 'avalid_next_logic/1', 'avalid_state_z/1');
    safe_add_line(subPath, 'start_or_hold/1', 'rd_avalid/1');

    safe_add_line(subPath, 'burst_count_z/1', 'count_inc/1');
    safe_add_line(subPath, 'one_const/1', 'count_inc/2');
    safe_add_line(subPath, 'burst_active_z/1', 'beat_fire_logic/1');
    safe_add_line(subPath, 'rd_dvalid/1', 'beat_fire_logic/2');
    safe_add_line(subPath, 'dready_const/1', 'beat_fire_logic/3');
    safe_add_line(subPath, 'count_inc/1', 'count_done_cmp/1');
    safe_add_line(subPath, 'burst_len/1', 'count_done_cmp/2');
    safe_add_line(subPath, 'beat_fire_logic/1', 'burst_done_logic/1');
    safe_add_line(subPath, 'count_done_cmp/1', 'burst_done_logic/2');
    safe_add_line(subPath, 'burst_done_logic/1', 'not_burst_done/1');
    safe_add_line(subPath, 'burst_active_z/1', 'burst_hold_logic/1');
    safe_add_line(subPath, 'not_burst_done/1', 'burst_hold_logic/2');
    safe_add_line(subPath, 'addr_hs_logic/1', 'burst_next_logic/1');
    safe_add_line(subPath, 'burst_hold_logic/1', 'burst_next_logic/2');
    safe_add_line(subPath, 'burst_next_logic/1', 'burst_active_z/1');

    safe_add_line(subPath, 'burst_count_z/1', 'count_on_beat_sw/1');
    safe_add_line(subPath, 'beat_fire_logic/1', 'count_on_beat_sw/2');
    safe_add_line(subPath, 'count_inc/1', 'count_on_beat_sw/3');
    safe_add_line(subPath, 'count_on_beat_sw/1', 'count_on_start_sw/1');
    safe_add_line(subPath, 'addr_hs_logic/1', 'count_on_start_sw/2');
    safe_add_line(subPath, 'zero_const/1', 'count_on_start_sw/3');
    safe_add_line(subPath, 'count_on_start_sw/1', 'burst_count_z/1');

    safe_add_line(subPath, 'rd_dvalid/1', 'rd_valid_gate/1');
    safe_add_line(subPath, 'burst_active_z/1', 'rd_valid_gate/2');
    safe_add_line(subPath, 'rd_valid_gate/1', 'rd_dvalid_out/1');
    safe_add_line(subPath, 'dready_const/1', 'rd_dready/1');
end

function configure_axi_master_wr(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/wr_data'], 'Position', [20, 30, 50, 44]);
    add_block('simulink/Sources/In1', [subPath '/wr_dvalid'], 'Position', [20, 70, 50, 84]);
    add_block('simulink/Sources/In1', [subPath '/wr_complete'], 'Position', [20, 110, 50, 124]);
    add_block('simulink/Sources/In1', [subPath '/addr_base'], 'Position', [20, 150, 50, 164]);
    add_block('simulink/Sources/In1', [subPath '/burst_len'], 'Position', [20, 190, 50, 204]);

    % Write request-hold: keep wr_valid high until burst completion acknowledged.
    add_block('simulink/Discrete/Unit Delay', [subPath '/wvalid_state_z'], ...
        'InitialCondition', '0', 'Position', [120, 60, 150, 90]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/request_or_hold'], ...
        'Operator', 'OR', 'Position', [180, 60, 210, 90]);

    % Burst activity/counter path: start on new request and clear on write_done.
    add_block('simulink/Discrete/Unit Delay', [subPath '/write_active_z'], ...
        'InitialCondition', '0', 'Position', [180, 110, 210, 140]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/write_count_z'], ...
        'InitialCondition', '0', 'Position', [180, 150, 210, 180]);
    add_block('simulink/Sources/Constant', [subPath '/zero_const'], ...
        'Value', '0', 'Position', [120, 250, 160, 270]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [180, 250, 220, 270]);

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/not_write_active'], ...
        'Operator', 'NOT', 'Position', [240, 110, 270, 140]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/start_write_logic'], ...
        'Operator', 'AND', 'Position', [300, 110, 330, 140]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/active_or_start'], ...
        'Operator', 'OR', 'Position', [360, 110, 390, 140]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/beat_fire_logic'], ...
        'Operator', 'AND', 'Position', [420, 110, 450, 140]);
    add_block('simulink/Math Operations/Add', [subPath '/count_inc'], ...
        'Inputs', '++', 'Position', [300, 150, 330, 180]);
    add_block('simulink/Logic and Bit Operations/Relational Operator', [subPath '/count_done_cmp'], ...
        'Operator', '>=', 'Position', [360, 150, 390, 180]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/write_done_logic'], ...
        'Operator', 'AND', 'Position', [420, 150, 450, 180]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/not_write_done'], ...
        'Operator', 'NOT', 'Position', [480, 150, 510, 180]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/active_hold_logic'], ...
        'Operator', 'AND', 'Position', [540, 110, 570, 140]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/active_next_logic'], ...
        'Operator', 'OR', 'Position', [600, 110, 630, 140]);
    add_block('simulink/Signal Routing/Switch', [subPath '/count_on_beat_sw'], ...
        'Threshold', '0.5', 'Position', [480, 190, 530, 240]);
    add_block('simulink/Signal Routing/Switch', [subPath '/count_on_start_sw'], ...
        'Threshold', '0.5', 'Position', [560, 190, 610, 240]);

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/wvalid_next_logic'], ...
        'Operator', 'AND', 'Position', [660, 60, 690, 90]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/next_line_logic'], ...
        'Operator', 'AND', 'Position', [660, 150, 690, 180]);

    add_block('simulink/Sinks/Out1', [subPath '/wr_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_valid'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/request_next_line'], 'Position', [370, 190, 400, 204]);

    safe_add_line(subPath, 'wr_data/1', 'wr_data_out/1');
    safe_add_line(subPath, 'addr_base/1', 'wr_addr/1');
    safe_add_line(subPath, 'burst_len/1', 'wr_len/1');

    safe_add_line(subPath, 'wr_dvalid/1', 'request_or_hold/1');
    safe_add_line(subPath, 'wvalid_state_z/1', 'request_or_hold/2');

    safe_add_line(subPath, 'write_active_z/1', 'not_write_active/1');
    safe_add_line(subPath, 'request_or_hold/1', 'start_write_logic/1');
    safe_add_line(subPath, 'not_write_active/1', 'start_write_logic/2');

    safe_add_line(subPath, 'write_active_z/1', 'active_or_start/1');
    safe_add_line(subPath, 'start_write_logic/1', 'active_or_start/2');
    safe_add_line(subPath, 'active_or_start/1', 'beat_fire_logic/1');
    safe_add_line(subPath, 'wr_dvalid/1', 'beat_fire_logic/2');

    safe_add_line(subPath, 'write_count_z/1', 'count_inc/1');
    safe_add_line(subPath, 'one_const/1', 'count_inc/2');
    safe_add_line(subPath, 'count_inc/1', 'count_done_cmp/1');
    safe_add_line(subPath, 'burst_len/1', 'count_done_cmp/2');
    safe_add_line(subPath, 'wr_complete/1', 'write_done_logic/1');
    safe_add_line(subPath, 'count_done_cmp/1', 'write_done_logic/2');
    safe_add_line(subPath, 'write_done_logic/1', 'not_write_done/1');

    safe_add_line(subPath, 'write_active_z/1', 'active_hold_logic/1');
    safe_add_line(subPath, 'not_write_done/1', 'active_hold_logic/2');
    safe_add_line(subPath, 'start_write_logic/1', 'active_next_logic/1');
    safe_add_line(subPath, 'active_hold_logic/1', 'active_next_logic/2');
    safe_add_line(subPath, 'active_next_logic/1', 'write_active_z/1');

    safe_add_line(subPath, 'write_count_z/1', 'count_on_beat_sw/1');
    safe_add_line(subPath, 'beat_fire_logic/1', 'count_on_beat_sw/2');
    safe_add_line(subPath, 'count_inc/1', 'count_on_beat_sw/3');
    safe_add_line(subPath, 'count_on_beat_sw/1', 'count_on_start_sw/1');
    safe_add_line(subPath, 'start_write_logic/1', 'count_on_start_sw/2');
    safe_add_line(subPath, 'zero_const/1', 'count_on_start_sw/3');
    safe_add_line(subPath, 'count_on_start_sw/1', 'write_count_z/1');

    safe_add_line(subPath, 'request_or_hold/1', 'wvalid_next_logic/1');
    safe_add_line(subPath, 'not_write_done/1', 'wvalid_next_logic/2');
    safe_add_line(subPath, 'wvalid_next_logic/1', 'wvalid_state_z/1');
    safe_add_line(subPath, 'request_or_hold/1', 'wr_valid/1');

    safe_add_line(subPath, 'write_done_logic/1', 'next_line_logic/1');
    safe_add_line(subPath, 'write_active_z/1', 'next_line_logic/2');
    safe_add_line(subPath, 'next_line_logic/1', 'request_next_line/1');
end

function configure_ctrl_fsm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/start'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Sources/In1', [subPath '/stop_req'], 'Position', [20, 140, 50, 154]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/state_z'], ...
        'InitialCondition', '0', 'Position', [120, 170, 150, 200]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/fsm_core'], ...
        'Position', [210, 55, 320, 185]);

    fsmCode = sprintf([ ...
        'function [done_out,busy_out,state_next] = fsm_core(mode_decode,start,stop_req,state_in)\n' ...
        '%%#codegen\n' ...
        '[done_out,busy_out,state_next] = ctrl_fsm_step(mode_decode,start,stop_req,state_in);\n' ...
        'end\n']);
    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subPath '/fsm_core']);
    if isempty(chart)
        error('configure_ctrl_fsm:MissingEMChart', ...
            'Expected MATLAB Function chart at %s/fsm_core', subPath);
    end
    chart.Script = fsmCode;

    add_block('simulink/Sinks/Out1', [subPath '/done_out'], 'Position', [350, 95, 380, 109]);
    add_block('simulink/Sinks/Out1', [subPath '/busy_out'], 'Position', [350, 55, 380, 69]);

    safe_add_line(subPath, 'mode_decode/1', 'fsm_core/1');
    safe_add_line(subPath, 'start/1', 'fsm_core/2');
    safe_add_line(subPath, 'stop_req/1', 'fsm_core/3');
    safe_add_line(subPath, 'state_z/1', 'fsm_core/4');
    safe_add_line(subPath, 'fsm_core/1', 'done_out/1');
    safe_add_line(subPath, 'fsm_core/2', 'busy_out/1');
    safe_add_line(subPath, 'fsm_core/3', 'state_z/1');
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

function configure_kv_cache_if(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/qkv_new'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/kv_hist'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 140, 50, 154]);
    add_block('simulink/Signal Routing/Switch', [subPath '/mode_switch'], ...
        'Threshold', '0.5', 'Position', [120, 60, 170, 140]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_to_attn'], 'Position', [260, 95, 290, 109]);

    safe_add_line(subPath, 'qkv_new/1', 'mode_switch/1');
    safe_add_line(subPath, 'mode_decode/1', 'mode_switch/2');
    safe_add_line(subPath, 'kv_hist/1', 'mode_switch/3');
    safe_add_line(subPath, 'mode_switch/1', 'kv_to_attn/1');
end

function configure_kv_addr_gen(subPath, cfg)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/seq_len'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 120, 50, 134]);

    % Software-configured constants (base/stride/decode burst), with hardware doing token math.
    add_block('simulink/Sources/Constant', [subPath '/rd_base_const'], ...
        'Value', num2str(cfg.rd_base), 'Position', [90, 10, 140, 30]);
    add_block('simulink/Sources/Constant', [subPath '/wr_base_const'], ...
        'Value', num2str(cfg.wr_base), 'Position', [90, 35, 140, 55]);
    add_block('simulink/Sources/Constant', [subPath '/stride_const'], ...
        'Value', num2str(cfg.stride_bytes), 'Position', [90, 60, 150, 80]);
    add_block('simulink/Sources/Constant', [subPath '/decode_burst_const'], ...
        'Value', num2str(cfg.decode_burst_len), 'Position', [90, 85, 170, 105]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [90, 150, 130, 170]);

    add_block('simulink/Math Operations/Add', [subPath '/rd_tok_prev'], ...
        'Inputs', '+-', 'Position', [150, 30, 180, 55]);
    add_block('simulink/Discontinuities/Saturation', [subPath '/rd_tok_prev_sat'], ...
        'UpperLimit', 'inf', 'LowerLimit', '0', 'Position', [210, 30, 240, 55]);
    add_block('simulink/Math Operations/Product', [subPath '/rd_addr_scale'], ...
        'Inputs', '**', 'Position', [280, 35, 340, 60]);
    add_block('simulink/Math Operations/Product', [subPath '/rd_addr_prefill_scale'], ...
        'Inputs', '**', 'Position', [280, 85, 340, 110]);
    add_block('simulink/Signal Routing/Switch', [subPath '/rd_addr_mode_sel'], ...
        'Threshold', '0.5', 'Position', [390, 45, 440, 125]);
    add_block('simulink/Math Operations/Add', [subPath '/rd_addr_add_base'], ...
        'Inputs', '++', 'Position', [470, 65, 500, 95]);

    add_block('simulink/Math Operations/Add', [subPath '/wr_tok_next'], ...
        'Inputs', '++', 'Position', [150, 105, 180, 130]);
    add_block('simulink/Math Operations/Product', [subPath '/wr_addr_scale'], ...
        'Inputs', '**', 'Position', [280, 145, 340, 170]);
    add_block('simulink/Math Operations/Product', [subPath '/wr_addr_add'], ...
        'Inputs', '**', 'Position', [280, 190, 340, 215]);
    add_block('simulink/Signal Routing/Switch', [subPath '/wr_addr_mode_sel'], ...
        'Threshold', '0.5', 'Position', [390, 145, 440, 225]);
    add_block('simulink/Math Operations/Add', [subPath '/wr_addr_add_base'], ...
        'Inputs', '++', 'Position', [470, 165, 500, 195]);

    add_block('simulink/Signal Routing/Switch', [subPath '/rd_len_mode_sel'], ...
        'Threshold', '0.5', 'Position', [505, 60, 555, 140]);
    add_block('simulink/Signal Routing/Switch', [subPath '/wr_len_mode_sel'], ...
        'Threshold', '0.5', 'Position', [505, 155, 555, 235]);

    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [620, 60, 650, 74]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [620, 100, 650, 114]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [620, 175, 650, 189]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [620, 215, 650, 229]);

    safe_add_line(subPath, 'token_pos/1', 'rd_tok_prev/1');
    safe_add_line(subPath, 'one_const/1', 'rd_tok_prev/2');
    safe_add_line(subPath, 'rd_tok_prev/1', 'rd_tok_prev_sat/1');
    safe_add_line(subPath, 'rd_tok_prev_sat/1', 'rd_addr_scale/1');
    safe_add_line(subPath, 'stride_const/1', 'rd_addr_scale/2');
    safe_add_line(subPath, 'token_pos/1', 'rd_addr_prefill_scale/1');
    safe_add_line(subPath, 'stride_const/1', 'rd_addr_prefill_scale/2');
    safe_add_line(subPath, 'rd_addr_scale/1', 'rd_addr_mode_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'rd_addr_mode_sel/2');
    safe_add_line(subPath, 'rd_addr_prefill_scale/1', 'rd_addr_mode_sel/3');
    safe_add_line(subPath, 'rd_addr_mode_sel/1', 'rd_addr_add_base/1');
    safe_add_line(subPath, 'rd_base_const/1', 'rd_addr_add_base/2');
    safe_add_line(subPath, 'rd_addr_add_base/1', 'rd_addr/1');

    safe_add_line(subPath, 'token_pos/1', 'wr_addr_scale/1');
    safe_add_line(subPath, 'stride_const/1', 'wr_addr_scale/2');
    safe_add_line(subPath, 'token_pos/1', 'wr_tok_next/1');
    safe_add_line(subPath, 'one_const/1', 'wr_tok_next/2');
    safe_add_line(subPath, 'wr_tok_next/1', 'wr_addr_add/1');
    safe_add_line(subPath, 'stride_const/1', 'wr_addr_add/2');
    safe_add_line(subPath, 'wr_addr_add/1', 'wr_addr_mode_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'wr_addr_mode_sel/2');
    safe_add_line(subPath, 'wr_addr_scale/1', 'wr_addr_mode_sel/3');
    safe_add_line(subPath, 'wr_addr_mode_sel/1', 'wr_addr_add_base/1');
    safe_add_line(subPath, 'wr_base_const/1', 'wr_addr_add_base/2');
    safe_add_line(subPath, 'wr_addr_add_base/1', 'wr_addr/1');

    safe_add_line(subPath, 'decode_burst_const/1', 'rd_len_mode_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'rd_len_mode_sel/2');
    safe_add_line(subPath, 'seq_len/1', 'rd_len_mode_sel/3');
    safe_add_line(subPath, 'rd_len_mode_sel/1', 'rd_len/1');

    safe_add_line(subPath, 'decode_burst_const/1', 'wr_len_mode_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'wr_len_mode_sel/2');
    safe_add_line(subPath, 'seq_len/1', 'wr_len_mode_sel/3');
    safe_add_line(subPath, 'wr_len_mode_sel/1', 'wr_len/1');
end

function configure_weight_addr_map(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/num_heads'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/page_base'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/Sources/In1', [subPath '/page_stride'], 'Position', [20, 160, 50, 174]);

    add_block('simulink/Math Operations/Product', [subPath '/tok_head_prod'], ...
        'Inputs', '**', 'Position', [100, 55, 135, 95]);
    add_block('simulink/Math Operations/Product', [subPath '/page_offset'], ...
        'Inputs', '**', 'Position', [170, 80, 205, 120]);
    add_block('simulink/Math Operations/Add', [subPath '/addr_base_sum'], ...
        'Inputs', '++', 'Position', [240, 95, 275, 125]);

    add_block('simulink/Sources/Constant', [subPath '/off0'], 'Value', '0', 'Position', [310, 30, 340, 50]);
    add_block('simulink/Sources/Constant', [subPath '/off1'], 'Value', '1', 'Position', [310, 60, 340, 80]);
    add_block('simulink/Sources/Constant', [subPath '/off2'], 'Value', '2', 'Position', [310, 90, 340, 110]);
    add_block('simulink/Sources/Constant', [subPath '/off3'], 'Value', '3', 'Position', [310, 120, 340, 140]);
    add_block('simulink/Sources/Constant', [subPath '/off4'], 'Value', '4', 'Position', [310, 150, 340, 170]);
    add_block('simulink/Sources/Constant', [subPath '/off5'], 'Value', '5', 'Position', [310, 180, 340, 200]);
    add_block('simulink/Sources/Constant', [subPath '/off6'], 'Value', '6', 'Position', [310, 210, 340, 230]);
    add_block('simulink/Sources/Constant', [subPath '/off7'], 'Value', '7', 'Position', [310, 240, 340, 260]);
    add_block('simulink/Sources/Constant', [subPath '/off8'], 'Value', '8', 'Position', [310, 270, 340, 290]);

    outNames = {'gamma_addr', 'q_addr', 'k_addr', 'v_addr', 'attn_q_addr', 'attn_k_addr', 'attn_v_addr', 'up_addr', 'gate_addr'};
    y0 = 40;
    for i = 1:numel(outNames)
        add_block('simulink/Math Operations/Add', [subPath '/' outNames{i} '_sum'], ...
            'Inputs', '++', 'Position', [370, y0 + 30 * (i - 1), 405, y0 + 30 * (i - 1) + 25]);
        add_block('simulink/Sinks/Out1', [subPath '/' outNames{i}], ...
            'Position', [450, y0 + 30 * (i - 1) + 5, 480, y0 + 30 * (i - 1) + 19]);
        safe_add_line(subPath, 'addr_base_sum/1', [outNames{i} '_sum/1']);
        safe_add_line(subPath, ['off' num2str(i - 1) '/1'], [outNames{i} '_sum/2']);
        safe_add_line(subPath, [outNames{i} '_sum/1'], [outNames{i} '/1']);
    end

    safe_add_line(subPath, 'token_pos/1', 'tok_head_prod/1');
    safe_add_line(subPath, 'num_heads/1', 'tok_head_prod/2');
    safe_add_line(subPath, 'tok_head_prod/1', 'page_offset/1');
    safe_add_line(subPath, 'page_stride/1', 'page_offset/2');
    safe_add_line(subPath, 'page_base/1', 'addr_base_sum/1');
    safe_add_line(subPath, 'page_offset/1', 'addr_base_sum/2');
end

function configure_axi_weight_rd(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 90, 50, 104]);
    add_or_reset_demux(subPath, 'req_demux', 18, [90, 30, 120, 330]);
    add_or_reset_mux(subPath, 'rsp_mux', 18, [360, 30, 390, 330]);
    add_block('simulink/Sinks/Out1', [subPath '/rsp_bus'], 'Position', [430, 120, 460, 134]);

    safe_add_line(subPath, 'req_bus/1', 'req_demux/1');
    safe_add_line(subPath, 'rsp_mux/1', 'rsp_bus/1');

    scaleVals = {'1.0','0.6','0.4','1.0','0.6','0.4','1.0','1.4','0.9'};
    for i = 1:9
        baseY = 20 + 30 * (i - 1);
        add_block('simulink/Math Operations/Gain', [subPath '/data_gain_' num2str(i)], ...
            'Gain', scaleVals{i}, 'Position', [180, baseY, 220, baseY + 20]);
        add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/data_u8_' num2str(i)], ...
            'OutDataTypeStr', 'uint8', 'Position', [240, baseY, 280, baseY + 20]);
        safe_add_line(subPath, ['req_demux/' num2str(2 * i - 1)], ['data_gain_' num2str(i) '/1']);
        safe_add_line(subPath, ['data_gain_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);
        safe_add_line(subPath, ['data_u8_' num2str(i) '/1'], ['rsp_mux/' num2str(2 * i - 1)]);
        safe_add_line(subPath, ['req_demux/' num2str(2 * i)], ['rsp_mux/' num2str(2 * i)]);
    end
end

function add_or_reset_mux(sys, name, n, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Routing/Mux', p, 'Position', pos);
    end
    set_param(p, 'Inputs', num2str(n), 'Position', pos);
end

function add_or_reset_demux(sys, name, n, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Routing/Demux', p, 'Position', pos);
    end
    set_param(p, 'Outputs', num2str(n), 'Position', pos);
end

function add_or_reset_dtc(sys, name, outType, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Attributes/Data Type Conversion', p, 'Position', pos);
    end
    set_param(p, 'OutDataTypeStr', outType, 'Position', pos);
end

function configure_rmsnorm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 40, 60, 54]);
    add_block('simulink/Sources/In1', [subPath '/eps_in'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/gamma_ddr_data'], 'Position', [30, 165, 60, 179]);
    add_block('simulink/Sources/In1', [subPath '/gamma_ddr_valid'], 'Position', [30, 200, 60, 214]);
    add_block('simulink/Sources/In1', [subPath '/gamma_addr_in'], 'Position', [30, 235, 60, 249]);
    add_block('simulink/Math Operations/Product', [subPath '/x_square'], ...
        'Inputs', '**', 'Position', [110, 35, 145, 65]);
    add_block('simulink/Math Operations/Add', [subPath '/var_eps_sum'], ...
        'Inputs', '++', 'Position', [190, 60, 225, 100]);
    add_block('simulink/Math Operations/Math Function', [subPath '/sqrt_denom'], ...
        'Operator', 'sqrt', 'Position', [260, 65, 300, 95]);
    add_block('simulink/Math Operations/Divide', [subPath '/x_norm'], ...
        'Position', [330, 50, 365, 100]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [560, 75, 590, 89]);

    safe_add_line(subPath, 'x_in/1', 'x_square/1');
    safe_add_line(subPath, 'x_in/1', 'x_square/2');
    safe_add_line(subPath, 'x_square/1', 'var_eps_sum/1');
    safe_add_line(subPath, 'eps_in/1', 'var_eps_sum/2');
    safe_add_line(subPath, 'var_eps_sum/1', 'sqrt_denom/1');
    safe_add_line(subPath, 'x_in/1', 'x_norm/1');
    safe_add_line(subPath, 'sqrt_denom/1', 'x_norm/2');

    gammaOut = add_streamed_weight_mul(subPath, 'gamma', 'x_norm/1', ...
        'gamma_ddr_data/1', 'gamma_ddr_valid/1', 'gamma_addr_in/1', 410, 55, '1.0');
    safe_add_line(subPath, gammaOut, 'y_out/1');
end

function configure_qkv_proj(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/q_ddr_data'], 'Position', [30, 10, 60, 24]);
    add_block('simulink/Sources/In1', [subPath '/q_ddr_valid'], 'Position', [30, 30, 60, 44]);
    add_block('simulink/Sources/In1', [subPath '/q_addr_in'], 'Position', [30, 50, 60, 64]);
    add_block('simulink/Sources/In1', [subPath '/k_ddr_data'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/k_ddr_valid'], 'Position', [30, 130, 60, 144]);
    add_block('simulink/Sources/In1', [subPath '/k_addr_in'], 'Position', [30, 150, 60, 164]);
    add_block('simulink/Sources/In1', [subPath '/v_ddr_data'], 'Position', [30, 210, 60, 224]);
    add_block('simulink/Sources/In1', [subPath '/v_ddr_valid'], 'Position', [30, 230, 60, 244]);
    add_block('simulink/Sources/In1', [subPath '/v_addr_in'], 'Position', [30, 250, 60, 264]);
    add_block('simulink/Math Operations/Add', [subPath '/qk_sum'], ...
        'Inputs', '++', 'Position', [500, 45, 535, 85]);
    add_block('simulink/Math Operations/Add', [subPath '/qkv_sum'], ...
        'Inputs', '++', 'Position', [570, 60, 605, 100]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [650, 75, 680, 89]);

    qOut = add_streamed_weight_mul(subPath, 'q', 'x_in/1', ...
        'q_ddr_data/1', 'q_ddr_valid/1', 'q_addr_in/1', 110, 15, '0.6');
    kOut = add_streamed_weight_mul(subPath, 'k', 'x_in/1', ...
        'k_ddr_data/1', 'k_ddr_valid/1', 'k_addr_in/1', 110, 105, '0.4');
    vOut = add_streamed_weight_mul(subPath, 'v', 'x_in/1', ...
        'v_ddr_data/1', 'v_ddr_valid/1', 'v_addr_in/1', 110, 195, '1.0');

    safe_add_line(subPath, qOut, 'qk_sum/1');
    safe_add_line(subPath, kOut, 'qk_sum/2');
    safe_add_line(subPath, 'qk_sum/1', 'qkv_sum/1');
    safe_add_line(subPath, vOut, 'qkv_sum/2');
    safe_add_line(subPath, 'qkv_sum/1', 'y_out/1');
end

function configure_attention(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/q_ddr_data'], 'Position', [30, 10, 60, 24]);
    add_block('simulink/Sources/In1', [subPath '/q_ddr_valid'], 'Position', [30, 30, 60, 44]);
    add_block('simulink/Sources/In1', [subPath '/q_addr_in'], 'Position', [30, 50, 60, 64]);
    add_block('simulink/Sources/In1', [subPath '/k_ddr_data'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/k_ddr_valid'], 'Position', [30, 130, 60, 144]);
    add_block('simulink/Sources/In1', [subPath '/k_addr_in'], 'Position', [30, 150, 60, 164]);
    add_block('simulink/Sources/In1', [subPath '/v_ddr_data'], 'Position', [30, 210, 60, 224]);
    add_block('simulink/Sources/In1', [subPath '/v_ddr_valid'], 'Position', [30, 230, 60, 244]);
    add_block('simulink/Sources/In1', [subPath '/v_addr_in'], 'Position', [30, 250, 60, 264]);
    add_block('simulink/Math Operations/Product', [subPath '/score_mul'], ...
        'Inputs', '**', 'Position', [210, 45, 245, 85]);
    add_block('simulink/Math Operations/Abs', [subPath '/score_abs'], ...
        'Position', [280, 45, 315, 75]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [280, 95, 320, 115]);
    add_block('simulink/Math Operations/Add', [subPath '/score_den'], ...
        'Inputs', '++', 'Position', [350, 55, 385, 95]);
    add_block('simulink/Math Operations/Divide', [subPath '/score_norm'], ...
        'Position', [420, 50, 455, 100]);
    add_block('simulink/Math Operations/Product', [subPath '/value_weight'], ...
        'Inputs', '**', 'Position', [500, 75, 535, 115]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [580, 90, 610, 104]);

    qOut = add_streamed_weight_mul(subPath, 'q', 'x_in/1', ...
        'q_ddr_data/1', 'q_ddr_valid/1', 'q_addr_in/1', 110, 15, '0.6');
    kOut = add_streamed_weight_mul(subPath, 'k', 'x_in/1', ...
        'k_ddr_data/1', 'k_ddr_valid/1', 'k_addr_in/1', 110, 95, '0.4');
    vOut = add_streamed_weight_mul(subPath, 'v', 'x_in/1', ...
        'v_ddr_data/1', 'v_ddr_valid/1', 'v_addr_in/1', 110, 175, '1.0');

    safe_add_line(subPath, qOut, 'score_mul/1');
    safe_add_line(subPath, kOut, 'score_mul/2');
    safe_add_line(subPath, 'score_mul/1', 'score_abs/1');
    safe_add_line(subPath, 'score_abs/1', 'score_den/1');
    safe_add_line(subPath, 'one_const/1', 'score_den/2');
    safe_add_line(subPath, 'score_mul/1', 'score_norm/1');
    safe_add_line(subPath, 'score_den/1', 'score_norm/2');
    safe_add_line(subPath, 'score_norm/1', 'value_weight/1');
    safe_add_line(subPath, vOut, 'value_weight/2');
    safe_add_line(subPath, 'value_weight/1', 'y_out/1');
end

function configure_ffn_swiglu(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/up_ddr_data'], 'Position', [30, 10, 60, 24]);
    add_block('simulink/Sources/In1', [subPath '/up_ddr_valid'], 'Position', [30, 30, 60, 44]);
    add_block('simulink/Sources/In1', [subPath '/up_addr_in'], 'Position', [30, 50, 60, 64]);
    add_block('simulink/Sources/In1', [subPath '/gate_ddr_data'], 'Position', [30, 155, 60, 169]);
    add_block('simulink/Sources/In1', [subPath '/gate_ddr_valid'], 'Position', [30, 175, 60, 189]);
    add_block('simulink/Sources/In1', [subPath '/gate_addr_in'], 'Position', [30, 195, 60, 209]);
    add_block('simulink/Math Operations/Abs', [subPath '/gate_abs'], ...
        'Position', [210, 85, 245, 115]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [210, 125, 250, 145]);
    add_block('simulink/Math Operations/Add', [subPath '/gate_den'], ...
        'Inputs', '++', 'Position', [280, 95, 315, 125]);
    add_block('simulink/Math Operations/Divide', [subPath '/gate_norm'], ...
        'Position', [350, 85, 385, 135]);
    add_block('simulink/Math Operations/Product', [subPath '/swiglu_mul'], ...
        'Inputs', '**', 'Position', [430, 70, 465, 120]);
    add_block('simulink/Math Operations/Gain', [subPath '/down_proj'], ...
        'Gain', '0.8', 'Position', [500, 80, 560, 105]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [620, 90, 650, 104]);

    upOut = add_streamed_weight_mul(subPath, 'up', 'x_in/1', ...
        'up_ddr_data/1', 'up_ddr_valid/1', 'up_addr_in/1', 110, 15, '1.4');
    gateOut = add_streamed_weight_mul(subPath, 'gate', 'x_in/1', ...
        'gate_ddr_data/1', 'gate_ddr_valid/1', 'gate_addr_in/1', 110, 145, '0.9');

    safe_add_line(subPath, gateOut, 'gate_abs/1');
    safe_add_line(subPath, 'gate_abs/1', 'gate_den/1');
    safe_add_line(subPath, 'one_const/1', 'gate_den/2');
    safe_add_line(subPath, gateOut, 'gate_norm/1');
    safe_add_line(subPath, 'gate_den/1', 'gate_norm/2');
    safe_add_line(subPath, upOut, 'swiglu_mul/1');
    safe_add_line(subPath, 'gate_norm/1', 'swiglu_mul/2');
    safe_add_line(subPath, 'swiglu_mul/1', 'down_proj/1');
    safe_add_line(subPath, 'down_proj/1', 'y_out/1');
end

function mulOut = add_streamed_weight_mul(subPath, prefix, inSig, ddrDataSig, ddrValidSig, reqAddrSig, x0, y0, defaultWeight)
    % Model off-chip DDR fetch + on-chip SRAM cache using HDL RAMS blocks.
    reqAddrCast = [prefix '_req_addr_u8'];
    reqNeeded = [prefix '_req_needed'];
    sram = [prefix '_sram'];
    ddrDataCast = [prefix '_ddr_data_u8'];
    ddrValidCast = [prefix '_ddr_valid_bool'];
    sramDataCast = [prefix '_sram_data_double'];
    sramValid = [prefix '_sram_data_valid_z'];
    reqAddrOut = [prefix '_ddr_req_addr'];
    reqValidOut = [prefix '_ddr_req_valid'];
    sramSel = [prefix '_sram_data_sel'];
    validOr = [prefix '_valid_or'];
    defaultConst = [prefix '_default_w'];
    mulBlk = [prefix '_mul'];

    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqAddrCast], ...
        'OutDataTypeStr', 'uint8', 'Position', [x0 + 145, y0 - 2, x0 + 185, y0 + 22]);

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' reqNeeded], ...
        'Operator', 'NOT', 'Position', [x0 + 5, y0 + 70, x0 + 35, y0 + 90]);
    add_block('simulink/Sources/Constant', [subPath '/' defaultConst], ...
        'Value', defaultWeight, 'Position', [x0 + 5, y0 + 45, x0 + 45, y0 + 65]);
    add_block('hdlsllib/HDL RAMs/Simple Dual Port RAM', [subPath '/' sram], ...
        'Position', [x0 + 95, y0 + 30, x0 + 165, y0 + 120]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' ddrDataCast], ...
        'OutDataTypeStr', 'uint8', 'Position', [x0 + 55, y0 + 45, x0 + 90, y0 + 65]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' ddrValidCast], ...
        'OutDataTypeStr', 'boolean', 'Position', [x0 + 55, y0 + 75, x0 + 90, y0 + 95]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' sramDataCast], ...
        'OutDataTypeStr', 'double', 'Position', [x0 + 180, y0 + 45, x0 + 215, y0 + 65]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/' sramValid], ...
        'InitialCondition', '0', 'Position', [x0 + 180, y0 + 70, x0 + 210, y0 + 90]);
    add_block('simulink/Sinks/Out1', [subPath '/' reqAddrOut], 'Position', [x0 + 180, y0 - 5, x0 + 210, y0 + 9]);
    add_block('simulink/Sinks/Out1', [subPath '/' reqValidOut], 'Position', [x0 + 180, y0 + 20, x0 + 210, y0 + 34]);
    add_block('simulink/Signal Routing/Switch', [subPath '/' sramSel], ...
        'Threshold', '0.5', 'Position', [x0 + 230, y0 + 40, x0 + 280, y0 + 90]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' validOr], ...
        'Operator', 'OR', 'Position', [x0 + 230, y0 + 95, x0 + 260, y0 + 125]);

    add_block('simulink/Math Operations/Product', [subPath '/' mulBlk], ...
        'Inputs', '**', 'Position', [x0 + 305, y0 + 45, x0 + 340, y0 + 85]);

    safe_add_line(subPath, reqAddrSig, [reqAddrCast '/1']);
    safe_add_line(subPath, ddrDataSig, [ddrDataCast '/1']);
    safe_add_line(subPath, ddrValidSig, [ddrValidCast '/1']);

    safe_add_line(subPath, [sramValid '/1'], [reqNeeded '/1']);
    safe_add_line(subPath, [reqAddrCast '/1'], [reqAddrOut '/1']);
    safe_add_line(subPath, [reqNeeded '/1'], [reqValidOut '/1']);

    % Simple Dual Port RAM expected ports: wr_addr, din, we, rd_addr -> dout.
    safe_add_line(subPath, [reqAddrCast '/1'], [sram '/1']);
    safe_add_line(subPath, [ddrDataCast '/1'], [sram '/2']);
    safe_add_line(subPath, [ddrValidCast '/1'], [sram '/3']);
    safe_add_line(subPath, [reqAddrCast '/1'], [sram '/4']);

    safe_add_line(subPath, [sram '/1'], [sramDataCast '/1']);
    safe_add_line(subPath, [sramDataCast '/1'], [sramSel '/1']);
    safe_add_line(subPath, [sramValid '/1'], [sramSel '/2']);
    safe_add_line(subPath, [defaultConst '/1'], [sramSel '/3']);

    safe_add_line(subPath, [sramValid '/1'], [validOr '/1']);
    safe_add_line(subPath, [ddrValidCast '/1'], [validOr '/2']);
    safe_add_line(subPath, [validOr '/1'], [sramValid '/1']);

    safe_add_line(subPath, inSig, [mulBlk '/1']);
    safe_add_line(subPath, [sramSel '/1'], [mulBlk '/2']);

    mulOut = [mulBlk '/1'];
end

function configure_residual(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_main'], 'Position', [30, 55, 60, 69]);
    add_block('simulink/Sources/In1', [subPath '/x_skip'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/main_delay'], ...
        'InitialCondition', '0', 'Position', [110, 55, 150, 75]);
    add_block('simulink/Math Operations/Gain', [subPath '/main_scale'], ...
        'Gain', '0.8', 'Position', [180, 50, 240, 75]);
    add_block('simulink/Math Operations/Gain', [subPath '/skip_scale'], ...
        'Gain', '1.0', 'Position', [180, 105, 240, 130]);
    add_block('simulink/Math Operations/Sum', [subPath '/res_sum'], ...
        'Inputs', '++', 'Position', [280, 70, 310, 110]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [360, 85, 390, 99]);

    safe_add_line(subPath, 'x_main/1', 'main_delay/1');
    safe_add_line(subPath, 'main_delay/1', 'main_scale/1');
    safe_add_line(subPath, 'x_skip/1', 'skip_scale/1');
    safe_add_line(subPath, 'main_scale/1', 'res_sum/1');
    safe_add_line(subPath, 'skip_scale/1', 'res_sum/2');
    safe_add_line(subPath, 'res_sum/1', 'y_out/1');
end

function configure_rope(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [30, 130, 60, 144]);
    add_block('simulink/Sources/In1', [subPath '/theta_scale'], 'Position', [30, 170, 60, 184]);
    add_block('simulink/Sources/In1', [subPath '/sin_mix_scale'], 'Position', [30, 205, 60, 219]);
    add_block('simulink/Math Operations/Product', [subPath '/phase_mul'], ...
        'Inputs', '**', 'Position', [110, 135, 145, 165]);
    add_block('hdlsllib/Lookup Tables/Cosine HDL Optimized', [subPath '/cos_phase'], ...
        'Position', [200, 95, 260, 125]);
    add_block('hdlsllib/Lookup Tables/Sine HDL Optimized', [subPath '/sin_phase'], ...
        'Position', [200, 140, 260, 170]);
    add_block('simulink/Math Operations/Product', [subPath '/sin_scaled'], ...
        'Inputs', '**', 'Position', [300, 140, 335, 170]);
    add_block('simulink/Math Operations/Add', [subPath '/rot_scale_sum'], ...
        'Inputs', '++', 'Position', [390, 105, 420, 145]);
    add_block('simulink/Math Operations/Product', [subPath '/apply_rot_scale'], ...
        'Inputs', '**', 'Position', [460, 80, 500, 120]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [560, 90, 590, 104]);

    safe_add_line(subPath, 'token_pos/1', 'phase_mul/1');
    safe_add_line(subPath, 'theta_scale/1', 'phase_mul/2');
    safe_add_line(subPath, 'phase_mul/1', 'cos_phase/1');
    safe_add_line(subPath, 'phase_mul/1', 'sin_phase/1');
    safe_add_line(subPath, 'cos_phase/1', 'rot_scale_sum/1');
    safe_add_line(subPath, 'sin_phase/1', 'sin_scaled/1');
    safe_add_line(subPath, 'sin_mix_scale/1', 'sin_scaled/2');
    safe_add_line(subPath, 'sin_scaled/1', 'rot_scale_sum/2');
    safe_add_line(subPath, 'x_in/1', 'apply_rot_scale/1');
    safe_add_line(subPath, 'rot_scale_sum/1', 'apply_rot_scale/2');
    safe_add_line(subPath, 'apply_rot_scale/1', 'y_out/1');
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

function force_add_line(sys, src, dst)
    dstParts = split(string(dst), '/');
    dstBlk = [sys '/' char(dstParts(1))];
    dstPort = str2double(dstParts(2));
    try
        ph = get_param(dstBlk, 'PortHandles');
        if numel(ph.Inport) >= dstPort
            inH = ph.Inport(dstPort);
            oldLine = get_param(inH, 'Line');
            if oldLine ~= -1
                delete_line(oldLine);
            end
        end
    catch
    end
    safe_add_line(sys, src, dst);
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
