function implement_stage1_rmsnorm_qkv(rootDir, options)
%IMPLEMENT_STAGE1_RMSNORM_QKV Build staged internals for qwen2_block_top.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    stageProfile = lower(string(getFieldOr(options, 'StageProfile', 'stage1')));
    useExternalWeightRsp = logical(getFieldOr(options, 'UseExternalWeightRsp', false));
    prefillCfg = resolve_prefill_schedule_config(rootDir, options);

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s. Run create_qwen2_block_top_placeholder first.', mdlPath);
    end

    [~, mdlName] = fileparts(mdlPath);
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
    load_system(mdlPath);
    configure_model_timing(mdlName);

    configure_rmsnorm([mdlName '/rmsnorm_u']);
    configure_qkv_proj([mdlName '/qkv_proj_u']);
    configure_attention([mdlName '/attention_u']);
    configure_ffn_swiglu([mdlName '/ffn_swiglu_u']);
    configure_residual([mdlName '/residual_u']);
    configure_rope([mdlName '/rope_u']);

    if stageProfile == "stage2_memory_ready"
        addpath(fullfile(rootDir, 'simulink', 'fsm'));
        kvCfg = resolve_kv_addr_config(options);
        ensure_weight_bus_objects();
        ensure_stage2_ports(mdlName, useExternalWeightRsp);
        remove_top_level_weight_req_merge_blocks(mdlName);
        remove_unused_top_level_blocks(mdlName);
        ensure_memory_subsystems(mdlName);
        configure_ctrl_fsm([mdlName '/ctrl_fsm_u']);
        configure_kv_cache_if([mdlName '/kv_cache_if_u']);
        configure_kv_addr_gen([mdlName '/kv_addr_gen_u'], kvCfg);
        configure_axi_master_rd([mdlName '/axi_master_rd_u']);
        configure_axi_master_wr([mdlName '/axi_master_wr_u']);
        configure_ddr_model_if([mdlName '/ddr_model_if_u']);
        configure_weight_addr_map([mdlName '/weight_addr_map_u']);
        configure_prefill_scheduler([mdlName '/prefill_sched_u'], prefillCfg);
        configure_axi_weight_rd([mdlName '/axi_weight_rd_u']);
    end

    build_prefill_path(mdlName, stageProfile, useExternalWeightRsp);
    build_decode_path(mdlName, stageProfile);
    build_kv_memory_stubs(mdlName, stageProfile);
    cleanup_disconnected_lines(mdlName);
    arrange_model_systems(mdlName, stageProfile);
    cleanup_disconnected_lines(mdlName);

    save_system(mdlName, mdlPath, 'OverwriteIfChangedOnDisk', true);
    close_system(mdlName, 0);

    fprintf('Implemented %s internals for rmsnorm_u and qkv_proj_u in %s\n', stageProfile, mdlPath);
end

function configure_model_timing(mdlName)
    set_param(mdlName, 'SolverType', 'Fixed-step');
    set_param(mdlName, 'Solver', 'FixedStepDiscrete');
    set_param(mdlName, 'FixedStep', '0.2');
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

function cfg = resolve_prefill_schedule_config(rootDir, options)
    if isfield(options, 'PrefillScheduleConfig') && isstruct(options.PrefillScheduleConfig)
        cfg = options.PrefillScheduleConfig;
        return;
    end

    addpath(fullfile(rootDir, 'scripts'));
    cfg = prefill_attention_schedule_32x32();
end

function build_prefill_path(mdlName, stageProfile, useExternalWeightRsp)
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

        add_or_reset_bus_creator(mdlName, 'rms_addr_bc', 1, [1060, 700, 1100, 730], 'WeightAddrRmsBus');
        add_or_reset_bus_creator(mdlName, 'qkv_addr_bc', 3, [1060, 740, 1100, 810], 'WeightAddrQkvBus');
        add_or_reset_bus_creator(mdlName, 'attn_addr_bc', 3, [1060, 820, 1100, 890], 'WeightAddrAttnBus');
        add_or_reset_bus_creator(mdlName, 'ffn_addr_bc', 2, [1060, 900, 1100, 950], 'WeightAddrFfnBus');

        force_add_line(mdlName, 'weight_addr_map_u/1', 'rms_addr_bc/1');
        force_add_line(mdlName, 'weight_addr_map_u/2', 'qkv_addr_bc/1');
        force_add_line(mdlName, 'weight_addr_map_u/3', 'qkv_addr_bc/2');
        force_add_line(mdlName, 'weight_addr_map_u/4', 'qkv_addr_bc/3');
        force_add_line(mdlName, 'weight_addr_map_u/5', 'attn_addr_bc/1');
        force_add_line(mdlName, 'weight_addr_map_u/6', 'attn_addr_bc/2');
        force_add_line(mdlName, 'weight_addr_map_u/7', 'attn_addr_bc/3');
        force_add_line(mdlName, 'weight_addr_map_u/8', 'ffn_addr_bc/1');
        force_add_line(mdlName, 'weight_addr_map_u/9', 'ffn_addr_bc/2');

        force_add_line(mdlName, 'rms_addr_bc/1', 'rmsnorm_u/4');
        force_add_line(mdlName, 'qkv_addr_bc/1', 'qkv_proj_u/3');
        force_add_line(mdlName, 'attn_addr_bc/1', 'attention_u/3');
        force_add_line(mdlName, 'ffn_addr_bc/1', 'ffn_swiglu_u/3');

        % Shared response bus distributed to each module; wrapper mode can source it externally.
        if useExternalWeightRsp
            force_add_line(mdlName, 'w_rd_rsp_bus/1', 'rmsnorm_u/3');
            force_add_line(mdlName, 'w_rd_rsp_bus/1', 'qkv_proj_u/2');
            force_add_line(mdlName, 'w_rd_rsp_bus/1', 'attention_u/2');
            force_add_line(mdlName, 'w_rd_rsp_bus/1', 'ffn_swiglu_u/2');
        else
            force_add_line(mdlName, 'axi_weight_rd_u/1', 'rmsnorm_u/3');
            force_add_line(mdlName, 'axi_weight_rd_u/1', 'qkv_proj_u/2');
            force_add_line(mdlName, 'axi_weight_rd_u/1', 'attention_u/2');
            force_add_line(mdlName, 'axi_weight_rd_u/1', 'ffn_swiglu_u/2');
        end

        add_or_reset_bus_selector(mdlName, 'rms_req_sel', 'gamma_addr,gamma_valid', [1240, 700, 1280, 750]);
        add_or_reset_bus_selector(mdlName, 'qkv_req_sel', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid', [1240, 760, 1280, 860]);
        add_or_reset_bus_selector(mdlName, 'attn_req_sel', 'attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid', [1240, 870, 1280, 970]);
        add_or_reset_bus_selector(mdlName, 'ffn_req_sel', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid', [1240, 980, 1280, 1060]);
        force_add_line(mdlName, 'rmsnorm_u/2', 'rms_req_sel/1');
        force_add_line(mdlName, 'qkv_proj_u/3', 'qkv_req_sel/1');
        force_add_line(mdlName, 'attention_u/2', 'attn_req_sel/1');
        force_add_line(mdlName, 'ffn_swiglu_u/2', 'ffn_req_sel/1');

        add_or_reset_bus_creator(mdlName, 'w_req_bc', 18, [1320, 900, 1360, 1120]);
        try
            set_param([mdlName '/w_req_bc'], 'InputSignalNames', ...
                'gamma_addr,gamma_valid,qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid,attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid,ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid');
        catch
        end
        force_add_line(mdlName, 'rms_req_sel/1', 'w_req_bc/1');
        force_add_line(mdlName, 'rms_req_sel/2', 'w_req_bc/2');
        force_add_line(mdlName, 'qkv_req_sel/1', 'w_req_bc/3');
        force_add_line(mdlName, 'qkv_req_sel/2', 'w_req_bc/4');
        force_add_line(mdlName, 'qkv_req_sel/3', 'w_req_bc/5');
        force_add_line(mdlName, 'qkv_req_sel/4', 'w_req_bc/6');
        force_add_line(mdlName, 'qkv_req_sel/5', 'w_req_bc/7');
        force_add_line(mdlName, 'qkv_req_sel/6', 'w_req_bc/8');
        force_add_line(mdlName, 'attn_req_sel/1', 'w_req_bc/9');
        force_add_line(mdlName, 'attn_req_sel/2', 'w_req_bc/10');
        force_add_line(mdlName, 'attn_req_sel/3', 'w_req_bc/11');
        force_add_line(mdlName, 'attn_req_sel/4', 'w_req_bc/12');
        force_add_line(mdlName, 'attn_req_sel/5', 'w_req_bc/13');
        force_add_line(mdlName, 'attn_req_sel/6', 'w_req_bc/14');
        force_add_line(mdlName, 'ffn_req_sel/1', 'w_req_bc/15');
        force_add_line(mdlName, 'ffn_req_sel/2', 'w_req_bc/16');
        force_add_line(mdlName, 'ffn_req_sel/3', 'w_req_bc/17');
        force_add_line(mdlName, 'ffn_req_sel/4', 'w_req_bc/18');
        set_line_name_by_src_port(mdlName, 'rms_req_sel', 1, 'gamma_addr');
        set_line_name_by_src_port(mdlName, 'rms_req_sel', 2, 'gamma_valid');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 1, 'qkv_q_addr');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 2, 'qkv_q_valid');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 3, 'qkv_k_addr');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 4, 'qkv_k_valid');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 5, 'qkv_v_addr');
        set_line_name_by_src_port(mdlName, 'qkv_req_sel', 6, 'qkv_v_valid');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 1, 'attn_q_addr');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 2, 'attn_q_valid');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 3, 'attn_k_addr');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 4, 'attn_k_valid');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 5, 'attn_v_addr');
        set_line_name_by_src_port(mdlName, 'attn_req_sel', 6, 'attn_v_valid');
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 1, 'ffn_up_addr');
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 2, 'ffn_up_valid');
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 3, 'ffn_gate_addr');
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 4, 'ffn_gate_valid');
        force_add_line(mdlName, 'w_req_bc/1', 'w_rd_req_bus/1');
        force_add_line(mdlName, 'rms_req_sel/1', 'axi_weight_rd_u/1');
        force_add_line(mdlName, 'rms_req_sel/2', 'axi_weight_rd_u/2');
        force_add_line(mdlName, 'qkv_req_sel/1', 'axi_weight_rd_u/3');
        force_add_line(mdlName, 'qkv_req_sel/2', 'axi_weight_rd_u/4');
        force_add_line(mdlName, 'qkv_req_sel/3', 'axi_weight_rd_u/5');
        force_add_line(mdlName, 'qkv_req_sel/4', 'axi_weight_rd_u/6');
        force_add_line(mdlName, 'qkv_req_sel/5', 'axi_weight_rd_u/7');
        force_add_line(mdlName, 'qkv_req_sel/6', 'axi_weight_rd_u/8');
        force_add_line(mdlName, 'attn_req_sel/1', 'axi_weight_rd_u/9');
        force_add_line(mdlName, 'attn_req_sel/2', 'axi_weight_rd_u/10');
        force_add_line(mdlName, 'attn_req_sel/3', 'axi_weight_rd_u/11');
        force_add_line(mdlName, 'attn_req_sel/4', 'axi_weight_rd_u/12');
        force_add_line(mdlName, 'attn_req_sel/5', 'axi_weight_rd_u/13');
        force_add_line(mdlName, 'attn_req_sel/6', 'axi_weight_rd_u/14');
        force_add_line(mdlName, 'ffn_req_sel/1', 'axi_weight_rd_u/15');
        force_add_line(mdlName, 'ffn_req_sel/2', 'axi_weight_rd_u/16');
        force_add_line(mdlName, 'ffn_req_sel/3', 'axi_weight_rd_u/17');
        force_add_line(mdlName, 'ffn_req_sel/4', 'axi_weight_rd_u/18');

        safe_add_line(mdlName, 'cfg_token_pos/1', 'prefill_sched_u/1');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'prefill_sched_u/2');
        safe_add_line(mdlName, 'mode_decode/1', 'prefill_sched_u/3');

        safe_add_line(mdlName, 'cfg_rope_theta_scale/1', 'rope_u/3');
        safe_add_line(mdlName, 'cfg_rope_sin_mix_scale/1', 'rope_u/4');

        safe_add_line(mdlName, 'qkv_proj_u/1', 'kv_cache_if_u/1');
        safe_add_line(mdlName, 'mode_decode/1', 'kv_cache_if_u/3');
        safe_add_line(mdlName, 'prefill_sched_u/1', 'kv_cache_if_u/4');
        safe_add_line(mdlName, 'prefill_sched_u/2', 'kv_cache_if_u/5');
        safe_add_line(mdlName, 'prefill_sched_u/6', 'kv_cache_if_u/6');
        safe_add_line(mdlName, 'prefill_sched_u/8', 'kv_cache_if_u/7');
        safe_add_line(mdlName, 'prefill_sched_u/9', 'kv_cache_if_u/8');
        safe_add_line(mdlName, 'prefill_sched_u/12', 'kv_cache_if_u/9');
        safe_add_line(mdlName, 'prefill_sched_u/8', 'kv_cache_if_u/10');
        safe_add_line(mdlName, 'prefill_sched_u/9', 'kv_cache_if_u/11');
        force_add_line(mdlName, 'kv_cache_if_u/1', 'attention_u/1');
        force_add_line(mdlName, 'prefill_sched_u/2', 'attention_u/4');
        force_add_line(mdlName, 'prefill_sched_u/3', 'attention_u/5');
        force_add_line(mdlName, 'prefill_sched_u/4', 'attention_u/6');
        force_add_line(mdlName, 'prefill_sched_u/5', 'attention_u/7');
        force_add_line(mdlName, 'prefill_sched_u/7', 'attention_u/8');
        force_add_line(mdlName, 'prefill_sched_u/10', 'attention_u/9');
        force_add_line(mdlName, 'prefill_sched_u/11', 'attention_u/10');
        force_add_line(mdlName, 'prefill_sched_u/12', 'attention_u/11');
        force_add_line(mdlName, 'prefill_sched_u/13', 'attention_u/12');
        force_add_line(mdlName, 'prefill_sched_u/14', 'attention_u/13');
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
                    safe_add_line(mdlName, 'kv_cache_if_u/2', 'axi_master_wr_u/1');
                    safe_add_line(mdlName, 'kv_cache_if_u/3', 'axi_master_wr_u/2');
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

function ensure_stage2_ports(mdlName, useExternalWeightRsp)
    remove_legacy_weight_ports(mdlName);

    ensure_inport(mdlName, 'stop_req', [20, 520, 50, 534]);
    ensure_inport(mdlName, 'kv_mem_rd_ready', [20, 560, 50, 574]);
    ensure_inport(mdlName, 'kv_mem_wr_ready', [20, 600, 50, 614]);
    ensure_inport(mdlName, 'cfg_weight_num_heads', [20, 900, 50, 914]);
    ensure_inport(mdlName, 'cfg_weight_page_base', [20, 940, 50, 954]);
    ensure_inport(mdlName, 'cfg_weight_page_stride', [20, 980, 50, 994]);
    ensure_inport(mdlName, 'cfg_rope_theta_scale', [20, 1020, 50, 1034]);
    ensure_inport(mdlName, 'cfg_rope_sin_mix_scale', [20, 1060, 50, 1074]);
    if useExternalWeightRsp
        ensure_inport(mdlName, 'w_rd_rsp_bus', [20, 1100, 50, 1114]);
    else
        remove_top_block_if_exists(mdlName, 'w_rd_rsp_bus');
    end
    set_root_inport_type(mdlName, 'start', 'boolean');
    set_root_inport_type(mdlName, 'in_valid', 'boolean');
    set_root_inport_type(mdlName, 'out_ready', 'boolean');
    set_root_inport_type(mdlName, 'in_hidden', 'fixdt(1,64,30)');
    set_root_inport_type(mdlName, 'in_residual', 'fixdt(1,64,30)');
    set_root_inport_type(mdlName, 'cfg_seq_len', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_token_pos', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_eps', 'fixdt(1,256,120)');
    set_root_inport_type(mdlName, 'stop_req', 'boolean');
    set_root_inport_type(mdlName, 'kv_cache_rd_valid', 'boolean');
    set_root_inport_type(mdlName, 'kv_mem_rd_ready', 'boolean');
    set_root_inport_type(mdlName, 'kv_mem_wr_ready', 'boolean');
    set_root_inport_type(mdlName, 'cfg_rope_theta_scale', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_rope_sin_mix_scale', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_weight_num_heads', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_weight_page_base', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_weight_page_stride', 'fixdt(1,17,15)');
    if useExternalWeightRsp
        set_root_inport_type(mdlName, 'w_rd_rsp_bus', 'Bus: WeightRspBus');
        try
            set_param([mdlName '/w_rd_rsp_bus'], 'SampleTime', '-1');
        catch
        end
    end

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
    try
        set_param([mdlName '/w_rd_req_bus'], 'OutDataTypeStr', 'Bus: WeightReqBus');
        set_param([mdlName '/w_rd_req_bus'], 'SampleTime', '-1');
    catch
    end
end

function ensure_memory_subsystems(mdlName)
    ensure_subsystem(mdlName, 'prefill_sched_u', [700, 430, 920, 520]);
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

function remove_top_level_weight_req_merge_blocks(mdlName)
    names = {'rms_req_sel', 'qkv_req_sel', 'attn_req_sel', 'ffn_req_sel', 'w_req_bc'};
    for i = 1:numel(names)
        remove_top_block_if_exists(mdlName, names{i});
    end
end

function remove_unused_top_level_blocks(mdlName)
    names = {'w_req_bridge_u', 'w_rsp_demux', 'w_req_v1_u8', 'w_req_v2_u8', 'clk', 'rst_n'};
    for i = 1:numel(names)
        remove_top_block_if_exists(mdlName, names{i});
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
    blk = [mdlName '/' name];
    hits = find_system(mdlName, 'SearchDepth', 1, 'Name', name);
    if ~isempty(hits)
        try
            if ~strcmp(get_param(blk, 'BlockType'), 'Outport')
                delete_block(blk);
                hits = {};
            end
        catch
            try
                delete_block(blk);
                hits = {};
            catch
            end
        end
    end
    if isempty(hits)
        add_block('simulink/Sinks/Out1', blk, 'Position', pos);
    else
        set_param(blk, 'Position', pos);
    end
end

function set_root_inport_type(mdlName, name, dt)
    blk = [mdlName '/' name];
    try
        set_param(blk, 'OutDataTypeStr', dt);
    catch
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
    add_block('simulink/Sources/In1', [subPath '/kv_phase_first'], 'Position', [20, 190, 50, 204]);
    add_block('simulink/Sources/In1', [subPath '/score_scale'], 'Position', [20, 230, 50, 244]);
    add_block('simulink/Sources/In1', [subPath '/x_bank_count'], 'Position', [20, 270, 50, 284]);
    add_block('simulink/Sources/In1', [subPath '/kv_bank_count'], 'Position', [20, 300, 50, 314]);
    add_block('simulink/Sources/In1', [subPath '/tile_seq'], 'Position', [20, 330, 50, 344]);
    add_block('simulink/Sources/In1', [subPath '/active_seq_len'], 'Position', [20, 360, 50, 374]);
    add_block('simulink/Sources/In1', [subPath '/tile_k'], 'Position', [20, 390, 50, 404]);
    add_block('simulink/Sources/In1', [subPath '/tile_out'], 'Position', [20, 420, 50, 434]);
    add_block('simulink/Math Operations/Gain', [subPath '/q_stream_gain'], ...
        'Gain', '0.6', 'Position', [90, 20, 130, 45]);
    add_block('simulink/Math Operations/Gain', [subPath '/k_cache_gain'], ...
        'Gain', '0.4', 'Position', [90, 65, 130, 90]);
    add_block('simulink/Math Operations/Gain', [subPath '/v_cache_gain'], ...
        'Gain', '1.0', 'Position', [90, 110, 130, 135]);
    add_block('simulink/Math Operations/Add', [subPath '/bank_sum'], ...
        'Inputs', '++', 'Position', [90, 285, 125, 315]);
    add_block('simulink/Math Operations/Add', [subPath '/seq_window_sum'], ...
        'Inputs', '++', 'Position', [90, 335, 125, 365]);
    add_block('simulink/Math Operations/Product', [subPath '/bank_addr_base'], ...
        'Inputs', '**', 'Position', [150, 335, 185, 365]);
    add_block('simulink/Math Operations/Add', [subPath '/bank_addr'], ...
        'Inputs', '++', 'Position', [220, 335, 255, 365]);
    add_block('simulink/Math Operations/Product', [subPath '/bank_sel'], ...
        'Inputs', '**', 'Position', [150, 285, 185, 315]);
    add_block('simulink/Signal Routing/Switch', [subPath '/k_cache_sel'], ...
        'Threshold', '0.5', 'Position', [190, 45, 240, 95]);
    add_block('simulink/Signal Routing/Switch', [subPath '/v_cache_sel'], ...
        'Threshold', '0.5', 'Position', [190, 105, 240, 155]);
    add_block('simulink/Math Operations/Add', [subPath '/kv_write_pack'], ...
        'Inputs', '++', 'Position', [280, 95, 315, 125]);
    add_block('simulink/Math Operations/Add', [subPath '/kv_write_banked'], ...
        'Inputs', '++', 'Position', [340, 95, 375, 125]);
    add_block('simulink/Math Operations/Product', [subPath '/kv_write_gate'], ...
        'Inputs', '**', 'Position', [280, 145, 315, 175]);
    add_block('simulink/Math Operations/Product', [subPath '/kv_seq_gate'], ...
        'Inputs', '**', 'Position', [340, 145, 375, 175]);
    add_block('simulink/Math Operations/Add', [subPath '/attn_compose'], ...
        'Inputs', '++++', 'Position', [410, 60, 445, 130]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_to_attn'], 'Position', [500, 85, 530, 99]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_write_data'], 'Position', [500, 145, 530, 159]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_write_valid'], 'Position', [500, 185, 530, 199]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_bank_addr'], 'Position', [500, 225, 530, 239]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_bank_sel'], 'Position', [500, 265, 530, 279]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_bank_wr_en'], 'Position', [500, 305, 530, 319]);

    safe_add_line(subPath, 'qkv_new/1', 'q_stream_gain/1');
    safe_add_line(subPath, 'qkv_new/1', 'k_cache_gain/1');
    safe_add_line(subPath, 'qkv_new/1', 'v_cache_gain/1');
    safe_add_line(subPath, 'x_bank_count/1', 'bank_sum/1');
    safe_add_line(subPath, 'kv_bank_count/1', 'bank_sum/2');
    safe_add_line(subPath, 'tile_seq/1', 'seq_window_sum/1');
    safe_add_line(subPath, 'active_seq_len/1', 'seq_window_sum/2');
    safe_add_line(subPath, 'seq_window_sum/1', 'bank_addr_base/1');
    safe_add_line(subPath, 'tile_k/1', 'bank_addr_base/2');
    safe_add_line(subPath, 'bank_addr_base/1', 'bank_addr/1');
    safe_add_line(subPath, 'tile_out/1', 'bank_addr/2');
    safe_add_line(subPath, 'bank_sum/1', 'bank_sel/1');
    safe_add_line(subPath, 'tile_out/1', 'bank_sel/2');
    safe_add_line(subPath, 'k_cache_gain/1', 'k_cache_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'k_cache_sel/2');
    safe_add_line(subPath, 'kv_hist/1', 'k_cache_sel/3');
    safe_add_line(subPath, 'v_cache_gain/1', 'v_cache_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'v_cache_sel/2');
    safe_add_line(subPath, 'kv_hist/1', 'v_cache_sel/3');
    safe_add_line(subPath, 'k_cache_sel/1', 'kv_write_pack/1');
    safe_add_line(subPath, 'v_cache_sel/1', 'kv_write_pack/2');
    safe_add_line(subPath, 'kv_write_pack/1', 'kv_write_banked/1');
    safe_add_line(subPath, 'bank_sum/1', 'kv_write_banked/2');
    safe_add_line(subPath, 'mode_decode/1', 'kv_write_gate/1');
    safe_add_line(subPath, 'kv_phase_first/1', 'kv_write_gate/2');
    safe_add_line(subPath, 'kv_write_gate/1', 'kv_seq_gate/1');
    safe_add_line(subPath, 'seq_window_sum/1', 'kv_seq_gate/2');
    safe_add_line(subPath, 'q_stream_gain/1', 'attn_compose/1');
    safe_add_line(subPath, 'k_cache_sel/1', 'attn_compose/2');
    safe_add_line(subPath, 'score_scale/1', 'attn_compose/3');
    safe_add_line(subPath, 'seq_window_sum/1', 'attn_compose/4');
    safe_add_line(subPath, 'attn_compose/1', 'kv_to_attn/1');
    safe_add_line(subPath, 'kv_write_banked/1', 'kv_write_data/1');
    safe_add_line(subPath, 'kv_seq_gate/1', 'kv_write_valid/1');
    safe_add_line(subPath, 'bank_addr/1', 'kv_bank_addr/1');
    safe_add_line(subPath, 'bank_sel/1', 'kv_bank_sel/1');
    safe_add_line(subPath, 'kv_seq_gate/1', 'kv_bank_wr_en/1');
end

function configure_prefill_scheduler(subPath, cfg)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/seq_len'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/Sources/Constant', [subPath '/tile_seq_const'], ...
        'Value', num2str(cfg.tile_seq), 'Position', [90, 20, 150, 40]);
    add_block('simulink/Sources/Constant', [subPath '/tile_k_const'], ...
        'Value', num2str(cfg.tile_k), 'Position', [90, 45, 150, 65]);
    add_block('simulink/Sources/Constant', [subPath '/tile_out_const'], ...
        'Value', num2str(cfg.tile_out), 'Position', [90, 70, 150, 90]);
    add_block('simulink/Sources/Constant', [subPath '/array_rows_const'], ...
        'Value', num2str(cfg.array_rows), 'Position', [90, 95, 150, 115]);
    add_block('simulink/Sources/Constant', [subPath '/array_cols_const'], ...
        'Value', num2str(cfg.array_cols), 'Position', [90, 120, 150, 140]);
    add_block('simulink/Sources/Constant', [subPath '/x_banks_const'], ...
        'Value', num2str(cfg.x_bank_count), 'Position', [90, 145, 150, 165]);
    add_block('simulink/Sources/Constant', [subPath '/psum_banks_const'], ...
        'Value', num2str(cfg.psum_bank_count), 'Position', [90, 170, 150, 190]);
    add_block('simulink/Sources/Constant', [subPath '/kv_banks_const'], ...
        'Value', num2str(cfg.kv_bank_count), 'Position', [90, 195, 150, 215]);
    add_block('simulink/Sources/Constant', [subPath '/qpkv_const'], ...
        'Value', num2str(cfg.q_heads_per_kv), 'Position', [90, 220, 150, 240]);
    add_block('simulink/Sources/Constant', [subPath '/kv_first_const'], ...
        'Value', num2str(cfg.kv_phase_first), 'Position', [90, 245, 150, 265]);
    add_block('simulink/Sources/Constant', [subPath '/score_scale_const'], ...
        'Value', num2str(cfg.score_scale), 'Position', [90, 270, 150, 290]);
    add_block('simulink/Sources/Constant', [subPath '/online_softmax_en_const'], ...
        'Value', num2str(cfg.online_softmax_en), 'Position', [90, 295, 170, 315]);
    add_block('simulink/Sources/Constant', [subPath '/scorev_enable_const'], ...
        'Value', num2str(cfg.scorev_enable), 'Position', [90, 320, 170, 340]);
    add_block('simulink/Math Operations/MinMax', [subPath '/active_seq_min'], ...
        'Function', 'min', 'Inputs', '2', 'Position', [190, 55, 230, 95]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_phase_first'], 'Position', [390, 40, 420, 54]);
    add_block('simulink/Sinks/Out1', [subPath '/score_scale'], 'Position', [390, 80, 420, 94]);
    add_block('simulink/Sinks/Out1', [subPath '/q_heads_per_kv'], 'Position', [390, 120, 420, 134]);
    add_block('simulink/Sinks/Out1', [subPath '/array_rows'], 'Position', [390, 160, 420, 174]);
    add_block('simulink/Sinks/Out1', [subPath '/array_cols'], 'Position', [390, 200, 420, 214]);
    add_block('simulink/Sinks/Out1', [subPath '/x_bank_count'], 'Position', [390, 240, 420, 254]);
    add_block('simulink/Sinks/Out1', [subPath '/psum_bank_count'], 'Position', [390, 280, 420, 294]);
    add_block('simulink/Sinks/Out1', [subPath '/kv_bank_count'], 'Position', [390, 320, 420, 334]);
    add_block('simulink/Sinks/Out1', [subPath '/tile_seq'], 'Position', [500, 40, 530, 54]);
    add_block('simulink/Sinks/Out1', [subPath '/tile_k'], 'Position', [500, 80, 530, 94]);
    add_block('simulink/Sinks/Out1', [subPath '/tile_out'], 'Position', [500, 120, 530, 134]);
    add_block('simulink/Sinks/Out1', [subPath '/active_seq_len'], 'Position', [500, 160, 530, 174]);
    add_block('simulink/Sinks/Out1', [subPath '/online_softmax_en'], 'Position', [500, 200, 530, 214]);
    add_block('simulink/Sinks/Out1', [subPath '/scorev_enable'], 'Position', [500, 240, 530, 254]);

    safe_add_line(subPath, 'seq_len/1', 'active_seq_min/1');
    safe_add_line(subPath, 'tile_seq_const/1', 'active_seq_min/2');
    safe_add_line(subPath, 'kv_first_const/1', 'kv_phase_first/1');
    safe_add_line(subPath, 'score_scale_const/1', 'score_scale/1');
    safe_add_line(subPath, 'qpkv_const/1', 'q_heads_per_kv/1');
    safe_add_line(subPath, 'array_rows_const/1', 'array_rows/1');
    safe_add_line(subPath, 'array_cols_const/1', 'array_cols/1');
    safe_add_line(subPath, 'x_banks_const/1', 'x_bank_count/1');
    safe_add_line(subPath, 'psum_banks_const/1', 'psum_bank_count/1');
    safe_add_line(subPath, 'kv_banks_const/1', 'kv_bank_count/1');
    safe_add_line(subPath, 'tile_seq_const/1', 'tile_seq/1');
    safe_add_line(subPath, 'tile_k_const/1', 'tile_k/1');
    safe_add_line(subPath, 'tile_out_const/1', 'tile_out/1');
    safe_add_line(subPath, 'active_seq_min/1', 'active_seq_len/1');
    safe_add_line(subPath, 'online_softmax_en_const/1', 'online_softmax_en/1');
    safe_add_line(subPath, 'scorev_enable_const/1', 'scorev_enable/1');
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

    reqNames = { ...
        'gamma_addr', 'gamma_valid', ...
        'qkv_q_addr', 'qkv_q_valid', 'qkv_k_addr', 'qkv_k_valid', 'qkv_v_addr', 'qkv_v_valid', ...
        'attn_q_addr', 'attn_q_valid', 'attn_k_addr', 'attn_k_valid', 'attn_v_addr', 'attn_v_valid', ...
        'ffn_up_addr', 'ffn_up_valid', 'ffn_gate_addr', 'ffn_gate_valid'};
    for i = 1:numel(reqNames)
        y = 20 + 16 * (i - 1);
        add_block('simulink/Sources/In1', [subPath '/' reqNames{i}], 'Position', [20, y, 50, y + 14]);
    end
    add_or_reset_bus_creator(subPath, 'rsp_bc', 18, [430, 20, 470, 330], 'WeightRspBus');
    try
        set_param([subPath '/rsp_bc'], 'InputSignalNames', ...
            'gamma_data,gamma_valid,qkv_q_data,qkv_q_valid,qkv_k_data,qkv_k_valid,qkv_v_data,qkv_v_valid,attn_q_data,attn_q_valid,attn_k_data,attn_k_valid,attn_v_data,attn_v_valid,ffn_up_data,ffn_up_valid,ffn_gate_data,ffn_gate_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/rsp_bus'], 'Position', [430, 120, 460, 134], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');

    safe_add_line(subPath, 'rsp_bc/1', 'rsp_bus/1');

    scaleVals = {'1.0','0.6','0.4','1.0','0.6','0.4','1.0','1.4','0.9'};
    for i = 1:9
        baseY = 20 + 30 * (i - 1);
        add_block('simulink/Sources/Constant', [subPath '/arready_' num2str(i)], ...
            'Value', '1', 'Position', [150, baseY, 180, baseY + 20]);
        add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/ar_hs_' num2str(i)], ...
            'Operator', 'AND', 'Position', [200, baseY, 230, baseY + 20]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/rdv_d1_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [250, baseY, 280, baseY + 20]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/rdv_d2_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [300, baseY, 330, baseY + 20]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/addr_d1_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [250, baseY + 12, 280, baseY + 32]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/addr_d2_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [300, baseY + 12, 330, baseY + 32]);
        add_block('simulink/Math Operations/Gain', [subPath '/data_gain_' num2str(i)], ...
            'Gain', scaleVals{i}, 'Position', [350, baseY + 10, 390, baseY + 30]);
        add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/data_u8_' num2str(i)], ...
            'OutDataTypeStr', 'uint8', 'Position', [400, baseY + 10, 430, baseY + 30]);

        reqAddrName = reqNames{2 * i - 1};
        reqValName = reqNames{2 * i};
        rspDataPort = 2 * i - 1;
        rspValPort = 2 * i;

        safe_add_line(subPath, [reqValName '/1'], ['ar_hs_' num2str(i) '/1']);
        safe_add_line(subPath, ['arready_' num2str(i) '/1'], ['ar_hs_' num2str(i) '/2']);
        safe_add_line(subPath, ['ar_hs_' num2str(i) '/1'], ['rdv_d1_' num2str(i) '/1']);
        safe_add_line(subPath, ['rdv_d1_' num2str(i) '/1'], ['rdv_d2_' num2str(i) '/1']);

        safe_add_line(subPath, [reqAddrName '/1'], ['addr_d1_' num2str(i) '/1']);
        safe_add_line(subPath, ['addr_d1_' num2str(i) '/1'], ['addr_d2_' num2str(i) '/1']);
        safe_add_line(subPath, ['addr_d2_' num2str(i) '/1'], ['data_gain_' num2str(i) '/1']);
        safe_add_line(subPath, ['data_gain_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1']);

        safe_add_line(subPath, ['data_u8_' num2str(i) '/1'], ['rsp_bc/' num2str(rspDataPort)]);
        safe_add_line(subPath, ['rdv_d2_' num2str(i) '/1'], ['rsp_bc/' num2str(rspValPort)]);
        set_line_name_by_dst_port(subPath, 'rsp_bc', rspDataPort, strrep(reqAddrName, '_addr', '_data'));
        set_line_name_by_dst_port(subPath, 'rsp_bc', rspValPort, reqValName);
    end
end

function add_or_reset_bus_creator(sys, name, n, pos, busObj)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Routing/Bus Creator', p, 'Position', pos);
    end
    set_param(p, 'Inputs', num2str(n), 'Position', pos);
    if nargin >= 5 && strlength(string(busObj)) > 0
        set_param(p, 'UseBusObject', 'on', 'BusObject', busObj);
    else
        try
            set_param(p, 'UseBusObject', 'off');
        catch
        end
        try
            set_param(p, 'BusObject', '');
        catch
        end
        try
            set_param(p, 'NonVirtualBus', 'off');
        catch
        end
    end
end

function add_or_reset_bus_selector(sys, name, outputs, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Routing/Bus Selector', p, 'Position', pos);
    end
    set_param(p, 'OutputSignals', outputs, 'Position', pos);
end

function ensure_weight_bus_objects()
    define_bus('WeightReqRmsBus', {'gamma_addr','gamma_valid'});
    define_bus('WeightReqQkvBus', {'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid'});
    define_bus('WeightReqAttnBus', {'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid'});
    define_bus('WeightReqFfnBus', {'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid'});

    define_bus('WeightAddrRmsBus', {'gamma_addr'});
    define_bus('WeightAddrQkvBus', {'q_addr','k_addr','v_addr'});
    define_bus('WeightAddrAttnBus', {'attn_q_addr','attn_k_addr','attn_v_addr'});
    define_bus('WeightAddrFfnBus', {'up_addr','gate_addr'});

    define_bus('WeightReqBus', {
        'gamma_addr','gamma_valid', ...
        'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid', ...
        'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid', ...
        'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid'});

    define_bus_typed('WeightRspBus', {
        'gamma_data','gamma_valid', ...
        'qkv_q_data','qkv_q_valid','qkv_k_data','qkv_k_valid','qkv_v_data','qkv_v_valid', ...
        'attn_q_data','attn_q_valid','attn_k_data','attn_k_valid','attn_v_data','attn_v_valid', ...
        'ffn_up_data','ffn_up_valid','ffn_gate_data','ffn_gate_valid'});

    define_bus('QkvStreamBus', {'q_stream','k_stream','v_stream','q_valid','kv_valid','group_idx'});
    define_bus('AttentionFlowBus', {'q_stream','k_cache','v_cache','group_idx','score_scale'});
    define_bus('PrefillScheduleBus', {
        'array_rows','array_cols','tile_seq','tile_k','tile_out', ...
        'x_bank_count','psum_bank_count','kv_bank_count','q_heads_per_kv', ...
        'active_seq_len','decode_mode','kv_phase_first','score_scale'});
end

function define_bus(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        elems(i).DataType = 'double';
        elems(i).Dimensions = 1;
    end
    b = Simulink.Bus;
    b.Elements = elems;
    assignin('base', name, b);
end

function define_bus_typed(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        if endsWith(fieldNames{i}, '_valid')
            elems(i).DataType = 'boolean';
        elseif endsWith(fieldNames{i}, '_data')
            elems(i).DataType = 'uint8';
        else
            elems(i).DataType = 'double';
        end
        elems(i).Dimensions = 1;
    end
    b = Simulink.Bus;
    b.Elements = elems;
    assignin('base', name, b);
end

function configure_rmsnorm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 40, 60, 54]);
    add_block('simulink/Sources/In1', [subPath '/eps_in'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/w_rsp_bus'], 'Position', [30, 165, 60, 179], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_block('simulink/Sources/In1', [subPath '/w_addr_bus'], 'Position', [30, 200, 60, 214], ...
        'OutDataTypeStr', 'Bus: WeightAddrRmsBus');
    add_or_reset_bus_selector(subPath, 'rsp_sel', 'gamma_data,gamma_valid', [90, 145, 130, 200]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'gamma_addr', [90, 205, 130, 235]);
    add_block('simulink/Math Operations/Product', [subPath '/x_square'], ...
        'Inputs', '**', 'Position', [110, 35, 145, 65]);
    add_block('simulink/Math Operations/Add', [subPath '/var_eps_sum'], ...
        'Inputs', '++', 'Position', [190, 60, 225, 100]);
    add_block('simulink/Math Operations/Math Function', [subPath '/sqrt_denom'], ...
        'Operator', 'sqrt', 'Position', [260, 65, 300, 95]);
    add_block('simulink/Math Operations/Divide', [subPath '/x_norm'], ...
        'Position', [330, 50, 365, 100]);
    add_or_reset_bus_creator(subPath, 'req_bc', 2, [560, 130, 600, 190], 'WeightReqRmsBus');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'gamma_addr,gamma_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [620, 75, 650, 89]);
    add_block('simulink/Sinks/Out1', [subPath '/w_req_bus'], 'Position', [620, 165, 650, 179], ...
        'OutDataTypeStr', 'Bus: WeightReqRmsBus');

    safe_add_line(subPath, 'x_in/1', 'x_square/1');
    safe_add_line(subPath, 'x_in/1', 'x_square/2');
    safe_add_line(subPath, 'x_square/1', 'var_eps_sum/1');
    safe_add_line(subPath, 'eps_in/1', 'var_eps_sum/2');
    safe_add_line(subPath, 'var_eps_sum/1', 'sqrt_denom/1');
    safe_add_line(subPath, 'x_in/1', 'x_norm/1');
    safe_add_line(subPath, 'sqrt_denom/1', 'x_norm/2');
    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    [gammaOut, reqAddr, reqValid] = add_streamed_weight_mul(subPath, 'gamma', 'x_norm/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 410, 55, '1.0');
    safe_add_line(subPath, reqAddr, 'req_bc/1');
    safe_add_line(subPath, reqValid, 'req_bc/2');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'gamma_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'gamma_valid');
    safe_add_line(subPath, 'req_bc/1', 'w_req_bus/1');
    safe_add_line(subPath, gammaOut, 'y_out/1');
end

function configure_qkv_proj(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/w_rsp_bus'], 'Position', [30, 15, 60, 29], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_block('simulink/Sources/In1', [subPath '/w_addr_bus'], 'Position', [30, 45, 60, 59], ...
        'OutDataTypeStr', 'Bus: WeightAddrQkvBus');
    add_or_reset_bus_selector(subPath, 'rsp_sel', ...
        'qkv_q_data,qkv_q_valid,qkv_k_data,qkv_k_valid,qkv_v_data,qkv_v_valid', [90, 5, 130, 120]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'q_addr,k_addr,v_addr', [90, 130, 130, 190]);
    add_block('simulink/Math Operations/Add', [subPath '/qk_sum'], ...
        'Inputs', '++', 'Position', [500, 45, 535, 85]);
    add_block('simulink/Math Operations/Add', [subPath '/qkv_sum'], ...
        'Inputs', '++', 'Position', [570, 60, 605, 100]);
    add_block('simulink/Sources/Constant', [subPath '/group_idx_const'], ...
        'Value', '0', 'Position', [500, 120, 540, 140]);
    add_block('simulink/Signal Attributes/Signal Conversion', [subPath '/q_valid_alias'], ...
        'Position', [505, 170, 540, 190]);
    add_block('simulink/Signal Attributes/Signal Conversion', [subPath '/kv_valid_alias'], ...
        'Position', [505, 205, 540, 225]);
    add_or_reset_bus_creator(subPath, 'qkv_stream_bc', 6, [560, 245, 600, 355], 'QkvStreamBus');
    try
        set_param([subPath '/qkv_stream_bc'], 'InputSignalNames', ...
            'q_stream,k_stream,v_stream,q_valid,kv_valid,group_idx');
    catch
    end
    add_or_reset_bus_creator(subPath, 'req_bc', 6, [560, 130, 600, 240], 'WeightReqQkvBus');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [650, 75, 680, 89]);
    add_block('simulink/Sinks/Out1', [subPath '/qkv_bus'], 'Position', [650, 145, 680, 159], ...
        'OutDataTypeStr', 'Bus: QkvStreamBus');
    add_block('simulink/Sinks/Out1', [subPath '/w_req_bus'], 'Position', [650, 225, 680, 239], ...
        'OutDataTypeStr', 'Bus: WeightReqQkvBus');

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    [qOut, qReqAddr, qReqValid] = add_streamed_weight_mul(subPath, 'q', 'x_in/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15, '0.6');
    [kOut, kReqAddr, kReqValid] = add_streamed_weight_mul(subPath, 'k', 'x_in/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 105, '0.4');
    [vOut, vReqAddr, vReqValid] = add_streamed_weight_mul(subPath, 'v', 'x_in/1', ...
        'rsp_sel/5', 'rsp_sel/6', 'addr_sel/3', 110, 195, '1.0');

    safe_add_line(subPath, qOut, 'qk_sum/1');
    safe_add_line(subPath, kOut, 'qk_sum/2');
    safe_add_line(subPath, 'qk_sum/1', 'qkv_sum/1');
    safe_add_line(subPath, vOut, 'qkv_sum/2');
    safe_add_line(subPath, 'qkv_sum/1', 'y_out/1');
    safe_add_line(subPath, qOut, 'qkv_stream_bc/1');
    safe_add_line(subPath, kOut, 'qkv_stream_bc/2');
    safe_add_line(subPath, vOut, 'qkv_stream_bc/3');
    safe_add_line(subPath, qReqValid, 'q_valid_alias/1');
    safe_add_line(subPath, kReqValid, 'kv_valid_alias/1');
    safe_add_line(subPath, 'q_valid_alias/1', 'qkv_stream_bc/4');
    safe_add_line(subPath, 'kv_valid_alias/1', 'qkv_stream_bc/5');
    safe_add_line(subPath, 'group_idx_const/1', 'qkv_stream_bc/6');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 1, 'q_stream');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 2, 'k_stream');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 3, 'v_stream');
    set_line_name_by_src_port(subPath, 'q_valid_alias', 1, 'q_valid');
    set_line_name_by_src_port(subPath, 'kv_valid_alias', 1, 'kv_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 4, 'q_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 5, 'kv_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 6, 'group_idx');
    safe_add_line(subPath, 'qkv_stream_bc/1', 'qkv_bus/1');

    safe_add_line(subPath, qReqAddr, 'req_bc/1');
    safe_add_line(subPath, qReqValid, 'req_bc/2');
    safe_add_line(subPath, kReqAddr, 'req_bc/3');
    safe_add_line(subPath, kReqValid, 'req_bc/4');
    safe_add_line(subPath, vReqAddr, 'req_bc/5');
    safe_add_line(subPath, vReqValid, 'req_bc/6');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'qkv_q_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'qkv_q_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'qkv_k_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 4, 'qkv_k_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 5, 'qkv_v_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 6, 'qkv_v_valid');
    safe_add_line(subPath, 'req_bc/1', 'w_req_bus/1');
end

function configure_attention(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/w_rsp_bus'], 'Position', [30, 15, 60, 29], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_block('simulink/Sources/In1', [subPath '/w_addr_bus'], 'Position', [30, 45, 60, 59], ...
        'OutDataTypeStr', 'Bus: WeightAddrAttnBus');
    add_block('simulink/Sources/In1', [subPath '/score_scale'], 'Position', [30, 225, 60, 239]);
    add_block('simulink/Sources/In1', [subPath '/q_heads_per_kv'], 'Position', [30, 255, 60, 269]);
    add_block('simulink/Sources/In1', [subPath '/array_rows'], 'Position', [30, 285, 60, 299]);
    add_block('simulink/Sources/In1', [subPath '/array_cols'], 'Position', [30, 315, 60, 329]);
    add_block('simulink/Sources/In1', [subPath '/psum_bank_count'], 'Position', [30, 345, 60, 359]);
    add_block('simulink/Sources/In1', [subPath '/tile_k'], 'Position', [30, 375, 60, 389]);
    add_block('simulink/Sources/In1', [subPath '/tile_out'], 'Position', [30, 405, 60, 419]);
    add_block('simulink/Sources/In1', [subPath '/active_seq_len'], 'Position', [30, 435, 60, 449]);
    add_block('simulink/Sources/In1', [subPath '/online_softmax_en'], 'Position', [30, 465, 60, 479]);
    add_block('simulink/Sources/In1', [subPath '/scorev_enable'], 'Position', [30, 495, 60, 509]);
    add_or_reset_bus_selector(subPath, 'rsp_sel', ...
        'attn_q_data,attn_q_valid,attn_k_data,attn_k_valid,attn_v_data,attn_v_valid', [90, 5, 130, 120]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'attn_q_addr,attn_k_addr,attn_v_addr', [90, 130, 130, 190]);
    add_block('simulink/Math Operations/Gain', [subPath '/q_head_stream_gain'], ...
        'Gain', '1', 'Position', [180, 165, 220, 190]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/head_group_stage_z'], ...
        'InitialCondition', '0', 'Position', [250, 15, 280, 35]);
    add_block('simulink/Math Operations/Product', [subPath '/score_mul'], ...
        'Inputs', '**', 'Position', [210, 45, 245, 85]);
    add_block('simulink/Math Operations/Abs', [subPath '/score_abs'], ...
        'Position', [280, 45, 315, 75]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/row_max_z'], ...
        'InitialCondition', '0', 'Position', [350, 15, 380, 35]);
    add_block('simulink/Math Operations/MinMax', [subPath '/row_max'], ...
        'Function', 'max', 'Inputs', '2', 'Position', [410, 10, 450, 40]);
    add_block('simulink/Math Operations/Add', [subPath '/score_shift'], ...
        'Inputs', '+-', 'Position', [470, 10, 505, 40]);
    add_block('simulink/Math Operations/Add', [subPath '/array_dim_sum'], ...
        'Inputs', '++', 'Position', [180, 295, 215, 325]);
    add_block('simulink/Math Operations/Add', [subPath '/head_group_bias'], ...
        'Inputs', '++', 'Position', [250, 250, 285, 280]);
    add_block('simulink/Math Operations/Divide', [subPath '/head_group_norm'], ...
        'Position', [310, 250, 345, 285]);
    add_block('simulink/Math Operations/Add', [subPath '/score_den'], ...
        'Inputs', '++', 'Position', [350, 55, 385, 95]);
    add_block('simulink/Math Operations/Product', [subPath '/softmax_gate'], ...
        'Inputs', '**', 'Position', [420, 30, 455, 60]);
    add_block('simulink/Math Operations/Add', [subPath '/softmax_online_den'], ...
        'Inputs', '++', 'Position', [420, 70, 455, 100]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/row_sum_z'], ...
        'InitialCondition', '0', 'Position', [470, 70, 500, 90]);
    add_block('simulink/Math Operations/Add', [subPath '/row_sum_accum'], ...
        'Inputs', '++', 'Position', [520, 70, 555, 100]);
    add_block('simulink/Math Operations/Divide', [subPath '/score_norm'], ...
        'Position', [490, 50, 525, 100]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/softmax_stage_z'], ...
        'InitialCondition', '0', 'Position', [550, 55, 580, 75]);
    add_block('simulink/Math Operations/Product', [subPath '/scorev_gate'], ...
        'Inputs', '**', 'Position', [600, 35, 635, 65]);
    add_block('simulink/Math Operations/Product', [subPath '/value_weight'], ...
        'Inputs', '**', 'Position', [600, 80, 635, 120]);
    add_block('simulink/Math Operations/Divide', [subPath '/scorev_reduce'], ...
        'Position', [670, 80, 705, 120]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/scorev_stage_z'], ...
        'InitialCondition', '0', 'Position', [730, 90, 760, 110]);
    add_or_reset_bus_creator(subPath, 'req_bc', 6, [560, 130, 600, 240], 'WeightReqAttnBus');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [810, 95, 840, 109]);
    add_block('simulink/Sinks/Out1', [subPath '/w_req_bus'], 'Position', [650, 185, 680, 199], ...
        'OutDataTypeStr', 'Bus: WeightReqAttnBus');

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    safe_add_line(subPath, 'x_in/1', 'q_head_stream_gain/1');
    safe_add_line(subPath, 'array_rows/1', 'array_dim_sum/1');
    safe_add_line(subPath, 'array_cols/1', 'array_dim_sum/2');
    safe_add_line(subPath, 'score_scale/1', 'head_group_bias/1');
    safe_add_line(subPath, 'q_heads_per_kv/1', 'head_group_bias/2');
    safe_add_line(subPath, 'head_group_bias/1', 'head_group_norm/1');
    safe_add_line(subPath, 'active_seq_len/1', 'head_group_norm/2');

    [qOut, qReqAddr, qReqValid] = add_streamed_weight_mul(subPath, 'q', 'q_head_stream_gain/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15, '0.6');
    [kOut, kReqAddr, kReqValid] = add_streamed_weight_mul(subPath, 'k', 'x_in/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 95, '0.4');
    [vOut, vReqAddr, vReqValid] = add_streamed_weight_mul(subPath, 'v', 'x_in/1', ...
        'rsp_sel/5', 'rsp_sel/6', 'addr_sel/3', 110, 175, '1.0');

    safe_add_line(subPath, qOut, 'head_group_stage_z/1');
    safe_add_line(subPath, 'head_group_stage_z/1', 'score_mul/1');
    safe_add_line(subPath, kOut, 'score_mul/2');
    safe_add_line(subPath, 'score_mul/1', 'score_abs/1');
    safe_add_line(subPath, 'score_abs/1', 'row_max/1');
    safe_add_line(subPath, 'row_max_z/1', 'row_max/2');
    safe_add_line(subPath, 'row_max/1', 'row_max_z/1');
    safe_add_line(subPath, 'score_abs/1', 'score_shift/1');
    safe_add_line(subPath, 'row_max/1', 'score_shift/2');
    safe_add_line(subPath, 'score_abs/1', 'score_den/1');
    safe_add_line(subPath, 'head_group_norm/1', 'score_den/2');
    safe_add_line(subPath, 'score_shift/1', 'softmax_gate/1');
    safe_add_line(subPath, 'online_softmax_en/1', 'softmax_gate/2');
    safe_add_line(subPath, 'softmax_gate/1', 'softmax_online_den/1');
    safe_add_line(subPath, 'score_den/1', 'softmax_online_den/2');
    safe_add_line(subPath, 'row_sum_z/1', 'row_sum_accum/1');
    safe_add_line(subPath, 'softmax_online_den/1', 'row_sum_accum/2');
    safe_add_line(subPath, 'row_sum_accum/1', 'row_sum_z/1');
    safe_add_line(subPath, 'score_mul/1', 'score_norm/1');
    safe_add_line(subPath, 'row_sum_accum/1', 'score_norm/2');
    safe_add_line(subPath, 'score_norm/1', 'softmax_stage_z/1');
    safe_add_line(subPath, 'softmax_stage_z/1', 'scorev_gate/1');
    safe_add_line(subPath, 'scorev_enable/1', 'scorev_gate/2');
    safe_add_line(subPath, 'scorev_gate/1', 'value_weight/1');
    safe_add_line(subPath, vOut, 'value_weight/2');
    safe_add_line(subPath, 'value_weight/1', 'scorev_reduce/1');
    safe_add_line(subPath, 'psum_bank_count/1', 'scorev_reduce/2');
    safe_add_line(subPath, 'scorev_reduce/1', 'scorev_stage_z/1');
    safe_add_line(subPath, 'scorev_stage_z/1', 'y_out/1');

    safe_add_line(subPath, qReqAddr, 'req_bc/1');
    safe_add_line(subPath, qReqValid, 'req_bc/2');
    safe_add_line(subPath, kReqAddr, 'req_bc/3');
    safe_add_line(subPath, kReqValid, 'req_bc/4');
    safe_add_line(subPath, vReqAddr, 'req_bc/5');
    safe_add_line(subPath, vReqValid, 'req_bc/6');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'attn_q_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'attn_q_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'attn_k_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 4, 'attn_k_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 5, 'attn_v_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 6, 'attn_v_valid');
    safe_add_line(subPath, 'req_bc/1', 'w_req_bus/1');
end

function configure_ffn_swiglu(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Position', [30, 75, 60, 89]);
    add_block('simulink/Sources/In1', [subPath '/w_rsp_bus'], 'Position', [30, 15, 60, 29], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_block('simulink/Sources/In1', [subPath '/w_addr_bus'], 'Position', [30, 45, 60, 59], ...
        'OutDataTypeStr', 'Bus: WeightAddrFfnBus');
    add_or_reset_bus_selector(subPath, 'rsp_sel', 'ffn_up_data,ffn_up_valid,ffn_gate_data,ffn_gate_valid', [90, 5, 130, 90]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'up_addr,gate_addr', [90, 100, 130, 150]);
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
    add_or_reset_bus_creator(subPath, 'req_bc', 4, [560, 130, 600, 210], 'WeightReqFfnBus');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [650, 90, 680, 104]);
    add_block('simulink/Sinks/Out1', [subPath '/w_req_bus'], 'Position', [650, 175, 680, 189], ...
        'OutDataTypeStr', 'Bus: WeightReqFfnBus');

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    [upOut, upReqAddr, upReqValid] = add_streamed_weight_mul(subPath, 'up', 'x_in/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15, '1.4');
    [gateOut, gateReqAddr, gateReqValid] = add_streamed_weight_mul(subPath, 'gate', 'x_in/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 145, '0.9');

    safe_add_line(subPath, gateOut, 'gate_abs/1');
    safe_add_line(subPath, 'gate_abs/1', 'gate_den/1');
    safe_add_line(subPath, 'one_const/1', 'gate_den/2');
    safe_add_line(subPath, gateOut, 'gate_norm/1');
    safe_add_line(subPath, 'gate_den/1', 'gate_norm/2');
    safe_add_line(subPath, upOut, 'swiglu_mul/1');
    safe_add_line(subPath, 'gate_norm/1', 'swiglu_mul/2');
    safe_add_line(subPath, 'swiglu_mul/1', 'down_proj/1');
    safe_add_line(subPath, 'down_proj/1', 'y_out/1');

    safe_add_line(subPath, upReqAddr, 'req_bc/1');
    safe_add_line(subPath, upReqValid, 'req_bc/2');
    safe_add_line(subPath, gateReqAddr, 'req_bc/3');
    safe_add_line(subPath, gateReqValid, 'req_bc/4');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'ffn_up_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'ffn_up_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'ffn_gate_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 4, 'ffn_gate_valid');
    safe_add_line(subPath, 'req_bc/1', 'w_req_bus/1');
end

function [mulOut, reqAddrOutSig, reqValidOutSig] = add_streamed_weight_mul(subPath, prefix, inSig, ddrDataSig, ddrValidSig, reqAddrSig, x0, y0, defaultWeight)
    % Model off-chip DDR fetch + on-chip SRAM cache using HDL RAMS blocks.
    reqAddrCast = [prefix '_req_addr_u8'];
    reqAddrOutCast = [prefix '_req_addr_double'];
    reqAddrDelay = [prefix '_req_addr_z'];
    reqNeeded = [prefix '_req_needed'];
    reqValidCast = [prefix '_req_valid_double'];
    sram = [prefix '_sram'];
    ddrDataCast = [prefix '_ddr_data_u8'];
    ddrValidCast = [prefix '_ddr_valid_bool'];
    sramDataCast = [prefix '_sram_data_double'];
    sramValid = [prefix '_sram_data_valid_z'];
    sramSel = [prefix '_sram_data_sel'];
    validOr = [prefix '_valid_or'];
    defaultConst = [prefix '_default_w'];
    mulBlk = [prefix '_mul'];

    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqAddrCast], ...
        'OutDataTypeStr', 'uint8', 'Position', [x0 + 145, y0 - 2, x0 + 185, y0 + 22]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqAddrOutCast], ...
        'OutDataTypeStr', 'double', 'Position', [x0 + 195, y0 - 2, x0 + 235, y0 + 22]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/' reqAddrDelay], ...
        'InitialCondition', '0', 'Position', [x0 + 245, y0 - 2, x0 + 275, y0 + 22]);

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' reqNeeded], ...
        'Operator', 'NOT', 'Position', [x0 + 5, y0 + 70, x0 + 35, y0 + 90]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqValidCast], ...
        'OutDataTypeStr', 'double', 'Position', [x0 + 45, y0 + 70, x0 + 85, y0 + 90]);
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
    add_block('simulink/Signal Routing/Switch', [subPath '/' sramSel], ...
        'Threshold', '0.5', 'Position', [x0 + 230, y0 + 40, x0 + 280, y0 + 90]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' validOr], ...
        'Operator', 'OR', 'Position', [x0 + 230, y0 + 95, x0 + 260, y0 + 125]);

    add_block('simulink/Math Operations/Product', [subPath '/' mulBlk], ...
        'Inputs', '**', 'Position', [x0 + 305, y0 + 45, x0 + 340, y0 + 85]);

    safe_add_line(subPath, reqAddrSig, [reqAddrCast '/1']);
    safe_add_line(subPath, [reqAddrCast '/1'], [reqAddrOutCast '/1']);
    safe_add_line(subPath, [reqAddrOutCast '/1'], [reqAddrDelay '/1']);
    safe_add_line(subPath, ddrDataSig, [ddrDataCast '/1']);
    safe_add_line(subPath, ddrValidSig, [ddrValidCast '/1']);

    safe_add_line(subPath, [sramValid '/1'], [reqNeeded '/1']);
    safe_add_line(subPath, [reqNeeded '/1'], [reqValidCast '/1']);

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
    reqAddrOutSig = [reqAddrDelay '/1'];
    reqValidOutSig = [reqValidCast '/1'];
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

function cleanup_disconnected_lines(sysPath)
    try
        dangling = find_system(sysPath, 'findall', 'on', 'Type', 'Line', 'Connected', 'off');
        for i = 1:numel(dangling)
            try
                delete_line(dangling(i));
            catch
            end
        end
    catch
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

function set_line_name_by_dst_port(sys, dstBlockName, dstPort, lineName)
    try
        ph = get_param([sys '/' dstBlockName], 'PortHandles');
        if numel(ph.Inport) >= dstPort
            ln = get_param(ph.Inport(dstPort), 'Line');
            if ln ~= -1
                set_param(ln, 'Name', lineName);
            end
        end
    catch
    end
end

function set_line_name_by_src_port(sys, srcBlockName, srcPort, lineName)
    try
        ph = get_param([sys '/' srcBlockName], 'PortHandles');
        if srcPort > numel(ph.Outport)
            return;
        end
        ln = get_param(ph.Outport(srcPort), 'Line');
        if ln ~= -1
            set_param(ln, 'Name', lineName);
            try
                set_param(ln, 'SignalPropagation', 'on');
            catch
            end
        end
    catch
    end
end

function arrange_model_systems(mdlName, stageProfile)
    try
        Simulink.BlockDiagram.arrangeSystem(mdlName);
    catch
    end

    targets = { ...
        [mdlName '/rmsnorm_u'], [mdlName '/qkv_proj_u'], [mdlName '/kv_cache_if_u'], ...
        [mdlName '/attention_u'], [mdlName '/ffn_swiglu_u'], [mdlName '/rope_u'], ...
        [mdlName '/residual_u']};

    if stageProfile == "stage2_memory_ready"
        targets = [targets, { ...
            [mdlName '/prefill_sched_u'], ...
            [mdlName '/axi_weight_rd_u'], [mdlName '/weight_addr_map_u'], ...
            [mdlName '/axi_master_rd_u'], [mdlName '/axi_master_wr_u'], ...
            [mdlName '/kv_addr_gen_u'], [mdlName '/ddr_model_if_u']}];
    end

    for i = 1:numel(targets)
        try
            Simulink.BlockDiagram.arrangeSystem(targets{i});
        catch
        end
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
