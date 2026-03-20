function implement_stage1_rmsnorm_qkv(rootDir, options)
%IMPLEMENT_STAGE1_RMSNORM_QKV Build staged internals for qwen2_block_top.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    stageProfile = lower(string(getFieldOr(options, 'StageProfile', 'stage1')));
    defaultExternalWeightRsp = (stageProfile == "stage2_memory_ready");
    useExternalWeightRsp = logical(getFieldOr(options, 'UseExternalWeightRsp', defaultExternalWeightRsp));
    saveModel = logical(getFieldOr(options, 'SaveModel', true));
    closeModel = logical(getFieldOr(options, 'CloseModel', true));
    applyAttentionScoreNormGuard = logical(getFieldOr(options, 'ApplyAttentionScoreNormGuard', stageProfile == "stage2_memory_ready"));
    useDelayedRamReadAddr = logical(getFieldOr(options, 'UseDelayedRamReadAddr', false));
    ffnGateValidExtraDelay = double(getFieldOr(options, 'FfnGateValidExtraDelay', 0));
    ffnSwigluValidExtraDelay = double(getFieldOr(options, 'FfnSwigluValidExtraDelay', 0));
    prefillCfg = resolve_prefill_schedule_config(rootDir, options);

    mdlPath = char(getFieldOr(options, 'ModelPath', fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx')));
    if ~exist(mdlPath, 'file')
        error('Model not found: %s. Run create_qwen2_block_top_placeholder first.', mdlPath);
    end
    if stageProfile == "stage2_memory_ready" && ~useExternalWeightRsp
        error('implement_stage1_rmsnorm_qkv:Stage2RequiresExternalWeightRsp', ...
            'stage2_memory_ready requires UseExternalWeightRsp=true; internal synthetic weight responses are no longer supported.');
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
    apply_stage2_verification_structure_patches(mdlName, stageProfile, applyAttentionScoreNormGuard, useDelayedRamReadAddr, ffnGateValidExtraDelay, ffnSwigluValidExtraDelay);

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
    end

    build_prefill_path(mdlName, stageProfile);
    build_decode_path(mdlName, stageProfile);
    build_kv_memory_stubs(mdlName, stageProfile);
    terminate_unused_stage2_ports(mdlName, stageProfile, useExternalWeightRsp);
    cleanup_disconnected_lines(mdlName);
    arrange_model_systems(mdlName, stageProfile);
    cleanup_disconnected_lines(mdlName);

    if saveModel
        save_system(mdlName, mdlPath, 'OverwriteIfChangedOnDisk', true);
    end
    if closeModel
        close_system(mdlName, 0);
    end

    fprintf('Implemented %s internals for rmsnorm_u and qkv_proj_u in %s\n', stageProfile, mdlPath);
end

function configure_model_timing(mdlName)
    set_param(mdlName, 'SolverType', 'Fixed-step');
    set_param(mdlName, 'Solver', 'FixedStepDiscrete');
    set_param(mdlName, 'FixedStep', '0.2');
    initCmd = stage2_model_init_callback();
    set_param(mdlName, 'PreLoadFcn', initCmd);
    set_param(mdlName, 'PostLoadFcn', initCmd);
    set_param(mdlName, 'InitFcn', initCmd);
end

function cmd = stage2_model_init_callback()
    cmd = sprintf([ ...
        'try\n' ...
        '  modelFile = get_param(bdroot, ''FileName'');\n' ...
        '  modelDir = fileparts(modelFile);\n' ...
        '  repoRoot = fileparts(fileparts(modelDir));\n' ...
        '  scriptsDir = fullfile(repoRoot, ''scripts'');\n' ...
        '  fsmDir = fullfile(repoRoot, ''simulink'', ''fsm'');\n' ...
        '  if exist(scriptsDir, ''dir'') == 7, addpath(scriptsDir); end\n' ...
        '  if exist(fsmDir, ''dir'') == 7, addpath(fsmDir); end\n' ...
        '  if exist(''ensure_stage2_weight_bus_objects'', ''file'') == 2, ensure_stage2_weight_bus_objects(); end\n' ...
        'catch ME\n' ...
        '  warning(''qwen2_block_top:InitFcnFailed'', ''%%s'', ME.message);\n' ...
        'end']);
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

function build_prefill_path(mdlName, stageProfile)
    safe_add_line(mdlName, 'in_hidden/1', 'rope_u/1');
    safe_add_line(mdlName, 'cfg_token_pos/1', 'rope_u/2');
    force_add_line(mdlName, 'rope_u/1', 'rmsnorm_u/1');
    safe_add_line(mdlName, 'cfg_eps/1', 'rmsnorm_u/2');
    safe_add_line(mdlName, 'rmsnorm_u/1', 'qkv_proj_u/1');
    safe_add_line(mdlName, 'in_valid/1', 'rmsnorm_u/6');

    if stageProfile == "stage2_memory_ready"
        delete_blocks_if_exist(mdlName, { ...
            'gamma_addr_bus_cast', 'q_addr_bus_cast', 'k_addr_bus_cast', 'v_addr_bus_cast', ...
            'attn_q_addr_bus_cast', 'attn_k_addr_bus_cast', 'attn_v_addr_bus_cast', ...
            'up_addr_bus_cast', 'gate_addr_bus_cast', 'down_addr_bus_cast', ...
            'rms_addr_bc', 'qkv_addr_bc', 'attn_addr_bc', 'ffn_addr_bc', 'prefill_sched_bc'});

        safe_add_line(mdlName, 'cfg_token_pos/1', 'weight_addr_map_u/1');
        safe_add_line(mdlName, 'cfg_weight_num_heads/1', 'weight_addr_map_u/2');
        safe_add_line(mdlName, 'cfg_weight_page_base/1', 'weight_addr_map_u/3');
        safe_add_line(mdlName, 'cfg_weight_page_stride/1', 'weight_addr_map_u/4');

        force_add_line(mdlName, 'weight_addr_map_u/1', 'rmsnorm_u/5');
        force_add_line(mdlName, 'weight_addr_map_u/2', 'qkv_proj_u/3');
        force_add_line(mdlName, 'weight_addr_map_u/3', 'attention_u/3');
        force_add_line(mdlName, 'weight_addr_map_u/4', 'ffn_swiglu_u/3');

        % Shared response bus distributed to each module; wrapper mode can source it externally.
        force_add_line(mdlName, 'w_rd_rsp_bus/1', 'rmsnorm_u/3');
        force_add_line(mdlName, 'w_rd_rsp_bus/1', 'qkv_proj_u/2');
        force_add_line(mdlName, 'w_rd_rsp_bus/1', 'attention_u/2');
        force_add_line(mdlName, 'w_rd_rsp_bus/1', 'ffn_swiglu_u/2');

        add_or_reset_bus_selector(mdlName, 'rms_req_sel', 'gamma_addr,gamma_valid', [1240, 700, 1280, 750]);
        add_or_reset_bus_selector(mdlName, 'qkv_req_sel', 'qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid', [1240, 760, 1280, 860]);
        add_or_reset_bus_selector(mdlName, 'attn_req_sel', 'attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid', [1240, 870, 1280, 970]);
        add_or_reset_bus_selector(mdlName, 'ffn_req_sel', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid,ffn_down_addr,ffn_down_valid', [1240, 980, 1280, 1100]);
        force_add_line(mdlName, 'rmsnorm_u/2', 'rms_req_sel/1');
        force_add_line(mdlName, 'qkv_proj_u/3', 'qkv_req_sel/1');
        force_add_line(mdlName, 'attention_u/2', 'attn_req_sel/1');
        force_add_line(mdlName, 'ffn_swiglu_u/2', 'ffn_req_sel/1');

        add_or_reset_bus_creator(mdlName, 'w_req_bc', 20, [1320, 900, 1360, 1160]);
        try
            set_param([mdlName '/w_req_bc'], 'InputSignalNames', ...
            'gamma_addr,gamma_valid,qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid,attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid,ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid,ffn_down_addr,ffn_down_valid');
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
        force_add_line(mdlName, 'ffn_req_sel/5', 'w_req_bc/19');
        force_add_line(mdlName, 'ffn_req_sel/6', 'w_req_bc/20');
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
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 5, 'ffn_down_addr');
        set_line_name_by_src_port(mdlName, 'ffn_req_sel', 6, 'ffn_down_valid');
        force_add_line(mdlName, 'w_req_bc/1', 'w_rd_req_bus/1');

        safe_add_line(mdlName, 'cfg_token_pos/1', 'prefill_sched_u/1');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'prefill_sched_u/2');
        safe_add_line(mdlName, 'mode_decode/1', 'prefill_sched_u/3');

        safe_add_line(mdlName, 'cfg_rope_theta_scale/1', 'rope_u/3');
        safe_add_line(mdlName, 'cfg_rope_sin_mix_scale/1', 'rope_u/4');

        safe_add_line(mdlName, 'in_valid/1', 'qkv_proj_u/4');
        safe_add_line(mdlName, 'qkv_proj_u/1', 'kv_cache_if_u/1');
        safe_add_line(mdlName, 'mode_decode/1', 'kv_cache_if_u/3');
        safe_add_line(mdlName, 'prefill_sched_u/1', 'kv_cache_if_u/4');
        force_add_line(mdlName, 'kv_cache_if_u/1', 'attention_u/1');
        force_add_line(mdlName, 'qkv_proj_u/4', 'attention_u/5');
        force_add_line(mdlName, 'prefill_sched_u/1', 'attention_u/4');
        force_add_line(mdlName, 'attention_u/1', 'ffn_swiglu_u/1');
        force_add_line(mdlName, 'attention_u/3', 'ffn_swiglu_u/4');
        force_add_line(mdlName, 'ffn_swiglu_u/1', 'residual_u/1');
        force_add_line(mdlName, 'ffn_swiglu_u/3', 'residual_u/3');
        force_add_line(mdlName, 'in_residual/1', 'residual_u/2');
        force_add_line(mdlName, 'residual_u/1', 'out_hidden/1');
    else
        safe_add_line(mdlName, 'qkv_proj_u/1', 'out_hidden/1');
    end

    if stageProfile == "stage2_memory_ready"
        configure_stage2_top_protocol(mdlName);
    else
        safe_add_line(mdlName, 'in_valid/1', 'out_valid/1');
        safe_add_line(mdlName, 'out_ready/1', 'in_ready/1');
    end
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
    if stageProfile ~= "stage2_memory_ready"
        safe_add_line(mdlName, 'kv_cache_rd_data/1', 'kv_cache_wr_data/1');
        safe_add_line(mdlName, 'kv_cache_rd_valid/1', 'kv_cache_wr_en/1');
        safe_add_line(mdlName, 'eos_in/1', 'eos_out/1');
    end

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
        force_add_line(mdlName, 'axi_master_wr_u/1', 'kv_cache_wr_data/1');
        force_add_line(mdlName, 'kv_cache_if_u/3', 'kv_cache_wr_en/1');
          force_add_line(mdlName, 'axi_master_wr_u/2', 'kv_mem_wr_addr/1');
          force_add_line(mdlName, 'axi_master_wr_u/3', 'kv_mem_wr_len/1');
          force_add_line(mdlName, 'axi_master_wr_u/4', 'kv_mem_wr_valid/1');

            % Temporary task-level status wiring for V2 ports.
          force_add_line(mdlName, 'ctrl_fsm_u/2', 'busy/1');
          force_add_line(mdlName, 'ctrl_fsm_u/1', 'done/1');
          force_add_line(mdlName, 'ctrl_fsm_u/1', 'irq/1');
          force_add_line(mdlName, 'cfg_token_pos/1', 'error_code/1');
        safe_add_line(mdlName, 'stop_req/1', 'ctrl_fsm_u/3');
        safe_add_line(mdlName, 'cfg_seq_len/1', 'ctrl_fsm_u/4');

        % Feed DDR model interface counter block (3rd fcn-style subsystem).
          safe_add_line(mdlName, 'axi_master_rd_u/5', 'ddr_model_if_u/1');
          safe_add_line(mdlName, 'axi_master_wr_u/4', 'ddr_model_if_u/2');
          safe_add_line(mdlName, 'axi_master_wr_u/5', 'ddr_model_if_u/5');
        safe_add_line(mdlName, 'kv_mem_rd_ready/1', 'ddr_model_if_u/3');
        safe_add_line(mdlName, 'kv_mem_wr_ready/1', 'ddr_model_if_u/4');
    end
end

function configure_stage2_top_protocol(mdlName)
    readyNot = [mdlName '/stage2_in_ready_not'];
    eosGate = [mdlName '/stage2_eos_gate'];
    outFireGate = [mdlName '/stage2_out_fire_gate'];

    if getSimulinkBlockHandle(readyNot) == -1
        add_block('simulink/Logic and Bit Operations/Logical Operator', readyNot, ...
            'Operator', 'NOT', 'Position', [1510, 500, 1540, 530]);
    end
    if getSimulinkBlockHandle(eosGate) == -1
        add_block('simulink/Logic and Bit Operations/Logical Operator', eosGate, ...
            'Operator', 'AND', 'Position', [1510, 540, 1540, 570]);
    end
    if getSimulinkBlockHandle(outFireGate) == -1
        add_block('simulink/Logic and Bit Operations/Logical Operator', outFireGate, ...
            'Operator', 'AND', 'Position', [1510, 580, 1540, 610]);
    end

    safe_add_line(mdlName, 'ctrl_fsm_u/2', 'stage2_in_ready_not/1');
    force_add_line(mdlName, 'stage2_in_ready_not/1', 'in_ready/1');
    force_add_line(mdlName, 'residual_u/2', 'out_valid/1');
    safe_add_line(mdlName, 'residual_u/2', 'stage2_eos_gate/1');
    safe_add_line(mdlName, 'residual_u/2', 'stage2_out_fire_gate/1');
    safe_add_line(mdlName, 'out_ready/1', 'stage2_out_fire_gate/2');
    safe_add_line(mdlName, 'stage2_out_fire_gate/1', 'ctrl_fsm_u/5');
    safe_add_line(mdlName, 'eos_in/1', 'stage2_eos_gate/2');
    force_add_line(mdlName, 'stage2_eos_gate/1', 'eos_out/1');
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
    set_root_inport_type(mdlName, 'mode_decode', 'boolean');
    set_root_inport_type(mdlName, 'eos_in', 'boolean');
    set_root_inport_type(mdlName, 'in_valid', 'boolean');
    set_root_inport_type(mdlName, 'out_ready', 'boolean');
    set_root_inport_type(mdlName, 'in_hidden', 'fixdt(1,64,30)');
    set_root_inport_type(mdlName, 'in_residual', 'fixdt(1,64,30)');
    set_root_inport_type(mdlName, 'kv_cache_rd_data', 'single');
    set_root_inport_type(mdlName, 'cfg_seq_len', 'fixdt(0,17,0)');
    set_root_inport_type(mdlName, 'cfg_token_pos', 'fixdt(0,17,0)');
    set_root_inport_type(mdlName, 'cfg_eps', 'fixdt(1,256,120)');
    set_root_inport_type(mdlName, 'stop_req', 'boolean');
    set_root_inport_type(mdlName, 'kv_cache_rd_valid', 'boolean');
    set_root_inport_type(mdlName, 'kv_mem_rd_ready', 'boolean');
    set_root_inport_type(mdlName, 'kv_mem_wr_ready', 'boolean');
    set_root_inport_type(mdlName, 'cfg_rope_theta_scale', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_rope_sin_mix_scale', 'fixdt(1,17,15)');
    set_root_inport_type(mdlName, 'cfg_weight_num_heads', 'fixdt(1,17,0)');
    set_root_inport_type(mdlName, 'cfg_weight_page_base', 'fixdt(1,17,0)');
    set_root_inport_type(mdlName, 'cfg_weight_page_stride', 'fixdt(1,17,0)');
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
    remove_top_block_if_exists(mdlName, 'axi_weight_rd_u');
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

function apply_stage2_verification_structure_patches(mdlName, stageProfile, applyAttentionScoreNormGuard, useDelayedRamReadAddr, ffnGateValidExtraDelay, ffnSwigluValidExtraDelay)
    if stageProfile ~= "stage2_memory_ready"
        return;
    end

    if applyAttentionScoreNormGuard
        ensure_attention_score_norm_guard(mdlName, 1e-6);
    end
    if useDelayedRamReadAddr
        ensure_ram_read_addr_delay(mdlName);
    end
    if ffnGateValidExtraDelay > 0
        ensure_ffn_gate_valid_delay(mdlName, ffnGateValidExtraDelay);
    end
    if ffnSwigluValidExtraDelay > 0
        ensure_ffn_swiglu_valid_delay(mdlName, ffnSwigluValidExtraDelay);
    end
end

function ensure_attention_score_norm_guard(mdlName, epsilon)
    if nargin < 2 || isempty(epsilon)
        epsilon = 1e-6;
    end

    subPath = [char(mdlName) '/attention_u'];
    scoreNormPath = [subPath '/score_norm'];
    rowSumPath = [subPath '/row_sum_accum'];
    guardPath = [subPath '/row_sum_guard'];
    epsPath = [subPath '/row_sum_guard_eps'];

    if getSimulinkBlockHandle(scoreNormPath) == -1 || getSimulinkBlockHandle(rowSumPath) == -1
        error('implement_stage1_rmsnorm_qkv:MissingAttentionGuardBlocks', ...
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

    safe_delete_line(subPath, 'row_sum_accum/1', 'score_norm/2');
    safe_delete_line(subPath, 'row_sum_accum/1', 'row_sum_guard/1');
    safe_delete_line(subPath, 'row_sum_guard_eps/1', 'row_sum_guard/2');
    safe_delete_line(subPath, 'row_sum_guard/1', 'score_norm/2');

    safe_add_line(subPath, 'row_sum_accum/1', 'row_sum_guard/1');
    safe_add_line(subPath, 'row_sum_guard_eps/1', 'row_sum_guard/2');
    safe_add_line(subPath, 'row_sum_guard/1', 'score_norm/2');
end

function ensure_ffn_gate_valid_delay(mdlName, extraDelay)
    if extraDelay <= 0
        return;
    end

    subPath = [mdlName '/ffn_swiglu_u'];
    safe_delete_line(subPath, 'gateup_pair_valid_z/1', 'gate_norm_gate/2');

    prevBlock = 'gateup_pair_valid_z';
    for i = 1:extraDelay
        delayName = sprintf('gateup_pair_valid_gate_z%d', i);
        delayPath = [subPath '/' delayName];
        if getSimulinkBlockHandle(delayPath) == -1
            add_block('simulink/Discrete/Unit Delay', delayPath, ...
                'InitialCondition', '0', 'Position', [320 + 40 * i, 20, 350 + 40 * i, 45]);
        end
        safe_delete_line(subPath, [prevBlock '/1'], [delayName '/1']);
        safe_add_line(subPath, [prevBlock '/1'], [delayName '/1']);
        prevBlock = delayName;
    end

    safe_delete_line(subPath, [prevBlock '/1'], 'gate_norm_gate/2');
    safe_add_line(subPath, [prevBlock '/1'], 'gate_norm_gate/2');
end

function ensure_ffn_swiglu_valid_delay(mdlName, extraDelay)
    if extraDelay <= 0
        return;
    end

    subPath = [mdlName '/ffn_swiglu_u'];
    safe_delete_line(subPath, 'swiglu_valid_z/1', 'swiglu_stage_gate/2');

    prevBlock = 'swiglu_valid_z';
    for i = 1:extraDelay
        delayName = sprintf('swiglu_valid_gate_z%d', i);
        delayPath = [subPath '/' delayName];
        if getSimulinkBlockHandle(delayPath) == -1
            add_block('simulink/Discrete/Unit Delay', delayPath, ...
                'InitialCondition', '0', 'Position', [540 + 40 * i, 20, 570 + 40 * i, 45]);
        end
        safe_delete_line(subPath, [prevBlock '/1'], [delayName '/1']);
        safe_add_line(subPath, [prevBlock '/1'], [delayName '/1']);
        prevBlock = delayName;
    end

    safe_delete_line(subPath, [prevBlock '/1'], 'swiglu_stage_gate/2');
    safe_add_line(subPath, [prevBlock '/1'], 'swiglu_stage_gate/2');
end

function ensure_ram_read_addr_delay(mdlName)
    specs = {
        struct('path', [mdlName '/rmsnorm_u'], 'prefix', 'gamma'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'q'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'k'), ...
        struct('path', [mdlName '/qkv_proj_u'], 'prefix', 'v'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'q'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'k'), ...
        struct('path', [mdlName '/attention_u'], 'prefix', 'v'), ...
        struct('path', [mdlName '/ffn_swiglu_u'], 'prefix', 'up'), ...
        struct('path', [mdlName '/ffn_swiglu_u'], 'prefix', 'gate')};

    for i = 1:numel(specs)
        subPath = specs{i}.path;
        prefix = specs{i}.prefix;
        sramPath = [subPath '/' prefix '_sram'];
        reqAddrCastPath = [subPath '/' prefix '_req_addr_u8'];
        reqAddrDelayPath = [subPath '/' prefix '_req_addr_z'];
        reqAddrDelayU8Path = [subPath '/' prefix '_req_addr_z_u8_exp'];

        safe_delete_line(subPath, [prefix '_sram_addr_alias/1'], [prefix '_sram/4']);
        safe_delete_line(subPath, [prefix '_req_addr_z_u8_exp/1'], [prefix '_sram/4']);
        if getSimulinkBlockHandle(sramPath) ~= -1 && ...
                getSimulinkBlockHandle(reqAddrCastPath) ~= -1 && ...
                getSimulinkBlockHandle(reqAddrDelayPath) ~= -1
            delete_blocks_if_exist(subPath, {[prefix '_req_addr_z_u8_exp']});
            add_block('simulink/User-Defined Functions/MATLAB Function', reqAddrDelayU8Path, ...
                'Position', [365, 8, 455, 52]);
            set_matlab_function_script(reqAddrDelayU8Path, cache_addr_alias_code(), ...
                'ensure_ram_read_addr_delay:MissingEMChart');
            safe_add_line(subPath, [prefix '_req_addr_z/1'], [prefix '_req_addr_z_u8_exp/1']);
            safe_add_line(subPath, [prefix '_req_addr_z_u8_exp/1'], [prefix '_sram/4']);
        end
    end
end

function safe_delete_line(sysPath, src, dst)
    try
        delete_line(sysPath, src, dst);
    catch
    end
end

function remove_unused_top_level_blocks(mdlName)
    names = {'w_req_bridge_u', 'w_rsp_demux', 'w_req_mux', ...
        'w_req_v1_u8', 'w_req_v2_u8', 'w_req_v3_u8', 'w_req_v4_u8', 'w_req_v5_u8', ...
        'w_req_v6_u8', 'w_req_v7_u8', 'w_req_v8_u8', 'w_req_v9_u8', 'clk', 'rst_n'};
    for i = 1:numel(names)
        remove_top_block_if_exists(mdlName, names{i});
    end
end

function terminate_unused_stage2_ports(mdlName, stageProfile, ~)
    if stageProfile ~= "stage2_memory_ready"
        return;
    end

    specs = {
        'axi_master_rd_u', 2; ...
        'axi_master_rd_u', 6; ...
        'axi_master_wr_u', 1; ...
        'axi_master_wr_u', 5; ...
        'ddr_model_if_u', 1; ...
        'ddr_model_if_u', 2; ...
        'ddr_model_if_u', 3; ...
        'kv_cache_if_u', 4; ...
        'kv_cache_if_u', 5; ...
        'kv_cache_if_u', 6; ...
        'qkv_proj_u', 2};

    for i = 1:size(specs, 1)
        ensure_output_terminator(mdlName, specs{i, 1}, specs{i, 2});
    end
end

function ensure_output_terminator(mdlName, blockName, portIndex)
    srcPath = [mdlName '/' blockName];
    if getSimulinkBlockHandle(srcPath) == -1
        return;
    end

    ph = get_param(srcPath, 'PortHandles');
    if numel(ph.Outport) < portIndex
        return;
    end

    lineHandle = get_param(ph.Outport(portIndex), 'Line');
    if lineHandle ~= -1
        return;
    end

    termName = sprintf('%s_unused_out%d_term', blockName, portIndex);
    termPath = [mdlName '/' termName];
    if getSimulinkBlockHandle(termPath) == -1
        add_block('simulink/Sinks/Terminator', termPath, 'Position', [1400, 100 + 35 * portIndex, 1420, 120 + 35 * portIndex]);
    end
    force_add_line(mdlName, sprintf('%s/%d', blockName, portIndex), [termName '/1']);
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
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/axi_rd_core'], ...
        'Position', [120, 60, 320, 220]);

    add_block('simulink/Sinks/Out1', [subPath '/rd_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dvalid_out'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_avalid'], 'Position', [370, 190, 400, 204]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_dready'], 'Position', [370, 230, 400, 244]);

    rdCode = [ ...
        'function [rd_data_out, rd_dvalid_out, rd_addr, rd_len, rd_avalid, rd_dready] = axi_rd_core(rd_data, rd_aready, rd_dvalid, start, addr_base, burst_len)' newline ...
        '%#codegen' newline ...
        'persistent avalid_state burst_active burst_count' newline ...
        'if isempty(avalid_state)' newline ...
        '    avalid_state = false;' newline ...
        '    burst_active = false;' newline ...
        '    burst_count = int32(0);' newline ...
        'end' newline ...
        'rd_data_out = rd_data;' newline ...
        'rd_addr = addr_base;' newline ...
        'rd_len = burst_len;' newline ...
        'rd_dready = true;' newline ...
        'start_or_hold = logical(start) || avalid_state;' newline ...
        'addr_hs = start_or_hold && logical(rd_aready);' newline ...
        'rd_avalid = start_or_hold;' newline ...
        'beat_fire = burst_active && logical(rd_dvalid);' newline ...
        'count_inc = burst_count + int32(1);' newline ...
        'count_done = count_inc >= int32(burst_len);' newline ...
        'burst_done = beat_fire && count_done;' newline ...
        'rd_dvalid_out = logical(rd_dvalid) && burst_active;' newline ...
        'if beat_fire' newline ...
        '    next_count = count_inc;' newline ...
        'else' newline ...
        '    next_count = burst_count;' newline ...
        'end' newline ...
        'if addr_hs' newline ...
        '    next_count = int32(0);' newline ...
        'end' newline ...
        'avalid_state = start_or_hold && ~addr_hs;' newline ...
        'burst_active = addr_hs || (burst_active && ~burst_done);' newline ...
        'burst_count = next_count;' newline ...
        'end'];
    set_matlab_function_script([subPath '/axi_rd_core'], rdCode, 'configure_axi_master_rd:MissingEMChart');

    safe_add_line(subPath, 'rd_data/1', 'axi_rd_core/1');
    safe_add_line(subPath, 'rd_aready/1', 'axi_rd_core/2');
    safe_add_line(subPath, 'rd_dvalid/1', 'axi_rd_core/3');
    safe_add_line(subPath, 'start/1', 'axi_rd_core/4');
    safe_add_line(subPath, 'addr_base/1', 'axi_rd_core/5');
    safe_add_line(subPath, 'burst_len/1', 'axi_rd_core/6');
    safe_add_line(subPath, 'axi_rd_core/1', 'rd_data_out/1');
    safe_add_line(subPath, 'axi_rd_core/2', 'rd_dvalid_out/1');
    safe_add_line(subPath, 'axi_rd_core/3', 'rd_addr/1');
    safe_add_line(subPath, 'axi_rd_core/4', 'rd_len/1');
    safe_add_line(subPath, 'axi_rd_core/5', 'rd_avalid/1');
    safe_add_line(subPath, 'axi_rd_core/6', 'rd_dready/1');
end

function configure_axi_master_wr(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/wr_data'], 'Position', [20, 30, 50, 44]);
    add_block('simulink/Sources/In1', [subPath '/wr_dvalid'], 'Position', [20, 70, 50, 84]);
    add_block('simulink/Sources/In1', [subPath '/wr_complete'], 'Position', [20, 110, 50, 124]);
    add_block('simulink/Sources/In1', [subPath '/addr_base'], 'Position', [20, 150, 50, 164]);
    add_block('simulink/Sources/In1', [subPath '/burst_len'], 'Position', [20, 190, 50, 204]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/axi_wr_core'], ...
        'Position', [120, 60, 320, 210]);

    add_block('simulink/Sinks/Out1', [subPath '/wr_data_out'], 'Position', [370, 30, 400, 44]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [370, 70, 400, 84]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [370, 110, 400, 124]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_valid'], 'Position', [370, 150, 400, 164]);
    add_block('simulink/Sinks/Out1', [subPath '/request_next_line'], 'Position', [370, 190, 400, 204]);

    wrCode = [ ...
        'function [wr_data_out, wr_addr, wr_len, wr_valid, request_next_line] = axi_wr_core(wr_data, wr_dvalid, wr_complete, addr_base, burst_len)' newline ...
        '%#codegen' newline ...
        'persistent wvalid_state write_active write_count' newline ...
        'if isempty(wvalid_state)' newline ...
        '    wvalid_state = false;' newline ...
        '    write_active = false;' newline ...
        '    write_count = int32(0);' newline ...
        'end' newline ...
        'wr_data_out = wr_data;' newline ...
        'wr_addr = addr_base;' newline ...
        'wr_len = burst_len;' newline ...
        'request_or_hold = logical(wr_dvalid) || wvalid_state;' newline ...
        'start_write = request_or_hold && ~write_active;' newline ...
        'active_or_start = write_active || start_write;' newline ...
        'beat_fire = active_or_start && logical(wr_dvalid);' newline ...
        'count_inc = write_count + int32(1);' newline ...
        'count_done = count_inc >= int32(burst_len);' newline ...
        'write_done = logical(wr_complete) && count_done;' newline ...
        'request_next_line = write_done && write_active;' newline ...
        'wr_valid = request_or_hold;' newline ...
        'if beat_fire' newline ...
        '    next_count = count_inc;' newline ...
        'else' newline ...
        '    next_count = write_count;' newline ...
        'end' newline ...
        'if start_write' newline ...
        '    next_count = int32(0);' newline ...
        'end' newline ...
        'wvalid_state = request_or_hold && ~write_done;' newline ...
        'write_active = start_write || (write_active && ~write_done);' newline ...
        'write_count = next_count;' newline ...
        'end'];
    set_matlab_function_script([subPath '/axi_wr_core'], wrCode, 'configure_axi_master_wr:MissingEMChart');

    safe_add_line(subPath, 'wr_data/1', 'axi_wr_core/1');
    safe_add_line(subPath, 'wr_dvalid/1', 'axi_wr_core/2');
    safe_add_line(subPath, 'wr_complete/1', 'axi_wr_core/3');
    safe_add_line(subPath, 'addr_base/1', 'axi_wr_core/4');
    safe_add_line(subPath, 'burst_len/1', 'axi_wr_core/5');
    safe_add_line(subPath, 'axi_wr_core/1', 'wr_data_out/1');
    safe_add_line(subPath, 'axi_wr_core/2', 'wr_addr/1');
    safe_add_line(subPath, 'axi_wr_core/3', 'wr_len/1');
    safe_add_line(subPath, 'axi_wr_core/4', 'wr_valid/1');
    safe_add_line(subPath, 'axi_wr_core/5', 'request_next_line/1');
end

function configure_ctrl_fsm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/start'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Sources/In1', [subPath '/stop_req'], 'Position', [20, 140, 50, 154]);
    add_block('simulink/Sources/In1', [subPath '/seq_len'], 'Position', [20, 190, 50, 204]);
    add_block('simulink/Sources/In1', [subPath '/out_fire'], 'Position', [20, 240, 50, 254]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/state_z'], ...
        'InitialCondition', '0', 'Position', [120, 210, 150, 240]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/count_z'], ...
        'InitialCondition', '0', 'Position', [120, 255, 150, 285]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/fsm_core'], ...
        'Position', [210, 70, 360, 250]);

    fsmCode = sprintf([ ...
        'function [done_out,busy_out,state_next,count_next] = fsm_core(mode_decode,start,stop_req,seq_len,out_fire,state_in,count_in)\n' ...
        '%%#codegen\n' ...
        '[done_tmp,busy_tmp,state_tmp,count_tmp] = ctrl_fsm_step(mode_decode,start,stop_req,seq_len,out_fire,state_in,count_in);\n' ...
        'done_out = double(done_tmp);\n' ...
        'busy_out = double(busy_tmp);\n' ...
        'state_next = double(state_tmp);\n' ...
        'count_next = double(count_tmp);\n' ...
        'end\n']);
    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subPath '/fsm_core']);
    if isempty(chart)
        error('configure_ctrl_fsm:MissingEMChart', ...
            'Expected MATLAB Function chart at %s/fsm_core', subPath);
    end
    chart.Script = fsmCode;

    add_block('simulink/Sinks/Out1', [subPath '/done_out'], 'Position', [390, 125, 420, 139]);
    add_block('simulink/Sinks/Out1', [subPath '/busy_out'], 'Position', [390, 85, 420, 99]);

    safe_add_line(subPath, 'mode_decode/1', 'fsm_core/1');
    safe_add_line(subPath, 'start/1', 'fsm_core/2');
    safe_add_line(subPath, 'stop_req/1', 'fsm_core/3');
    safe_add_line(subPath, 'seq_len/1', 'fsm_core/4');
    safe_add_line(subPath, 'out_fire/1', 'fsm_core/5');
    safe_add_line(subPath, 'state_z/1', 'fsm_core/6');
    safe_add_line(subPath, 'count_z/1', 'fsm_core/7');
    safe_add_line(subPath, 'fsm_core/1', 'done_out/1');
    safe_add_line(subPath, 'fsm_core/2', 'busy_out/1');
    safe_add_line(subPath, 'fsm_core/3', 'state_z/1');
    safe_add_line(subPath, 'fsm_core/4', 'count_z/1');
end

function configure_ddr_model_if(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/rd_valid'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/wr_valid'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/rd_ready'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/Sources/In1', [subPath '/wr_ready'], 'Position', [20, 160, 50, 174]);
    add_block('simulink/Sources/In1', [subPath '/request_next_line'], 'Position', [20, 200, 50, 214]);

    add_block('simulink/Math Operations/Add', [subPath '/stall_sum'], ...
        'Position', [120, 70, 160, 130]);
    add_block('simulink/Sources/Constant', [subPath '/drop_const'], ...
        'Value', '0', 'Position', [120, 140, 160, 160]);
    add_block('simulink/Math Operations/Add', [subPath '/accepted_sum'], ...
        'Inputs', '+++', 'Position', [120, 170, 160, 230]);

    add_block('simulink/Sinks/Out1', [subPath '/stall_count'], 'Position', [370, 60, 400, 74]);
    add_block('simulink/Sinks/Out1', [subPath '/dropped_burst_count'], 'Position', [370, 100, 400, 114]);
    add_block('simulink/Sinks/Out1', [subPath '/accepted_beats'], 'Position', [370, 140, 400, 154]);

    safe_add_line(subPath, 'rd_valid/1', 'stall_sum/1');
    safe_add_line(subPath, 'wr_valid/1', 'stall_sum/2');
    safe_add_line(subPath, 'stall_sum/1', 'stall_count/1');
    safe_add_line(subPath, 'drop_const/1', 'dropped_burst_count/1');
    safe_add_line(subPath, 'rd_ready/1', 'accepted_sum/1');
    safe_add_line(subPath, 'wr_ready/1', 'accepted_sum/2');
    safe_add_line(subPath, 'request_next_line/1', 'accepted_sum/3');
    safe_add_line(subPath, 'accepted_sum/1', 'accepted_beats/1');
end

function configure_kv_cache_if(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/qkv_new'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/kv_hist'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 140, 50, 154]);
    add_block('simulink/Sources/In1', [subPath '/sched_bus'], 'Position', [20, 190, 50, 204], ...
        'OutDataTypeStr', 'Bus: PrefillScheduleBus');
    add_or_reset_bus_selector(subPath, 'schedule_sel', ...
        'kv_phase_first,score_scale,x_bank_count,kv_bank_count,tile_seq,active_seq_len,tile_k,tile_out', [90, 180, 140, 330]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/q_stream_alias'], ...
        'OutDataTypeStr', 'single', 'Position', [90, 20, 130, 45]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/k_stream_alias'], ...
        'OutDataTypeStr', 'single', 'Position', [90, 65, 130, 90]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/v_stream_alias'], ...
        'OutDataTypeStr', 'single', 'Position', [90, 110, 130, 135]);
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

    safe_add_line(subPath, 'qkv_new/1', 'q_stream_alias/1');
    safe_add_line(subPath, 'qkv_new/1', 'k_stream_alias/1');
    safe_add_line(subPath, 'qkv_new/1', 'v_stream_alias/1');
    safe_add_line(subPath, 'sched_bus/1', 'schedule_sel/1');
    safe_add_line(subPath, 'schedule_sel/3', 'bank_sum/1');
    safe_add_line(subPath, 'schedule_sel/4', 'bank_sum/2');
    safe_add_line(subPath, 'schedule_sel/5', 'seq_window_sum/1');
    safe_add_line(subPath, 'schedule_sel/6', 'seq_window_sum/2');
    safe_add_line(subPath, 'seq_window_sum/1', 'bank_addr_base/1');
    safe_add_line(subPath, 'schedule_sel/7', 'bank_addr_base/2');
    safe_add_line(subPath, 'bank_addr_base/1', 'bank_addr/1');
    safe_add_line(subPath, 'schedule_sel/8', 'bank_addr/2');
    safe_add_line(subPath, 'bank_sum/1', 'bank_sel/1');
    safe_add_line(subPath, 'schedule_sel/8', 'bank_sel/2');
    safe_add_line(subPath, 'k_stream_alias/1', 'k_cache_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'k_cache_sel/2');
    safe_add_line(subPath, 'kv_hist/1', 'k_cache_sel/3');
    safe_add_line(subPath, 'v_stream_alias/1', 'v_cache_sel/1');
    safe_add_line(subPath, 'mode_decode/1', 'v_cache_sel/2');
    safe_add_line(subPath, 'kv_hist/1', 'v_cache_sel/3');
    safe_add_line(subPath, 'k_cache_sel/1', 'kv_write_pack/1');
    safe_add_line(subPath, 'v_cache_sel/1', 'kv_write_pack/2');
    safe_add_line(subPath, 'kv_write_pack/1', 'kv_write_banked/1');
    safe_add_line(subPath, 'bank_sum/1', 'kv_write_banked/2');
    safe_add_line(subPath, 'mode_decode/1', 'kv_write_gate/1');
    safe_add_line(subPath, 'schedule_sel/1', 'kv_write_gate/2');
    safe_add_line(subPath, 'kv_write_gate/1', 'kv_seq_gate/1');
    safe_add_line(subPath, 'seq_window_sum/1', 'kv_seq_gate/2');
    safe_add_line(subPath, 'q_stream_alias/1', 'attn_compose/1');
    safe_add_line(subPath, 'k_cache_sel/1', 'attn_compose/2');
    safe_add_line(subPath, 'schedule_sel/2', 'attn_compose/3');
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
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/sched_core'], ...
        'Position', [100, 40, 290, 210]);
    add_or_reset_bus_creator(subPath, 'schedule_bc', 14, [520, 65, 565, 315], 'PrefillScheduleBus');
    add_block('simulink/Sinks/Out1', [subPath '/schedule_bus'], 'Position', [620, 180, 650, 194], ...
        'OutDataTypeStr', 'Bus: PrefillScheduleBus');

    schedCode = sprintf([ ...
        'function [array_rows,array_cols,tile_seq,tile_k,tile_out,x_bank_count,psum_bank_count,kv_bank_count,q_heads_per_kv,active_seq_len,kv_phase_first,score_scale,online_softmax_en,scorev_enable] = sched_core(token_pos,seq_len,mode_decode)\n' ...
        '%%#codegen\n' ...
        'tile_seq = %d;\n' ...
        'tile_k = %d;\n' ...
        'tile_out = %d;\n' ...
        'array_rows = %d;\n' ...
        'array_cols = %d;\n' ...
        'x_bank_count = %d;\n' ...
        'psum_bank_count = %d;\n' ...
        'kv_bank_count = %d;\n' ...
        'q_heads_per_kv = %d;\n' ...
        'kv_phase_first = %d;\n' ...
        'score_scale = %d;\n' ...
        'online_softmax_en = %d;\n' ...
        'scorev_enable = %d;\n' ...
        'active_seq_len = seq_len;\n' ...
        'if active_seq_len > tile_seq\n' ...
        '    active_seq_len = tile_seq;\n' ...
        'end\n' ...
        'if token_pos < 0 || mode_decode < 0\n' ...
        '    active_seq_len = 0;\n' ...
        'end\n' ...
        'end\n'], ...
        cfg.tile_seq, cfg.tile_k, cfg.tile_out, cfg.array_rows, cfg.array_cols, ...
        cfg.x_bank_count, cfg.psum_bank_count, cfg.kv_bank_count, cfg.q_heads_per_kv, ...
        cfg.kv_phase_first, cfg.score_scale, cfg.online_softmax_en, cfg.scorev_enable);
    set_matlab_function_script([subPath '/sched_core'], schedCode, 'configure_prefill_scheduler:MissingEMChart');

    scheduleFields = {'array_rows','array_cols','tile_seq','tile_k','tile_out', ...
        'x_bank_count','psum_bank_count','kv_bank_count','q_heads_per_kv', ...
        'active_seq_len','kv_phase_first','score_scale', ...
        'online_softmax_en','scorev_enable'};
    for i = 1:14
        castName = [scheduleFields{i} '_cast'];
        y0 = 35 + 20 * (i - 1);
        add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' castName], ...
            'OutDataTypeStr', 'int32', 'Position', [350, y0, 390, y0 + 18]);
        safe_add_line(subPath, sprintf('sched_core/%d', i), [castName '/1']);
        set_line_name_by_src_port(subPath, castName, 1, scheduleFields{i});
    end
    safe_add_line(subPath, 'array_rows_cast/1', 'schedule_bc/1');
    safe_add_line(subPath, 'array_cols_cast/1', 'schedule_bc/2');
    safe_add_line(subPath, 'tile_seq_cast/1', 'schedule_bc/3');
    safe_add_line(subPath, 'tile_k_cast/1', 'schedule_bc/4');
    safe_add_line(subPath, 'tile_out_cast/1', 'schedule_bc/5');
    safe_add_line(subPath, 'x_bank_count_cast/1', 'schedule_bc/6');
    safe_add_line(subPath, 'psum_bank_count_cast/1', 'schedule_bc/7');
    safe_add_line(subPath, 'kv_bank_count_cast/1', 'schedule_bc/8');
    safe_add_line(subPath, 'q_heads_per_kv_cast/1', 'schedule_bc/9');
    safe_add_line(subPath, 'active_seq_len_cast/1', 'schedule_bc/10');
    safe_add_line(subPath, 'kv_phase_first_cast/1', 'schedule_bc/11');
    safe_add_line(subPath, 'score_scale_cast/1', 'schedule_bc/12');
    safe_add_line(subPath, 'online_softmax_en_cast/1', 'schedule_bc/13');
    safe_add_line(subPath, 'scorev_enable_cast/1', 'schedule_bc/14');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 1, 'array_rows');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 2, 'array_cols');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 3, 'tile_seq');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 4, 'tile_k');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 5, 'tile_out');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 6, 'x_bank_count');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 7, 'psum_bank_count');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 8, 'kv_bank_count');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 9, 'q_heads_per_kv');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 10, 'active_seq_len');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 11, 'kv_phase_first');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 12, 'score_scale');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 13, 'online_softmax_en');
    set_line_name_by_dst_port(subPath, 'schedule_bc', 14, 'scorev_enable');
    safe_add_line(subPath, 'schedule_bc/1', 'schedule_bus/1');
end

function configure_kv_addr_gen(subPath, cfg)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/seq_len'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/mode_decode'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/kv_addr_core'], ...
        'Position', [120, 40, 330, 180]);

    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [620, 60, 650, 74]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [620, 100, 650, 114]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [620, 175, 650, 189]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [620, 215, 650, 229]);

    kvAddrCode = sprintf([ ...
        'function [rd_addr, rd_len, wr_addr, wr_len] = kv_addr_core(token_pos, seq_len, mode_decode)\n' ...
        '%%#codegen\n' ...
        'rd_base = int32(%d);\n' ...
        'wr_base = int32(%d);\n' ...
        'stride = int32(%d);\n' ...
        'decode_burst = int32(%d);\n' ...
        'rd_tok_prev = int32(token_pos) - int32(1);\n' ...
        'if rd_tok_prev < 0\n' ...
        '    rd_tok_prev = int32(0);\n' ...
        'end\n' ...
        'rd_addr_decode = rd_tok_prev * stride;\n' ...
        'rd_addr_prefill = int32(token_pos) * stride;\n' ...
        'if logical(mode_decode)\n' ...
        '    rd_addr = rd_base + rd_addr_decode;\n' ...
        '    rd_len = decode_burst;\n' ...
        '    wr_addr = wr_base + (int32(token_pos) + int32(1)) * stride;\n' ...
        '    wr_len = decode_burst;\n' ...
        'else\n' ...
        '    rd_addr = rd_base + rd_addr_prefill;\n' ...
        '    rd_len = int32(seq_len);\n' ...
        '    wr_addr = wr_base + int32(token_pos) * stride;\n' ...
        '    wr_len = int32(seq_len);\n' ...
        'end\n' ...
        'end\n'], cfg.rd_base, cfg.wr_base, cfg.stride_bytes, cfg.decode_burst_len);
    set_matlab_function_script([subPath '/kv_addr_core'], kvAddrCode, 'configure_kv_addr_gen:MissingEMChart');

    safe_add_line(subPath, 'token_pos/1', 'kv_addr_core/1');
    safe_add_line(subPath, 'seq_len/1', 'kv_addr_core/2');
    safe_add_line(subPath, 'mode_decode/1', 'kv_addr_core/3');
    safe_add_line(subPath, 'kv_addr_core/1', 'rd_addr/1');
    safe_add_line(subPath, 'kv_addr_core/2', 'rd_len/1');
    safe_add_line(subPath, 'kv_addr_core/3', 'wr_addr/1');
    safe_add_line(subPath, 'kv_addr_core/4', 'wr_len/1');
end

function configure_weight_addr_map(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/token_pos'], 'Position', [20, 40, 50, 54]);
    add_block('simulink/Sources/In1', [subPath '/num_heads'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Sources/In1', [subPath '/page_base'], 'Position', [20, 120, 50, 134]);
    add_block('simulink/Sources/In1', [subPath '/page_stride'], 'Position', [20, 160, 50, 174]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/addr_map_core'], ...
        'Position', [90, 45, 320, 240]);
    add_or_reset_bus_creator(subPath, 'rms_addr_bc', 1, [430, 40, 470, 70], 'WeightAddrRmsBus');
    add_or_reset_bus_creator(subPath, 'qkv_addr_bc', 3, [430, 90, 470, 160], 'WeightAddrQkvBus');
    add_or_reset_bus_creator(subPath, 'attn_addr_bc', 3, [430, 180, 470, 250], 'WeightAddrAttnBus');
    add_or_reset_bus_creator(subPath, 'ffn_addr_bc', 3, [430, 270, 470, 340], 'WeightAddrFfnBus');
    add_block('simulink/Sinks/Out1', [subPath '/rms_addr_bus'], 'Position', [530, 50, 560, 64], ...
        'OutDataTypeStr', 'Bus: WeightAddrRmsBus');
    add_block('simulink/Sinks/Out1', [subPath '/qkv_addr_bus'], 'Position', [530, 120, 560, 134], ...
        'OutDataTypeStr', 'Bus: WeightAddrQkvBus');
    add_block('simulink/Sinks/Out1', [subPath '/attn_addr_bus'], 'Position', [530, 210, 560, 224], ...
        'OutDataTypeStr', 'Bus: WeightAddrAttnBus');
    add_block('simulink/Sinks/Out1', [subPath '/ffn_addr_bus'], 'Position', [530, 300, 560, 314], ...
        'OutDataTypeStr', 'Bus: WeightAddrFfnBus');

    addrCode = sprintf([ ...
        'function [gamma_addr,q_addr,k_addr,v_addr,attn_q_addr,attn_k_addr,attn_v_addr,up_addr,gate_addr,down_addr] = addr_map_core(token_pos,num_heads,page_base,page_stride)\n' ...
        '%%#codegen\n' ...
        'base_addr = page_base + (token_pos * num_heads) * page_stride;\n' ...
        'gamma_addr = base_addr + 0;\n' ...
        'q_addr = base_addr + 1;\n' ...
        'k_addr = base_addr + 2;\n' ...
        'v_addr = base_addr + 3;\n' ...
        'attn_q_addr = base_addr + 4;\n' ...
        'attn_k_addr = base_addr + 5;\n' ...
        'attn_v_addr = base_addr + 6;\n' ...
        'up_addr = base_addr + 7;\n' ...
        'gate_addr = base_addr + 8;\n' ...
        'down_addr = base_addr + 9;\n' ...
        'end\n']);
    set_matlab_function_script([subPath '/addr_map_core'], addrCode, 'configure_weight_addr_map:MissingEMChart');

    addrNames = {'gamma_addr','q_addr','k_addr','v_addr','attn_q_addr','attn_k_addr','attn_v_addr','up_addr','gate_addr','down_addr'};
    safe_add_line(subPath, 'token_pos/1', 'addr_map_core/1');
    safe_add_line(subPath, 'num_heads/1', 'addr_map_core/2');
    safe_add_line(subPath, 'page_base/1', 'addr_map_core/3');
    safe_add_line(subPath, 'page_stride/1', 'addr_map_core/4');
    for i = 1:numel(addrNames)
        castName = [addrNames{i} '_cast'];
        y0 = 20 + 25 * (i - 1);
        add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' castName], ...
            'OutDataTypeStr', 'int32', 'Position', [345, y0, 385, y0 + 18]);
        safe_add_line(subPath, sprintf('addr_map_core/%d', i), [castName '/1']);
        set_line_name_by_src_port(subPath, castName, 1, addrNames{i});
    end
    safe_add_line(subPath, 'gamma_addr_cast/1', 'rms_addr_bc/1');
    safe_add_line(subPath, 'q_addr_cast/1', 'qkv_addr_bc/1');
    safe_add_line(subPath, 'k_addr_cast/1', 'qkv_addr_bc/2');
    safe_add_line(subPath, 'v_addr_cast/1', 'qkv_addr_bc/3');
    safe_add_line(subPath, 'attn_q_addr_cast/1', 'attn_addr_bc/1');
    safe_add_line(subPath, 'attn_k_addr_cast/1', 'attn_addr_bc/2');
    safe_add_line(subPath, 'attn_v_addr_cast/1', 'attn_addr_bc/3');
    safe_add_line(subPath, 'up_addr_cast/1', 'ffn_addr_bc/1');
    safe_add_line(subPath, 'gate_addr_cast/1', 'ffn_addr_bc/2');
    safe_add_line(subPath, 'down_addr_cast/1', 'ffn_addr_bc/3');
    set_line_name_by_dst_port(subPath, 'rms_addr_bc', 1, 'gamma_addr');
    set_line_name_by_dst_port(subPath, 'qkv_addr_bc', 1, 'q_addr');
    set_line_name_by_dst_port(subPath, 'qkv_addr_bc', 2, 'k_addr');
    set_line_name_by_dst_port(subPath, 'qkv_addr_bc', 3, 'v_addr');
    set_line_name_by_dst_port(subPath, 'attn_addr_bc', 1, 'attn_q_addr');
    set_line_name_by_dst_port(subPath, 'attn_addr_bc', 2, 'attn_k_addr');
    set_line_name_by_dst_port(subPath, 'attn_addr_bc', 3, 'attn_v_addr');
    set_line_name_by_dst_port(subPath, 'ffn_addr_bc', 1, 'up_addr');
    set_line_name_by_dst_port(subPath, 'ffn_addr_bc', 2, 'gate_addr');
    set_line_name_by_dst_port(subPath, 'ffn_addr_bc', 3, 'down_addr');
    safe_add_line(subPath, 'rms_addr_bc/1', 'rms_addr_bus/1');
    safe_add_line(subPath, 'qkv_addr_bc/1', 'qkv_addr_bus/1');
    safe_add_line(subPath, 'attn_addr_bc/1', 'attn_addr_bus/1');
    safe_add_line(subPath, 'ffn_addr_bc/1', 'ffn_addr_bus/1');
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

function add_or_reset_in_bus_element(sys, name, portNum, elementName, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Ports & Subsystems/In Bus Element', p, 'Position', pos);
    end
    set_param(p, 'Port', num2str(portNum), 'Element', elementName, 'Position', pos);
end

function add_or_reset_data_type_conversion(sys, name, outType, pos)
    p = [sys '/' name];
    if isempty(find_system(sys, 'SearchDepth', 1, 'Name', name))
        add_block('simulink/Signal Attributes/Data Type Conversion', p, 'Position', pos);
    end
    set_param(p, 'OutDataTypeStr', outType, 'Position', pos);
end

function ensure_weight_bus_objects()
    define_bus('WeightReqRmsBus', {'gamma_addr','gamma_valid'});
    define_bus('WeightReqQkvBus', {'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid'});
    define_bus('WeightReqAttnBus', {'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid'});
    define_bus('WeightReqFfnBus', {'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid','ffn_down_addr','ffn_down_valid'});

    define_bus_from_specs('WeightAddrRmsBus', {'gamma_addr', 'int32'});
    define_bus_from_specs('WeightAddrQkvBus', {
        'q_addr', 'int32';
        'k_addr', 'int32';
        'v_addr', 'int32'});
    define_bus_from_specs('WeightAddrAttnBus', {
        'attn_q_addr', 'int32';
        'attn_k_addr', 'int32';
        'attn_v_addr', 'int32'});
    define_bus_from_specs('WeightAddrFfnBus', {
        'up_addr', 'int32';
        'gate_addr', 'int32';
        'down_addr', 'int32'});

    define_bus('WeightReqBus', {
        'gamma_addr','gamma_valid', ...
        'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid', ...
        'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid', ...
        'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid','ffn_down_addr','ffn_down_valid'});

    define_bus_typed('WeightRspBus', {
        'gamma_data','gamma_valid', ...
        'qkv_q_data','qkv_q_valid','qkv_k_data','qkv_k_valid','qkv_v_data','qkv_v_valid', ...
        'attn_q_data','attn_q_valid','attn_k_data','attn_k_valid','attn_v_data','attn_v_valid', ...
        'ffn_up_data','ffn_up_valid','ffn_gate_data','ffn_gate_valid','ffn_down_data','ffn_down_valid'});

    define_bus('QkvStreamBus', {'q_stream','k_stream','v_stream','q_valid','kv_valid','group_idx'});
    define_bus('AttentionFlowBus', {'q_stream','k_cache','v_cache','group_idx','score_scale'});
    define_bus_from_specs('PrefillScheduleBus', {
        'array_rows', 'int32';
        'array_cols', 'int32';
        'tile_seq', 'int32';
        'tile_k', 'int32';
        'tile_out', 'int32';
        'x_bank_count', 'int32';
        'psum_bank_count', 'int32';
        'kv_bank_count', 'int32';
        'q_heads_per_kv', 'int32';
        'active_seq_len', 'int32';
        'kv_phase_first', 'int32';
        'score_scale', 'int32';
        'online_softmax_en', 'int32';
        'scorev_enable', 'int32'});
end

function define_bus(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        elems(i).DataType = 'single';
        elems(i).Dimensions = 1;
    end
    b = Simulink.Bus;
    b.Elements = elems;
    assignin('base', name, b);
end

function define_bus_from_specs(name, specs)
    elems = repmat(Simulink.BusElement, size(specs, 1), 1);
    for i = 1:size(specs, 1)
        elems(i).Name = specs{i, 1};
        elems(i).DataType = specs{i, 2};
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
            elems(i).DataType = 'single';
        end
        elems(i).Dimensions = 1;
    end
    b = Simulink.Bus;
    b.Elements = elems;
    assignin('base', name, b);
end

function configure_rmsnorm(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_in'], 'Port', '1', 'Position', [30, 40, 60, 54]);
    add_block('simulink/Sources/In1', [subPath '/eps_in'], 'Port', '2', 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/w_rsp_bus'], 'Position', [30, 165, 60, 179], ...
        'Port', '3', ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_block('simulink/Sources/In1', [subPath '/w_addr_bus'], 'Position', [30, 200, 60, 214], ...
        'Port', '4', ...
        'OutDataTypeStr', 'Bus: WeightAddrRmsBus');
    add_block('simulink/Sources/In1', [subPath '/x_valid'], 'Port', '5', 'Position', [30, 235, 60, 249]);
    add_or_reset_bus_selector(subPath, 'rsp_sel', 'gamma_data,gamma_valid', [90, 145, 130, 200]);
    add_or_reset_in_bus_element(subPath, 'gamma_addr_be', 4, 'gamma_addr', [90, 205, 140, 225]);
    add_block('simulink/Math Operations/Product', [subPath '/x_square'], ...
        'Inputs', '**', 'Position', [110, 35, 145, 65]);
    add_block('simulink/Math Operations/Add', [subPath '/var_eps_sum'], ...
        'Inputs', '++', 'Position', [190, 60, 225, 100]);
    add_block('simulink/Math Operations/Math Function', [subPath '/sqrt_denom'], ...
        'Operator', 'sqrt', 'Position', [260, 65, 300, 95]);
    add_block('simulink/Math Operations/Divide', [subPath '/x_norm'], ...
        'OutDataTypeStr', 'single', ...
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

    [gammaOut, reqAddr, reqValid] = add_streamed_weight_mul(subPath, 'gamma', 'x_norm/1', 'x_valid/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'gamma_addr_be/1', 410, 55);
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
    add_block('simulink/Sources/In1', [subPath '/x_valid'], 'Position', [30, 105, 60, 119]);
    add_or_reset_bus_selector(subPath, 'rsp_sel', ...
        'qkv_q_data,qkv_q_valid,qkv_k_data,qkv_k_valid,qkv_v_data,qkv_v_valid', [90, 5, 130, 120]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'q_addr,k_addr,v_addr', [90, 130, 130, 190]);
    add_block('simulink/Math Operations/Add', [subPath '/qk_sum'], ...
        'Inputs', '++', 'Position', [500, 45, 535, 85]);
    add_block('simulink/Math Operations/Add', [subPath '/qkv_sum'], ...
        'Inputs', '++', 'Position', [570, 60, 605, 100]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/kv_pair_valid'], ...
        'Operator', 'AND', 'Position', [440, 145, 470, 170]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/kv_pair_valid_z'], ...
        'InitialCondition', '0', 'Position', [490, 145, 520, 170]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/x_valid_z'], ...
        'InitialCondition', '0', 'Position', [390, 185, 420, 210]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/fused_qkv_valid'], ...
        'Operator', 'AND', 'Position', [440, 185, 470, 210]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/fused_qkv_valid_z'], ...
        'InitialCondition', '0', 'Position', [490, 185, 520, 210]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/q_valid_alias'], ...
        'OutDataTypeStr', 'single', ...
        'Position', [505, 170, 540, 190]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/kv_valid_alias'], ...
        'OutDataTypeStr', 'single', ...
        'Position', [505, 205, 540, 225]);
    add_block('simulink/Math Operations/Gain', [subPath '/kv_group_gain'], ...
        'Gain', '2', 'Position', [500, 110, 540, 135]);
    add_block('simulink/Math Operations/Add', [subPath '/group_idx_sum'], ...
        'Inputs', '++', 'Position', [560, 110, 595, 140]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/group_idx_alias'], ...
        'OutDataTypeStr', 'single', 'Position', [605, 110, 645, 140]);
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
    add_block('simulink/Sinks/Out1', [subPath '/valid_out'], 'Position', [650, 185, 680, 199]);

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    [qOut, qReqAddr, qReqValid, ~] = add_streamed_weight_mul(subPath, 'q', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15);
    [kOut, kReqAddr, kReqValid, kDataValid] = add_streamed_weight_mul(subPath, 'k', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 105);
    [vOut, vReqAddr, vReqValid, vDataValid] = add_streamed_weight_mul(subPath, 'v', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/5', 'rsp_sel/6', 'addr_sel/3', 110, 195);

    safe_add_line(subPath, qOut, 'qk_sum/1');
    safe_add_line(subPath, kOut, 'qk_sum/2');
    safe_add_line(subPath, 'qk_sum/1', 'qkv_sum/1');
    safe_add_line(subPath, vOut, 'qkv_sum/2');
    safe_add_line(subPath, 'qkv_sum/1', 'y_out/1');
    safe_add_line(subPath, qOut, 'qkv_stream_bc/1');
    safe_add_line(subPath, kOut, 'qkv_stream_bc/2');
    safe_add_line(subPath, vOut, 'qkv_stream_bc/3');
    safe_add_line(subPath, kDataValid, 'kv_pair_valid/1');
    safe_add_line(subPath, vDataValid, 'kv_pair_valid/2');
    safe_add_line(subPath, 'kv_pair_valid/1', 'kv_pair_valid_z/1');
    safe_add_line(subPath, 'x_valid/1', 'x_valid_z/1');
    safe_add_line(subPath, 'x_valid_z/1', 'fused_qkv_valid/1');
    safe_add_line(subPath, 'kv_pair_valid_z/1', 'fused_qkv_valid/2');
    safe_add_line(subPath, 'fused_qkv_valid/1', 'fused_qkv_valid_z/1');
    safe_add_line(subPath, 'fused_qkv_valid_z/1', 'q_valid_alias/1');
    safe_add_line(subPath, 'kv_pair_valid_z/1', 'kv_valid_alias/1');
    safe_add_line(subPath, 'q_valid_alias/1', 'qkv_stream_bc/4');
    safe_add_line(subPath, 'kv_valid_alias/1', 'qkv_stream_bc/5');
    safe_add_line(subPath, 'kv_valid_alias/1', 'kv_group_gain/1');
    safe_add_line(subPath, 'q_valid_alias/1', 'group_idx_sum/1');
    safe_add_line(subPath, 'kv_group_gain/1', 'group_idx_sum/2');
    safe_add_line(subPath, 'group_idx_sum/1', 'group_idx_alias/1');
    safe_add_line(subPath, 'group_idx_alias/1', 'qkv_stream_bc/6');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 1, 'q_stream');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 2, 'k_stream');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 3, 'v_stream');
    set_line_name_by_src_port(subPath, 'q_valid_alias', 1, 'q_valid');
    set_line_name_by_src_port(subPath, 'kv_valid_alias', 1, 'kv_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 4, 'q_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 5, 'kv_valid');
    set_line_name_by_dst_port(subPath, 'qkv_stream_bc', 6, 'group_idx');
    set_line_name_by_src_port(subPath, 'kv_pair_valid', 1, 'kv_pair_valid');
    set_line_name_by_src_port(subPath, 'kv_pair_valid_z', 1, 'kv_pair_valid_z');
    set_line_name_by_src_port(subPath, 'fused_qkv_valid', 1, 'fused_qkv_valid');
    set_line_name_by_src_port(subPath, 'fused_qkv_valid_z', 1, 'fused_qkv_valid_z');
    set_line_name_by_src_port(subPath, 'group_idx_alias', 1, 'group_idx');
    safe_add_line(subPath, 'qkv_stream_bc/1', 'qkv_bus/1');
    safe_add_line(subPath, 'fused_qkv_valid_z/1', 'valid_out/1');

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
    add_block('simulink/Sources/In1', [subPath '/sched_bus'], 'Position', [30, 225, 60, 239], ...
        'OutDataTypeStr', 'Bus: PrefillScheduleBus');
    add_block('simulink/Sources/In1', [subPath '/x_valid'], 'Position', [30, 105, 60, 119]);
    add_or_reset_bus_selector(subPath, 'rsp_sel', ...
        'attn_q_data,attn_q_valid,attn_k_data,attn_k_valid,attn_v_data,attn_v_valid', [90, 5, 130, 120]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'attn_q_addr,attn_k_addr,attn_v_addr', [90, 130, 130, 190]);
    add_or_reset_bus_selector(subPath, 'schedule_sel', ...
        'score_scale,q_heads_per_kv,array_rows,array_cols,psum_bank_count,tile_k,tile_out,active_seq_len,online_softmax_en,scorev_enable', [90, 225, 140, 390]);
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
    add_block('simulink/Math Operations/Add', [subPath '/head_group_den'], ...
        'Inputs', '++', 'Position', [250, 295, 285, 325]);
    add_block('simulink/Math Operations/Divide', [subPath '/head_group_norm'], ...
        'Position', [310, 250, 345, 285]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/qk_pair_valid'], ...
        'Operator', 'AND', 'Position', [250, 95, 280, 120]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/qk_pair_valid_z'], ...
        'InitialCondition', '0', 'Position', [300, 95, 330, 120]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/x_valid_z'], ...
        'InitialCondition', '0', 'Position', [200, 95, 230, 120]);
    add_block('simulink/Math Operations/Product', [subPath '/score_stage_gate'], ...
        'Inputs', '**', 'Position', [340, 55, 375, 95]);
    add_block('simulink/Math Operations/Add', [subPath '/score_tile_bias'], ...
        'Inputs', '++', 'Position', [350, 250, 385, 280]);
    add_block('simulink/Math Operations/Add', [subPath '/score_den_pre'], ...
        'Inputs', '+', 'Position', [350, 55, 385, 95]);
    add_block('simulink/Math Operations/Add', [subPath '/score_den'], ...
        'Inputs', '++', 'Position', [420, 55, 455, 95]);
    add_block('simulink/Math Operations/Product', [subPath '/softmax_gate'], ...
        'Inputs', '**', 'Position', [420, 30, 455, 60]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/softmax_valid'], ...
        'Operator', 'AND', 'Position', [470, 110, 500, 135]);
    add_block('simulink/Math Operations/Add', [subPath '/softmax_online_den'], ...
        'Inputs', '++', 'Position', [420, 70, 455, 100]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/row_sum_z'], ...
        'InitialCondition', '0', 'Position', [470, 70, 500, 90]);
    add_block('simulink/Math Operations/Add', [subPath '/row_sum_accum'], ...
        'Inputs', '++', 'Position', [520, 70, 555, 100]);
    add_block('simulink/Math Operations/Divide', [subPath '/score_norm'], ...
        'Position', [490, 50, 525, 100]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/softmax_valid_z'], ...
        'InitialCondition', '0', 'Position', [550, 100, 580, 125]);
    add_block('simulink/Math Operations/Product', [subPath '/softmax_value_gate'], ...
        'Inputs', '**', 'Position', [550, 25, 585, 55]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/softmax_stage_z'], ...
        'InitialCondition', '0', 'Position', [550, 55, 580, 75]);
    add_block('simulink/Math Operations/Product', [subPath '/scorev_gate'], ...
        'Inputs', '**', 'Position', [600, 35, 635, 65]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/scorev_input_valid'], ...
        'Operator', 'AND', 'Position', [600, 130, 630, 155]);
    add_block('simulink/Math Operations/Product', [subPath '/value_weight'], ...
        'Inputs', '**', 'Position', [600, 80, 635, 120]);
    add_block('simulink/Math Operations/Add', [subPath '/scorev_den'], ...
        'Inputs', '++', 'Position', [650, 130, 685, 160]);
    add_block('simulink/Math Operations/Divide', [subPath '/scorev_reduce'], ...
        'Position', [670, 80, 705, 120]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/scorev_valid_z'], ...
        'InitialCondition', '0', 'Position', [730, 130, 760, 155]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/scorev_valid_out_z'], ...
        'InitialCondition', '0', 'Position', [780, 130, 810, 155]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/y_valid_z'], ...
        'InitialCondition', '0', 'Position', [780, 160, 810, 185]);
    add_block('simulink/Math Operations/Product', [subPath '/output_valid_gate'], ...
        'Inputs', '**', 'Position', [730, 80, 765, 120]);
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
    add_block('simulink/Sinks/Out1', [subPath '/valid_out'], 'Position', [810, 130, 840, 144]);

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');
    safe_add_line(subPath, 'sched_bus/1', 'schedule_sel/1');

    safe_add_line(subPath, 'schedule_sel/3', 'array_dim_sum/1');
    safe_add_line(subPath, 'schedule_sel/4', 'array_dim_sum/2');
    safe_add_line(subPath, 'schedule_sel/1', 'head_group_bias/1');
    safe_add_line(subPath, 'schedule_sel/2', 'head_group_bias/2');
    safe_add_line(subPath, 'schedule_sel/8', 'head_group_den/1');
    safe_add_line(subPath, 'array_dim_sum/1', 'head_group_den/2');
    safe_add_line(subPath, 'head_group_bias/1', 'head_group_norm/1');
    safe_add_line(subPath, 'head_group_den/1', 'head_group_norm/2');

    [qOut, qReqAddr, qReqValid, qDataValid] = add_streamed_weight_mul(subPath, 'q', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15);
    [kOut, kReqAddr, kReqValid, kDataValid] = add_streamed_weight_mul(subPath, 'k', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 95);
    [vOut, vReqAddr, vReqValid, vDataValid] = add_streamed_weight_mul(subPath, 'v', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/5', 'rsp_sel/6', 'addr_sel/3', 110, 175);

    safe_add_line(subPath, qOut, 'head_group_stage_z/1');
    safe_add_line(subPath, 'head_group_stage_z/1', 'score_mul/1');
    safe_add_line(subPath, kOut, 'score_mul/2');
    safe_add_line(subPath, 'x_valid/1', 'x_valid_z/1');
    safe_add_line(subPath, qDataValid, 'qk_pair_valid/1');
    safe_add_line(subPath, kDataValid, 'qk_pair_valid/2');
    safe_add_line(subPath, 'qk_pair_valid/1', 'qk_pair_valid_z/1');
    safe_add_line(subPath, 'score_mul/1', 'score_stage_gate/1');
    safe_add_line(subPath, 'qk_pair_valid_z/1', 'score_stage_gate/2');
    safe_add_line(subPath, 'score_stage_gate/1', 'score_abs/1');
    safe_add_line(subPath, 'score_abs/1', 'row_max/1');
    safe_add_line(subPath, 'row_max_z/1', 'row_max/2');
    safe_add_line(subPath, 'row_max/1', 'row_max_z/1');
    safe_add_line(subPath, 'score_abs/1', 'score_shift/1');
    safe_add_line(subPath, 'row_max/1', 'score_shift/2');
    safe_add_line(subPath, 'score_abs/1', 'score_den_pre/1');
    safe_add_line(subPath, 'head_group_norm/1', 'score_tile_bias/1');
    safe_add_line(subPath, 'schedule_sel/6', 'score_tile_bias/2');
    safe_add_line(subPath, 'score_den_pre/1', 'score_den/1');
    safe_add_line(subPath, 'score_tile_bias/1', 'score_den/2');
    safe_add_line(subPath, 'score_shift/1', 'softmax_gate/1');
    safe_add_line(subPath, 'schedule_sel/9', 'softmax_gate/2');
    safe_add_line(subPath, 'qk_pair_valid_z/1', 'softmax_valid/1');
    safe_add_line(subPath, 'schedule_sel/9', 'softmax_valid/2');
    safe_add_line(subPath, 'softmax_gate/1', 'softmax_online_den/1');
    safe_add_line(subPath, 'score_den/1', 'softmax_online_den/2');
    safe_add_line(subPath, 'row_sum_z/1', 'row_sum_accum/1');
    safe_add_line(subPath, 'softmax_online_den/1', 'row_sum_accum/2');
    safe_add_line(subPath, 'row_sum_accum/1', 'row_sum_z/1');
    safe_add_line(subPath, 'score_stage_gate/1', 'score_norm/1');
    safe_add_line(subPath, 'row_sum_accum/1', 'score_norm/2');
    safe_add_line(subPath, 'softmax_valid/1', 'softmax_valid_z/1');
    safe_add_line(subPath, 'score_norm/1', 'softmax_value_gate/1');
    safe_add_line(subPath, 'softmax_valid_z/1', 'softmax_value_gate/2');
    safe_add_line(subPath, 'softmax_value_gate/1', 'softmax_stage_z/1');
    safe_add_line(subPath, 'softmax_stage_z/1', 'scorev_gate/1');
    safe_add_line(subPath, 'schedule_sel/10', 'scorev_gate/2');
    safe_add_line(subPath, 'softmax_valid_z/1', 'scorev_input_valid/1');
    safe_add_line(subPath, vDataValid, 'scorev_input_valid/2');
    safe_add_line(subPath, 'scorev_gate/1', 'value_weight/1');
    safe_add_line(subPath, vOut, 'value_weight/2');
    safe_add_line(subPath, 'schedule_sel/5', 'scorev_den/1');
    safe_add_line(subPath, 'schedule_sel/7', 'scorev_den/2');
    safe_add_line(subPath, 'value_weight/1', 'scorev_reduce/1');
    safe_add_line(subPath, 'scorev_den/1', 'scorev_reduce/2');
    safe_add_line(subPath, 'scorev_input_valid/1', 'scorev_valid_z/1');
    safe_add_line(subPath, 'scorev_valid_z/1', 'scorev_valid_out_z/1');
    safe_add_line(subPath, 'scorev_valid_out_z/1', 'y_valid_z/1');
    safe_add_line(subPath, 'scorev_reduce/1', 'output_valid_gate/1');
    safe_add_line(subPath, 'scorev_valid_out_z/1', 'output_valid_gate/2');
    safe_add_line(subPath, 'output_valid_gate/1', 'scorev_stage_z/1');
    safe_add_line(subPath, 'scorev_stage_z/1', 'y_out/1');
    safe_add_line(subPath, 'y_valid_z/1', 'valid_out/1');

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
    add_block('simulink/Sources/In1', [subPath '/x_valid'], 'Position', [30, 105, 60, 119]);
    add_or_reset_bus_selector(subPath, 'rsp_sel', 'ffn_up_data,ffn_up_valid,ffn_gate_data,ffn_gate_valid,ffn_down_data,ffn_down_valid', [90, 5, 130, 120]);
    add_or_reset_bus_selector(subPath, 'addr_sel', 'up_addr,gate_addr,down_addr', [90, 135, 130, 195]);
    add_block('simulink/Math Operations/Abs', [subPath '/gate_abs'], ...
        'Position', [210, 85, 245, 115]);
    add_block('simulink/Sources/Constant', [subPath '/one_const'], ...
        'Value', '1', 'Position', [210, 125, 250, 145]);
    add_block('simulink/Math Operations/Add', [subPath '/gate_den'], ...
        'Inputs', '++', 'Position', [280, 95, 315, 125]);
    add_block('simulink/Math Operations/Divide', [subPath '/gate_norm'], ...
        'Position', [350, 85, 385, 135]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/gateup_pair_valid'], ...
        'Operator', 'AND', 'Position', [210, 20, 240, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/x_valid_z'], ...
        'InitialCondition', '0', 'Position', [155, 20, 185, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/gateup_pair_valid_z'], ...
        'InitialCondition', '0', 'Position', [265, 20, 295, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/gateup_pair_valid_gate_z1'], ...
        'InitialCondition', '0', 'Position', [320, 20, 350, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/gateup_pair_valid_gate_z2'], ...
        'InitialCondition', '0', 'Position', [375, 20, 405, 45]);
    add_block('simulink/Math Operations/Product', [subPath '/gate_norm_gate'], ...
        'Inputs', '**', 'Position', [400, 85, 435, 135]);
    add_block('simulink/Math Operations/Product', [subPath '/swiglu_mul'], ...
        'Inputs', '**', 'Position', [430, 70, 465, 120]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/swiglu_valid_z'], ...
        'InitialCondition', '0', 'Position', [490, 20, 520, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/swiglu_valid_gate_z1'], ...
        'InitialCondition', '0', 'Position', [545, 20, 575, 45]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/swiglu_valid_gate_z2'], ...
        'InitialCondition', '0', 'Position', [600, 20, 630, 45]);
    add_block('simulink/Math Operations/Product', [subPath '/swiglu_stage_gate'], ...
        'Inputs', '**', 'Position', [500, 115, 535, 145]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/down_pair_valid_z1'], ...
        'InitialCondition', '0', 'Position', [570, 50, 600, 75]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/down_pair_valid_z2'], ...
        'InitialCondition', '0', 'Position', [625, 50, 655, 75]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/down_pair_valid'], ...
        'Operator', 'AND', 'Position', [520, 50, 550, 75]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/down_valid_z'], ...
        'InitialCondition', '0', 'Position', [570, 20, 600, 45]);
    add_block('simulink/Math Operations/Product', [subPath '/down_stage_gate'], ...
        'Inputs', '**', 'Position', [580, 80, 615, 110]);
    add_or_reset_bus_creator(subPath, 'req_bc', 6, [560, 130, 600, 250], 'WeightReqFfnBus');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid,ffn_down_addr,ffn_down_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [650, 90, 680, 104]);
    add_block('simulink/Sinks/Out1', [subPath '/w_req_bus'], 'Position', [650, 175, 680, 189], ...
        'OutDataTypeStr', 'Bus: WeightReqFfnBus');
    add_block('simulink/Sinks/Out1', [subPath '/valid_out'], 'Position', [650, 130, 680, 144]);

    safe_add_line(subPath, 'w_rsp_bus/1', 'rsp_sel/1');
    safe_add_line(subPath, 'w_addr_bus/1', 'addr_sel/1');

    [upOut, upReqAddr, upReqValid, upDataValid] = add_streamed_weight_mul(subPath, 'up', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/1', 'rsp_sel/2', 'addr_sel/1', 110, 15);
    [gateOut, gateReqAddr, gateReqValid, gateDataValid] = add_streamed_weight_mul(subPath, 'gate', 'x_in/1', 'x_valid/1', ...
        'rsp_sel/3', 'rsp_sel/4', 'addr_sel/2', 110, 145);
    [downOut, downReqAddr, downReqValid, downDataValid] = add_streamed_weight_mul(subPath, 'down', 'swiglu_stage_gate/1', 'swiglu_valid_gate_z2/1', ...
        'rsp_sel/5', 'rsp_sel/6', 'addr_sel/3', 580, 145);

    safe_add_line(subPath, 'x_valid/1', 'x_valid_z/1');
    safe_add_line(subPath, upDataValid, 'gateup_pair_valid/1');
    safe_add_line(subPath, gateDataValid, 'gateup_pair_valid/2');
    safe_add_line(subPath, 'gateup_pair_valid/1', 'gateup_pair_valid_z/1');
    safe_add_line(subPath, 'gateup_pair_valid_z/1', 'gateup_pair_valid_gate_z1/1');
    safe_add_line(subPath, 'gateup_pair_valid_gate_z1/1', 'gateup_pair_valid_gate_z2/1');
    safe_add_line(subPath, gateOut, 'gate_abs/1');
    safe_add_line(subPath, 'gate_abs/1', 'gate_den/1');
    safe_add_line(subPath, 'one_const/1', 'gate_den/2');
    safe_add_line(subPath, gateOut, 'gate_norm/1');
    safe_add_line(subPath, 'gate_den/1', 'gate_norm/2');
    safe_add_line(subPath, 'gate_norm/1', 'gate_norm_gate/1');
    safe_add_line(subPath, gateDataValid, 'gate_norm_gate/2');
    safe_add_line(subPath, upOut, 'swiglu_mul/1');
    safe_add_line(subPath, 'gate_norm_gate/1', 'swiglu_mul/2');
    safe_add_line(subPath, 'gateup_pair_valid_z/1', 'swiglu_valid_z/1');
    safe_add_line(subPath, 'swiglu_valid_z/1', 'swiglu_valid_gate_z1/1');
    safe_add_line(subPath, 'swiglu_valid_gate_z1/1', 'swiglu_valid_gate_z2/1');
    safe_add_line(subPath, 'swiglu_mul/1', 'swiglu_stage_gate/1');
    safe_add_line(subPath, 'swiglu_valid_gate_z2/1', 'swiglu_stage_gate/2');
    safe_add_line(subPath, 'swiglu_valid_gate_z2/1', 'down_valid_z/1');
    safe_add_line(subPath, 'down_valid_z/1', 'down_pair_valid/1');
    safe_add_line(subPath, downDataValid, 'down_pair_valid/2');
    safe_add_line(subPath, 'down_pair_valid/1', 'down_pair_valid_z1/1');
    safe_add_line(subPath, 'down_pair_valid_z1/1', 'down_pair_valid_z2/1');
    safe_add_line(subPath, downOut, 'down_stage_gate/1');
    safe_add_line(subPath, 'down_pair_valid_z2/1', 'down_stage_gate/2');
    safe_add_line(subPath, 'down_stage_gate/1', 'y_out/1');
    safe_add_line(subPath, 'down_pair_valid_z2/1', 'valid_out/1');

    safe_add_line(subPath, upReqAddr, 'req_bc/1');
    safe_add_line(subPath, upReqValid, 'req_bc/2');
    safe_add_line(subPath, gateReqAddr, 'req_bc/3');
    safe_add_line(subPath, gateReqValid, 'req_bc/4');
    safe_add_line(subPath, downReqAddr, 'req_bc/5');
    safe_add_line(subPath, downReqValid, 'req_bc/6');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'ffn_up_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'ffn_up_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'ffn_gate_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 4, 'ffn_gate_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 5, 'ffn_down_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 6, 'ffn_down_valid');
    safe_add_line(subPath, 'req_bc/1', 'w_req_bus/1');
end

function [mulOut, reqAddrOutSig, reqValidOutSig, dataValidOutSig] = add_streamed_weight_mul(subPath, prefix, inSig, reqEnableSig, ddrDataSig, ddrValidSig, reqAddrSig, x0, y0)
    % Model off-chip DDR fetch + on-chip SRAM cache using HDL RAMS blocks.
    reqAddrCast = [prefix '_req_addr_u8'];
    reqAddrOutCast = [prefix '_req_addr_double'];
    reqAddrDelay = [prefix '_req_addr_z'];
    reqNeeded = [prefix '_req_needed'];
    reqValidCast = [prefix '_req_valid_double'];
    sram = [prefix '_sram'];
    ddrDataCast = [prefix '_ddr_data_u8'];
    ddrValidCast = [prefix '_ddr_valid_bool'];
    sramAddrAlias = [prefix '_sram_addr_alias'];
    sramDinAlias = [prefix '_sram_din_alias'];
    sramWeAlias = [prefix '_sram_we_alias'];
    sramDoutAlias = [prefix '_sram_dout_alias'];
    sramAddrTerm = [prefix '_sram_addr_term'];
    sramDinTerm = [prefix '_sram_din_term'];
    sramWeTerm = [prefix '_sram_we_term'];
    sramDoutTerm = [prefix '_sram_dout_term'];
    sramDataCast = [prefix '_sram_data_double'];
    sramValid = [prefix '_sram_data_valid_z'];
    sramSel = [prefix '_sram_data_sel'];
    validOr = [prefix '_valid_or'];
    mulBlk = [prefix '_mul'];

    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqAddrCast], ...
        'OutDataTypeStr', 'int32', 'Position', [x0 + 145, y0 - 2, x0 + 185, y0 + 22]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqAddrOutCast], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 195, y0 - 2, x0 + 235, y0 + 22]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/' reqAddrDelay], ...
        'InitialCondition', '0', 'Position', [x0 + 245, y0 - 2, x0 + 275, y0 + 22]);

    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' reqNeeded], ...
        'Operator', 'NOT', 'Position', [x0 + 5, y0 + 70, x0 + 35, y0 + 90]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' reqValidCast], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 45, y0 + 70, x0 + 85, y0 + 90]);
    add_block('hdlsllib/HDL RAMs/Simple Dual Port RAM', [subPath '/' sram], ...
        'Position', [x0 + 95, y0 + 30, x0 + 165, y0 + 120]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' ddrDataCast], ...
        'OutDataTypeStr', 'uint8', 'Position', [x0 + 55, y0 + 45, x0 + 90, y0 + 65]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' ddrValidCast], ...
        'OutDataTypeStr', 'boolean', 'Position', [x0 + 55, y0 + 75, x0 + 90, y0 + 95]);
    add_block('simulink/User-Defined Functions/MATLAB Function', [subPath '/' sramAddrAlias], ...
        'Position', [x0 + 135, y0 - 40, x0 + 220, y0 - 5]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' sramDinAlias], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 55, y0 + 5, x0 + 95, y0 + 25]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' sramWeAlias], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 55, y0 + 105, x0 + 95, y0 + 125]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' sramDataCast], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 180, y0 + 45, x0 + 215, y0 + 65]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/' sramDoutAlias], ...
        'OutDataTypeStr', 'single', 'Position', [x0 + 180, y0 + 5, x0 + 215, y0 + 25]);
    add_block('simulink/Sinks/Terminator', [subPath '/' sramAddrTerm], ...
        'Position', [x0 + 205, y0 - 35, x0 + 225, y0 - 15]);
    add_block('simulink/Sinks/Terminator', [subPath '/' sramDinTerm], ...
        'Position', [x0 + 115, y0 + 5, x0 + 135, y0 + 25]);
    add_block('simulink/Sinks/Terminator', [subPath '/' sramWeTerm], ...
        'Position', [x0 + 115, y0 + 105, x0 + 135, y0 + 125]);
    add_block('simulink/Sinks/Terminator', [subPath '/' sramDoutTerm], ...
        'Position', [x0 + 235, y0 + 5, x0 + 255, y0 + 25]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/' sramValid], ...
        'InitialCondition', '0', 'Position', [x0 + 180, y0 + 70, x0 + 210, y0 + 90]);
    add_block('simulink/Signal Routing/Switch', [subPath '/' sramSel], ...
        'Threshold', '0.5', 'Position', [x0 + 230, y0 + 40, x0 + 280, y0 + 90]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/' validOr], ...
        'Operator', 'OR', 'Position', [x0 + 230, y0 + 95, x0 + 260, y0 + 125]);

    add_block('simulink/Math Operations/Product', [subPath '/' mulBlk], ...
        'Inputs', '**', 'OutDataTypeStr', 'single', 'Position', [x0 + 305, y0 + 45, x0 + 340, y0 + 85]);
    set_matlab_function_script([subPath '/' sramAddrAlias], cache_addr_alias_code(), ...
        'add_streamed_weight_mul:MissingSramAddrAlias');

    safe_add_line(subPath, reqAddrSig, [reqAddrCast '/1']);
    safe_add_line(subPath, [reqAddrCast '/1'], [reqAddrOutCast '/1']);
    safe_add_line(subPath, [reqAddrOutCast '/1'], [reqAddrDelay '/1']);
    safe_add_line(subPath, [reqAddrCast '/1'], [sramAddrAlias '/1']);
    safe_add_line(subPath, ddrDataSig, [ddrDataCast '/1']);
    safe_add_line(subPath, ddrValidSig, [ddrValidCast '/1']);
    safe_add_line(subPath, [ddrDataCast '/1'], [sramDinAlias '/1']);
    safe_add_line(subPath, [ddrValidCast '/1'], [sramWeAlias '/1']);
    safe_add_line(subPath, [sramAddrAlias '/1'], [sramAddrTerm '/1']);
    safe_add_line(subPath, [sramDinAlias '/1'], [sramDinTerm '/1']);
    safe_add_line(subPath, [sramWeAlias '/1'], [sramWeTerm '/1']);

    safe_add_line(subPath, reqEnableSig, [reqValidCast '/1']);

    % Simple Dual Port RAM expected ports: din, wr_addr, we, rd_addr -> dout.
    safe_add_line(subPath, [ddrDataCast '/1'], [sram '/1']);
    safe_add_line(subPath, [sramAddrAlias '/1'], [sram '/2']);
    safe_add_line(subPath, [ddrValidCast '/1'], [sram '/3']);
    safe_add_line(subPath, [sramAddrAlias '/1'], [sram '/4']);

    safe_add_line(subPath, [sram '/1'], [sramDataCast '/1']);
    safe_add_line(subPath, [sram '/1'], [sramDoutAlias '/1']);
    safe_add_line(subPath, [sramDoutAlias '/1'], [sramDoutTerm '/1']);
    safe_add_line(subPath, [sramDataCast '/1'], [sramSel '/1']);
    safe_add_line(subPath, [sramValid '/1'], [sramSel '/2']);
    safe_add_line(subPath, [ddrDataCast '/1'], [sramSel '/3']);

    safe_add_line(subPath, [ddrValidCast '/1'], [sramValid '/1']);
    safe_add_line(subPath, [ddrValidCast '/1'], [validOr '/1']);
    safe_add_line(subPath, [sramValid '/1'], [validOr '/2']);

    safe_add_line(subPath, inSig, [mulBlk '/1']);
    safe_add_line(subPath, [sramSel '/1'], [mulBlk '/2']);

    mulOut = [mulBlk '/1'];
    reqAddrOutSig = [reqAddrDelay '/1'];
    reqValidOutSig = [reqValidCast '/1'];
    dataValidOutSig = [validOr '/1'];
end

function code = cache_addr_alias_code()
    code = sprintf([ ...
        'function addr_alias = cache_addr_alias(addr_full)\n' ...
        '%%#codegen\n' ...
        'addr_i32 = int32(addr_full);\n' ...
        'addr_alias = uint8(bitand(addr_i32, int32(255)));\n' ...
        'end\n']);
end

function set_matlab_function_script(blockPath, code, errorId)
    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', blockPath);
    if isempty(chart)
        error(errorId, 'Expected MATLAB Function chart at %s', blockPath);
    end
    chart.Script = code;
end

function configure_residual(subPath)
    clear_subsystem_contents(subPath);

    add_block('simulink/Sources/In1', [subPath '/x_main'], 'Position', [30, 55, 60, 69]);
    add_block('simulink/Sources/In1', [subPath '/x_skip'], 'Position', [30, 110, 60, 124]);
    add_block('simulink/Sources/In1', [subPath '/x_valid'], 'Position', [30, 165, 60, 179]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/main_delay'], ...
        'InitialCondition', '0', 'Position', [110, 55, 150, 75]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/valid_delay'], ...
        'InitialCondition', '0', 'Position', [110, 165, 150, 185]);
    add_block('simulink/Math Operations/Sum', [subPath '/res_sum'], ...
        'Inputs', '++', 'Position', [210, 70, 240, 110]);
    add_block('simulink/Sinks/Out1', [subPath '/y_out'], 'Position', [290, 85, 320, 99]);
    add_block('simulink/Sinks/Out1', [subPath '/y_valid'], 'Position', [290, 165, 320, 179]);

    safe_add_line(subPath, 'x_main/1', 'main_delay/1');
    safe_add_line(subPath, 'main_delay/1', 'res_sum/1');
    safe_add_line(subPath, 'x_skip/1', 'res_sum/2');
    safe_add_line(subPath, 'res_sum/1', 'y_out/1');
    safe_add_line(subPath, 'x_valid/1', 'valid_delay/1');
    safe_add_line(subPath, 'valid_delay/1', 'y_valid/1');
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

function delete_blocks_if_exist(sys, blockNames)
    for i = 1:numel(blockNames)
        blkPath = [sys '/' blockNames{i}];
        if getSimulinkBlockHandle(blkPath) ~= -1
            try
                delete_block(blkPath);
            catch
            end
        end
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
            [mdlName '/weight_addr_map_u'], ...
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
