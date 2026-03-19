function summary = run_stage2_smoke_suite_fast(rootDir)
%RUN_STAGE2_SMOKE_SUITE_FAST Fast stage2 smoke suite with model-build reuse.
%   This suite accelerates local iteration by avoiding repeated Simulink rebuilds
%   across smoke checks. It runs:
%   1) default KvAddressConfig: hardware interface contract + decode + wrapper TB + top KV IO + KV banking + attention pipeline + FFN pipeline + axi rd functional + axi wr functional
%      plus real first-block direct-bus and output-delta regressions
%   2) non-default KvAddressConfig: decode internal smoke for parameter coverage

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));

    cfgDefault = struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1);
    cfgVariant = struct('rd_base', 64, 'wr_base', 128, 'stride_bytes', 4, 'decode_burst_len', 2);

    rebuildEachCase = false;

    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', cfgDefault));
    rModelSelfInit = run_stage2_model_self_init_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    assert_model_upgrade_markers(rootDir);
    rHwInterface = run_stage2_hardware_interface_contract_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rWeightPath = run_stage2_weight_path_assertions(rootDir, struct('BuildModel', rebuildEachCase));
    rWeightAddrRange = run_stage2_weight_addr_range_audit(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rDecodeDefault = run_stage2_decode_internal_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rKvBoundary = run_stage2_kv_cache_boundary_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rPrefillAttention = run_stage2_prefill_attention_functional_smoke(rootDir, struct('BuildModel', rebuildEachCase));
    rWrapperTb = run_stage2_wrapper_tb_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rTopKvIo = run_stage2_top_kv_io_tb_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rKvBanking = run_stage2_kv_banking_pipeline_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rAttentionPipe = run_stage2_attention_pipeline_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rFfnPipe = run_stage2_ffn_pipeline_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rQkvPipe = run_stage2_qkv_pipeline_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rAttentionDdr = run_stage2_attention_ddr_integration_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rAxiRd = run_stage2_axi_rd_functional_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rAxiWr = run_stage2_axi_wr_functional_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rRealDirectBus = run_stage2_real_first_block_direct_bus_regression(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));
    rRealOutputDelta = run_stage2_real_first_block_output_delta_regression(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgDefault));

    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', cfgVariant));
    rDecodeVariant = run_stage2_decode_internal_smoke(rootDir, struct('BuildModel', rebuildEachCase, 'KvAddressConfig', cfgVariant));

    summary = struct();
    summary.model_self_init = rModelSelfInit;
    summary.hardware_interface = rHwInterface;
    summary.default_decode = rDecodeDefault;
    summary.default_weight_path = rWeightPath;
    summary.default_weight_addr_range = rWeightAddrRange;
    summary.default_kv_boundary = rKvBoundary;
    summary.default_prefill_attention = rPrefillAttention;
    summary.default_wrapper_tb = rWrapperTb;
    summary.default_top_kv_io = rTopKvIo;
    summary.default_kv_banking = rKvBanking;
    summary.default_attention_pipeline = rAttentionPipe;
    summary.default_ffn_pipeline = rFfnPipe;
    summary.default_qkv_pipeline = rQkvPipe;
    summary.default_attention_ddr = rAttentionDdr;
    summary.default_axi_rd = rAxiRd;
    summary.default_axi_wr = rAxiWr;
    summary.default_real_direct_bus = rRealDirectBus;
    summary.default_real_output_delta = rRealOutputDelta;
    summary.variant_decode = rDecodeVariant;
    summary.pass = rModelSelfInit.pass && rHwInterface.pass && rDecodeDefault.pass && rWeightPath.pass && ...
        rWeightAddrRange.pass && rKvBoundary.pass && ...
        rPrefillAttention.pass && rWrapperTb.pass && rTopKvIo.pass && rKvBanking.pass && ...
        rAttentionPipe.pass && rFfnPipe.pass && rQkvPipe.pass && ...
        rAttentionDdr.pass && rAxiRd.pass && rAxiWr.pass && ...
        rRealDirectBus.pass && rRealOutputDelta.pass && ...
        rDecodeVariant.pass;

    if summary.pass
        fprintf('Stage2 fast smoke suite PASS\n');
    else
        fprintf('Stage2 fast smoke suite FAIL\n');
        error('run_stage2_smoke_suite_fast:Failed', 'One or more stage2 smoke checks failed');
    end
end

function assert_model_upgrade_markers(rootDir)
    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    [~, mdlName] = fileparts(mdlPath);
    load_system(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    requiredBlocks = {
        [mdlName '/rmsnorm_u/gamma_sram'];
        [mdlName '/qkv_proj_u/q_sram'];
        [mdlName '/attention_u/v_sram'];
        [mdlName '/ffn_swiglu_u/gate_sram'];
        [mdlName '/rope_u/cos_phase'];
        [mdlName '/rope_u/sin_phase'];
    };

    for i = 1:numel(requiredBlocks)
        if getSimulinkBlockHandle(requiredBlocks{i}) == -1
            error('run_stage2_smoke_suite_fast:MissingUpgradeBlock', ...
                'Missing required upgraded block: %s', requiredBlocks{i});
        end
    end
end
