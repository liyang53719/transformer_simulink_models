function summary = run_stage2_smoke_suite_fast(rootDir)
%RUN_STAGE2_SMOKE_SUITE_FAST Fast stage2 smoke suite with model-build reuse.
%   This suite accelerates local iteration by avoiding repeated Simulink rebuilds
%   across smoke checks. It runs:
%   1) default KvAddressConfig: decode + wrapper TB + KV banking + attention pipeline + FFN pipeline + axi rd functional + axi wr functional
%   2) non-default KvAddressConfig: decode internal smoke for parameter coverage

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));

    cfgDefault = struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1);
    cfgVariant = struct('rd_base', 64, 'wr_base', 128, 'stride_bytes', 4, 'decode_burst_len', 2);

    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', cfgDefault));
    assert_model_upgrade_markers(rootDir);
    rWeightPath = run_stage2_weight_path_assertions(rootDir, struct('BuildModel', false));
    rDecodeDefault = run_stage2_decode_internal_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rKvBoundary = run_stage2_kv_cache_boundary_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rPrefillAttention = run_stage2_prefill_attention_functional_smoke(rootDir, struct('BuildModel', false));
    rWrapperTb = run_stage2_wrapper_tb_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rKvBanking = run_stage2_kv_banking_pipeline_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rAttentionPipe = run_stage2_attention_pipeline_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rFfnPipe = run_stage2_ffn_pipeline_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rAttentionDdr = run_stage2_attention_ddr_integration_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rAxiRd = run_stage2_axi_rd_functional_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));
    rAxiWr = run_stage2_axi_wr_functional_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgDefault));

    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', cfgVariant));
    rDecodeVariant = run_stage2_decode_internal_smoke(rootDir, struct('BuildModel', false, 'KvAddressConfig', cfgVariant));

    summary = struct();
    summary.default_decode = rDecodeDefault;
    summary.default_weight_path = rWeightPath;
    summary.default_kv_boundary = rKvBoundary;
    summary.default_prefill_attention = rPrefillAttention;
    summary.default_wrapper_tb = rWrapperTb;
    summary.default_kv_banking = rKvBanking;
    summary.default_attention_pipeline = rAttentionPipe;
    summary.default_ffn_pipeline = rFfnPipe;
    summary.default_attention_ddr = rAttentionDdr;
    summary.default_axi_rd = rAxiRd;
    summary.default_axi_wr = rAxiWr;
    summary.variant_decode = rDecodeVariant;
    summary.pass = rDecodeDefault.pass && rWeightPath.pass && rKvBoundary.pass && ...
        rPrefillAttention.pass && rWrapperTb.pass && rKvBanking.pass && ...
        rAttentionPipe.pass && rFfnPipe.pass && ...
        rAttentionDdr.pass && rAxiRd.pass && ...
        rAxiWr.pass && rDecodeVariant.pass;

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
