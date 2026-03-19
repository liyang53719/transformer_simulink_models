function result = run_stage2_reference_readiness_audit(rootDir, options)
%RUN_STAGE2_REFERENCE_READINESS_AUDIT Audit how the real reference maps onto the stage2 hardware contract.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    paramsSource = char(getFieldOr(options, 'ParamsSource', 'module_awq'));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    addpath(fullfile(rootDir, 'matlab_ref'));

    [params, sourceInfo] = load_qwen_reference_params(paramsSource, rootDir);
    contract = build_default_stage2_contract();
    ctx = struct('Parameters', params, 'LayerIndex', 1);
    [adapterSummary, adapterDetail] = qwen2_block_ref_stage2_contract_adapter(contract, ctx);

    result = struct();
    result.pass = true;
    result.params_source = sourceInfo;
    result.hidden_size = double(params.Hyperparameters.HiddenSize);
    result.directly_mapped_fields = { ...
        'mode_decode', 'start', 'eos_in', 'in_valid', 'out_ready', ...
        'cfg_seq_len', 'cfg_token_pos', 'cfg_eps', ...
        'cfg_rope_theta_scale', 'cfg_rope_sin_mix_scale', ...
        'cfg_weight_num_heads', 'cfg_weight_page_base', 'cfg_weight_page_stride'};
    result.adapter_mapped_fields = { ...
        'in_hidden', 'in_residual', 'kv_cache_rd_data', 'kv_cache_rd_valid', 'out_hidden'};
    result.not_directly_comparable_fields = { ...
        'w_rd_req_bus', 'w_rd_rsp_bus', 'kv_cache_wr_data', 'kv_cache_wr_en', ...
        'kv_mem_rd_addr', 'kv_mem_rd_len', 'kv_mem_rd_valid', ...
        'kv_mem_wr_addr', 'kv_mem_wr_len', 'kv_mem_wr_valid', ...
        'busy', 'irq', 'error_code', 'cycle_latency'};
    result.adapter_broadcast_policy = adapterDetail.broadcast_policy;
    result.adapter_output_reduction = adapterDetail.output_reduction;
    result.adapter_kv_policy = adapterDetail.kv_policy;
    result.reference_out_hidden_finite = adapterSummary.out_hidden_finite;
    result.reference_out_hidden_nonzero = adapterSummary.out_hidden_nonzero;
    result.reference_seq_len = adapterSummary.seq_len;
    result.reference_kv_len = adapterSummary.kv_len;

    if ~(result.reference_out_hidden_finite && result.reference_out_hidden_nonzero)
        error('run_stage2_reference_readiness_audit:AdapterNotReady', ...
            'Reference adapter did not produce finite, non-trivial output under the stage2 contract.');
    end

    fprintf('Stage2 reference readiness audit PASS\n');
    fprintf('  params_source=%s\n', result.params_source);
    fprintf('  hidden_size=%d seq_len=%d kv_len=%d\n', ...
        result.hidden_size, result.reference_seq_len, result.reference_kv_len);
    fprintf('  direct_fields=%s\n', strjoin(string(result.directly_mapped_fields), ', '));
    fprintf('  adapter_fields=%s\n', strjoin(string(result.adapter_mapped_fields), ', '));
    fprintf('  noncomparable_fields=%s\n', strjoin(string(result.not_directly_comparable_fields), ', '));
end

function contract = build_default_stage2_contract()
    contract = struct();
    contract.mode_decode = true;
    contract.start = true;
    contract.eos_in = false;
    contract.in_valid = true;
    contract.out_ready = true;
    contract.in_hidden = single(2);
    contract.in_residual = single(1);
    contract.kv_cache_rd_data = single(0);
    contract.kv_cache_rd_valid = false;
    contract.cfg_seq_len = 1;
    contract.cfg_token_pos = 1;
    contract.cfg_eps = single(1e-5);
    contract.stop_req = false;
    contract.cfg_weight_num_heads = 12;
    contract.cfg_weight_page_base = 64;
    contract.cfg_weight_page_stride = 8;
    contract.cfg_rope_theta_scale = 1;
    contract.cfg_rope_sin_mix_scale = 1;
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end