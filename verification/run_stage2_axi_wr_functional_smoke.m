function result = run_stage2_axi_wr_functional_smoke(rootDir, options)
%RUN_STAGE2_AXI_WR_FUNCTIONAL_SMOKE DUT-level write-side smoke via wrapper TB.
%   This smoke delegates simulation to run_stage2_wrapper_tb_smoke and asserts
%   write-side timing semantics from real DUT observations only.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));
    assert_stage2_manual_model_policy(buildModel, mfilename);

    addpath(fullfile(rootDir, 'verification'));
    wrapperResult = run_stage2_wrapper_tb_smoke(rootDir, struct('BuildModel', buildModel, 'KvAddressConfig', kvCfg));

    result = struct();
    result.tb_ready = getFieldOr(wrapperResult, 'tb_ready', false);
    result.kv_wr_valid_seen = getFieldOr(wrapperResult, 'kv_wr_valid_seen', false);
    result.kv_wr_en_seen = getFieldOr(wrapperResult, 'kv_wr_en_seen', false);
    result.kv_wr_addr_nonzero = getFieldOr(wrapperResult, 'kv_wr_addr_nonzero', false);
    result.kv_wr_data_nonzero = getFieldOr(wrapperResult, 'kv_wr_data_nonzero', false);
    result.request_next_line_seen = getFieldOr(wrapperResult, 'wr_request_next_line_seen', false);
    result.request_next_line_count = getFieldOr(wrapperResult, 'wr_request_next_line_count', 0);
    result.request_next_line_on_write = getFieldOr(wrapperResult, 'wr_request_next_line_on_write', false);
    result.request_next_line_after_first_write = getFieldOr(wrapperResult, 'wr_request_next_line_after_first_write', false);
    result.wr_valid_clears_after_pulse = getFieldOr(wrapperResult, 'wr_valid_clears_after_pulse', false);
    result.pass = result.tb_ready && result.kv_wr_valid_seen && result.kv_wr_en_seen && ...
        result.kv_wr_addr_nonzero && result.kv_wr_data_nonzero && ...
        result.request_next_line_seen && result.request_next_line_on_write && ...
        result.request_next_line_after_first_write;

    if result.pass
        fprintf('Stage2 axi_master_wr functional smoke PASS\n');
    else
        fprintf('Stage2 axi_master_wr functional smoke FAIL\n');
        fprintf('  tb_ready=%d kv_wr_valid_seen=%d kv_wr_en_seen=%d kv_wr_addr_nonzero=%d kv_wr_data_nonzero=%d\n', ...
            result.tb_ready, result.kv_wr_valid_seen, result.kv_wr_en_seen, result.kv_wr_addr_nonzero, result.kv_wr_data_nonzero);
        fprintf('  request_next_line_seen=%d request_next_line_count=%d on_write=%d after_first_write=%d wr_valid_clears_after_pulse=%d\n', ...
            result.request_next_line_seen, result.request_next_line_count, result.request_next_line_on_write, ...
            result.request_next_line_after_first_write, result.wr_valid_clears_after_pulse);
        if isfield(wrapperResult, 'reason')
            result.reason = wrapperResult.reason;
            fprintf('  reason=%s\n', result.reason);
        end
        error('run_stage2_axi_wr_functional_smoke:Failed', ...
            'axi_master_wr DUT-level functional checks failed');
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

