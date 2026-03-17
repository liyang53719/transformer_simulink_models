function result = run_stage2_attention_ddr_integration_smoke(rootDir, options)
%RUN_STAGE2_ATTENTION_DDR_INTEGRATION_SMOKE Validate mainline attention + DDR activity together.
%   This smoke reuses the visible wrapper TB and checks that:
%   1) attention weight request valid bits are asserted,
%   2) DDR read responses are observed,
%   3) block output becomes non-zero, and
%   4) KV writeback reaches the external DDR wrapper path.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct('BuildModel', buildModel, 'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');

    attnQReq = extract_signal(yout, 'tb_attn_q_req_valid');
    attnKReq = extract_signal(yout, 'tb_attn_k_req_valid');
    attnVReq = extract_signal(yout, 'tb_attn_v_req_valid');
    rdRspValid = extract_signal(yout, 'tb_rd_rsp_valid');
    rdRspData = extract_signal(yout, 'tb_rd_rsp_data');
    outHidden = extract_signal(yout, 'out_hidden');
    kvWrValid = extract_signal(yout, 'kv_mem_wr_valid');
    kvWrData = extract_signal(yout, 'kv_cache_wr_data');
    kvWrEn = extract_signal(yout, 'kv_cache_wr_en');

    result = struct();
    result.attn_q_req_seen = any(attnQReq > 0.5);
    result.attn_k_req_seen = any(attnKReq > 0.5);
    result.attn_v_req_seen = any(attnVReq > 0.5);
    result.rd_rsp_valid_seen = any(rdRspValid > 0.5);
    result.rd_rsp_data_nonzero = any(abs(rdRspData) > 0);
    result.out_hidden_nonzero = any(abs(outHidden) > 0);
    result.kv_wr_valid_seen = any(kvWrValid > 0.5);
    result.kv_wr_en_seen = any(kvWrEn > 0.5);
    result.kv_wr_data_nonzero = any(abs(kvWrData) > 0);
    result.pass = result.attn_q_req_seen && result.attn_k_req_seen && result.attn_v_req_seen && ...
        result.rd_rsp_valid_seen && result.rd_rsp_data_nonzero && ...
        result.out_hidden_nonzero && result.kv_wr_valid_seen && ...
        result.kv_wr_en_seen && result.kv_wr_data_nonzero;

    if result.pass
        fprintf('Stage2 attention DDR integration smoke PASS\n');
    else
        fprintf('Stage2 attention DDR integration smoke FAIL\n');
        fprintf(['  attn_q=%d attn_k=%d attn_v=%d rd_rsp_valid=%d rd_rsp_data=%d ' ...
            'out_hidden=%d kv_wr_valid=%d kv_wr_en=%d kv_wr_data=%d\n'], ...
            result.attn_q_req_seen, result.attn_k_req_seen, result.attn_v_req_seen, ...
            result.rd_rsp_valid_seen, result.rd_rsp_data_nonzero, ...
            result.out_hidden_nonzero, result.kv_wr_valid_seen, ...
            result.kv_wr_en_seen, result.kv_wr_data_nonzero);
        error('run_stage2_attention_ddr_integration_smoke:Failed', ...
            'attention + DDR integration checks failed');
    end
end

function values = extract_signal(yout, name)
    for i = 1:yout.numElements
        sig = yout.get(i);
        sigName = string(sig.Name);
        blockPath = string('');
        try
            blockPath = string(sig.BlockPath.getBlock(1));
        catch
        end
        if sigName == string(name) || endsWith(blockPath, "/" + string(name))
            values = sig.Values.Data;
            return;
        end
    end
    error('run_stage2_attention_ddr_integration_smoke:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function safe_close_models(tbName, mdlName)
    if bdIsLoaded(tbName)
        close_system(tbName, 0);
    end
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
end