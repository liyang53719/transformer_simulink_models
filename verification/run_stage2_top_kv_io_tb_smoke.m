function result = run_stage2_top_kv_io_tb_smoke(rootDir, options)
%RUN_STAGE2_TOP_KV_IO_TB_SMOKE Simulate top-level KV IO via a typed wrapper TB.
%   This reuses the stable typed wrapper harness so root fixed-point ports do
%   not depend on direct external-input loader behavior.

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

    kvRdValid = extract_signal(yout, 'kv_mem_rd_valid');
    kvWrValid = extract_signal(yout, 'kv_mem_wr_valid');
    kvRdAddr = extract_signal(yout, 'kv_mem_rd_addr');
    kvWrAddr = extract_signal(yout, 'kv_mem_wr_addr');
    kvWrData = extract_signal(yout, 'kv_cache_wr_data');
    kvWrEn = extract_signal(yout, 'kv_cache_wr_en');

    noLoopbackData = ~all(abs(kvWrData - 7) < 1e-9);

    result = struct();
    result.tb_ready = true;
    result.kv_rd_valid_seen = any(kvRdValid > 0.5);
    result.kv_wr_valid_seen = any(kvWrValid > 0.5);
    result.kv_wr_addr_nonzero = any(kvWrAddr > 0);
    result.kv_wr_en_seen = any(kvWrEn > 0.5);
    result.kv_cache_not_loopback_data = noLoopbackData;
    result.pass = result.kv_rd_valid_seen && result.kv_wr_valid_seen && ...
        result.kv_wr_addr_nonzero && ...
        result.kv_wr_en_seen && result.kv_cache_not_loopback_data;

    if result.pass
        fprintf('Stage2 top KV IO TB smoke PASS\n');
    else
        fprintf('Stage2 top KV IO TB smoke FAIL\n');
        fprintf('  kv_rd_valid_seen=%d kv_wr_valid_seen=%d kv_wr_addr_nonzero=%d\n', ...
            result.kv_rd_valid_seen, result.kv_wr_valid_seen, result.kv_wr_addr_nonzero);
        fprintf('  kv_wr_en_seen=%d kv_cache_not_loopback_data=%d\n', ...
            result.kv_wr_en_seen, result.kv_cache_not_loopback_data);
        error('run_stage2_top_kv_io_tb_smoke:Failed', ...
            'Top-level KV IO TB-style smoke failed');
    end
end

function values = extract_signal(yout, name)
    if isa(yout, 'Simulink.SimulationData.Dataset')
        for i = 1:yout.numElements
            elem = yout.getElement(i);
            if strcmp(string(elem.Name), string(name))
                values = extract_values(elem.Values);
                return;
            end
            if dataset_element_matches_name(elem, name)
                values = extract_values(elem.Values);
                return;
            end
        end
    end
    error('run_stage2_top_kv_io_tb_smoke:MissingOutput', 'Missing output signal: %s', name);
end

function match = dataset_element_matches_name(elem, name)
    match = false;
    try
        blockPathText = string(elem.BlockPath);
        if any(endsWith(blockPathText, "/" + string(name)))
            match = true;
            return;
        end
    catch
    end
    try
        blockPath = elem.BlockPath;
        lastBlock = string(blockPath.getBlock(blockPath.getLength));
        match = endsWith(lastBlock, "/" + string(name));
    catch
    end
end

function values = extract_values(ts)
    if isa(ts, 'timeseries')
        values = squeeze(ts.Data);
    else
        values = squeeze(ts);
    end
    values = double(values(:));
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
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