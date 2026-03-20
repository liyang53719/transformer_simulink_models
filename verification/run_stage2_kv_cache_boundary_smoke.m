function result = run_stage2_kv_cache_boundary_smoke(rootDir, options)
%RUN_STAGE2_KV_CACHE_BOUNDARY_SMOKE Validate kv_cache_if_u boundary ports and top-level connectivity.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    kvPath = [mdlName '/kv_cache_if_u'];
    requiredInports = {'qkv_new','kv_hist','mode_decode','sched_bus'};
    requiredOutports = {'kv_to_attn','kv_write_data','kv_write_valid','kv_bank_addr','kv_bank_sel','kv_bank_wr_en'};

    missingKvInports = missing_named_ports(kvPath, 'Inport', requiredInports);
    missingKvOutports = missing_named_ports(kvPath, 'Outport', requiredOutports);

    requiredEdges = {
        'qkv_proj_u/1', 'kv_cache_if_u/1';
        'axi_master_rd_u/1', 'kv_cache_if_u/2';
        'mode_decode/1', 'kv_cache_if_u/3';
        'prefill_sched_u/1', 'kv_cache_if_u/4';
        'kv_cache_if_u/1', 'attention_u/1';
        'kv_cache_if_u/2', 'axi_master_wr_u/1';
        'kv_cache_if_u/3', 'axi_master_wr_u/2';
        'kv_cache_rd_data/1', 'axi_master_rd_u/1';
        'kv_cache_rd_valid/1', 'axi_master_rd_u/3';
        'kv_mem_rd_ready/1', 'axi_master_rd_u/2';
        'axi_master_rd_u/3', 'kv_mem_rd_addr/1';
        'axi_master_rd_u/4', 'kv_mem_rd_len/1';
        'axi_master_rd_u/5', 'kv_mem_rd_valid/1';
        'axi_master_wr_u/1', 'kv_cache_wr_data/1';
        'axi_master_wr_u/2', 'kv_mem_wr_addr/1';
        'axi_master_wr_u/3', 'kv_mem_wr_len/1';
        'axi_master_wr_u/4', 'kv_mem_wr_valid/1';
        'kv_cache_if_u/3', 'kv_cache_wr_en/1'};

    forbiddenEdges = {
        'kv_cache_rd_data/1', 'kv_cache_wr_data/1';
        'kv_cache_rd_valid/1', 'kv_cache_wr_en/1'};

    missingEdges = {};
    for i = 1:size(requiredEdges, 1)
        if ~has_connection(mdlName, requiredEdges{i, 1}, requiredEdges{i, 2})
            missingEdges{end+1} = sprintf('%s -> %s', requiredEdges{i, 1}, requiredEdges{i, 2}); %#ok<AGROW>
        end
    end

    forbiddenPresent = {};
    for i = 1:size(forbiddenEdges, 1)
        if has_connection(mdlName, forbiddenEdges{i, 1}, forbiddenEdges{i, 2})
            forbiddenPresent{end+1} = sprintf('%s -> %s', forbiddenEdges{i, 1}, forbiddenEdges{i, 2}); %#ok<AGROW>
        end
    end

    result = struct();
    result.missing_kv_inports = missingKvInports;
    result.missing_kv_outports = missingKvOutports;
    result.missing_edges = missingEdges;
    result.forbidden_edges = forbiddenPresent;
    result.kv_boundary_ok = isempty(missingKvInports) && isempty(missingKvOutports);
    result.top_connectivity_ok = isempty(missingEdges) && isempty(forbiddenPresent);
    result.pass = result.kv_boundary_ok && result.top_connectivity_ok;

    close_system(mdlName, 0);

    if result.pass
        fprintf('Stage2 kv_cache_if boundary smoke PASS\n');
    else
        fprintf('Stage2 kv_cache_if boundary smoke FAIL\n');
        if ~isempty(missingKvInports)
            fprintf('  Missing kv_cache_if_u inports: %s\n', strjoin(missingKvInports, ', '));
        end
        if ~isempty(missingKvOutports)
            fprintf('  Missing kv_cache_if_u outports: %s\n', strjoin(missingKvOutports, ', '));
        end
        if ~isempty(missingEdges)
            fprintf('  Missing boundary/top edges:\n');
            for i = 1:numel(missingEdges)
                fprintf('    - %s\n', missingEdges{i});
            end
        end
        if ~isempty(forbiddenPresent)
            fprintf('  Forbidden placeholder edges still present:\n');
            for i = 1:numel(forbiddenPresent)
                fprintf('    - %s\n', forbiddenPresent{i});
            end
        end
        error('run_stage2_kv_cache_boundary_smoke:Failed', ...
            'kv_cache_if_u boundary or top-level connectivity check failed');
    end
end

function missing = missing_named_ports(sysPath, blockType, portNames)
    missing = {};
    for i = 1:numel(portNames)
        if isempty(find_system(sysPath, 'SearchDepth', 1, 'BlockType', blockType, 'Name', portNames{i}))
            missing{end+1} = portNames{i}; %#ok<AGROW>
        end
    end
end

function yes = has_connection(mdlName, srcSpec, dstSpec)
    src = split_port_spec(srcSpec);
    dst = split_port_spec(dstSpec);

    srcBlk = [mdlName '/' src.block];
    dstBlk = [mdlName '/' dst.block];

    yes = false;
    try
        srcPh = get_param(srcBlk, 'PortHandles');
        dstPh = get_param(dstBlk, 'PortHandles');

        if numel(srcPh.Outport) < src.port || numel(dstPh.Inport) < dst.port
            return;
        end

        srcPortHandle = srcPh.Outport(src.port);
        dstPortHandle = dstPh.Inport(dst.port);
        lines = find_system(mdlName, 'FindAll', 'on', 'Type', 'line');
        for i = 1:numel(lines)
            s = get_param(lines(i), 'SrcPortHandle');
            d = get_param(lines(i), 'DstPortHandle');
            if s == srcPortHandle && any(d == dstPortHandle)
                yes = true;
                return;
            end
        end
    catch
        yes = false;
    end
end

function out = split_port_spec(spec)
    parts = split(string(spec), '/');
    out = struct();
    out.port = str2double(parts(end));
    out.block = char(strjoin(parts(1:end-1), '/'));
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end