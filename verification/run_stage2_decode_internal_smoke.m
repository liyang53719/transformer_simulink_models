function result = run_stage2_decode_internal_smoke(rootDir)
%RUN_STAGE2_DECODE_INTERNAL_SMOKE Validate stage2 decode internal wiring.
%
% This smoke test focuses on deterministic checks:
% 1) Model compile/update succeeds after stage2 build.
% 2) Required top-level decode-path and kv-address wiring exists.
% 3) kv_addr_gen_u contains expected address-generation blocks.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    addpath(fullfile(rootDir, 'scripts'));
    kvCfg = struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1);
    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', kvCfg));

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    set_param(mdlName, 'SimulationCommand', 'update');

    requiredEdges = {
        'qkv_proj_u/1', 'kv_cache_if_u/1';
        'mode_decode/1', 'kv_cache_if_u/3';
        'axi_master_rd_u/1', 'kv_cache_if_u/2';
        'kv_cache_if_u/1', 'attention_u/1';
        'cfg_token_pos/1', 'kv_addr_gen_u/1';
        'cfg_seq_len/1', 'kv_addr_gen_u/2';
        'kv_addr_gen_u/1', 'axi_master_rd_u/5';
        'kv_addr_gen_u/2', 'axi_master_rd_u/6';
        'kv_addr_gen_u/3', 'axi_master_wr_u/4';
        'kv_addr_gen_u/4', 'axi_master_wr_u/5'};

    missingEdges = {};
    for i = 1:size(requiredEdges, 1)
        if ~has_connection(mdlName, requiredEdges{i, 1}, requiredEdges{i, 2})
            missingEdges{end+1} = sprintf('%s -> %s', requiredEdges{i, 1}, requiredEdges{i, 2}); %#ok<AGROW>
        end
    end

    kvAddrPath = [mdlName '/kv_addr_gen_u'];
    kvAddrBlocks = {'rd_base_const', 'wr_base_const', 'stride_const', 'decode_burst_const', ...
        'rd_addr_scale', 'rd_addr_mode_sel', 'rd_addr_add_base', ...
        'wr_addr_scale', 'wr_addr_mode_sel', 'wr_addr_add_base', ...
        'rd_len_mode_sel', 'wr_len_mode_sel'};
    missingKvAddrBlocks = {};
    for i = 1:numel(kvAddrBlocks)
        if isempty(find_system(kvAddrPath, 'SearchDepth', 1, 'Name', kvAddrBlocks{i}))
            missingKvAddrBlocks{end+1} = kvAddrBlocks{i}; %#ok<AGROW>
        end
    end

    result = struct();
    result.compile_update_ok = true;
    result.missing_edges = missingEdges;
    result.missing_kv_addr_blocks = missingKvAddrBlocks;
    result.decode_path_wiring_ok = isempty(missingEdges);
    result.addr_gen_structure_ok = isempty(missingKvAddrBlocks);

    kvConstChecks = {
        [kvAddrPath '/rd_base_const'], num2str(kvCfg.rd_base);
        [kvAddrPath '/wr_base_const'], num2str(kvCfg.wr_base);
        [kvAddrPath '/stride_const'], num2str(kvCfg.stride_bytes);
        [kvAddrPath '/decode_burst_const'], num2str(kvCfg.decode_burst_len)};
    badKvConstValues = {};
    for i = 1:size(kvConstChecks, 1)
        actualVal = strtrim(get_param(kvConstChecks{i, 1}, 'Value'));
        if ~strcmp(actualVal, kvConstChecks{i, 2})
            badKvConstValues{end+1} = sprintf('%s expected=%s actual=%s', ...
                kvConstChecks{i, 1}, kvConstChecks{i, 2}, actualVal); %#ok<AGROW>
        end
    end

    axiRdPath = [mdlName '/axi_master_rd_u'];
    axiRdBlocks = {'avalid_state_z', 'addr_hs_logic', 'burst_active_z', 'burst_count_z', 'burst_done_logic'};
    missingAxiRdBlocks = {};
    for i = 1:numel(axiRdBlocks)
        if isempty(find_system(axiRdPath, 'SearchDepth', 1, 'Name', axiRdBlocks{i}))
            missingAxiRdBlocks{end+1} = axiRdBlocks{i}; %#ok<AGROW>
        end
    end

    result.missing_axi_rd_blocks = missingAxiRdBlocks;
    result.axi_rd_structure_ok = isempty(missingAxiRdBlocks);
    result.bad_kv_constant_values = badKvConstValues;
    result.kv_constant_values_ok = isempty(badKvConstValues);
    result.pass = result.compile_update_ok && result.decode_path_wiring_ok && ...
        result.addr_gen_structure_ok && result.kv_constant_values_ok && result.axi_rd_structure_ok;

    if result.pass
        fprintf('Stage2 decode internal smoke PASS\n');
    else
        fprintf('Stage2 decode internal smoke FAIL\n');
        if ~isempty(missingEdges)
            fprintf('  Missing edges:\n');
            for i = 1:numel(missingEdges)
                fprintf('    - %s\n', missingEdges{i});
            end
        end
        if ~isempty(missingKvAddrBlocks)
            fprintf('  Missing kv_addr_gen blocks: %s\n', strjoin(missingKvAddrBlocks, ', '));
        end
        if ~isempty(badKvConstValues)
            fprintf('  Bad kv_addr_gen constant values:\n');
            for i = 1:numel(badKvConstValues)
                fprintf('    - %s\n', badKvConstValues{i});
            end
        end
        if ~isempty(missingAxiRdBlocks)
            fprintf('  Missing axi_master_rd blocks: %s\n', strjoin(missingAxiRdBlocks, ', '));
        end
        error('run_stage2_decode_internal_smoke:Failed', ...
            'stage2 decode path or kv address generation check failed');
    end

    close_system(mdlName, 0);
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
    out.block = char(parts(1));
    out.port = str2double(parts(2));
end