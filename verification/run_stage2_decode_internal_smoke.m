function result = run_stage2_decode_internal_smoke(rootDir, options)
%RUN_STAGE2_DECODE_INTERNAL_SMOKE Validate stage2 decode internal wiring.
%
% This smoke test focuses on deterministic checks:
% 1) Model compile/update succeeds after stage2 build.
% 2) Required top-level decode-path and kv-address wiring exists.
% 3) kv_addr_gen_u contains expected address-generation blocks.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    if buildModel
        implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', kvCfg));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    set_param(mdlName, 'SimulationCommand', 'update');

    requiredEdges = {
          'in_hidden/1', 'rope_u/1';
          'cfg_token_pos/1', 'rope_u/2';
          'rope_u/1', 'rmsnorm_u/1';
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

    kvInternalEdges = {
        'kv_addr_gen_u/rd_addr_scale/1', 'kv_addr_gen_u/rd_addr_mode_sel/1';
        'kv_addr_gen_u/rd_addr_prefill_scale/1', 'kv_addr_gen_u/rd_addr_mode_sel/3';
        'kv_addr_gen_u/mode_decode/1', 'kv_addr_gen_u/rd_addr_mode_sel/2';
        'kv_addr_gen_u/rd_addr_mode_sel/1', 'kv_addr_gen_u/rd_addr_add_base/1';
        'kv_addr_gen_u/rd_base_const/1', 'kv_addr_gen_u/rd_addr_add_base/2';
        'kv_addr_gen_u/wr_addr_scale/1', 'kv_addr_gen_u/wr_addr_mode_sel/3';
        'kv_addr_gen_u/wr_addr_add/1', 'kv_addr_gen_u/wr_addr_mode_sel/1';
        'kv_addr_gen_u/decode_burst_const/1', 'kv_addr_gen_u/rd_len_mode_sel/1';
        'kv_addr_gen_u/seq_len/1', 'kv_addr_gen_u/rd_len_mode_sel/3'};
    missingKvInternalEdges = {};
    for i = 1:size(kvInternalEdges, 1)
        if ~has_connection(mdlName, kvInternalEdges{i, 1}, kvInternalEdges{i, 2})
            missingKvInternalEdges{end+1} = sprintf('%s -> %s', kvInternalEdges{i, 1}, kvInternalEdges{i, 2}); %#ok<AGROW>
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

    axiRdInternalEdges = {
        'axi_master_rd_u/start/1', 'axi_master_rd_u/start_or_hold/1';
        'axi_master_rd_u/avalid_state_z/1', 'axi_master_rd_u/start_or_hold/2';
        'axi_master_rd_u/start_or_hold/1', 'axi_master_rd_u/addr_hs_logic/1';
        'axi_master_rd_u/rd_aready/1', 'axi_master_rd_u/addr_hs_logic/2';
        'axi_master_rd_u/addr_hs_logic/1', 'axi_master_rd_u/burst_next_logic/1';
        'axi_master_rd_u/burst_next_logic/1', 'axi_master_rd_u/burst_active_z/1';
        'axi_master_rd_u/burst_active_z/1', 'axi_master_rd_u/beat_fire_logic/1';
        'axi_master_rd_u/rd_dvalid/1', 'axi_master_rd_u/beat_fire_logic/2';
        'axi_master_rd_u/count_done_cmp/1', 'axi_master_rd_u/burst_done_logic/2';
        'axi_master_rd_u/rd_valid_gate/1', 'axi_master_rd_u/rd_dvalid_out/1'};
    missingAxiRdInternalEdges = {};
    for i = 1:size(axiRdInternalEdges, 1)
        if ~has_connection(mdlName, axiRdInternalEdges{i, 1}, axiRdInternalEdges{i, 2})
            missingAxiRdInternalEdges{end+1} = sprintf('%s -> %s', axiRdInternalEdges{i, 1}, axiRdInternalEdges{i, 2}); %#ok<AGROW>
        end
    end

    result.missing_axi_rd_blocks = missingAxiRdBlocks;
    result.axi_rd_structure_ok = isempty(missingAxiRdBlocks);
    result.missing_kv_internal_edges = missingKvInternalEdges;
    result.kv_internal_semantics_ok = isempty(missingKvInternalEdges);
    result.missing_axi_rd_internal_edges = missingAxiRdInternalEdges;
    result.axi_rd_semantics_ok = isempty(missingAxiRdInternalEdges);
    result.bad_kv_constant_values = badKvConstValues;
    result.kv_constant_values_ok = isempty(badKvConstValues);

    axiWrPath = [mdlName '/axi_master_wr_u'];
    axiWrBlocks = {'wvalid_state_z', 'request_or_hold', 'write_active_z', 'write_count_z', 'write_done_logic', 'next_line_logic'};
    missingAxiWrBlocks = {};
    for i = 1:numel(axiWrBlocks)
        if isempty(find_system(axiWrPath, 'SearchDepth', 1, 'Name', axiWrBlocks{i}))
            missingAxiWrBlocks{end+1} = axiWrBlocks{i}; %#ok<AGROW>
        end
    end

    axiWrInternalEdges = {
        'axi_master_wr_u/wr_dvalid/1', 'axi_master_wr_u/request_or_hold/1';
        'axi_master_wr_u/wvalid_state_z/1', 'axi_master_wr_u/request_or_hold/2';
        'axi_master_wr_u/request_or_hold/1', 'axi_master_wr_u/start_write_logic/1';
        'axi_master_wr_u/write_active_z/1', 'axi_master_wr_u/not_write_active/1';
        'axi_master_wr_u/beat_fire_logic/1', 'axi_master_wr_u/count_on_beat_sw/2';
        'axi_master_wr_u/count_done_cmp/1', 'axi_master_wr_u/write_done_logic/2';
        'axi_master_wr_u/write_done_logic/1', 'axi_master_wr_u/next_line_logic/1';
        'axi_master_wr_u/next_line_logic/1', 'axi_master_wr_u/request_next_line/1'};
    missingAxiWrInternalEdges = {};
    for i = 1:size(axiWrInternalEdges, 1)
        if ~has_connection(mdlName, axiWrInternalEdges{i, 1}, axiWrInternalEdges{i, 2})
            missingAxiWrInternalEdges{end+1} = sprintf('%s -> %s', axiWrInternalEdges{i, 1}, axiWrInternalEdges{i, 2}); %#ok<AGROW>
        end
    end

    result.missing_axi_wr_blocks = missingAxiWrBlocks;
    result.axi_wr_structure_ok = isempty(missingAxiWrBlocks);
    result.missing_axi_wr_internal_edges = missingAxiWrInternalEdges;
    result.axi_wr_semantics_ok = isempty(missingAxiWrInternalEdges);
    result.pass = result.compile_update_ok && result.decode_path_wiring_ok && ...
        result.addr_gen_structure_ok && result.kv_constant_values_ok && ...
        result.kv_internal_semantics_ok && result.axi_rd_structure_ok && result.axi_rd_semantics_ok && ...
        result.axi_wr_structure_ok && result.axi_wr_semantics_ok;

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
        if ~isempty(missingKvInternalEdges)
            fprintf('  Missing kv_addr_gen internal edges:\n');
            for i = 1:numel(missingKvInternalEdges)
                fprintf('    - %s\n', missingKvInternalEdges{i});
            end
        end
        if ~isempty(missingAxiRdBlocks)
            fprintf('  Missing axi_master_rd blocks: %s\n', strjoin(missingAxiRdBlocks, ', '));
        end
        if ~isempty(missingAxiRdInternalEdges)
            fprintf('  Missing axi_master_rd internal edges:\n');
            for i = 1:numel(missingAxiRdInternalEdges)
                fprintf('    - %s\n', missingAxiRdInternalEdges{i});
            end
        end
        if ~isempty(missingAxiWrBlocks)
            fprintf('  Missing axi_master_wr blocks: %s\n', strjoin(missingAxiWrBlocks, ', '));
        end
        if ~isempty(missingAxiWrInternalEdges)
            fprintf('  Missing axi_master_wr internal edges:\n');
            for i = 1:numel(missingAxiWrInternalEdges)
                fprintf('    - %s\n', missingAxiWrInternalEdges{i});
            end
        end
        error('run_stage2_decode_internal_smoke:Failed', ...
            'stage2 decode path or kv address generation check failed');
    end

    close_system(mdlName, 0);
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
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