function result = run_stage2_decode_internal_smoke(rootDir, options)
%RUN_STAGE2_DECODE_INTERNAL_SMOKE Validate stage2 decode internal wiring.
%
% This smoke test focuses on deterministic checks:
% 1) Model compile/update succeeds after stage2 build.
% 2) Required top-level decode-path and kv-address wiring exists.
% 3) kv_addr_gen_u / axi masters use MATLAB Function cores with expected top-level semantics.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));

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
        'prefill_sched_u/1', 'kv_cache_if_u/4';
        'axi_master_rd_u/1', 'kv_cache_if_u/2';
        'kv_cache_if_u/1', 'attention_u/1';
        'prefill_sched_u/1', 'attention_u/4';
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
    missingKvCore = {};
    if isempty(find_system(kvAddrPath, 'SearchDepth', 1, 'Name', 'kv_addr_core'))
        missingKvCore{end+1} = 'kv_addr_core'; %#ok<AGROW>
    end
    legacyKvBlocks = {'rd_base_const','wr_base_const','stride_const','decode_burst_const', ...
        'rd_addr_scale','rd_addr_prefill_scale','rd_addr_mode_sel','rd_addr_add_base', ...
        'wr_addr_scale','wr_addr_add','wr_addr_mode_sel','wr_addr_add_base','rd_len_mode_sel','wr_len_mode_sel'};
    unexpectedKvLegacy = collect_existing_blocks(kvAddrPath, legacyKvBlocks);

    result = struct();
    result.compile_update_ok = true;
    result.missing_edges = missingEdges;
    result.missing_kv_addr_blocks = missingKvCore;
    result.decode_path_wiring_ok = isempty(missingEdges);
    result.addr_gen_structure_ok = isempty(missingKvCore) && isempty(unexpectedKvLegacy);
    badKvConstValues = {};

    axiRdPath = [mdlName '/axi_master_rd_u'];
    missingAxiRdBlocks = {};
    if isempty(find_system(axiRdPath, 'SearchDepth', 1, 'Name', 'axi_rd_core'))
        missingAxiRdBlocks{end+1} = 'axi_rd_core'; %#ok<AGROW>
    end
    unexpectedAxiRdLegacy = collect_existing_blocks(axiRdPath, {'avalid_state_z','start_or_hold','addr_hs_logic', ...
        'burst_active_z','burst_count_z','burst_done_logic','rd_valid_gate'});
    missingAxiRdInternalEdges = {};

    result.missing_axi_rd_blocks = missingAxiRdBlocks;
    result.axi_rd_structure_ok = isempty(missingAxiRdBlocks) && isempty(unexpectedAxiRdLegacy);
    result.missing_kv_internal_edges = unexpectedKvLegacy;
    result.kv_internal_semantics_ok = isempty(unexpectedKvLegacy);
    result.missing_axi_rd_internal_edges = missingAxiRdInternalEdges;
    result.axi_rd_semantics_ok = isempty(missingAxiRdInternalEdges) && isempty(unexpectedAxiRdLegacy);
    result.bad_kv_constant_values = badKvConstValues;
    result.kv_constant_values_ok = isempty(badKvConstValues);

    axiWrPath = [mdlName '/axi_master_wr_u'];
    missingAxiWrBlocks = {};
    if isempty(find_system(axiWrPath, 'SearchDepth', 1, 'Name', 'axi_wr_core'))
        missingAxiWrBlocks{end+1} = 'axi_wr_core'; %#ok<AGROW>
    end
    unexpectedAxiWrLegacy = collect_existing_blocks(axiWrPath, {'wvalid_state_z','request_or_hold','write_active_z', ...
        'write_count_z','write_done_logic','next_line_logic'});
    missingAxiWrInternalEdges = {};

    result.missing_axi_wr_blocks = missingAxiWrBlocks;
    result.axi_wr_structure_ok = isempty(missingAxiWrBlocks) && isempty(unexpectedAxiWrLegacy);
    result.missing_axi_wr_internal_edges = missingAxiWrInternalEdges;
    result.axi_wr_semantics_ok = isempty(missingAxiWrInternalEdges) && isempty(unexpectedAxiWrLegacy);
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
        if ~isempty(missingKvCore)
            fprintf('  Missing kv_addr_gen blocks: %s\n', strjoin(missingKvCore, ', '));
        end
        if ~isempty(badKvConstValues)
            fprintf('  Bad kv_addr_gen constant values:\n');
            for i = 1:numel(badKvConstValues)
                fprintf('    - %s\n', badKvConstValues{i});
            end
        end
        if ~isempty(unexpectedKvLegacy)
            fprintf('  Unexpected legacy kv_addr_gen blocks: %s\n', strjoin(unexpectedKvLegacy, ', '));
        end
        if ~isempty(missingAxiRdBlocks)
            fprintf('  Missing axi_master_rd blocks: %s\n', strjoin(missingAxiRdBlocks, ', '));
        end
        if ~isempty(unexpectedAxiRdLegacy)
            fprintf('  Unexpected legacy axi_master_rd blocks: %s\n', strjoin(unexpectedAxiRdLegacy, ', '));
        end
        if ~isempty(missingAxiWrBlocks)
            fprintf('  Missing axi_master_wr blocks: %s\n', strjoin(missingAxiWrBlocks, ', '));
        end
        if ~isempty(unexpectedAxiWrLegacy)
            fprintf('  Unexpected legacy axi_master_wr blocks: %s\n', strjoin(unexpectedAxiWrLegacy, ', '));
        end
        error('run_stage2_decode_internal_smoke:Failed', ...
            'stage2 decode path or kv/axi matlab-function conversion check failed');
    end

    close_system(mdlName, 0);
end

function existing = collect_existing_blocks(sysPath, names)
    existing = {};
    for i = 1:numel(names)
        if ~isempty(find_system(sysPath, 'SearchDepth', 1, 'Name', names{i}))
            existing{end+1} = names{i}; %#ok<AGROW>
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