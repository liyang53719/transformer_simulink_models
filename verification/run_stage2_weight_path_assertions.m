function result = run_stage2_weight_path_assertions(rootDir, options)
%RUN_STAGE2_WEIGHT_PATH_ASSERTIONS Structural+semantic assertions for DDR->RAM weight path.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    [~, mdlName] = fileparts(mdlPath);

    if buildModel
        addpath(fullfile(rootDir, 'scripts'));
        implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready'));
    end

    load_system(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    require_blocks(mdlName, {
        [mdlName '/weight_addr_map_u'], ...
        [mdlName '/prefill_sched_u'], ...
        [mdlName '/axi_weight_rd_u'], ...
        [mdlName '/w_req_bc'], ...
        [mdlName '/rms_addr_bc'], ...
        [mdlName '/qkv_addr_bc'], ...
        [mdlName '/attn_addr_bc'], ...
        [mdlName '/ffn_addr_bc'], ...
        [mdlName '/rmsnorm_u/gamma_sram'], ...
        [mdlName '/qkv_proj_u/q_sram'], ...
        [mdlName '/attention_u/q_sram'], ...
        [mdlName '/ffn_swiglu_u/up_sram']});

    require_edge(mdlName, 'cfg_token_pos/1', 'weight_addr_map_u/1');
    require_edge(mdlName, 'weight_addr_map_u/1', 'rms_addr_bc/1');
    require_edge(mdlName, 'weight_addr_map_u/2', 'qkv_addr_bc/1');
    require_edge(mdlName, 'weight_addr_map_u/5', 'attn_addr_bc/1');
    require_edge(mdlName, 'weight_addr_map_u/8', 'ffn_addr_bc/1');
    require_edge(mdlName, 'rms_addr_bc/1', 'rmsnorm_u/4');
    require_edge(mdlName, 'qkv_addr_bc/1', 'qkv_proj_u/3');
    require_edge(mdlName, 'attn_addr_bc/1', 'attention_u/3');
    require_edge(mdlName, 'ffn_addr_bc/1', 'ffn_swiglu_u/3');
    require_edge(mdlName, 'prefill_sched_u/1', 'kv_cache_if_u/4');
    require_edge(mdlName, 'prefill_sched_u/2', 'kv_cache_if_u/5');
    require_edge(mdlName, 'prefill_sched_u/6', 'kv_cache_if_u/6');
    require_edge(mdlName, 'prefill_sched_u/8', 'kv_cache_if_u/7');
    require_edge(mdlName, 'qkv_proj_u/1', 'kv_cache_if_u/1');
    require_edge(mdlName, 'rms_req_sel/1', 'axi_weight_rd_u/1');
    require_edge(mdlName, 'axi_weight_rd_u/1', 'rmsnorm_u/3');

    % First-load + cache-hit semantics are enforced by req_needed = NOT(sram_valid)
    % and mux select by sram_valid inside each stream helper.
    require_block_param([mdlName '/rmsnorm_u/gamma_req_needed'], 'Operator', 'NOT');
    require_block_param([mdlName '/rmsnorm_u/gamma_sram_data_valid_z'], 'InitialCondition', '0');
    require_block_param([mdlName '/rmsnorm_u/gamma_sram_data_sel'], 'Threshold', '0.5');

    % Address increment semantics now delegated to weight_addr_map_u.
    require_blocks(mdlName, {
        [mdlName '/weight_addr_map_u/tok_head_prod'], ...
        [mdlName '/weight_addr_map_u/page_offset'], ...
        [mdlName '/weight_addr_map_u/addr_base_sum']});

    close_system(mdlName, 0);
    result = struct('pass', true);
    fprintf('Stage2 weight path assertions PASS\n');
end

function require_blocks(mdlName, paths)
    for i = 1:numel(paths)
        if getSimulinkBlockHandle(paths{i}) == -1
            error('run_stage2_weight_path_assertions:MissingBlock', 'Missing block: %s', paths{i});
        end
    end
end

function require_edge(sys, src, dst)
    dstParts = split(string(dst), '/');
    dstBlk = [sys '/' char(dstParts(1))];
    dstPort = str2double(dstParts(2));
    phDst = get_param(dstBlk, 'PortHandles');
    if dstPort > numel(phDst.Inport)
        error('run_stage2_weight_path_assertions:BadDst', 'Invalid dst port %s', dst);
    end
    ln = get_param(phDst.Inport(dstPort), 'Line');
    if ln == -1
        error('run_stage2_weight_path_assertions:MissingEdge', 'Missing edge %s -> %s', src, dst);
    end
    srcH = get_param(ln, 'SrcPortHandle');
    srcFull = getfullname(srcH);
    srcParts = split(string(src), '/');
    want = [sys '/' char(srcParts(1)) '/'];
    if ~startsWith(srcFull, want)
        error('run_stage2_weight_path_assertions:WrongEdge', 'Edge into %s is from %s, expected %s', dst, srcFull, src);
    end
end

function require_block_param(path, paramName, expected)
    v = get_param(path, paramName);
    if ~strcmp(string(v), string(expected))
        error('run_stage2_weight_path_assertions:BadParam', ...
            'Block %s param %s = %s, expected %s', path, paramName, string(v), string(expected));
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
