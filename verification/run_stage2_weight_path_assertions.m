function result = run_stage2_weight_path_assertions(rootDir, options)
%RUN_STAGE2_WEIGHT_PATH_ASSERTIONS Structural+semantic assertions for DDR->RAM weight path.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    [~, mdlName] = fileparts(mdlPath);

    load_system(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    require_blocks(mdlName, {
        [mdlName '/weight_addr_map_u'], ...
        [mdlName '/prefill_sched_u'], ...
        [mdlName '/w_req_bc'], ...
        [mdlName '/weight_addr_map_u/addr_map_core'], ...
        [mdlName '/weight_addr_map_u/rms_addr_bc'], ...
        [mdlName '/weight_addr_map_u/qkv_addr_bc'], ...
        [mdlName '/weight_addr_map_u/attn_addr_bc'], ...
        [mdlName '/weight_addr_map_u/ffn_addr_bc'], ...
        [mdlName '/prefill_sched_u/schedule_bc'], ...
        [mdlName '/prefill_sched_u/sched_core'], ...
        [mdlName '/rmsnorm_u/gamma_addr_be'], ...
        [mdlName '/qkv_proj_u/addr_sel'], ...
        [mdlName '/attention_u/addr_sel'], ...
        [mdlName '/attention_u/schedule_sel'], ...
        [mdlName '/kv_cache_if_u/schedule_sel'], ...
        [mdlName '/rmsnorm_u/gamma_sram'], ...
        [mdlName '/qkv_proj_u/q_sram'], ...
        [mdlName '/attention_u/q_sram'], ...
        [mdlName '/ffn_swiglu_u/up_sram']});

    require_edge(mdlName, 'cfg_token_pos/1', 'weight_addr_map_u/1');
    require_edge(mdlName, 'weight_addr_map_u/1', 'rmsnorm_u/4');
    require_edge(mdlName, 'weight_addr_map_u/2', 'qkv_proj_u/3');
    require_edge(mdlName, 'weight_addr_map_u/3', 'attention_u/3');
    require_edge(mdlName, 'weight_addr_map_u/4', 'ffn_swiglu_u/3');
    require_edge(mdlName, 'mode_decode/1', 'kv_cache_if_u/3');
    require_edge(mdlName, 'prefill_sched_u/1', 'kv_cache_if_u/4');
    require_edge(mdlName, 'prefill_sched_u/1', 'attention_u/4');
    require_edge(mdlName, 'qkv_proj_u/1', 'kv_cache_if_u/1');
    require_inport(mdlName, 'w_rd_rsp_bus');
    require_no_block([mdlName '/axi_weight_rd_u']);
    require_edge(mdlName, 'w_rd_rsp_bus/1', 'rmsnorm_u/3');
    require_edge(mdlName, 'w_rd_rsp_bus/1', 'qkv_proj_u/2');
    require_edge(mdlName, 'w_rd_rsp_bus/1', 'attention_u/2');
    require_edge(mdlName, 'w_rd_rsp_bus/1', 'ffn_swiglu_u/2');

    % First-load + cache-hit semantics are enforced by req_needed = NOT(sram_valid)
    % and mux select by sram_valid inside each stream helper.
    require_block_param([mdlName '/rmsnorm_u/gamma_req_needed'], 'Operator', 'NOT');
    require_block_param([mdlName '/rmsnorm_u/gamma_sram_data_valid_z'], 'InitialCondition', '0');
    require_block_param([mdlName '/rmsnorm_u/gamma_sram_data_sel'], 'Threshold', '0.5');

    require_block_param([mdlName '/weight_addr_map_u/rms_addr_bc'], 'UseBusObject', 'on');
    require_block_param([mdlName '/prefill_sched_u/schedule_bc'], 'UseBusObject', 'on');

    close_system(mdlName, 0);
    result = struct('pass', true);
    fprintf('Stage2 weight path assertions PASS\n');
end

function require_blocks(~, paths)
    for i = 1:numel(paths)
        if getSimulinkBlockHandle(paths{i}) == -1
            error('run_stage2_weight_path_assertions:MissingBlock', 'Missing block: %s', paths{i});
        end
    end
end

function require_no_block(path)
    if getSimulinkBlockHandle(path) ~= -1
        error('run_stage2_weight_path_assertions:UnexpectedBlock', 'Unexpected block still present: %s', path);
    end
end

function require_inport(sys, name)
    if isempty(find_system(sys, 'SearchDepth', 1, 'BlockType', 'Inport', 'Name', name))
        error('run_stage2_weight_path_assertions:MissingInport', 'Missing inport: %s/%s', sys, name);
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
