function result = check_block_interface_spec_consistency(rootDir, options)
%CHECK_BLOCK_INTERFACE_SPEC_CONSISTENCY Validate qwen2_block_top against a hardware-first top-level contract.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s', mdlPath);
    end

    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    profile = lower(string(getFieldOr(options, 'Profile', 'm1_frozen')));
    validateRootTypes = logical(getFieldOr(options, 'ValidateRootTypes', false));
    validateHardwareEdges = logical(getFieldOr(options, 'ValidateHardwareEdges', false));

    requiredIn = {'mode_decode','start','eos_in','in_valid','out_ready', ...
        'in_hidden','in_residual','kv_cache_rd_data','kv_cache_rd_valid', ...
        'cfg_seq_len','cfg_token_pos','cfg_eps'};
    requiredOut = {'done','eos_out','in_ready','out_valid','out_hidden','kv_cache_wr_data','kv_cache_wr_en'};
    requiredSubs = {'rmsnorm_u','rope_u','qkv_proj_u','attention_u','ffn_swiglu_u','residual_u','kv_cache_if_u','ctrl_fsm_u'};

    requiredCtrlIn = {};
    requiredCtrlOut = {};
    requiredAxiIn = {};
    requiredAxiOut = {};
    requiredMemSubs = {};

    if profile == "v2_memory_first" || profile == "stage2_memory_ready_hw"
        requiredCtrlIn = {'stop_req'};
        requiredCtrlOut = {'busy','irq','error_code'};
        requiredAxiIn = {'kv_mem_rd_ready','kv_mem_wr_ready'};
        requiredAxiOut = {'kv_mem_rd_addr','kv_mem_rd_len','kv_mem_rd_valid', ...
            'kv_mem_wr_addr','kv_mem_wr_len','kv_mem_wr_valid'};
        requiredMemSubs = {'axi_master_rd_u','axi_master_wr_u','ddr_model_if_u'};
        validateRootTypes = true;
        validateHardwareEdges = true;
    end

    result = struct();
    result.pass = true;
    result.missing_in = {};
    result.missing_out = {};
    result.missing_subs = {};
    result.profile = char(profile);
    result.missing_ctrl_ports = {};
    result.missing_axi_ports = {};
    result.root_type_mismatches = struct('port', {}, 'declared', {}, 'expected', {});
    result.missing_edges = {};
    result.wrong_edges = {};

    for i = 1:numel(requiredIn)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredIn{i}))
            result.pass = false;
            result.missing_in{end+1} = requiredIn{i}; %#ok<AGROW>
        end
    end

    for i = 1:numel(requiredOut)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredOut{i}))
            result.pass = false;
            result.missing_out{end+1} = requiredOut{i}; %#ok<AGROW>
        end
    end

    for i = 1:numel(requiredSubs)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredSubs{i}))
            result.pass = false;
            result.missing_subs{end+1} = requiredSubs{i}; %#ok<AGROW>
        end
    end

    for i = 1:numel(requiredCtrlIn)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredCtrlIn{i}))
            result.pass = false;
            result.missing_ctrl_ports{end+1} = requiredCtrlIn{i}; %#ok<AGROW>
        end
    end
    for i = 1:numel(requiredCtrlOut)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredCtrlOut{i}))
            result.pass = false;
            result.missing_ctrl_ports{end+1} = requiredCtrlOut{i}; %#ok<AGROW>
        end
    end

    for i = 1:numel(requiredAxiIn)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredAxiIn{i}))
            result.pass = false;
            result.missing_axi_ports{end+1} = requiredAxiIn{i}; %#ok<AGROW>
        end
    end
    for i = 1:numel(requiredAxiOut)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredAxiOut{i}))
            result.pass = false;
            result.missing_axi_ports{end+1} = requiredAxiOut{i}; %#ok<AGROW>
        end
    end

    for i = 1:numel(requiredMemSubs)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredMemSubs{i}))
            result.pass = false;
            result.missing_subs{end+1} = requiredMemSubs{i}; %#ok<AGROW>
        end
    end

    if validateRootTypes
        expectedInTypes = get_expected_inport_types(profile);
        inNames = fieldnames(expectedInTypes);
        for i = 1:numel(inNames)
            portName = inNames{i};
            blk = [mdlName '/' portName];
            if getSimulinkBlockHandle(blk) == -1
                continue;
            end

            declared = string(get_param(blk, 'OutDataTypeStr'));
            expected = string(expectedInTypes.(portName));
            if ~strcmp(normalize_dtype(declared), normalize_dtype(expected))
                result.pass = false;
                result.root_type_mismatches(end+1) = struct( ...
                    'port', portName, ...
                    'declared', char(declared), ...
                    'expected', char(expected)); %#ok<AGROW>
            end
        end

        expectedOutTypes = get_expected_outport_types(profile);
        outNames = fieldnames(expectedOutTypes);
        for i = 1:numel(outNames)
            portName = outNames{i};
            blk = [mdlName '/' portName];
            if getSimulinkBlockHandle(blk) == -1
                continue;
            end

            declared = string(get_param(blk, 'OutDataTypeStr'));
            expected = string(expectedOutTypes.(portName));
            if ~strcmp(normalize_dtype(declared), normalize_dtype(expected))
                result.pass = false;
                result.root_type_mismatches(end+1) = struct( ...
                    'port', portName, ...
                    'declared', char(declared), ...
                    'expected', char(expected)); %#ok<AGROW>
            end
        end
    end

    if validateHardwareEdges
        requiredEdges = get_required_hardware_edges(profile);
        for i = 1:size(requiredEdges, 1)
            [edgeOk, reason] = has_exact_edge(mdlName, requiredEdges{i, 1}, requiredEdges{i, 2});
            if ~edgeOk
                result.pass = false;
                if reason == "missing"
                    result.missing_edges{end+1} = sprintf('%s -> %s', requiredEdges{i, 1}, requiredEdges{i, 2}); %#ok<AGROW>
                else
                    result.wrong_edges{end+1} = sprintf('%s -> %s', requiredEdges{i, 1}, requiredEdges{i, 2}); %#ok<AGROW>
                end
            end
        end
    end

    if result.pass
        fprintf('Interface spec consistency PASS for %s\n', mdlName);
    else
        fprintf('Interface spec consistency FAIL for %s\n', mdlName);
        if ~isempty(result.missing_in)
            fprintf('Missing inports: %s\n', strjoin(string(result.missing_in), ', '));
        end
        if ~isempty(result.missing_out)
            fprintf('Missing outports: %s\n', strjoin(string(result.missing_out), ', '));
        end
        if ~isempty(result.missing_subs)
            fprintf('Missing subsystems: %s\n', strjoin(string(result.missing_subs), ', '));
        end
        if ~isempty(result.missing_ctrl_ports)
            fprintf('Missing control ports: %s\n', strjoin(string(result.missing_ctrl_ports), ', '));
        end
        if ~isempty(result.missing_axi_ports)
            fprintf('Missing AXI ports: %s\n', strjoin(string(result.missing_axi_ports), ', '));
        end
        if ~isempty(result.root_type_mismatches)
            for i = 1:numel(result.root_type_mismatches)
                mismatch = result.root_type_mismatches(i);
                fprintf('Type mismatch on %s: declared=%s expected=%s\n', ...
                    mismatch.port, mismatch.declared, mismatch.expected);
            end
        end
        if ~isempty(result.missing_edges)
            fprintf('Missing hardware edges: %s\n', strjoin(string(result.missing_edges), ', '));
        end
        if ~isempty(result.wrong_edges)
            fprintf('Wrong hardware edges: %s\n', strjoin(string(result.wrong_edges), ', '));
        end
        error('check_block_interface_spec_consistency:Mismatch', 'Top model does not match frozen interface spec.');
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

function expected = get_expected_inport_types(profile)
    expected = struct( ...
        'mode_decode', 'boolean', ...
        'start', 'boolean', ...
        'eos_in', 'boolean', ...
        'in_valid', 'boolean', ...
        'out_ready', 'boolean', ...
        'in_hidden', 'fixdt(1,64,30)', ...
        'in_residual', 'fixdt(1,64,30)', ...
        'kv_cache_rd_data', 'single', ...
        'kv_cache_rd_valid', 'boolean', ...
        'cfg_seq_len', 'fixdt(1,17,15)', ...
        'cfg_token_pos', 'fixdt(1,17,15)', ...
        'cfg_eps', 'fixdt(1,256,120)');

    if profile == "v2_memory_first" || profile == "stage2_memory_ready_hw"
        expected.stop_req = 'boolean';
        expected.kv_mem_rd_ready = 'boolean';
        expected.kv_mem_wr_ready = 'boolean';
        expected.cfg_rope_theta_scale = 'fixdt(1,17,15)';
        expected.cfg_rope_sin_mix_scale = 'fixdt(1,17,15)';
        expected.cfg_weight_num_heads = 'fixdt(1,17,0)';
        expected.cfg_weight_page_base = 'fixdt(1,17,0)';
        expected.cfg_weight_page_stride = 'fixdt(1,17,0)';
    end
end

function expected = get_expected_outport_types(profile)
    expected = struct();
    if profile == "v2_memory_first" || profile == "stage2_memory_ready_hw"
        expected.w_rd_req_bus = 'Bus: WeightReqBus';
    end
end

function edges = get_required_hardware_edges(profile)
    edges = {
        'cfg_eps/1', 'rmsnorm_u/2';
        'cfg_token_pos/1', 'prefill_sched_u/1';
        'cfg_seq_len/1', 'prefill_sched_u/2';
        'cfg_token_pos/1', 'kv_addr_gen_u/1';
        'cfg_seq_len/1', 'kv_addr_gen_u/2';
        'mode_decode/1', 'kv_addr_gen_u/3'};

    if profile == "v2_memory_first" || profile == "stage2_memory_ready_hw"
        edges = [edges; {
            'cfg_token_pos/1', 'weight_addr_map_u/1';
            'cfg_weight_num_heads/1', 'weight_addr_map_u/2';
            'cfg_weight_page_base/1', 'weight_addr_map_u/3';
            'cfg_weight_page_stride/1', 'weight_addr_map_u/4';
            'cfg_rope_theta_scale/1', 'rope_u/3';
            'cfg_rope_sin_mix_scale/1', 'rope_u/4'}];
    end
end

function normalized = normalize_dtype(dtype)
    normalized = regexprep(lower(strtrim(char(string(dtype)))), '\s+', '');
end

function [ok, reason] = has_exact_edge(sys, src, dst)
    ok = false;
    reason = "missing";

    dstParts = split(string(dst), '/');
    dstBlk = [sys '/' char(dstParts(1))];
    dstPort = str2double(dstParts(2));
    phDst = get_param(dstBlk, 'PortHandles');
    if dstPort > numel(phDst.Inport)
        return;
    end

    ln = get_param(phDst.Inport(dstPort), 'Line');
    if isequal(ln, -1)
        return;
    end

    srcH = get_param(ln, 'SrcPortHandle');
    srcFull = string(getfullname(srcH));
    srcParts = split(string(src), '/');
    wantPrefix = string([sys '/' char(srcParts(1)) '/']);
    ok = startsWith(srcFull, wantPrefix);
    if ~ok
        reason = "wrong";
    end
end
