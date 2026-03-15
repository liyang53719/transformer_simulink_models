function result = check_block_interface_spec_consistency(rootDir, options)
%CHECK_BLOCK_INTERFACE_SPEC_CONSISTENCY Validate qwen2_block_top ports against selected spec profile.

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

    requiredIn = {'clk','rst_n','mode_decode','start','eos_in','in_valid','out_ready', ...
        'in_hidden','in_residual','kv_cache_rd_data','kv_cache_rd_valid', ...
        'cfg_seq_len','cfg_token_pos','cfg_eps'};
    requiredOut = {'done','eos_out','in_ready','out_valid','out_hidden','kv_cache_wr_data','kv_cache_wr_en'};
    requiredSubs = {'rmsnorm_u','rope_u','qkv_proj_u','attention_u','ffn_swiglu_u','residual_u','kv_cache_if_u','ctrl_fsm_u'};

    requiredCtrlIn = {};
    requiredCtrlOut = {};
    requiredAxiIn = {};
    requiredAxiOut = {};
    requiredMemSubs = {};

    if profile == "v2_memory_first"
        requiredCtrlIn = {'stop_req'};
        requiredCtrlOut = {'busy','irq','error_code'};
        requiredAxiIn = {'kv_mem_rd_ready','kv_mem_wr_ready'};
        requiredAxiOut = {'kv_mem_rd_addr','kv_mem_rd_len','kv_mem_rd_valid', ...
            'kv_mem_wr_addr','kv_mem_wr_len','kv_mem_wr_valid'};
        requiredMemSubs = {'axi_master_rd_u','axi_master_wr_u','ddr_model_if_u'};
    end

    result = struct();
    result.pass = true;
    result.missing_in = {};
    result.missing_out = {};
    result.missing_subs = {};
    result.profile = char(profile);
    result.missing_ctrl_ports = {};
    result.missing_axi_ports = {};

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
