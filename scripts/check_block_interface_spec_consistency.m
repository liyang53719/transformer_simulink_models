function result = check_block_interface_spec_consistency(rootDir)
%CHECK_BLOCK_INTERFACE_SPEC_CONSISTENCY Validate qwen2_block_top ports against frozen M1 spec.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s', mdlPath);
    end

    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    requiredIn = {'clk','rst_n','mode_decode','start','eos_in','in_valid','out_ready', ...
        'in_hidden','in_residual','kv_cache_rd_data','kv_cache_rd_valid', ...
        'cfg_seq_len','cfg_token_pos','cfg_eps'};
    requiredOut = {'done','eos_out','in_ready','out_valid','out_hidden','kv_cache_wr_data','kv_cache_wr_en'};
    requiredSubs = {'rmsnorm_u','rope_u','qkv_proj_u','attention_u','ffn_swiglu_u','residual_u','kv_cache_if_u','ctrl_fsm_u'};

    result = struct();
    result.pass = true;
    result.missing_in = {};
    result.missing_out = {};
    result.missing_subs = {};

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
        error('check_block_interface_spec_consistency:Mismatch', 'Top model does not match frozen interface spec.');
    end

    close_system(mdlName, 0);
end
