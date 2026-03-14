function result = check_simulink_placeholder_consistency(rootDir)
%CHECK_SIMULINK_PLACEHOLDER_CONSISTENCY Verify top placeholder model structure.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    if ~exist(mdlPath, 'file')
        error('Model not found: %s. Run create_qwen2_block_top_placeholder first.', mdlPath);
    end

    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);

    requiredSubs = {'rmsnorm_u','rope_u','qkv_proj_u','attention_u','ffn_swiglu_u','residual_u','kv_cache_if_u','ctrl_fsm_u'};
    requiredIn = {'in_valid','out_ready','mode_decode','start','cfg_seq_len','cfg_token_pos','cfg_eps'};
    requiredOut = {'in_ready','out_valid','done','kv_cache_wr_en'};

    result = struct();
    result.model = mdlName;
    result.subsystems_ok = true;
    result.inports_ok = true;
    result.outports_ok = true;

    for i = 1:numel(requiredSubs)
        p = [mdlName '/' requiredSubs{i}];
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredSubs{i}))
            fprintf('Missing subsystem: %s\n', p);
            result.subsystems_ok = false;
        end
    end

    for i = 1:numel(requiredIn)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredIn{i}))
            fprintf('Missing inport: %s\n', requiredIn{i});
            result.inports_ok = false;
        end
    end

    for i = 1:numel(requiredOut)
        if isempty(find_system(mdlName, 'SearchDepth', 1, 'Name', requiredOut{i}))
            fprintf('Missing outport: %s\n', requiredOut{i});
            result.outports_ok = false;
        end
    end

    result.pass = result.subsystems_ok && result.inports_ok && result.outports_ok;
    if result.pass
        fprintf('Simulink placeholder consistency PASS\n');
    else
        error('Simulink placeholder consistency FAIL');
    end

    close_system(mdlName, 0);
end
