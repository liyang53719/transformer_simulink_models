function create_qwen2_block_top_placeholder(rootDir)
%CREATE_QWEN2_BLOCK_TOP_PLACEHOLDER Create placeholder Simulink top model.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    mdlDir = fullfile(rootDir, 'simulink', 'models');
    if ~exist(mdlDir, 'dir')
        mkdir(mdlDir);
    end

    mdlName = 'qwen2_block_top';
    mdlPath = fullfile(mdlDir, [mdlName '.slx']);

    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end

    new_system(mdlName);
    open_system(mdlName);

    inports = {'in_valid','out_ready','mode_decode','start','cfg_seq_len','cfg_token_pos','cfg_eps'};
    outports = {'in_ready','out_valid','done','kv_cache_wr_en'};

    for i = 1:numel(inports)
        add_block('simulink/Sources/In1', [mdlName '/' inports{i}], ...
            'Position', [30, 40 + 40*i, 60, 54 + 40*i]);
    end

    for i = 1:numel(outports)
        add_block('simulink/Sinks/Out1', [mdlName '/' outports{i}], ...
            'Position', [1180, 40 + 40*i, 1210, 54 + 40*i]);
    end

    subs = {'rmsnorm_u','rope_u','qkv_proj_u','attention_u','ffn_swiglu_u','residual_u','kv_cache_if_u','ctrl_fsm_u'};
    x0 = 180;
    y0 = 120;
    for i = 1:numel(subs)
        add_block('simulink/Ports & Subsystems/Subsystem', [mdlName '/' subs{i}], ...
            'Position', [x0 + 120*(i-1), y0, x0 + 120*(i-1) + 90, y0 + 80]);
    end

    % Minimal placeholder chain for first smoke run
    add_line(mdlName, 'in_valid/1', 'ctrl_fsm_u/1', 'autorouting', 'on');
    % start is reserved in top-level interface; current placeholder control path
    % uses in_valid as trigger until ctrl_fsm_u internals are implemented.
    add_line(mdlName, 'ctrl_fsm_u/1', 'done/1', 'autorouting', 'on');

    save_system(mdlName, mdlPath);
    close_system(mdlName, 0);

    fprintf('Created model: %s\n', mdlPath);
end
