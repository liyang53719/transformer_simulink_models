function result = run_stage2_model_self_init_smoke(rootDir, options)
%RUN_STAGE2_MODEL_SELF_INIT_SMOKE Verify qwen2_block_top can self-initialize bus objects.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    addpath(fullfile(rootDir, 'scripts'));

    clear_stage2_bus_objects();

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    [~, mdlName] = fileparts(mdlPath);
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end

    load_system(mdlPath);
    cleanup = onCleanup(@()safe_close_model(mdlName)); %#ok<NASGU>
    set_param(mdlName, 'SimulationCommand', 'update');

    result = struct();
    result.pass = true;
    fprintf('Stage2 model self-init smoke PASS\n');
end

function clear_stage2_bus_objects()
    names = { ...
        'WeightReqRmsBus', 'WeightReqQkvBus', 'WeightReqAttnBus', 'WeightReqFfnBus', ...
        'WeightAddrRmsBus', 'WeightAddrQkvBus', 'WeightAddrAttnBus', 'WeightAddrFfnBus', ...
        'WeightReqBus', 'WeightRspBus', 'QkvStreamBus', 'AttentionFlowBus', 'PrefillScheduleBus'};
    for i = 1:numel(names)
        evalin('base', sprintf('if exist(''%s'',''var''), clear(''%s''); end', names{i}, names{i}));
    end
end

function safe_close_model(mdlName)
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end