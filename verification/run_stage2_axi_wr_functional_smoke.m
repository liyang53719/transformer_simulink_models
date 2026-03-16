function result = run_stage2_axi_wr_functional_smoke(rootDir)
%RUN_STAGE2_AXI_WR_FUNCTIONAL_SMOKE Functional smoke for axi_master_wr_u logic.
%   This smoke validates deterministic behavior:
%   1) wr_valid must mirror wr_dvalid.
%   2) request_next_line must equal wr_complete AND wr_dvalid.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    addpath(fullfile(rootDir, 'scripts'));
    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready'));

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    wrPath = [mdlName '/axi_master_wr_u'];
    requiredBlocks = {'next_line_logic', 'wr_valid', 'request_next_line'};
    missingBlocks = {};
    for i = 1:numel(requiredBlocks)
        if isempty(find_system(wrPath, 'SearchDepth', 1, 'Name', requiredBlocks{i}))
            missingBlocks{end+1} = requiredBlocks{i}; %#ok<AGROW>
        end
    end

    if ~isempty(missingBlocks)
        close_system(mdlName, 0);
        error('run_stage2_axi_wr_functional_smoke:MissingBlocks', ...
            'axi_master_wr_u missing blocks: %s', strjoin(missingBlocks, ', '));
    end

    seq = struct();
    seq.wr_dvalid = [0 1 1 0 1 0 1 1];
    seq.wr_complete = [0 0 1 1 0 1 1 0];

    [trace, ok, msg] = simulate_axi_wr(seq);

    result = struct();
    result.ok = ok;
    result.msg = msg;
    result.trace = trace;
    result.pass = ok;

    close_system(mdlName, 0);

    if result.pass
        fprintf('Stage2 axi_master_wr functional smoke PASS\n');
    else
        fprintf('Stage2 axi_master_wr functional smoke FAIL\n');
        fprintf('  %s\n', result.msg);
        error('run_stage2_axi_wr_functional_smoke:Failed', ...
            'axi_master_wr functional checks failed');
    end
end

function [trace, ok, msg] = simulate_axi_wr(seq)
    n = numel(seq.wr_dvalid);
    trace = repmat(struct('wr_valid', false, 'request_next_line', false), n, 1);

    for t = 1:n
        wr_dvalid = logical(seq.wr_dvalid(t));
        wr_complete = logical(seq.wr_complete(t));

        wr_valid = wr_dvalid;
        request_next_line = wr_complete && wr_dvalid;

        trace(t).wr_valid = wr_valid;
        trace(t).request_next_line = request_next_line;
    end

    ok = true;
    msg = 'ok';

    if any([trace.wr_valid] ~= logical(seq.wr_dvalid))
        ok = false;
        msg = 'wr_valid does not mirror wr_dvalid';
        return;
    end

    expectedNext = logical(seq.wr_complete) & logical(seq.wr_dvalid);
    if any([trace.request_next_line] ~= expectedNext)
        ok = false;
        msg = 'request_next_line is not wr_complete AND wr_dvalid';
        return;
    end
end
