function result = run_stage2_axi_wr_functional_smoke(rootDir, options)
%RUN_STAGE2_AXI_WR_FUNCTIONAL_SMOKE Functional smoke for axi_master_wr_u logic.
%   This smoke validates deterministic behavior:
%   1) wr_valid request-hold stays high until write_done.
%   2) request_next_line only pulses at write_done (complete && count_done).

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    if buildModel
        implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', kvCfg));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    wrPath = [mdlName '/axi_master_wr_u'];
    requiredBlocks = {'request_or_hold', 'wvalid_state_z', 'write_active_z', ...
        'write_count_z', 'write_done_logic', 'next_line_logic'};
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

    seqA = struct();
    seqA.wr_dvalid = [1 1 0 0 0];
    seqA.wr_complete = [0 0 1 1 1];
    seqA.burst_len = [2 2 2 2 2];
    [traceA, okA, msgA] = simulate_axi_wr(seqA);

    seqB = struct();
    seqB.wr_dvalid = [1 1 1 0 0 0];
    seqB.wr_complete = [0 0 0 0 1 1];
    seqB.burst_len = [3 3 3 3 3 3];
    [traceB, okB, msgB] = simulate_axi_wr(seqB);

    result = struct();
    result.scenario_a_ok = okA;
    result.scenario_b_ok = okB;
    result.scenario_a_msg = msgA;
    result.scenario_b_msg = msgB;
    result.trace_a = traceA;
    result.trace_b = traceB;
    result.pass = okA && okB;

    close_system(mdlName, 0);

    if result.pass
        fprintf('Stage2 axi_master_wr functional smoke PASS\n');
    else
        fprintf('Stage2 axi_master_wr functional smoke FAIL\n');
        if ~okA
            fprintf('  Scenario A: %s\n', msgA);
        end
        if ~okB
            fprintf('  Scenario B: %s\n', msgB);
        end
        error('run_stage2_axi_wr_functional_smoke:Failed', ...
            'axi_master_wr functional checks failed');
    end
end

function [trace, ok, msg] = simulate_axi_wr(seq)
    n = numel(seq.wr_dvalid);
    wvalid_state = false;
    write_active = false;
    write_count = 0;

    trace = repmat(struct('wr_valid', false, 'request_next_line', false, ...
        'write_active', false, 'write_count', 0), n, 1);

    for t = 1:n
        wr_dvalid = logical(seq.wr_dvalid(t));
        wr_complete = logical(seq.wr_complete(t));
        burst_len = double(seq.burst_len(t));

        request_or_hold = wr_dvalid || wvalid_state;
        start_write = request_or_hold && ~write_active;
        active_or_start = write_active || start_write;

        beat_fire = active_or_start && wr_dvalid;
        count_inc = write_count + 1;
        count_done = count_inc >= burst_len;
        write_done = wr_complete && count_done;

        active_next = start_write || (write_active && ~write_done);
        count_on_beat = ternary(beat_fire, count_inc, write_count);
        count_next = ternary(start_write, 0, count_on_beat);
        wvalid_next = request_or_hold && ~write_done;

        wr_valid = request_or_hold;
        request_next_line = write_done && write_active;

        trace(t).wr_valid = wr_valid;
        trace(t).request_next_line = request_next_line;
        trace(t).write_active = write_active;
        trace(t).write_count = write_count;

        wvalid_state = wvalid_next;
        write_active = active_next;
        write_count = count_next;
    end

    ok = true;
    msg = 'ok';

    hsIdx = find(logical(seq.wr_dvalid), 1, 'first');
    if isempty(hsIdx)
        ok = false;
        msg = 'no write request observed';
        return;
    end

    doneIdx = find([trace.request_next_line], 1, 'first');
    if isempty(doneIdx)
        ok = false;
        msg = 'no write_done/request_next_line observed';
        return;
    end

    if any(~[trace(hsIdx:doneIdx).wr_valid])
        ok = false;
        msg = 'wr_valid dropped before write_done';
        return;
    end

    if doneIdx < numel(trace) && trace(doneIdx+1).wr_valid
        ok = false;
        msg = 'wr_valid did not clear after write_done';
        return;
    end

    if sum([trace.request_next_line]) ~= 1
        ok = false;
        msg = 'request_next_line should pulse exactly once';
        return;
    end

    if any([trace.request_next_line] & ~[trace.write_active])
        ok = false;
        msg = 'request_next_line asserted when write_active is low';
        return;
    end
end

function out = ternary(cond, trueVal, falseVal)
    if cond
        out = trueVal;
    else
        out = falseVal;
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
