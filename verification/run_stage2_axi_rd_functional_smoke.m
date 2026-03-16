function result = run_stage2_axi_rd_functional_smoke(rootDir)
%RUN_STAGE2_AXI_RD_FUNCTIONAL_SMOKE Functional smoke for axi_master_rd_u control logic.
%   This smoke validates two behavioral properties with deterministic vectors:
%   1) rd_avalid remains asserted until rd_aready handshake.
%   2) burst_active and beat counter clear exactly at configured burst completion.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end

    addpath(fullfile(rootDir, 'scripts'));
    implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready'));

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    axiPath = [mdlName '/axi_master_rd_u'];
    requiredBlocks = {'start_or_hold', 'addr_hs_logic', 'avalid_state_z', ...
        'burst_active_z', 'burst_count_z', 'burst_done_logic', 'rd_valid_gate'};
    missingBlocks = {};
    for i = 1:numel(requiredBlocks)
        if isempty(find_system(axiPath, 'SearchDepth', 1, 'Name', requiredBlocks{i}))
            missingBlocks{end+1} = requiredBlocks{i}; %#ok<AGROW>
        end
    end

    if ~isempty(missingBlocks)
        close_system(mdlName, 0);
        error('run_stage2_axi_rd_functional_smoke:MissingBlocks', ...
            'axi_master_rd_u missing blocks: %s', strjoin(missingBlocks, ', '));
    end

    % Scenario A: start asserted while rd_aready stalls for 2 cycles.
    seqA = struct();
    seqA.start = [1 0 0 0 0];
    seqA.rd_aready = [0 0 1 1 1];
    seqA.rd_dvalid = [0 0 0 0 0];
    seqA.burst_len = [2 2 2 2 2];
    [traceA, okA, msgA] = simulate_axi_rd(seqA);

    % Scenario B: burst_len=3, data valid 3 beats after handshake.
    seqB = struct();
    seqB.start = [1 0 0 0 0 0 0];
    seqB.rd_aready = [1 1 1 1 1 1 1];
    seqB.rd_dvalid = [0 1 1 1 0 0 0];
    seqB.burst_len = [3 3 3 3 3 3 3];
    [traceB, okB, msgB] = simulate_axi_rd(seqB);

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
        fprintf('Stage2 axi_master_rd functional smoke PASS\n');
    else
        fprintf('Stage2 axi_master_rd functional smoke FAIL\n');
        if ~okA
            fprintf('  Scenario A: %s\n', msgA);
        end
        if ~okB
            fprintf('  Scenario B: %s\n', msgB);
        end
        error('run_stage2_axi_rd_functional_smoke:Failed', ...
            'axi_master_rd functional checks failed');
    end
end

function [trace, ok, msg] = simulate_axi_rd(seq)
    n = numel(seq.start);

    avalid_state = false;
    burst_active = false;
    burst_count = 0;

    trace = repmat(struct('rd_avalid', false, 'addr_hs', false, 'burst_active', false, ...
        'burst_count', 0, 'rd_dvalid_out', false), n, 1);

    for t = 1:n
        start = logical(seq.start(t));
        rd_aready = logical(seq.rd_aready(t));
        rd_dvalid = logical(seq.rd_dvalid(t));
        burst_len = double(seq.burst_len(t));

        start_or_hold = start || avalid_state;
        addr_hs = start_or_hold && rd_aready;
        avalid_next = start_or_hold && ~addr_hs;

        beat_fire = burst_active && rd_dvalid;
        count_inc = burst_count + 1;
        count_done = count_inc >= burst_len;
        burst_done = beat_fire && count_done;
        burst_next = addr_hs || (burst_active && ~burst_done);

        count_on_beat = ternary(beat_fire, count_inc, burst_count);
        count_next = ternary(addr_hs, 0, count_on_beat);

        rd_dvalid_out = rd_dvalid && burst_active;

        trace(t).rd_avalid = start_or_hold;
        trace(t).addr_hs = addr_hs;
        trace(t).burst_active = burst_active;
        trace(t).burst_count = burst_count;
        trace(t).rd_dvalid_out = rd_dvalid_out;

        avalid_state = avalid_next;
        burst_active = burst_next;
        burst_count = count_next;
    end

    [ok, msg] = check_trace(seq, trace);
end

function [ok, msg] = check_trace(seq, trace)
    ok = true;
    msg = 'ok';

    % Rule 1: once started, rd_avalid must stay high until first addr handshake.
    hsIdx = find([trace.addr_hs], 1, 'first');
    if isempty(hsIdx)
        ok = false;
        msg = 'no address handshake observed';
        return;
    end

    if any(~[trace(1:hsIdx).rd_avalid])
        ok = false;
        msg = 'rd_avalid dropped before handshake';
        return;
    end

    % Rule 2: rd_dvalid_out can only be high while burst_active is high.
    if any([trace.rd_dvalid_out] & ~[trace.burst_active])
        ok = false;
        msg = 'rd_dvalid_out asserted when burst not active';
        return;
    end

    % Rule 3: after enough valid beats, burst_active must clear.
    burstLen = seq.burst_len(hsIdx);
    validAfterHs = seq.rd_dvalid(hsIdx+1:end);
    if sum(validAfterHs) >= burstLen
        % Find first cycle reaching burstLen valid beats after handshake.
        accum = cumsum(validAfterHs);
        doneLocal = find(accum >= burstLen, 1, 'first');
        doneIdx = hsIdx + doneLocal;
        if doneIdx < numel(trace)
            if trace(doneIdx+1).burst_active
                ok = false;
                msg = 'burst_active did not clear after burst completion';
                return;
            end
        end
    end
end

function out = ternary(cond, trueVal, falseVal)
    if cond
        out = trueVal;
    else
        out = falseVal;
    end
end
