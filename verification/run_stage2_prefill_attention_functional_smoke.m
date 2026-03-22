function result = run_stage2_prefill_attention_functional_smoke(rootDir, options)
%RUN_STAGE2_PREFILL_ATTENTION_FUNCTIONAL_SMOKE Interface-oriented smoke for KV banking and running-softmax flow.
%   This smoke validates deterministic prefill-path interface behavior in two scenarios:
%   1) KV banking outputs and running-softmax accumulation with all gates enabled.
%   2) Gating-off behavior for KV writes and score·V reduction.
%   It is intentionally a simplified hardware-flow smoke, not a canonical numerical model of attention.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    kvPath = [mdlName '/kv_cache_if_u'];
    attnPath = [mdlName '/attention_u'];

    requiredKvBlocks = {'bank_sum', 'seq_window_sum', 'bank_addr_base', 'bank_addr', ...
        'bank_sel', 'kv_write_gate', 'kv_seq_gate', 'kv_write_banked'};
    requiredAttnBlocks = {'head_group_stage_z', 'row_max_z', 'row_max', 'score_shift', ...
        'row_sum_z', 'row_sum_accum', 'score_norm', 'scorev_gate', 'value_weight', 'scorev_reduce'};

    missingBlocks = [missing_blocks(kvPath, requiredKvBlocks), missing_blocks(attnPath, requiredAttnBlocks)];
    if ~isempty(missingBlocks)
        close_system(mdlName, 0);
        error('run_stage2_prefill_attention_functional_smoke:MissingBlocks', ...
            'Prefill-attention flow missing blocks: %s', strjoin(missingBlocks, ', '));
    end

    seqA = make_sequence( ...
        [10 12 14 16], [3 3 3 3], [1 1 1 1], [1 1 1 1], [2 2 2 2], ...
        [4 4 4 4], [6 6 6 6], [8 8 8 8], [4 4 4 4], [3 3 3 3], [2 2 2 2], ...
        [2 2 2 2], [4 4 4 4], [32 32 32 32], [32 32 32 32], [1 1 1 1], [1 1 1 1]);
    [traceA, okA, msgA] = simulate_prefill_attention(seqA);

    seqB = make_sequence( ...
        [9 11 13 15], [5 5 5 5], [0 0 0 0], [0 0 0 0], [2 2 2 2], ...
        [4 4 4 4], [6 6 6 6], [8 8 8 8], [4 4 4 4], [3 3 3 3], [2 2 2 2], ...
        [2 2 2 2], [4 4 4 4], [32 32 32 32], [32 32 32 32], [0 0 0 0], [0 0 0 0]);
    [traceB, okB, msgB] = simulate_prefill_attention(seqB);

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
        fprintf('Stage2 prefill-attention functional smoke PASS\n');
        fprintf('  note: this smoke checks staged interface semantics only; real-model traces remain the numerical source of truth.\n');
    else
        fprintf('Stage2 prefill-attention functional smoke FAIL\n');
        if ~okA
            fprintf('  Scenario A: %s\n', msgA);
        end
        if ~okB
            fprintf('  Scenario B: %s\n', msgB);
        end
        error('run_stage2_prefill_attention_functional_smoke:Failed', ...
            'prefill-attention functional checks failed');
    end
end

function missing = missing_blocks(sysPath, blockNames)
    missing = {};
    for i = 1:numel(blockNames)
        if isempty(find_system(sysPath, 'SearchDepth', 1, 'Name', blockNames{i}))
            missing{end+1} = sprintf('%s/%s', sysPath, blockNames{i}); %#ok<AGROW>
        end
    end
end

function seq = make_sequence(qkvNew, kvHist, modeDecode, kvPhaseFirst, scoreScale, ...
        xBankCount, kvBankCount, tileSeq, activeSeqLen, tileK, tileOut, ...
        qHeadsPerKv, psumBankCount, arrayRows, arrayCols, onlineSoftmaxEn, scorevEnable)
    seq = struct();
    seq.qkv_new = qkvNew;
    seq.kv_hist = kvHist;
    seq.mode_decode = modeDecode;
    seq.kv_phase_first = kvPhaseFirst;
    seq.score_scale = scoreScale;
    seq.x_bank_count = xBankCount;
    seq.kv_bank_count = kvBankCount;
    seq.tile_seq = tileSeq;
    seq.active_seq_len = activeSeqLen;
    seq.tile_k = tileK;
    seq.tile_out = tileOut;
    seq.q_heads_per_kv = qHeadsPerKv;
    seq.psum_bank_count = psumBankCount;
    seq.array_rows = arrayRows;
    seq.array_cols = arrayCols;
    seq.online_softmax_en = onlineSoftmaxEn;
    seq.scorev_enable = scorevEnable;
end

function [trace, ok, msg] = simulate_prefill_attention(seq)
    n = numel(seq.qkv_new);
    headGroupStage = 0;
    rowMaxState = 0;
    rowSumState = 0;
    softmaxStage = 0;
    scorevStage = 0;

    trace = repmat(struct( ...
        'kv_to_attn', 0, 'kv_bank_addr', 0, 'kv_bank_sel', 0, 'kv_bank_wr_en', 0, ...
        'row_max', 0, 'row_sum_accum', 0, 'score_norm', 0, 'y_out', 0), n, 1);

    for t = 1:n
        qkvNew = double(seq.qkv_new(t));
        kvHist = double(seq.kv_hist(t));
        modeDecode = double(seq.mode_decode(t));
        kvPhaseFirst = double(seq.kv_phase_first(t));
        scoreScale = double(seq.score_scale(t));
        xBankCount = double(seq.x_bank_count(t));
        kvBankCount = double(seq.kv_bank_count(t));
        tileSeq = double(seq.tile_seq(t));
        activeSeqLen = double(seq.active_seq_len(t));
        tileK = double(seq.tile_k(t));
        tileOut = double(seq.tile_out(t));
        qHeadsPerKv = double(seq.q_heads_per_kv(t));
        psumBankCount = double(seq.psum_bank_count(t));
        arrayRows = double(seq.array_rows(t));
        arrayCols = double(seq.array_cols(t));
        onlineSoftmaxEn = double(seq.online_softmax_en(t));
        scorevEnable = double(seq.scorev_enable(t));

        qStream = 0.6 * qkvNew;
        kGain = 0.4 * qkvNew;
        vGain = 1.0 * qkvNew;
        kCache = switch_eval(modeDecode, kGain, kvHist);
        vCache = switch_eval(modeDecode, vGain, kvHist);

        bankSum = xBankCount + kvBankCount;
        seqWindowSum = tileSeq + activeSeqLen;
        bankAddrBase = seqWindowSum * tileK;
        bankAddr = bankAddrBase + tileOut;
        bankSel = bankSum * tileOut;
        kvWritePack = kCache + vCache;
        kvWriteBanked = kvWritePack + bankSum;
        kvWriteGate = modeDecode * kvPhaseFirst;
        kvSeqGate = kvWriteGate * seqWindowSum;
        kvToAttn = qStream + kCache + scoreScale + seqWindowSum;

        qOut = kvToAttn * 0.6;
        kOut = kvToAttn * 0.4;
        vOut = kvToAttn;
        scoreMul = headGroupStage * kOut;
        scoreAbs = abs(scoreMul);
        rowMax = max(scoreAbs, rowMaxState);
        scoreShift = scoreAbs - rowMax;
        arrayDimSum = arrayRows + arrayCols; %#ok<NASGU>
        headGroupBias = scoreScale + qHeadsPerKv;
        headGroupNorm = safe_divide(headGroupBias, activeSeqLen);
        scoreDen = scoreAbs + headGroupNorm;
        softmaxGate = scoreShift * onlineSoftmaxEn;
        softmaxOnlineDen = softmaxGate + scoreDen;
        rowSumAccum = rowSumState + softmaxOnlineDen;
        scoreNorm = safe_divide(scoreMul, rowSumAccum);
        scorevGate = softmaxStage * scorevEnable;
        valueWeight = scorevGate * vOut;
        scorevReduce = safe_divide(valueWeight, psumBankCount);

        trace(t).kv_to_attn = kvToAttn;
        trace(t).kv_bank_addr = bankAddr;
        trace(t).kv_bank_sel = bankSel;
        trace(t).kv_bank_wr_en = kvSeqGate;
        trace(t).row_max = rowMax;
        trace(t).row_sum_accum = rowSumAccum;
        trace(t).score_norm = scoreNorm;
        trace(t).y_out = scorevStage;

        headGroupStage = qOut;
        rowMaxState = rowMax;
        rowSumState = rowSumAccum;
        softmaxStage = scoreNorm;
        scorevStage = scorevReduce;

        kvWriteBanked = kvWriteBanked; %#ok<NASGU>
    end

    [ok, msg] = check_trace(seq, trace);
end

function [ok, msg] = check_trace(seq, trace)
    ok = true;
    msg = 'ok';

    expectedAddr = (double(seq.tile_seq) + double(seq.active_seq_len)) .* double(seq.tile_k) + double(seq.tile_out);
    expectedSel = (double(seq.x_bank_count) + double(seq.kv_bank_count)) .* double(seq.tile_out);
    expectedWrEn = double(seq.mode_decode) .* double(seq.kv_phase_first) .* ...
        (double(seq.tile_seq) + double(seq.active_seq_len));

    if any([trace.kv_bank_addr] ~= expectedAddr)
        ok = false;
        msg = 'kv_bank_addr does not match banked address formula';
        return;
    end

    if any([trace.kv_bank_sel] ~= expectedSel)
        ok = false;
        msg = 'kv_bank_sel does not match bank-select formula';
        return;
    end

    if any([trace.kv_bank_wr_en] ~= expectedWrEn)
        ok = false;
        msg = 'kv_bank_wr_en does not match gated write-enable formula';
        return;
    end

    rowMaxTrace = [trace.row_max];
    if any(diff(rowMaxTrace) < 0)
        ok = false;
        msg = 'row_max should be monotonic non-decreasing';
        return;
    end

    rowSumTrace = [trace.row_sum_accum];
    if any(diff(rowSumTrace) <= 0)
        ok = false;
        msg = 'row_sum_accum should increase each cycle for positive inputs';
        return;
    end

    if all(double(seq.scorev_enable) > 0.5)
        if trace(end).y_out <= 0
            ok = false;
            msg = 'score·V pipeline did not produce a positive delayed output';
            return;
        end
    else
        if any([trace.y_out] ~= 0)
            ok = false;
            msg = 'score·V output should remain zero when scorev_enable is disabled';
            return;
        end
    end

    if all(double(seq.mode_decode) < 0.5)
        expectedKvToAttn = 0.6 .* double(seq.qkv_new) + double(seq.kv_hist) + double(seq.score_scale) + ...
            (double(seq.tile_seq) + double(seq.active_seq_len));
        if any(abs([trace.kv_to_attn] - expectedKvToAttn) > 1e-9)
            ok = false;
            msg = 'kv_to_attn should use historical KV data when mode_decode is low';
            return;
        end
    end
end

function out = switch_eval(ctrl, topInput, bottomInput)
    if ctrl >= 0.5
        out = topInput;
    else
        out = bottomInput;
    end
end

function out = safe_divide(num, den)
    if abs(den) < 1e-12
        out = 0;
    else
        out = num / den;
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end