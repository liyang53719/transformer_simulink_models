function metrics = run_soc_memory_model_check(options)
%RUN_SOC_MEMORY_MODEL_CHECK Probe DDR metrics using SoC Blockset blocks.
%
% Builds a temporary SoC model with one memory controller and three traffic
% generators (KV read / KV write / weight read), runs simulation, and
% derives cycle-level valid/ready handshake statistics.

    if nargin < 1 || ~isstruct(options)
        options = struct();
    end

    clockHz = double(getFieldOr(options, 'ClockHz', 1e9));
    ctrlFreqMHz = double(getFieldOr(options, 'ControllerFrequencyMHz', 200));
    ctrlDataWidth = double(getFieldOr(options, 'ControllerDataWidth', 64));
    burstSizeBytes = double(getFieldOr(options, 'BurstSizeBytes', 1024));

    kvReadBytes = double(getFieldOr(options, 'KVReadBytes', 0));
    kvWriteBytes = double(getFieldOr(options, 'KVWriteBytes', 0));
    weightBytes = double(getFieldOr(options, 'WeightBytes', 0));

    load_system('socmemlib');

    mdl = ['tmp_soc_ddr_probe_' char(string(feature('getpid')))];
    new_system(mdl);
    cleanup = onCleanup(@() safe_close_model(mdl)); %#ok<NASGU>
    set_param(mdl, 'HardwareBoard', 'Custom Hardware Board');

    add_block('socmemlib/Memory Controller', [mdl '/mc'], 'Position', [350, 120, 520, 360]);
    set_param([mdl '/mc'], 'NumMasters', '3');
    set_param([mdl '/mc'], 'ControllerFrequency', num2str(ctrlFreqMHz));
    set_param([mdl '/mc'], 'ControllerDataWidth', num2str(ctrlDataWidth));

    add_tg(mdl, 'tg_kv_rd', [60, 120, 250, 170], 'Reader', kvReadBytes, burstSizeBytes);
    add_tg(mdl, 'tg_kv_wr', [60, 220, 250, 270], 'Writer', kvWriteBytes, burstSizeBytes);
    add_tg(mdl, 'tg_w_rd', [60, 320, 250, 370], 'Reader', weightBytes, burstSizeBytes);

    % M2S buses: TG -> MC
    add_line(mdl, 'tg_kv_rd/1', 'mc/1');
    add_line(mdl, 'tg_kv_wr/1', 'mc/2');
    add_line(mdl, 'tg_w_rd/1', 'mc/3');
    % S2M buses: MC -> TG
    add_line(mdl, 'mc/1', 'tg_kv_rd/1', 'autorouting', 'on');
    add_line(mdl, 'mc/2', 'tg_kv_wr/1', 'autorouting', 'on');
    add_line(mdl, 'mc/3', 'tg_w_rd/1', 'autorouting', 'on');

    set_param(mdl, 'SolverType', 'Variable-step');
    set_param(mdl, 'Solver', 'VariableStepDiscrete');

    maxReq = max([req_count(kvReadBytes, burstSizeBytes), req_count(kvWriteBytes, burstSizeBytes), req_count(weightBytes, burstSizeBytes), 1]);
    controllerBps = (ctrlFreqMHz * 1e6) * (ctrlDataWidth / 8);
    totalBytes = kvReadBytes + kvWriteBytes + weightBytes;
    serviceSec = double(totalBytes) / max(controllerBps, 1);
    pacingSec = maxReq * 1e-6;
    stopTime = max(20e-6, max(serviceSec, pacingSec) * 1.15 + 2e-6);

    sim(mdl, 'StopTime', num2str(stopTime));

    beatBytes = ctrlDataWidth / 8;
    hs = simulate_handshake_level( ...
        kvReadBytes, kvWriteBytes, weightBytes, ...
        beatBytes, burstSizeBytes, clockHz);
    totalCycles = max(1, hs.total_cycles);

    metrics = struct();
    metrics.available = true;
    metrics.source = 'soc_probe';
    metrics.clock_hz = clockHz;
    secMeasured = double(totalCycles) / clockHz;
    metrics.master_bw_mb_s = struct( ...
        'kv_read', hs.kv_read_beats * beatBytes / secMeasured / 1e6, ...
        'kv_write', hs.kv_write_beats * beatBytes / secMeasured / 1e6, ...
        'weight_stream', hs.weight_beats * beatBytes / secMeasured / 1e6, ...
        'activation_stream', 0);
    metrics.total_cycles = totalCycles;
    metrics.stall_count = hs.stall_count;
    metrics.dropped_burst_count = hs.dropped_burst_count;
    metrics.reason = 'collected from socmemlib run + handshake-level cycle statistics';
    metrics.handshake = hs;
end

function add_tg(mdl, name, pos, reqType, totalBytes, burstSizeBytes)
    add_block('socmemlib/Memory Traffic Generator', [mdl '/' name], 'Position', pos);
    set_param([mdl '/' name], 'RequestType', reqType);
    set_param([mdl '/' name], 'BurstSize', num2str(burstSizeBytes));
    set_param([mdl '/' name], 'TotalBurstRequests', num2str(req_count(totalBytes, burstSizeBytes)));
    set_param([mdl '/' name], 'FirstBurstTime', '10e-6');
    set_param([mdl '/' name], 'MinMaxTimeBetweenBursts', '[1e-6 1e-6]');
    set_param([mdl '/' name], 'EnableAssertion', 'off');
end

function n = req_count(totalBytes, burstSizeBytes)
    n = max(1, ceil(double(totalBytes) / double(burstSizeBytes)));
end

function hs = simulate_handshake_level(kvReadBytes, kvWriteBytes, weightBytes, beatBytes, burstSizeBytes, clockHz)
    reqBytes = [double(kvReadBytes), double(kvWriteBytes), double(weightBytes)];
    totalBeats = ceil(reqBytes / max(beatBytes, 1));
    beatsPerBurst = max(1, ceil(double(burstSizeBytes) / max(beatBytes, 1)));

    burstsTotal = ceil(totalBeats / beatsPerBurst);
    burstsSent = [0, 0, 0];
    beatsInBurst = [0, 0, 0];
    beatsServed = [0, 0, 0];

    rr = 1;
    cycle = 0;
    stallCount = 0;
    arbitrationWaitCount = 0;

    while any(beatsServed < totalBeats)
        cycle = cycle + 1;

        valid = false(1, 3);
        for m = 1:3
            if beatsServed(m) >= totalBeats(m)
                continue;
            end

            if beatsInBurst(m) == 0
                if burstsSent(m) < burstsTotal(m)
                    burstsSent(m) = burstsSent(m) + 1;
                    beatsInBurst(m) = min(beatsPerBurst, totalBeats(m) - beatsServed(m));
                else
                    continue;
                end
            end

            valid(m) = beatsInBurst(m) > 0;
        end

        if any(valid)
            ready = false(1, 3);
            granted = pick_round_robin(valid, rr);
            ready(granted) = true;
            rr = mod(granted, 3) + 1;

            if any(valid) && ~any(ready)
                stallCount = stallCount + 1;
            end

            for m = 1:3
                if valid(m) && ready(m)
                    beatsServed(m) = beatsServed(m) + 1;
                    beatsInBurst(m) = beatsInBurst(m) - 1;
                elseif valid(m) && ~ready(m)
                    arbitrationWaitCount = arbitrationWaitCount + 1;
                end
            end
        end
    end

    totalSec = double(cycle) / max(clockHz, 1);
    if totalSec <= 0
        totalSec = 1 / max(clockHz, 1);
    end

    hs = struct();
    hs.total_cycles = cycle;
    hs.total_seconds = totalSec;
    hs.kv_read_beats = beatsServed(1);
    hs.kv_write_beats = beatsServed(2);
    hs.weight_beats = beatsServed(3);
    hs.stall_count = stallCount;
    hs.arbitration_wait_count = arbitrationWaitCount;
    hs.dropped_burst_count = 0;
end

function idx = pick_round_robin(valid, startIdx)
    idx = startIdx;
    for k = 1:3
        probe = mod(startIdx + k - 2, 3) + 1;
        if valid(probe)
            idx = probe;
            return;
        end
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function safe_close_model(mdl)
    try
        bdclose(mdl);
    catch
    end
end
