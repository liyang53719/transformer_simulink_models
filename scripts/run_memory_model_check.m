function metrics = run_memory_model_check(rootDir, options)
%RUN_MEMORY_MODEL_CHECK Collect DDR-related metrics from vector workload.
%
% The metrics are derived from real regression vectors and a deterministic
% AXI-like transfer model (burst + overhead). This provides concrete values
% for hard-gate checks before full SoC DDR instrumentation is wired in.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    tvDir = fullfile(rootDir, 'verification', 'testvectors');
    prefillPath = fullfile(tvDir, 'block_prefill_case01.mat');
    decodePath = fullfile(tvDir, 'block_decode_case01.mat');

    assert(exist(prefillPath, 'file') == 2, 'Missing testvector: %s', prefillPath);
    assert(exist(decodePath, 'file') == 2, 'Missing testvector: %s', decodePath);

    prefill = load(prefillPath);
    decode = load(decodePath);

    clockHz = double(getFieldOr(options, 'ClockHz', 1e9));
    busWidthBytes = double(getFieldOr(options, 'BusWidthBytes', 16));
    burstLenBeats = double(getFieldOr(options, 'BurstLenBeats', 128));
    burstOverheadCycles = double(getFieldOr(options, 'BurstOverheadCycles', 12));
    bytesPerValue = double(getFieldOr(options, 'BytesPerValue', 2));

    kvReadElems = kv_elem_count(decode.input.kv_cache);
    kvWriteElems = kv_elem_count(prefill.golden.output_kv) + kv_elem_count(decode.golden.output_kv);

    % Use hidden tensors as a stand-in for external weight/activation stream volume.
    hiddenElems = numel(prefill.input.hidden) + numel(decode.input.hidden);
    residualElems = numel(prefill.input.residual) + numel(decode.input.residual);

    bytesKVRead = kvReadElems * bytesPerValue;
    bytesKVWrite = kvWriteElems * bytesPerValue;
    bytesWeight = hiddenElems * bytesPerValue;
    bytesAct = residualElems * bytesPerValue;

    [cyclesKVRead, droppedKVRead] = estimateAxiCycles(bytesKVRead, busWidthBytes, burstLenBeats, burstOverheadCycles);
    [cyclesKVWrite, droppedKVWrite] = estimateAxiCycles(bytesKVWrite, busWidthBytes, burstLenBeats, burstOverheadCycles);
    [cyclesWeight, droppedWeight] = estimateAxiCycles(bytesWeight, busWidthBytes, burstLenBeats, burstOverheadCycles);

    totalCycles = max([cyclesKVRead, cyclesKVWrite, cyclesWeight, 1]);
    sec = totalCycles / clockHz;

    bw = struct();
    bw.kv_read = bytesKVRead / sec / 1e6;
    bw.kv_write = bytesKVWrite / sec / 1e6;
    bw.weight_stream = bytesWeight / sec / 1e6;
    bw.activation_stream = bytesAct / sec / 1e6;

    metrics = struct();
    metrics.available = true;
    metrics.clock_hz = clockHz;
    metrics.master_bw_mb_s = bw;
    metrics.total_cycles = totalCycles;
    metrics.stall_count = max(0, round((cyclesKVRead + cyclesKVWrite + cyclesWeight) - 3 * totalCycles));
    metrics.dropped_burst_count = droppedKVRead + droppedKVWrite + droppedWeight;
    metrics.reason = 'derived from testvector workload and AXI burst transfer model';
end

function n = kv_elem_count(kv)
    if isstruct(kv)
        n = 0;
        if isfield(kv, 'keys')
            n = n + numel(kv.keys);
        end
        if isfield(kv, 'values')
            n = n + numel(kv.values);
        end
    else
        n = numel(kv);
    end
end

function [cycles, dropped] = estimateAxiCycles(totalBytes, busWidthBytes, burstLenBeats, burstOverheadCycles)
    beats = ceil(totalBytes / busWidthBytes);
    if beats == 0
        cycles = 1;
        dropped = 0;
        return;
    end

    bursts = ceil(beats / burstLenBeats);
    cycles = beats + bursts * burstOverheadCycles;

    % No dropped burst in current deterministic model.
    dropped = 0;
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
