function [outHidden, outKV] = qwen2_block_ref_real_adapter(inHidden, inResidual, kvReadData, ctx)
%QWEN2_BLOCK_REF_REAL_ADAPTER Adapter to existing +qwen2 / +qwen2_quant block API.
% This adapter intentionally keeps IO compatible with scaffold regression.

    arguments
        inHidden single
        inResidual single
        kvReadData single
        ctx struct
    end

    if ~isfield(ctx, 'Parameters')
        error('qwen2_block_ref_real_adapter:MissingParameters', 'ctx.Parameters is required.');
    end

    params = ctx.Parameters;
    hp = params.Hyperparameters;
    weights = params.Weights;

    layerIndex = getFieldOr(ctx, 'LayerIndex', 1);
    layerName = sprintf('h%d', layerIndex - 1);
    if ~isfield(weights, layerName)
        error('qwen2_block_ref_real_adapter:MissingLayer', 'Missing layer weights: %s', layerName);
    end
    layerWeights = weights.(layerName);

    x = single(inHidden) + single(inResidual);
    [hiddenSize, seqLen] = size(x);
    h = reshape(x, hiddenSize, seqLen, 1);

    past = buildPastFromKv(kvReadData, hp);
    freqs_cis = buildFreqsCis(hp, seqLen, sizePastLen(past), ctx);

    useQuantBlock = is_quantized_layer(layerWeights);

    if useQuantBlock && ~isempty(which('qwen2_quant.layer.block'))
        cfg = mergeStruct(defaultRuntimeCfg(), getFieldOr(ctx, 'RuntimeConfig', struct()));
        [hOut, present] = qwen2_quant.layer.block(h, past, layerWeights, hp, freqs_cis, cfg);
    elseif ~isempty(which('qwen2.layer.block'))
        [hOut, present] = qwen2.layer.block(h, past, layerWeights, hp, freqs_cis);
    elseif ~isempty(which('qwen2_quant.layer.block'))
        cfg = mergeStruct(defaultRuntimeCfg(), getFieldOr(ctx, 'RuntimeConfig', struct()));
        [hOut, present] = qwen2_quant.layer.block(h, past, layerWeights, hp, freqs_cis, cfg);
    else
        error('qwen2_block_ref_real_adapter:MissingDependency', ...
            'Neither qwen2.layer.block nor qwen2_quant.layer.block is available on MATLAB path.');
    end

    outHidden = reshape(single(hOut), hiddenSize, seqLen);
    outKV = struct();
    if isstruct(present) && isfield(present, 'keys') && isfield(present, 'values')
        outKV.keys = single(present.keys);
        outKV.values = single(present.values);
    else
        outKV.keys = single([]);
        outKV.values = single([]);
    end
end

function past = buildPastFromKv(kvReadData, hp)
    past = struct('keys', [], 'values', []);
    if isempty(kvReadData)
        return;
    end

    headDim = double(hp.HeadDim);
    numKVHeads = double(hp.NumKVHeads);
    total = headDim * numKVHeads;

    if mod(size(kvReadData, 1), total) ~= 0
        return;
    end

    cacheLen = size(kvReadData, 2);
    keys = reshape(single(kvReadData(1:total, :)), headDim, numKVHeads, cacheLen, 1);
    values = keys;

    past.keys = keys;
    past.values = values;
end

function pastLen = sizePastLen(past)
    pastLen = 0;
    if isstruct(past) && isfield(past, 'keys') && ~isempty(past.keys)
        dims = size(past.keys);
        if numel(dims) >= 3
            pastLen = dims(3);
        end
    end
end

function freqs_cis = buildFreqsCis(hp, seqLen, pastLen, ctx)
    ropeTheta = getFieldOr(hp, 'RopeTheta', 1000000.0);
    headDim = double(hp.HeadDim);

    startPos = pastLen + 1;
    maxSeq = startPos + seqLen + 16;

    if isfield(ctx, 'TokenPos')
        startPos = double(ctx.TokenPos);
        maxSeq = max(maxSeq, startPos + seqLen + 16);
    end

    if ~isempty(which('transformer.layer.precomputeFreqsCis'))
        freqs = transformer.layer.precomputeFreqsCis(headDim, maxSeq, ropeTheta);
        freqs = complex(single(real(freqs)), single(imag(freqs)));
    else
        freqs = complex(single(ones(headDim/2, maxSeq)), single(zeros(headDim/2, maxSeq)));
    end

    freqs_cis = freqs(:, startPos:startPos+seqLen-1);
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function tf = is_quantized_layer(layerWeights)
    tf = false;
    candidates = {'self_attn_q_proj', 'self_attn_k_proj', 'self_attn_v_proj', 'mlp_gate_proj'};
    for i = 1:numel(candidates)
        fn = candidates{i};
        if isfield(layerWeights, fn)
            w = layerWeights.(fn);
            cls = string(class(w));
            if contains(lower(cls), 'quantized_weight')
                tf = true;
                return;
            end
            if isstruct(w) && isfield(w, 'QuantType')
                tf = true;
                return;
            end
        end
    end
end

function cfg = defaultRuntimeCfg()
    cfg = struct();
    cfg.LinearMode = 'gptq_int4_matlab_sim';
    cfg.TracePrecision = false;
    cfg.TraceTensors = false;
    cfg.Int8WeightScaleMode = 'per_row';
    cfg.Int8ActivationScaleMode = 'per_col';
    cfg.EnablePackedFullChain = false;
end

function merged = mergeStruct(base, override)
    merged = base;
    if ~isstruct(override)
        return;
    end
    f = fieldnames(override);
    for i = 1:numel(f)
        merged.(f{i}) = override.(f{i});
    end
end
