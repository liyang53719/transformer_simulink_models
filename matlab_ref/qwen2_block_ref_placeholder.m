function [outHidden, outKV] = qwen2_block_ref_placeholder(inHidden, inResidual, kvReadData, cfg)
%QWEN2_BLOCK_REF_PLACEHOLDER Deterministic placeholder block reference.
% This function is only for M1 scaffolding and regression plumbing.

    arguments
        inHidden single
        inResidual single
        kvReadData single = single([])
        cfg struct = struct()
    end

    alpha = getFieldOr(cfg, "ResidualScale", single(0.1));
    beta = getFieldOr(cfg, "KvMixScale", single(0.05));

    x = single(inHidden) + alpha .* single(inResidual);
    x = tanh(x);

    if ~isempty(kvReadData)
        kvMean = mean(single(kvReadData), 2);
        x = x + beta .* kvMean;
    end

    outHidden = single(x);
    outKV = struct();
    outKV.keys = single(x);
    outKV.values = single(0.5 .* x);
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
