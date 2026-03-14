function [refFn, refInfo] = get_block_reference_fn(referenceMode, referenceContext)
%GET_BLOCK_REFERENCE_FN Resolve block reference function for regression.

    if nargin < 1 || strlength(string(referenceMode)) == 0
        referenceMode = "placeholder";
    end
    if nargin < 2 || ~isstruct(referenceContext)
        referenceContext = struct();
    end

    mode = lower(string(referenceMode));

    switch mode
        case "placeholder"
            refFn = @qwen2_block_ref_placeholder;
            refInfo = struct('mode', 'placeholder', 'reason', 'forced placeholder mode');

        case "real_auto"
            hasQwen2 = ~isempty(which('qwen2.layer.block'));
            hasQwen2Quant = ~isempty(which('qwen2_quant.layer.block'));

            if (hasQwen2 || hasQwen2Quant) && isfield(referenceContext, 'Parameters')
                refFn = @qwen2_block_ref_real_adapter;
                refInfo = struct('mode', 'real', 'reason', 'found package block function and parameters');
            else
                refFn = @qwen2_block_ref_placeholder;
                refInfo = struct('mode', 'placeholder', 'reason', ...
                    'missing qwen2/qwen2_quant path or Parameters; fallback to placeholder');
            end

        otherwise
            error('get_block_reference_fn:InvalidMode', ...
                'Unsupported reference mode: %s', string(referenceMode));
    end
end
