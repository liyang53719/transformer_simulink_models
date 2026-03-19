function [summary, detail] = qwen2_block_ref_stage2_contract_adapter(contract, ctx)
%QWEN2_BLOCK_REF_STAGE2_CONTRACT_ADAPTER Adapt the stage2 hardware contract onto the matrix reference API.
%   Scalar stage2 activations are broadcast across hidden lanes. The scalar
%   output contract is derived from the mean of the last reference output
%   column. This is an adapter contract, not a golden numeric equivalence.

    arguments
        contract struct
        ctx struct
    end

    if ~isfield(ctx, 'Parameters')
        error('qwen2_block_ref_stage2_contract_adapter:MissingParameters', ...
            'ctx.Parameters is required.');
    end

    hp = ctx.Parameters.Hyperparameters;
    hiddenSize = double(hp.HiddenSize);
    seqLen = max(1, round(double(getFieldOr(contract, 'cfg_seq_len', 1))));
    tokenPos = max(1, round(double(getFieldOr(contract, 'cfg_token_pos', 1))));
    kvValid = logical(getFieldOr(contract, 'kv_cache_rd_valid', false));
    modeDecode = logical(getFieldOr(contract, 'mode_decode', true));
    kvLen = derive_kv_len(seqLen, tokenPos, kvValid, modeDecode, ctx);

    inHiddenValue = single(getFieldOr(contract, 'in_hidden', 0));
    inResidualValue = single(getFieldOr(contract, 'in_residual', 0));
    kvReadValue = single(getFieldOr(contract, 'kv_cache_rd_data', 0));

    inHidden = repmat(inHiddenValue, hiddenSize, seqLen);
    inResidual = repmat(inResidualValue, hiddenSize, seqLen);
    if kvValid
        kvReadData = repmat(kvReadValue, hiddenSize, kvLen);
    else
        kvReadData = single([]);
    end

    refCtx = ctx;
    refCtx.TokenPos = tokenPos;
    [outHiddenMatrix, outKV] = qwen2_block_ref_real_adapter(inHidden, inResidual, kvReadData, refCtx);

    lastColumn = single(outHiddenMatrix(:, end));
    summary = struct();
    summary.out_hidden = single(mean(lastColumn, 'all'));
    summary.out_hidden_abs_mean = single(mean(abs(lastColumn), 'all'));
    summary.out_hidden_nonzero = any(abs(lastColumn) > 0);
    summary.out_hidden_finite = all(isfinite(lastColumn), 'all');
    summary.kv_present = isstruct(outKV) && isfield(outKV, 'keys') && ~isempty(outKV.keys);
    [kvWriteDataSummary, kvWriteAbsMean] = summarize_kv_write(outKV);
    summary.kv_write_en = summary.kv_present;
    summary.kv_write_data = kvWriteDataSummary;
    summary.kv_write_abs_mean = kvWriteAbsMean;
    summary.kv_write_finite = isfinite(kvWriteDataSummary) && isfinite(kvWriteAbsMean);
    summary.kv_len = kvLen;
    summary.seq_len = seqLen;
    summary.token_pos = tokenPos;

    detail = struct();
    detail.out_hidden_matrix = outHiddenMatrix;
    detail.out_kv = outKV;
    detail.broadcast_policy = 'scalar_to_hidden_lane_broadcast';
    detail.output_reduction = 'mean_last_output_column';
    detail.kv_policy = 'scalar_kv_value_broadcast_to_hidden_rows';
end

function [dataSummary, absMean] = summarize_kv_write(outKV)
    dataSummary = single(0);
    absMean = single(0);

    if ~isstruct(outKV)
        return;
    end

    valueTensor = single([]);
    if isfield(outKV, 'values') && ~isempty(outKV.values)
        valueTensor = single(outKV.values);
    elseif isfield(outKV, 'keys') && ~isempty(outKV.keys)
        valueTensor = single(outKV.keys);
    end

    if isempty(valueTensor)
        return;
    end

    lastSlice = extract_last_slice(valueTensor);
    dataSummary = single(mean(lastSlice, 'all'));
    absMean = single(mean(abs(lastSlice), 'all'));
end

function slice = extract_last_slice(valueTensor)
    dims = size(valueTensor);
    if numel(dims) >= 3
        idx = repmat({':'}, 1, numel(dims));
        idx{3} = dims(3);
        slice = single(valueTensor(idx{:}));
    else
        slice = single(valueTensor);
    end
end

function kvLen = derive_kv_len(seqLen, tokenPos, kvValid, modeDecode, ctx)
    if ~kvValid
        kvLen = 0;
        return;
    end

    if isfield(ctx, 'KvLen')
        kvLen = max(1, round(double(ctx.KvLen)));
        return;
    end

    if modeDecode
        kvLen = max(1, tokenPos);
    else
        kvLen = max(1, seqLen);
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end