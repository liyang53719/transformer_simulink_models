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
    runtimeCfg = getFieldOr(refCtx, 'RuntimeConfig', struct());
    if ~isfield(runtimeCfg, 'TraceTensors')
        runtimeCfg.TraceTensors = false;
    end
    refCtx.RuntimeConfig = runtimeCfg;
    [outHiddenMatrix, outKV, debugInfo] = qwen2_block_ref_real_adapter(inHidden, inResidual, kvReadData, refCtx);

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
    if isstruct(debugInfo)
        detail.debug_info = debugInfo;
    else
        detail.debug_info = struct();
    end
    detail.stage_contract_trace = reduce_stage_contract_trace(getFieldOr(detail.debug_info, 'TensorTrace', struct([])));
    detail.broadcast_policy = 'scalar_to_hidden_lane_broadcast';
    detail.output_reduction = 'mean_last_output_column';
    detail.kv_policy = 'scalar_kv_value_broadcast_to_hidden_rows';
end

function stageTrace = reduce_stage_contract_trace(tensorTrace)
    stageTrace = struct();
    if ~isstruct(tensorTrace) || isempty(tensorTrace)
        return;
    end

    for i = 1:numel(tensorTrace)
        opName = string(getFieldOr(tensorTrace(i), 'Op', ''));
        value = getFieldOr(tensorTrace(i), 'Value', []);
        if strlength(opName) == 0 || isempty(value)
            continue;
        end

        stageName = map_trace_op_to_stage(opName);
        if strlength(stageName) == 0
            continue;
        end

        stageTrace.(stageName) = reduce_trace_tensor_to_contract(value);
    end

    if isfield(stageTrace, 'q_proj_out') && isfield(stageTrace, 'k_proj_out') && isfield(stageTrace, 'v_proj_out')
        stageTrace.qkv_out = stageTrace.q_proj_out + stageTrace.k_proj_out + stageTrace.v_proj_out;
    end
    if isfield(stageTrace, 'o_proj_out')
        stageTrace.attn_out = stageTrace.o_proj_out;
        stageTrace.attn_scorev_reduce = stageTrace.o_proj_out;
    end
    if isfield(stageTrace, 'swiglu_out')
        stageTrace.ffn_swiglu_mul = stageTrace.swiglu_out;
    end
    if isfield(stageTrace, 'down_proj_out')
        stageTrace.ffn_down_stage = stageTrace.down_proj_out;
    end
end

function stageName = map_trace_op_to_stage(opName)
    switch char(opName)
        case 'block.input_norm.output'
            stageName = "rms_out";
        case 'attn.q_proj.output'
            stageName = "q_proj_out";
        case 'attn.k_proj.output'
            stageName = "k_proj_out";
        case 'attn.v_proj.output'
            stageName = "v_proj_out";
        case 'attn.o_proj.output'
            stageName = "o_proj_out";
        case 'block.attn_residual.output'
            stageName = "attn_residual_out";
        case 'block.post_attn_norm.output'
            stageName = "post_attn_norm_out";
        case 'mlp.up_proj.output'
            stageName = "ffn_up_mul";
        case 'mlp.gate_proj.output'
            stageName = "ffn_gate_mul";
        case 'mlp.swiglu.output'
            stageName = "swiglu_out";
        case 'mlp.down_proj.output'
            stageName = "down_proj_out";
        case 'block.ffn_residual.output'
            stageName = "residual_out";
        otherwise
            stageName = "";
    end
end

function scalarValue = reduce_trace_tensor_to_contract(value)
    data = single(value);
    dims = size(data);

    if isempty(data)
        scalarValue = single(0);
        return;
    end

    if numel(dims) >= 2
        idx = repmat({':'}, 1, numel(dims));
        idx{2} = dims(2);
        lastColumn = single(data(idx{:}));
    else
        lastColumn = data;
    end

    scalarValue = single(mean(lastColumn, 'all'));
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