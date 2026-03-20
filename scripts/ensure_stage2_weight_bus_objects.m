function ensure_stage2_weight_bus_objects()
%ENSURE_STAGE2_WEIGHT_BUS_OBJECTS Define stage2 weight-path bus objects in base workspace.

    define_bus('WeightReqRmsBus', {'gamma_addr','gamma_valid'});
    define_bus('WeightReqQkvBus', {'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid'});
    define_bus('WeightReqAttnBus', {'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid'});
    define_bus('WeightReqFfnBus', {'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid','ffn_down_addr','ffn_down_valid'});

    define_bus_from_specs('WeightAddrRmsBus', {'gamma_addr', 'int32'});
    define_bus_from_specs('WeightAddrQkvBus', {
        'q_addr', 'int32';
        'k_addr', 'int32';
        'v_addr', 'int32'});
    define_bus_from_specs('WeightAddrAttnBus', {
        'attn_q_addr', 'int32';
        'attn_k_addr', 'int32';
        'attn_v_addr', 'int32'});
    define_bus_from_specs('WeightAddrFfnBus', {
        'up_addr', 'int32';
        'gate_addr', 'int32';
        'down_addr', 'int32'});

    define_bus('WeightReqBus', {
        'gamma_addr','gamma_valid', ...
        'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid', ...
        'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid', ...
        'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid','ffn_down_addr','ffn_down_valid'});

    define_bus_typed('WeightRspBus', {
        'gamma_data','gamma_valid', ...
        'qkv_q_data','qkv_q_valid','qkv_k_data','qkv_k_valid','qkv_v_data','qkv_v_valid', ...
        'attn_q_data','attn_q_valid','attn_k_data','attn_k_valid','attn_v_data','attn_v_valid', ...
        'ffn_up_data','ffn_up_valid','ffn_gate_data','ffn_gate_valid','ffn_down_data','ffn_down_valid'});

    define_bus('QkvStreamBus', {'q_stream','k_stream','v_stream','q_valid','kv_valid','group_idx'});
    define_bus('AttentionFlowBus', {'q_stream','k_cache','v_cache','group_idx','score_scale'});
    define_bus_from_specs('PrefillScheduleBus', {
        'array_rows', 'int32';
        'array_cols', 'int32';
        'tile_seq', 'int32';
        'tile_k', 'int32';
        'tile_out', 'int32';
        'x_bank_count', 'int32';
        'psum_bank_count', 'int32';
        'kv_bank_count', 'int32';
        'q_heads_per_kv', 'int32';
        'active_seq_len', 'int32';
        'kv_phase_first', 'int32';
        'score_scale', 'int32';
        'online_softmax_en', 'int32';
        'scorev_enable', 'int32'});
end

function define_bus(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        elems(i).DataType = 'single';
        elems(i).Dimensions = 1;
    end
    busObj = Simulink.Bus;
    busObj.Elements = elems;
    assignin('base', name, busObj);
end

function define_bus_from_specs(name, specs)
    elems = repmat(Simulink.BusElement, size(specs, 1), 1);
    for i = 1:size(specs, 1)
        elems(i).Name = specs{i, 1};
        elems(i).DataType = specs{i, 2};
        elems(i).Dimensions = 1;
    end
    busObj = Simulink.Bus;
    busObj.Elements = elems;
    assignin('base', name, busObj);
end

function define_bus_typed(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        if endsWith(fieldNames{i}, '_valid')
            elems(i).DataType = 'boolean';
        elseif endsWith(fieldNames{i}, '_data')
            elems(i).DataType = 'uint8';
        else
            elems(i).DataType = 'single';
        end
        elems(i).Dimensions = 1;
    end
    busObj = Simulink.Bus;
    busObj.Elements = elems;
    assignin('base', name, busObj);
end