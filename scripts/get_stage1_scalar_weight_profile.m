function laneProfile = get_stage1_scalar_weight_profile(laneName)
%GET_STAGE1_SCALAR_WEIGHT_PROFILE Shared scalar-weight reduction and decode settings.

    key = char(string(laneName));
    switch key
        case 'gamma'
            laneProfile = make_profile(key, 1, 'empirical_constant', 0.63058);
        case 'qkv_q'
            laneProfile = make_profile(key, 1, 'mean_colsum', 0.115924);
        case 'qkv_k'
            laneProfile = make_profile(key, 8, 'mean_colsum', 7.97476);
        case 'qkv_v'
            laneProfile = make_profile(key, 1, 'mean_colsum', 0.105031);
        case {'attn_q', 'attn_k', 'attn_v'}
            laneProfile = make_profile(key, 1, 'mean_colsum', NaN);
        case 'ffn_up'
            laneProfile = make_profile(key, 2, 'mean_colsum', 1.11327);
        case 'ffn_gate'
            laneProfile = make_profile(key, 32, 'mean_colsum', 18.4333);
        case 'ffn_down'
            laneProfile = make_profile(key, 16, 'mean_colsum', 9.20565);
        otherwise
            laneProfile = make_profile(key, 1, 'identity', NaN);
    end
end

function laneProfile = make_profile(laneName, decodeScale, reductionMode, constantValue)
    laneProfile = struct();
    laneProfile.lane_name = string(laneName);
    laneProfile.decode_scale = single(decodeScale);
    laneProfile.reduction_mode = string(reductionMode);
    laneProfile.constant_value = single(constantValue);
end