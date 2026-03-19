function result = compute_stage2_weight_addr_range(contract)
%COMPUTE_STAGE2_WEIGHT_ADDR_RANGE Compute weight-request address envelope for the stage2 hardware contract.

    arguments
        contract struct
    end

    tokenPos = double(getFieldOr(contract, 'cfg_token_pos', 1));
    numHeads = double(getFieldOr(contract, 'cfg_weight_num_heads', 12));
    pageBase = double(getFieldOr(contract, 'cfg_weight_page_base', 64));
    pageStride = double(getFieldOr(contract, 'cfg_weight_page_stride', 8));
    laneOffsets = 0:9;

    baseAddr = pageBase + tokenPos * numHeads * pageStride;
    laneAddrs = baseAddr + laneOffsets;
    maxOffset = max(laneOffsets);
    denom = numHeads * pageStride;

    if denom <= 0
        maxSafeTokenPos = inf;
    else
        maxSafeTokenPos = floor((255 - pageBase - maxOffset) / denom);
    end

    result = struct();
    result.token_pos = tokenPos;
    result.num_heads = numHeads;
    result.page_base = pageBase;
    result.page_stride = pageStride;
    result.base_addr = baseAddr;
    result.lane_offsets = laneOffsets;
    result.lane_addrs = laneAddrs;
    result.min_addr = min(laneAddrs);
    result.max_addr = max(laneAddrs);
    result.max_safe_token_pos = maxSafeTokenPos;
    result.in_uint8_range = all(laneAddrs >= 0 & laneAddrs <= 255);
    result.will_wrap = ~result.in_uint8_range;
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end