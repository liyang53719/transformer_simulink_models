function retarget_weight_ref_to_sample_tables(tbName, weightRspCfg)
%RETARGET_WEIGHT_REF_TO_SAMPLE_TABLES Replace synthetic weight_ref_u lanes with request-driven table lookups.

    sampleTables = getFieldOr(weightRspCfg, 'sample_tables', {});
    if ~(iscell(sampleTables) && ~isempty(sampleTables))
        return;
    end

    subPath = [char(tbName) '/weight_ref_u'];
    if getSimulinkBlockHandle(subPath) == -1
        error('retarget_weight_ref_to_sample_tables:MissingWeightRef', ...
            'weight_ref_u subsystem not found under %s', char(tbName));
    end

    fprintf('weight_ref_u responder mode: request_driven_sample_tables (%d lanes)\n', numel(sampleTables));
    for laneIndex = 1:min(10, numel(sampleTables))
        rewire_lane(subPath, laneIndex, sampleTables{laneIndex});
    end
end

function rewire_lane(subPath, laneIndex, tableData)
    lookupName = ['table_lookup_' num2str(laneIndex)];
    lookupPath = [subPath '/' lookupName];
    baseY = 25 + 30 * (laneIndex - 1);

    delete_block_if_exists([subPath '/tag_idx_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/tag_stride_mul_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/tag_sum_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/lane_offset_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/tag_lane_sum_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/data_page_tag_' num2str(laneIndex)]);
    delete_block_if_exists([subPath '/sample_value_' num2str(laneIndex)]);

    if getSimulinkBlockHandle(lookupPath) == -1
        add_block('simulink/User-Defined Functions/MATLAB Function', lookupPath, ...
            'Position', [430, baseY - 2, 560, baseY + 30]);
    end
    configure_lookup_chart(lookupPath, tableData);

    force_connect_line(subPath, ['addr_d2_' num2str(laneIndex) '/1'], ['table_lookup_' num2str(laneIndex) '/1']);
    force_connect_line(subPath, ['table_lookup_' num2str(laneIndex) '/1'], ['data_u8_' num2str(laneIndex) '/1']);
end

function configure_lookup_chart(blockPath, tableData)
    tableVec = uint8(double(tableData(:)'));
    tableLiteral = strtrim(sprintf('%d ', double(tableVec)));
    code = sprintf([ ...
        'function data_out = table_lookup(addr_in)\n' ...
        '%%#codegen\n' ...
        'table = uint8([%s]);\n' ...
        'idx = floor(double(addr_in)) + 1;\n' ...
        'if idx < 1\n' ...
        '    idx = 1;\n' ...
        'elseif idx > numel(table)\n' ...
        '    idx = numel(table);\n' ...
        'end\n' ...
        'data_out = table(idx);\n' ...
        'end\n'], tableLiteral);

    rt = sfroot;
    chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', blockPath);
    if isempty(chart)
        error('retarget_weight_ref_to_sample_tables:MissingEMChart', ...
            'Expected MATLAB Function chart at %s', blockPath);
    end
    chart.Script = code;
end

function delete_block_if_exists(blockPath)
    if getSimulinkBlockHandle(blockPath) ~= -1
        delete_block(blockPath);
    end
end

function safe_delete_line(sys, src, dst)
    try
        delete_line(sys, src, dst);
    catch
    end
end

function force_connect_line(sys, src, dst)
    dstParts = split(string(dst), '/');
    dstBlk = [sys '/' char(dstParts(1))];
    dstPort = str2double(dstParts(2));
    phDst = get_param(dstBlk, 'PortHandles');
    if dstPort <= numel(phDst.Inport)
        ln = get_param(phDst.Inport(dstPort), 'Line');
        if ln ~= -1
            try
                delete_line(ln);
            catch
            end
        end
    end
    add_line(sys, src, dst, 'autorouting', 'on');
end

function out = getFieldOr(s, name, defaultValue)
    if isstruct(s) && isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
