function result = run_stage2_wrapper_tb_smoke(rootDir, options)
%RUN_STAGE2_WRAPPER_TB_SMOKE Build and simulate a dedicated wrapper TB for qwen2_block_top.
%   The wrapper follows the separated read/write DDR handshake style used by
%   the SoC Toolbox reference flow around soc_image_rotation_fpga: the DUT is
%   wrapped by typed stimulus sources and a lightweight DDR responder that
%   accepts kv_mem_* requests and returns kv_cache_rd_* data with fixed latency.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'verification'));
    modelInfo = build_stage2_wrapper_tb_model(rootDir, struct('BuildModel', buildModel, 'KvAddressConfig', kvCfg));
    tbName = char(modelInfo.tbName);
    mdlName = char(modelInfo.dutName);
    cleanup = onCleanup(@()safe_close_models(tbName, mdlName)); %#ok<NASGU>

    try
        simOut = sim(tbName, 'StopTime', '4', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
            'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
        yout = simOut.get('yout');

        kvRdValid = extract_signal(yout, 'kv_mem_rd_valid');
        kvWrValid = extract_signal(yout, 'kv_mem_wr_valid');
        kvRdAddr = extract_signal(yout, 'kv_mem_rd_addr');
        kvWrAddr = extract_signal(yout, 'kv_mem_wr_addr');
        kvWrData = extract_signal(yout, 'kv_cache_wr_data');
        kvWrEn = extract_signal(yout, 'kv_cache_wr_en');
        rdRspValid = extract_signal(yout, 'tb_rd_rsp_valid');
        rdRspData = extract_signal(yout, 'tb_rd_rsp_data');
        doneSig = extract_signal(yout, 'done');

        result = struct();
        result.tb_ready = true;
        result.kv_rd_valid_seen = any(kvRdValid > 0.5);
        result.kv_wr_valid_seen = any(kvWrValid > 0.5);
        result.kv_rd_addr_nonzero = any(kvRdAddr > 0);
        result.kv_wr_addr_nonzero = any(kvWrAddr > 0);
        result.kv_wr_en_seen = any(kvWrEn > 0.5);
        result.rd_rsp_valid_seen = any(rdRspValid > 0.5);
        result.rd_rsp_data_nonzero = any(abs(rdRspData) > 0);
        result.done_seen = any(doneSig > 0.5);
        result.kv_wr_data_nonzero = any(abs(kvWrData) > 0);
        result.pass = result.kv_rd_valid_seen && result.kv_wr_valid_seen && ...
            result.kv_wr_addr_nonzero && ...
            result.kv_wr_en_seen && result.rd_rsp_valid_seen && ...
            result.rd_rsp_data_nonzero && result.kv_wr_data_nonzero;

        if result.pass
            fprintf('Stage2 wrapper TB smoke PASS\n');
        else
            fprintf('Stage2 wrapper TB smoke FAIL\n');
            fprintf('  kv_rd_valid_seen=%d kv_wr_valid_seen=%d rd_rsp_valid_seen=%d done_seen=%d\n', ...
                result.kv_rd_valid_seen, result.kv_wr_valid_seen, result.rd_rsp_valid_seen, result.done_seen);
            fprintf('  kv_rd_addr_nonzero=%d kv_wr_addr_nonzero=%d kv_wr_en_seen=%d kv_wr_data_nonzero=%d rd_rsp_data_nonzero=%d\n', ...
                result.kv_rd_addr_nonzero, result.kv_wr_addr_nonzero, result.kv_wr_en_seen, ...
                result.kv_wr_data_nonzero, result.rd_rsp_data_nonzero);
        end
    catch ME
        if is_wrapper_tb_blocker(ME)
            result = struct();
            result.tb_ready = false;
            result.pass = false;
            result.reason = classify_wrapper_tb_blocker(ME);
            fprintf('Stage2 wrapper TB smoke BLOCKED\n');
            fprintf('  Reason: %s\n', result.reason);
            return;
        end
        rethrow(ME);
    end
end

function build_typed_sources(tbName, mdlName, inports)
    for i = 1:numel(inports)
        name = char(inports(i).name);
        blk = [tbName '/src_' name];
        add_block('simulink/Sources/Constant', blk, 'Position', [40, 40 + 35 * i, 100, 60 + 35 * i]);

        compiledType = get_inport_compiled_type(mdlName, inports(i).name);
        set_param(blk, 'OutDataTypeStr', compiled_type_to_param_string(compiledType));
        set_param(blk, 'Value', default_constant_for_name(name));
    end
end

function connect_dut_inputs(tbName, ~, inports)
    ddrInMap = containers.Map( ...
        {'kv_cache_rd_data', 'kv_cache_rd_valid', 'kv_mem_rd_ready', 'kv_mem_wr_ready'}, ...
        {'ddr_ref_u/3', 'ddr_ref_u/4', 'ddr_ref_u/1', 'ddr_ref_u/2'});

    for i = 1:numel(inports)
        name = char(inports(i).name);
        if isKey(ddrInMap, name)
            add_line(tbName, ddrInMap(name), ['dut/' num2str(inports(i).port)], 'autorouting', 'on');
        else
            add_line(tbName, ['src_' name '/1'], ['dut/' num2str(inports(i).port)], 'autorouting', 'on');
        end
    end
end

function connect_dut_outputs(tbName, outports)
    ddrSinkMap = containers.Map( ...
        {'kv_mem_rd_addr', 'kv_mem_rd_len', 'kv_mem_rd_valid', 'kv_mem_wr_addr', 'kv_mem_wr_len', 'kv_mem_wr_valid', 'kv_cache_wr_data', 'kv_cache_wr_en'}, ...
        {'ddr_ref_u/1', 'ddr_ref_u/2', 'ddr_ref_u/3', 'ddr_ref_u/4', 'ddr_ref_u/5', 'ddr_ref_u/6', 'ddr_ref_u/7', 'ddr_ref_u/8'});

    observed = {'done', 'kv_mem_rd_addr', 'kv_mem_rd_valid', 'kv_mem_wr_addr', 'kv_mem_wr_valid', ...
        'kv_cache_wr_data', 'kv_cache_wr_en'};
    y = 80;
    obsCount = 0;
    for i = 1:numel(outports)
        name = char(outports(i).name);
        src = ['dut/' num2str(outports(i).port)];

        if isKey(ddrSinkMap, name)
            add_line(tbName, src, ddrSinkMap(name), 'autorouting', 'on');
        end

        if any(strcmp(name, observed))
            obsCount = obsCount + 1;
            add_block('simulink/Sinks/Out1', [tbName '/' name], 'Position', [980, y + 40 * obsCount, 1010, y + 40 * obsCount + 14]);
            add_line(tbName, src, [name '/1'], 'autorouting', 'on');
        elseif ~isKey(ddrSinkMap, name)
            term = ['term_' name];
            add_block('simulink/Sinks/Terminator', [tbName '/' term], 'Position', [900, y + 25 * i, 920, y + 25 * i + 20]);
            add_line(tbName, src, [term '/1'], 'autorouting', 'on');
        end
    end

    add_block('simulink/Sinks/Out1', [tbName '/tb_rd_rsp_data'], 'Position', [980, 120, 1010, 134]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_rd_rsp_valid'], 'Position', [980, 160, 1010, 174]);
    add_line(tbName, 'ddr_ref_u/3', 'tb_rd_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'ddr_ref_u/4', 'tb_rd_rsp_valid/1', 'autorouting', 'on');
end

function configure_soc_style_ddr_ref(subPath)
    add_block('simulink/Ports & Subsystems/Subsystem', subPath, 'Position', [780, 180, 930, 460]);
    Simulink.SubSystem.deleteContents(subPath);

    inNames = {'rd_addr', 'rd_len', 'rd_valid', 'wr_addr', 'wr_len', 'wr_valid', 'wr_data', 'wr_en'};
    outNames = {'rd_ready', 'wr_ready', 'rd_data', 'rd_data_valid'};
    for i = 1:numel(inNames)
        add_block('simulink/Sources/In1', [subPath '/' inNames{i}], 'Position', [20, 30 + 35 * i, 50, 44 + 35 * i]);
    end
    for i = 1:numel(outNames)
        add_block('simulink/Sinks/Out1', [subPath '/' outNames{i}], 'Position', [430, 40 + 45 * i, 460, 54 + 45 * i]);
    end

    add_block('simulink/Sources/Constant', [subPath '/ready_const'], ...
        'Value', 'true', 'OutDataTypeStr', 'boolean', 'Position', [90, 20, 130, 40]);
    add_block('simulink/Sources/Constant', [subPath '/latency_seed'], 'Value', '7', 'Position', [90, 60, 130, 80]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/rd_fire'], ...
        'Operator', 'AND', 'Position', [150, 120, 180, 150]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_fire_z'], ...
        'InitialCondition', '0', 'Position', [210, 120, 240, 150]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_addr_z'], ...
        'InitialCondition', '0', 'Position', [210, 75, 240, 105]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_len_z'], ...
        'InitialCondition', '0', 'Position', [210, 165, 240, 195]);
    add_block('simulink/Math Operations/Add', [subPath '/rd_addr_len_sum'], ...
        'Inputs', '++', 'Position', [280, 100, 315, 130]);
    add_block('simulink/Math Operations/Add', [subPath '/rd_data_pack'], ...
        'Inputs', '++', 'Position', [340, 100, 375, 130]);

    add_line(subPath, 'ready_const/1', 'rd_ready/1', 'autorouting', 'on');
    add_line(subPath, 'ready_const/1', 'wr_ready/1', 'autorouting', 'on');
    add_line(subPath, 'rd_valid/1', 'rd_fire/1', 'autorouting', 'on');
    add_line(subPath, 'ready_const/1', 'rd_fire/2', 'autorouting', 'on');
    add_line(subPath, 'rd_fire/1', 'rd_fire_z/1', 'autorouting', 'on');
    add_line(subPath, 'rd_addr/1', 'rd_addr_z/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len/1', 'rd_len_z/1', 'autorouting', 'on');
    add_line(subPath, 'rd_addr_z/1', 'rd_addr_len_sum/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len_z/1', 'rd_addr_len_sum/2', 'autorouting', 'on');
    add_line(subPath, 'rd_addr_len_sum/1', 'rd_data_pack/1', 'autorouting', 'on');
    add_line(subPath, 'latency_seed/1', 'rd_data_pack/2', 'autorouting', 'on');
    add_line(subPath, 'rd_data_pack/1', 'rd_data/1', 'autorouting', 'on');
    add_line(subPath, 'rd_fire_z/1', 'rd_data_valid/1', 'autorouting', 'on');
end

function ports = get_root_ports(mdlName, blockType)
    hits = find_system(mdlName, 'SearchDepth', 1, 'BlockType', blockType);
    nums = zeros(numel(hits), 1);
    for i = 1:numel(hits)
        nums(i) = str2double(get_param(hits{i}, 'Port'));
    end
    [~, order] = sort(nums);
    hits = hits(order);
    ports = repmat(struct('name', "", 'port', 0), numel(hits), 1);
    for i = 1:numel(hits)
        ports(i).name = string(get_param(hits{i}, 'Name'));
        ports(i).port = str2double(get_param(hits{i}, 'Port'));
    end
end

function compiledType = get_inport_compiled_type(mdlName, name)
    blk = [mdlName '/' char(name)];
    ph = get_param(blk, 'PortHandles');
    declaredType = string(get_param(blk, 'OutDataTypeStr'));
    if strlength(declaredType) ~= 0 && ~strcmpi(declaredType, 'Inherit: auto')
        compiledType = char(declaredType);
        return;
    end
    compiledType = char(get_param(ph.Outport(1), 'CompiledPortDataType'));
    if strlength(string(compiledType)) == 0
        lineHandle = get_param(ph.Outport(1), 'Line');
        if lineHandle ~= -1
            dstPorts = get_param(lineHandle, 'DstPortHandle');
            dstPorts = dstPorts(dstPorts ~= -1);
            for i = 1:numel(dstPorts)
                compiledType = char(get_param(dstPorts(i), 'CompiledPortDataType'));
                if strlength(string(compiledType)) ~= 0
                    break;
                end
            end
        end
    end
    if strlength(string(compiledType)) == 0
        compiledType = fallback_inport_type_for_name(name);
    end
end

function compiledType = fallback_inport_type_for_name(name)
    switch char(name)
        case {'start', 'in_valid', 'out_ready', 'stop_req', 'kv_cache_rd_valid', 'kv_mem_rd_ready', 'kv_mem_wr_ready'}
            compiledType = 'boolean';
        otherwise
            compiledType = 'double';
    end
end

function dt = compiled_type_to_param_string(compiledType)
    token = regexp(compiledType, '^(s|u)fix(\d+)_En(\d+)$', 'tokens', 'once');
    if ~isempty(token)
        isSigned = strcmp(token{1}, 's');
        dt = sprintf('fixdt(%d,%s,%s)', isSigned, token{2}, token{3});
        return;
    end
    if strcmpi(compiledType, 'logical')
        dt = 'boolean';
        return;
    end
    dt = compiledType;
end

function value = default_constant_for_name(name)
    switch name
        case 'mode_decode'
            value = '1';
        case 'start'
            value = '1';
        case 'eos_in'
            value = '0';
        case 'in_valid'
            value = '1';
        case 'out_ready'
            value = '1';
        case 'in_hidden'
            value = '2';
        case 'in_residual'
            value = '1';
        case 'cfg_seq_len'
            value = '1';
        case 'cfg_token_pos'
            value = '1';
        case 'cfg_eps'
            value = '1e-5';
        case 'stop_req'
            value = '0';
        case 'cfg_weight_num_heads'
            value = '1';
        case 'cfg_weight_page_base'
            value = '1';
        case 'cfg_weight_page_stride'
            value = '1';
        case 'cfg_rope_theta_scale'
            value = '1';
        case 'cfg_rope_sin_mix_scale'
            value = '1';
        otherwise
            value = '0';
    end
end

function values = extract_signal(yout, name)
    if isa(yout, 'Simulink.SimulationData.Dataset')
        for i = 1:yout.numElements
            elem = yout.getElement(i);
            if strcmp(string(elem.Name), string(name))
                values = extract_values(elem.Values);
                return;
            end
            if dataset_element_matches_name(elem, name)
                values = extract_values(elem.Values);
                return;
            end
        end
    end
    error('run_stage2_wrapper_tb_smoke:MissingOutput', 'Missing output signal: %s', name);
end

function match = dataset_element_matches_name(elem, name)
    match = false;
    try
        blockPathText = string(elem.BlockPath);
        if any(endsWith(blockPathText, "/" + string(name)))
            match = true;
            return;
        end
    catch
    end
    try
        blockPath = elem.BlockPath;
        lastBlock = string(blockPath.getBlock(blockPath.getLength));
        match = endsWith(lastBlock, "/" + string(name));
    catch
    end
end

function values = extract_values(ts)
    if isa(ts, 'timeseries')
        values = squeeze(ts.Data);
    else
        values = squeeze(ts);
    end
    values = double(values(:));
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function yes = is_wrapper_tb_blocker(ME)
    msg = string(getReport(ME, 'basic', 'hyperlinks', 'off'));
    yes = contains(msg, 'w_rd_req_bus') || ...
        contains(msg, 'bus data type should be specified') || ...
        contains(msg, 'Sample time mismatch') || ...
        contains(msg, 'Element name mismatch') || ...
        contains(msg, 'duplicated names at input ports');
end

function reason = classify_wrapper_tb_blocker(ME)
    msg = string(getReport(ME, 'basic', 'hyperlinks', 'off'));
    if contains(msg, 'w_rd_req_bus') || contains(msg, 'bus data type should be specified')
        reason = 'wrapper compile blocked by top-level w_rd_req_bus not being model-reference-safe as an explicit bus output';
    elseif contains(msg, 'Sample time mismatch')
        reason = 'wrapper compile blocked by mixed sample times on the top-level weight request bus path';
    elseif contains(msg, 'duplicated names at input ports')
        reason = 'wrapper compile blocked by duplicated signal names entering w_req_bc under wrapper/model-reference compilation';
    elseif contains(msg, 'Element name mismatch')
        reason = 'wrapper compile blocked by WeightRspBus element-name mismatch under wrapper/model-reference compilation';
    else
        reason = 'wrapper TB blocked by model-reference compile incompatibility in qwen2_block_top';
    end
end

function safe_close_models(tbName, mdlName)
    try
        bdclose(tbName);
    catch
    end
    try
        bdclose(mdlName);
    catch
    end
end