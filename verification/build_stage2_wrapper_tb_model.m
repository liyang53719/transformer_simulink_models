function info = build_stage2_wrapper_tb_model(rootDir, options)
%BUILD_STAGE2_WRAPPER_TB_MODEL Build a visible wrapper TB around qwen2_block_top.
%   This creates the same DUT + typed stimulus + SoC-style DDR responder
%   structure used by the wrapper smoke. By default it builds a temporary
%   model; with PersistModel=true it saves a persistent model under
%   simulink/models/qwen2_block_top_wrapper_tb.slx.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildDutModel = getFieldOr(options, 'BuildModel', true);
    persistModel = getFieldOr(options, 'PersistModel', false);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    addpath(fullfile(rootDir, 'simulink', 'fsm'));
    ensure_weight_bus_objects_local();
    if buildDutModel
        implement_stage1_rmsnorm_qkv(rootDir, struct( ...
            'StageProfile', 'stage2_memory_ready', ...
            'KvAddressConfig', kvCfg, ...
            'UseExternalWeightRsp', true));
    end

    dutPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(dutPath);
    [~, dutName] = fileparts(dutPath);
    if ~has_root_inport(dutName, 'w_rd_rsp_bus')
        close_system(dutName, 0);
        implement_stage1_rmsnorm_qkv(rootDir, struct( ...
            'StageProfile', 'stage2_memory_ready', ...
            'KvAddressConfig', kvCfg, ...
            'UseExternalWeightRsp', true));
        load_system(dutPath);
        [~, dutName] = fileparts(dutPath);
    end
    set_param(dutName, 'SimulationCommand', 'update');

    if persistModel
        tbName = 'qwen2_block_top_wrapper_tb';
        tbPath = fullfile(rootDir, 'simulink', 'models', [tbName '.slx']);
        if bdIsLoaded(tbName)
            close_system(tbName, 0);
        end
        if exist(tbPath, 'file')
            load_system(tbPath);
            clear_model_contents(tbName);
        else
            new_system(tbName);
        end
    else
        tbName = ['tmp_qwen2_wrapper_tb_' char(string(feature('getpid')))];
        tbPath = "";
        if bdIsLoaded(tbName)
            close_system(tbName, 0);
        end
        new_system(tbName);
    end

    set_param(tbName, 'SolverType', 'Variable-step');
    set_param(tbName, 'Solver', 'VariableStepDiscrete');

    inports = get_root_ports(dutName, 'Inport');
    outports = get_root_ports(dutName, 'Outport');

    add_block('simulink/Ports & Subsystems/Model', [tbName '/dut'], ...
        'ModelName', dutName, 'Position', [500, 180, 760, 520]);
    configure_soc_style_ddr_ref([tbName '/ddr_ref_u']);
    configure_weight_rsp_ref([tbName '/weight_ref_u']);

    build_typed_sources(tbName, dutName, inports);
    connect_dut_inputs(tbName, inports);
    connect_dut_outputs(tbName, outports);

    try
        Simulink.BlockDiagram.arrangeSystem(tbName);
    catch
    end

    if persistModel
        save_system(tbName, tbPath, 'OverwriteIfChangedOnDisk', true);
    end

    info = struct();
    info.tbName = tbName;
    info.tbPath = string(tbPath);
    info.dutName = dutName;
    info.persistModel = persistModel;
end

function clear_model_contents(mdlName)
    blocks = find_system(mdlName, 'SearchDepth', 1, 'Type', 'Block');
    for i = 1:numel(blocks)
        blk = blocks{i};
        if strcmp(blk, mdlName)
            continue;
        end
        try
            delete_block(blk);
        catch
        end
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

function connect_dut_inputs(tbName, inports)
    ddrInMap = containers.Map( ...
        {'kv_cache_rd_data', 'kv_cache_rd_valid', 'kv_mem_rd_ready', 'kv_mem_wr_ready', 'w_rd_rsp_bus'}, ...
        {'ddr_ref_u/3', 'ddr_ref_u/4', 'ddr_ref_u/1', 'ddr_ref_u/2', 'weight_ref_u/1'});

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

    observed = {'done', 'out_hidden', 'kv_mem_rd_addr', 'kv_mem_rd_valid', 'kv_mem_wr_addr', 'kv_mem_wr_valid', ...
        'kv_cache_wr_data', 'kv_cache_wr_en'};
    y = 80;
    obsCount = 0;
    wReqSrc = '';
    for i = 1:numel(outports)
        name = char(outports(i).name);
        src = ['dut/' num2str(outports(i).port)];

        if strcmp(name, 'w_rd_req_bus')
            wReqSrc = src;
        end

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

    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_sel'], ...
        'OutputSignals', 'attn_q_valid,attn_k_valid,attn_v_valid', ...
        'Position', [820, 655, 870, 735]);
    add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_rsp_data_sel'], ...
        'OutputSignals', 'attn_q_data,attn_k_data,attn_v_data', ...
        'Position', [820, 745, 870, 825]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_q_rsp_valid'], 'Position', [980, 320, 1010, 334]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_k_rsp_valid'], 'Position', [980, 360, 1010, 374]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_v_rsp_valid'], 'Position', [980, 400, 1010, 414]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_q_rsp_data'], 'Position', [980, 440, 1010, 454]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_k_rsp_data'], 'Position', [980, 480, 1010, 494]);
    add_block('simulink/Sinks/Out1', [tbName '/tb_attn_v_rsp_data'], 'Position', [980, 520, 1010, 534]);
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_sel/1', 'autorouting', 'on');
    add_line(tbName, 'weight_ref_u/1', 'tb_w_rsp_data_sel/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_sel/1', 'tb_attn_q_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_sel/2', 'tb_attn_k_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_sel/3', 'tb_attn_v_rsp_valid/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_data_sel/1', 'tb_attn_q_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_data_sel/2', 'tb_attn_k_rsp_data/1', 'autorouting', 'on');
    add_line(tbName, 'tb_w_rsp_data_sel/3', 'tb_attn_v_rsp_data/1', 'autorouting', 'on');

    if ~isempty(wReqSrc)
        add_line(tbName, wReqSrc, 'weight_ref_u/1', 'autorouting', 'on');
        add_block('simulink/Signal Routing/Bus Selector', [tbName '/tb_w_req_sel'], ...
            'OutputSignals', 'attn_q_valid,attn_k_valid,attn_v_valid', ...
            'Position', [820, 560, 870, 640]);
        add_block('simulink/Sinks/Out1', [tbName '/tb_attn_q_req_valid'], 'Position', [980, 200, 1010, 214]);
        add_block('simulink/Sinks/Out1', [tbName '/tb_attn_k_req_valid'], 'Position', [980, 240, 1010, 254]);
        add_block('simulink/Sinks/Out1', [tbName '/tb_attn_v_req_valid'], 'Position', [980, 280, 1010, 294]);
        add_line(tbName, wReqSrc, 'tb_w_req_sel/1', 'autorouting', 'on');
        add_line(tbName, 'tb_w_req_sel/1', 'tb_attn_q_req_valid/1', 'autorouting', 'on');
        add_line(tbName, 'tb_w_req_sel/2', 'tb_attn_k_req_valid/1', 'autorouting', 'on');
        add_line(tbName, 'tb_w_req_sel/3', 'tb_attn_v_req_valid/1', 'autorouting', 'on');
    end
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

    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/Input Read Memory'], ...
        'Position', [110, 70, 290, 230]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/Output Write Memory'], ...
        'Position', [110, 260, 290, 430]);

    configure_input_read_memory([subPath '/Input Read Memory']);
    configure_output_write_memory([subPath '/Output Write Memory']);

    add_line(subPath, 'rd_addr/1', 'Input Read Memory/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len/1', 'Input Read Memory/2', 'autorouting', 'on');
    add_line(subPath, 'rd_valid/1', 'Input Read Memory/3', 'autorouting', 'on');
    add_line(subPath, 'Input Read Memory/1', 'rd_ready/1', 'autorouting', 'on');
    add_line(subPath, 'Input Read Memory/2', 'rd_data/1', 'autorouting', 'on');
    add_line(subPath, 'Input Read Memory/3', 'rd_data_valid/1', 'autorouting', 'on');

    add_line(subPath, 'wr_addr/1', 'Output Write Memory/1', 'autorouting', 'on');
    add_line(subPath, 'wr_len/1', 'Output Write Memory/2', 'autorouting', 'on');
    add_line(subPath, 'wr_valid/1', 'Output Write Memory/3', 'autorouting', 'on');
    add_line(subPath, 'wr_data/1', 'Output Write Memory/4', 'autorouting', 'on');
    add_line(subPath, 'wr_en/1', 'Output Write Memory/5', 'autorouting', 'on');
    add_line(subPath, 'Output Write Memory/1', 'wr_ready/1', 'autorouting', 'on');
end

function configure_weight_rsp_ref(subPath)
    add_block('simulink/Ports & Subsystems/Subsystem', subPath, 'Position', [780, 500, 930, 690]);
    Simulink.SubSystem.deleteContents(subPath);

    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 75, 50, 89], ...
        'OutDataTypeStr', 'Bus: WeightReqBus');
    add_block('simulink/Signal Routing/Bus Selector', [subPath '/req_sel'], 'Position', [95, 25, 145, 190]);
    set_param([subPath '/req_sel'], 'OutputSignals', ...
        'gamma_addr,gamma_valid,qkv_q_addr,qkv_q_valid,qkv_k_addr,qkv_k_valid,qkv_v_addr,qkv_v_valid,attn_q_addr,attn_q_valid,attn_k_addr,attn_k_valid,attn_v_addr,attn_v_valid,ffn_up_addr,ffn_up_valid,ffn_gate_addr,ffn_gate_valid');
    add_block('simulink/Signal Routing/Bus Creator', [subPath '/rsp_bc'], ...
        'Position', [505, 25, 545, 295], 'Inputs', '18');
    set_param([subPath '/rsp_bc'], 'UseBusObject', 'on', 'BusObject', 'WeightRspBus');
    try
        set_param([subPath '/rsp_bc'], 'InputSignalNames', ...
            'gamma_data,gamma_valid,qkv_q_data,qkv_q_valid,qkv_k_data,qkv_k_valid,qkv_v_data,qkv_v_valid,attn_q_data,attn_q_valid,attn_k_data,attn_k_valid,attn_v_data,attn_v_valid,ffn_up_data,ffn_up_valid,ffn_gate_data,ffn_gate_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/rsp_bus'], 'Position', [585, 145, 615, 159], ...
        'OutDataTypeStr', 'Bus: WeightRspBus');
    add_line(subPath, 'req_bus/1', 'req_sel/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_bc/1', 'rsp_bus/1', 'autorouting', 'on');

    pageSignatures = {'0','8','16','24','32','40','48','56','64'};
    reqNames = { ...
        'gamma_addr', 'gamma_valid', ...
        'qkv_q_addr', 'qkv_q_valid', 'qkv_k_addr', 'qkv_k_valid', 'qkv_v_addr', 'qkv_v_valid', ...
        'attn_q_addr', 'attn_q_valid', 'attn_k_addr', 'attn_k_valid', 'attn_v_addr', 'attn_v_valid', ...
        'ffn_up_addr', 'ffn_up_valid', 'ffn_gate_addr', 'ffn_gate_valid'};
    for i = 1:9
        baseY = 25 + 30 * (i - 1);
        add_block('simulink/Sources/Constant', [subPath '/ready_' num2str(i)], ...
            'Value', '1', 'Position', [170, baseY, 200, baseY + 18]);
        add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/req_hs_' num2str(i)], ...
            'Operator', 'AND', 'Position', [220, baseY, 250, baseY + 18]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/val_d1_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [270, baseY, 300, baseY + 18]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/val_d2_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [320, baseY, 350, baseY + 18]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/addr_d1_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [270, baseY + 12, 300, baseY + 30]);
        add_block('simulink/Discrete/Unit Delay', [subPath '/addr_d2_' num2str(i)], ...
            'InitialCondition', '0', 'Position', [320, baseY + 12, 350, baseY + 30]);
        add_block('simulink/Sources/Constant', [subPath '/page_sig_' num2str(i)], ...
            'Value', pageSignatures{i}, 'Position', [360, baseY + 2, 390, baseY + 18]);
        add_block('simulink/Math Operations/Add', [subPath '/data_page_tag_' num2str(i)], ...
            'Inputs', '++', 'Position', [400, baseY + 8, 435, baseY + 30]);
        add_block('simulink/Signal Attributes/Data Type Conversion', [subPath '/data_u8_' num2str(i)], ...
            'OutDataTypeStr', 'uint8', 'Position', [455, baseY + 8, 485, baseY + 30]);

        reqAddrPort = 2 * i - 1;
        reqValidPort = 2 * i;
        rspDataPort = 2 * i - 1;
        rspValidPort = 2 * i;
        reqAddrName = reqNames{reqAddrPort};
        reqValidName = reqNames{reqValidPort};

        add_line(subPath, ['req_sel/' num2str(reqAddrPort)], ['addr_d1_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['req_sel/' num2str(reqValidPort)], ['req_hs_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['ready_' num2str(i) '/1'], ['req_hs_' num2str(i) '/2'], 'autorouting', 'on');
        add_line(subPath, ['req_hs_' num2str(i) '/1'], ['val_d1_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['val_d1_' num2str(i) '/1'], ['val_d2_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['addr_d1_' num2str(i) '/1'], ['addr_d2_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['addr_d2_' num2str(i) '/1'], ['data_page_tag_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['page_sig_' num2str(i) '/1'], ['data_page_tag_' num2str(i) '/2'], 'autorouting', 'on');
        add_line(subPath, ['data_page_tag_' num2str(i) '/1'], ['data_u8_' num2str(i) '/1'], 'autorouting', 'on');
        add_line(subPath, ['data_u8_' num2str(i) '/1'], ['rsp_bc/' num2str(rspDataPort)], 'autorouting', 'on');
        add_line(subPath, ['val_d2_' num2str(i) '/1'], ['rsp_bc/' num2str(rspValidPort)], 'autorouting', 'on');
        set_line_name_by_dst_port(subPath, 'rsp_bc', rspDataPort, strrep(reqAddrName, '_addr', '_data'));
        set_line_name_by_dst_port(subPath, 'rsp_bc', rspValidPort, reqValidName);
    end
end

function configure_input_read_memory(subPath)
    Simulink.SubSystem.deleteContents(subPath);

    add_block('simulink/Sources/In1', [subPath '/rd_addr'], 'Position', [20, 45, 50, 59]);
    add_block('simulink/Sources/In1', [subPath '/rd_len'], 'Position', [20, 85, 50, 99]);
    add_block('simulink/Sources/In1', [subPath '/rd_valid'], 'Position', [20, 125, 50, 139]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterRead BusCreator'], ...
        'Position', [80, 30, 180, 160]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterReadController'], ...
        'Position', [215, 30, 345, 160]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterRead BusSelector'], ...
        'Position', [380, 30, 500, 160]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_ready'], 'Position', [330, 20, 360, 34]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_data'], 'Position', [330, 60, 360, 74]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_data_valid'], 'Position', [330, 120, 360, 134]);

    configure_axi4masterread_buscreator([subPath '/AXI4MasterRead BusCreator']);
    configure_axi4masterread_controller([subPath '/AXI4MasterReadController']);
    configure_axi4masterread_busselector([subPath '/AXI4MasterRead BusSelector']);

    add_line(subPath, 'rd_addr/1', 'AXI4MasterRead BusCreator/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len/1', 'AXI4MasterRead BusCreator/2', 'autorouting', 'on');
    add_line(subPath, 'rd_valid/1', 'AXI4MasterRead BusCreator/3', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusCreator/1', 'AXI4MasterReadController/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterReadController/1', 'rd_ready/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterReadController/2', 'AXI4MasterRead BusSelector/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusSelector/1', 'rd_data/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusSelector/2', 'rd_data_valid/1', 'autorouting', 'on');
end

function configure_output_write_memory(subPath)
    Simulink.SubSystem.deleteContents(subPath);

    add_block('simulink/Sources/In1', [subPath '/wr_addr'], 'Position', [20, 35, 50, 49]);
    add_block('simulink/Sources/In1', [subPath '/wr_len'], 'Position', [20, 70, 50, 84]);
    add_block('simulink/Sources/In1', [subPath '/wr_valid'], 'Position', [20, 105, 50, 119]);
    add_block('simulink/Sources/In1', [subPath '/wr_data'], 'Position', [20, 140, 50, 154]);
    add_block('simulink/Sources/In1', [subPath '/wr_en'], 'Position', [20, 175, 50, 189]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterWrite BusCreator'], ...
        'Position', [80, 25, 190, 185]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterWriteController'], ...
        'Position', [225, 25, 355, 185]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterWrite BusSelector'], ...
        'Position', [390, 25, 500, 185]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_ready'], 'Position', [310, 120, 340, 134]);
    add_block('simulink/Sinks/Terminator', [subPath '/term_request_next_line'], 'Position', [530, 120, 550, 140]);

    configure_axi4masterwrite_buscreator([subPath '/AXI4MasterWrite BusCreator']);
    configure_axi4masterwrite_controller([subPath '/AXI4MasterWriteController']);
    configure_axi4masterwrite_busselector([subPath '/AXI4MasterWrite BusSelector']);

    add_line(subPath, 'wr_addr/1', 'AXI4MasterWrite BusCreator/1', 'autorouting', 'on');
    add_line(subPath, 'wr_len/1', 'AXI4MasterWrite BusCreator/2', 'autorouting', 'on');
    add_line(subPath, 'wr_valid/1', 'AXI4MasterWrite BusCreator/3', 'autorouting', 'on');
    add_line(subPath, 'wr_data/1', 'AXI4MasterWrite BusCreator/4', 'autorouting', 'on');
    add_line(subPath, 'wr_en/1', 'AXI4MasterWrite BusCreator/5', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusCreator/1', 'AXI4MasterWriteController/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWriteController/1', 'AXI4MasterWrite BusSelector/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/1', 'wr_ready/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/2', 'term_request_next_line/1', 'autorouting', 'on');
end

function configure_axi4masterread_buscreator(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/rd_addr'], 'Position', [20, 35, 50, 49]);
    add_block('simulink/Sources/In1', [subPath '/rd_len'], 'Position', [20, 75, 50, 89]);
    add_block('simulink/Sources/In1', [subPath '/rd_valid'], 'Position', [20, 115, 50, 129]);
    add_block('simulink/Signal Routing/Bus Creator', [subPath '/req_bc'], 'Position', [110, 45, 150, 125]);
    set_param([subPath '/req_bc'], 'Inputs', '3');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'rd_addr,rd_len,rd_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/req_bus'], 'Position', [190, 80, 220, 94]);
    add_line(subPath, 'rd_addr/1', 'req_bc/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len/1', 'req_bc/2', 'autorouting', 'on');
    add_line(subPath, 'rd_valid/1', 'req_bc/3', 'autorouting', 'on');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'rd_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'rd_len');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'rd_valid');
    add_line(subPath, 'req_bc/1', 'req_bus/1', 'autorouting', 'on');
end

function configure_axi4masterread_controller(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 80, 50, 94]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterRead BusSelector'], ...
        'Position', [85, 45, 195, 130]);
    configure_axi4masterread_req_busselector([subPath '/AXI4MasterRead BusSelector']);
    add_block('simulink/Sources/Constant', [subPath '/ready_const'], ...
        'Value', 'true', 'OutDataTypeStr', 'boolean', 'Position', [225, 20, 265, 40]);
    add_block('simulink/Sources/Constant', [subPath '/latency_seed'], ...
        'Value', '7', 'Position', [225, 50, 265, 70]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/rd_fire'], ...
        'Operator', 'AND', 'Position', [285, 105, 315, 130]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_fire_z'], ...
        'InitialCondition', '0', 'Position', [335, 105, 365, 130]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_addr_z'], ...
        'InitialCondition', '0', 'Position', [335, 35, 365, 60]);
    add_block('simulink/Discrete/Unit Delay', [subPath '/rd_len_z'], ...
        'InitialCondition', '0', 'Position', [335, 70, 365, 95]);
    add_block('simulink/Math Operations/Add', [subPath '/rd_addr_len_sum'], ...
        'Inputs', '++', 'Position', [385, 45, 420, 75]);
    add_block('simulink/Math Operations/Add', [subPath '/rd_data_pack'], ...
        'Inputs', '++', 'Position', [440, 45, 475, 75]);
    add_block('simulink/Signal Routing/Bus Creator', [subPath '/rsp_bc'], 'Position', [510, 55, 550, 115]);
    set_param([subPath '/rsp_bc'], 'Inputs', '2');
    try
        set_param([subPath '/rsp_bc'], 'InputSignalNames', 'rd_data,rd_data_valid');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/rd_ready'], 'Position', [585, 20, 615, 34]);
    add_block('simulink/Sinks/Out1', [subPath '/rsp_bus'], 'Position', [585, 80, 615, 94]);

    add_line(subPath, 'req_bus/1', 'AXI4MasterRead BusSelector/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusSelector/1', 'rd_addr_z/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusSelector/2', 'rd_len_z/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterRead BusSelector/3', 'rd_fire/1', 'autorouting', 'on');
    add_line(subPath, 'ready_const/1', 'rd_fire/2', 'autorouting', 'on');
    add_line(subPath, 'rd_fire/1', 'rd_fire_z/1', 'autorouting', 'on');
    add_line(subPath, 'rd_addr_z/1', 'rd_addr_len_sum/1', 'autorouting', 'on');
    add_line(subPath, 'rd_len_z/1', 'rd_addr_len_sum/2', 'autorouting', 'on');
    add_line(subPath, 'rd_addr_len_sum/1', 'rd_data_pack/1', 'autorouting', 'on');
    add_line(subPath, 'latency_seed/1', 'rd_data_pack/2', 'autorouting', 'on');
    add_line(subPath, 'rd_data_pack/1', 'rsp_bc/1', 'autorouting', 'on');
    add_line(subPath, 'rd_fire_z/1', 'rsp_bc/2', 'autorouting', 'on');
    set_line_name_by_dst_port(subPath, 'rsp_bc', 1, 'rd_data');
    set_line_name_by_dst_port(subPath, 'rsp_bc', 2, 'rd_data_valid');
    add_line(subPath, 'ready_const/1', 'rd_ready/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_bc/1', 'rsp_bus/1', 'autorouting', 'on');
end

function configure_axi4masterread_req_busselector(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 65, 50, 79]);
    add_block('simulink/Signal Routing/Bus Selector', [subPath '/req_sel'], 'Position', [95, 35, 145, 105]);
    set_param([subPath '/req_sel'], 'OutputSignals', 'rd_addr,rd_len,rd_valid');
    add_block('simulink/Sinks/Out1', [subPath '/rd_addr'], 'Position', [190, 35, 220, 49]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_len'], 'Position', [190, 65, 220, 79]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_valid'], 'Position', [190, 95, 220, 109]);
    add_line(subPath, 'req_bus/1', 'req_sel/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/1', 'rd_addr/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/2', 'rd_len/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/3', 'rd_valid/1', 'autorouting', 'on');
end

function configure_axi4masterread_busselector(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/rsp_bus'], 'Position', [20, 65, 50, 79]);
    add_block('simulink/Signal Routing/Bus Selector', [subPath '/rsp_sel'], 'Position', [95, 35, 145, 105]);
    set_param([subPath '/rsp_sel'], 'OutputSignals', 'rd_data,rd_data_valid');
    add_block('simulink/Sinks/Out1', [subPath '/rd_data'], 'Position', [190, 35, 220, 49]);
    add_block('simulink/Sinks/Out1', [subPath '/rd_data_valid'], 'Position', [190, 95, 220, 109]);
    add_line(subPath, 'rsp_bus/1', 'rsp_sel/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_sel/1', 'rd_data/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_sel/2', 'rd_data_valid/1', 'autorouting', 'on');
end

function configure_axi4masterwrite_buscreator(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/wr_addr'], 'Position', [20, 35, 50, 49]);
    add_block('simulink/Sources/In1', [subPath '/wr_len'], 'Position', [20, 65, 50, 79]);
    add_block('simulink/Sources/In1', [subPath '/wr_valid'], 'Position', [20, 95, 50, 109]);
    add_block('simulink/Sources/In1', [subPath '/wr_data'], 'Position', [20, 125, 50, 139]);
    add_block('simulink/Sources/In1', [subPath '/wr_en'], 'Position', [20, 155, 50, 169]);
    add_block('simulink/Signal Routing/Bus Creator', [subPath '/req_bc'], 'Position', [110, 55, 150, 155]);
    set_param([subPath '/req_bc'], 'Inputs', '5');
    try
        set_param([subPath '/req_bc'], 'InputSignalNames', 'wr_addr,wr_len,wr_valid,wr_data,wr_en');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/req_bus'], 'Position', [190, 100, 220, 114]);
    add_line(subPath, 'wr_addr/1', 'req_bc/1', 'autorouting', 'on');
    add_line(subPath, 'wr_len/1', 'req_bc/2', 'autorouting', 'on');
    add_line(subPath, 'wr_valid/1', 'req_bc/3', 'autorouting', 'on');
    add_line(subPath, 'wr_data/1', 'req_bc/4', 'autorouting', 'on');
    add_line(subPath, 'wr_en/1', 'req_bc/5', 'autorouting', 'on');
    set_line_name_by_dst_port(subPath, 'req_bc', 1, 'wr_addr');
    set_line_name_by_dst_port(subPath, 'req_bc', 2, 'wr_len');
    set_line_name_by_dst_port(subPath, 'req_bc', 3, 'wr_valid');
    set_line_name_by_dst_port(subPath, 'req_bc', 4, 'wr_data');
    set_line_name_by_dst_port(subPath, 'req_bc', 5, 'wr_en');
    add_line(subPath, 'req_bc/1', 'req_bus/1', 'autorouting', 'on');
end

function configure_axi4masterwrite_controller(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Ports & Subsystems/Subsystem', [subPath '/AXI4MasterWrite BusSelector'], ...
        'Position', [85, 35, 195, 165]);
    configure_axi4masterwrite_req_busselector([subPath '/AXI4MasterWrite BusSelector']);
    add_block('simulink/Sources/Constant', [subPath '/ready_const'], ...
        'Value', 'true', 'OutDataTypeStr', 'boolean', 'Position', [225, 25, 265, 45]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/wr_fire'], ...
        'Operator', 'AND', 'Position', [290, 90, 320, 115]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [subPath '/wr_commit'], ...
        'Operator', 'AND', 'Position', [345, 110, 375, 135]);
    add_block('simulink/Sinks/Terminator', [subPath '/term_wr_addr'], 'Position', [405, 35, 425, 55]);
    add_block('simulink/Sinks/Terminator', [subPath '/term_wr_len'], 'Position', [405, 65, 425, 85]);
    add_block('simulink/Sinks/Terminator', [subPath '/term_wr_data'], 'Position', [405, 125, 425, 145]);
    add_block('simulink/Signal Routing/Bus Creator', [subPath '/rsp_bc'], 'Position', [450, 85, 490, 140]);
    set_param([subPath '/rsp_bc'], 'Inputs', '2');
    try
        set_param([subPath '/rsp_bc'], 'InputSignalNames', 'wr_ready,request_next_line');
    catch
    end
    add_block('simulink/Sinks/Out1', [subPath '/rsp_bus'], 'Position', [530, 100, 560, 114]);

    add_line(subPath, 'req_bus/1', 'AXI4MasterWrite BusSelector/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/1', 'term_wr_addr/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/2', 'term_wr_len/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/3', 'wr_fire/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/4', 'term_wr_data/1', 'autorouting', 'on');
    add_line(subPath, 'AXI4MasterWrite BusSelector/5', 'wr_commit/2', 'autorouting', 'on');
    add_line(subPath, 'ready_const/1', 'wr_fire/2', 'autorouting', 'on');
    add_line(subPath, 'wr_fire/1', 'wr_commit/1', 'autorouting', 'on');
    add_line(subPath, 'ready_const/1', 'rsp_bc/1', 'autorouting', 'on');
    add_line(subPath, 'wr_commit/1', 'rsp_bc/2', 'autorouting', 'on');
    set_line_name_by_dst_port(subPath, 'rsp_bc', 1, 'wr_ready');
    set_line_name_by_dst_port(subPath, 'rsp_bc', 2, 'request_next_line');
    add_line(subPath, 'rsp_bc/1', 'rsp_bus/1', 'autorouting', 'on');
end

function configure_axi4masterwrite_req_busselector(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/req_bus'], 'Position', [20, 90, 50, 104]);
    add_block('simulink/Signal Routing/Bus Selector', [subPath '/req_sel'], 'Position', [95, 35, 145, 145]);
    set_param([subPath '/req_sel'], 'OutputSignals', 'wr_addr,wr_len,wr_valid,wr_data,wr_en');
    add_block('simulink/Sinks/Out1', [subPath '/wr_addr'], 'Position', [190, 35, 220, 49]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_len'], 'Position', [190, 65, 220, 79]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_valid'], 'Position', [190, 95, 220, 109]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_data'], 'Position', [190, 125, 220, 139]);
    add_block('simulink/Sinks/Out1', [subPath '/wr_en'], 'Position', [190, 155, 220, 169]);
    add_line(subPath, 'req_bus/1', 'req_sel/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/1', 'wr_addr/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/2', 'wr_len/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/3', 'wr_valid/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/4', 'wr_data/1', 'autorouting', 'on');
    add_line(subPath, 'req_sel/5', 'wr_en/1', 'autorouting', 'on');
end

function configure_axi4masterwrite_busselector(subPath)
    Simulink.SubSystem.deleteContents(subPath);
    add_block('simulink/Sources/In1', [subPath '/rsp_bus'], 'Position', [20, 65, 50, 79]);
    add_block('simulink/Signal Routing/Bus Selector', [subPath '/rsp_sel'], 'Position', [95, 35, 145, 110]);
    set_param([subPath '/rsp_sel'], 'OutputSignals', 'wr_ready,request_next_line');
    add_block('simulink/Sinks/Out1', [subPath '/wr_ready'], 'Position', [190, 65, 220, 79]);
    add_block('simulink/Sinks/Out1', [subPath '/request_next_line'], 'Position', [190, 95, 220, 109]);
    add_line(subPath, 'rsp_bus/1', 'rsp_sel/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_sel/1', 'wr_ready/1', 'autorouting', 'on');
    add_line(subPath, 'rsp_sel/2', 'request_next_line/1', 'autorouting', 'on');
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
            value = '12';
        case 'cfg_weight_page_base'
            value = '64';
        case 'cfg_weight_page_stride'
            value = '8';
        case 'cfg_rope_theta_scale'
            value = '1';
        case 'cfg_rope_sin_mix_scale'
            value = '1';
        otherwise
            value = '0';
    end
end

function set_line_name_by_dst_port(sys, dstBlockName, dstPort, lineName)
    try
        ph = get_param([sys '/' dstBlockName], 'PortHandles');
        if numel(ph.Inport) >= dstPort
            ln = get_param(ph.Inport(dstPort), 'Line');
            if ln ~= -1
                set_param(ln, 'Name', lineName);
                try
                    set_param(ln, 'SignalPropagation', 'on');
                catch
                end
            end
        end
    catch
    end
end

function yes = has_root_inport(mdlName, name)
    yes = ~isempty(find_system(mdlName, 'SearchDepth', 1, 'BlockType', 'Inport', 'Name', name));
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end

function ensure_weight_bus_objects_local()
    define_bus_local('WeightReqRmsBus', {'gamma_addr','gamma_valid'});
    define_bus_local('WeightReqQkvBus', {'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid'});
    define_bus_local('WeightReqAttnBus', {'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid'});
    define_bus_local('WeightReqFfnBus', {'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid'});
    define_bus_local('WeightAddrRmsBus', {'gamma_addr'});
    define_bus_local('WeightAddrQkvBus', {'q_addr','k_addr','v_addr'});
    define_bus_local('WeightAddrAttnBus', {'attn_q_addr','attn_k_addr','attn_v_addr'});
    define_bus_local('WeightAddrFfnBus', {'up_addr','gate_addr'});
    define_bus_local('WeightReqBus', {
        'gamma_addr','gamma_valid', ...
        'qkv_q_addr','qkv_q_valid','qkv_k_addr','qkv_k_valid','qkv_v_addr','qkv_v_valid', ...
        'attn_q_addr','attn_q_valid','attn_k_addr','attn_k_valid','attn_v_addr','attn_v_valid', ...
        'ffn_up_addr','ffn_up_valid','ffn_gate_addr','ffn_gate_valid'});
    define_bus_typed_local('WeightRspBus', {
        'gamma_data','gamma_valid', ...
        'qkv_q_data','qkv_q_valid','qkv_k_data','qkv_k_valid','qkv_v_data','qkv_v_valid', ...
        'attn_q_data','attn_q_valid','attn_k_data','attn_k_valid','attn_v_data','attn_v_valid', ...
        'ffn_up_data','ffn_up_valid','ffn_gate_data','ffn_gate_valid'});
    define_bus_local('QkvStreamBus', {'q_stream','k_stream','v_stream','q_valid','kv_valid','group_idx'});
    define_bus_local('AttentionFlowBus', {'q_stream','k_cache','v_cache','group_idx','score_scale'});
    define_bus_local('PrefillScheduleBus', {
        'array_rows','array_cols','tile_seq','tile_k','tile_out', ...
        'x_bank_count','psum_bank_count','kv_bank_count','q_heads_per_kv', ...
        'active_seq_len','decode_mode','kv_phase_first','score_scale'});
end

function define_bus_local(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        elems(i).DataType = 'double';
        elems(i).Dimensions = 1;
    end
    busObj = Simulink.Bus;
    busObj.Elements = elems;
    assignin('base', name, busObj);
end

function define_bus_typed_local(name, fieldNames)
    elems = repmat(Simulink.BusElement, numel(fieldNames), 1);
    for i = 1:numel(fieldNames)
        elems(i).Name = fieldNames{i};
        if endsWith(fieldNames{i}, '_valid')
            elems(i).DataType = 'boolean';
        elseif endsWith(fieldNames{i}, '_data')
            elems(i).DataType = 'uint8';
        else
            elems(i).DataType = 'double';
        end
        elems(i).Dimensions = 1;
    end
    busObj = Simulink.Bus;
    busObj.Elements = elems;
    assignin('base', name, busObj);
end