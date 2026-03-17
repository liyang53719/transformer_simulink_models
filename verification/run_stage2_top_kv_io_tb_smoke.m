function result = run_stage2_top_kv_io_tb_smoke(rootDir, options)
%RUN_STAGE2_TOP_KV_IO_TB_SMOKE Audit or simulate top-level KV IO path with external inputs.
%   This is a minimal TB-style smoke. It does not model external weight DDR.
%   If root input datatypes are not directly drivable from simple workspace
%   external inputs, the function returns tb_ready=false with blockers listed.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', true);
    kvCfg = getFieldOr(options, 'KvAddressConfig', struct('rd_base', 0, 'wr_base', 0, 'stride_bytes', 2, 'decode_burst_len', 1));

    addpath(fullfile(rootDir, 'scripts'));
    if buildModel
        implement_stage1_rmsnorm_qkv(rootDir, struct('StageProfile', 'stage2_memory_ready', 'KvAddressConfig', kvCfg));
    end

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    load_system(mdlPath);
    [~, mdlName] = fileparts(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    inports = get_root_inports(mdlName);
    typeAudit = audit_tb_input_types(mdlName, inports);

    if ~typeAudit.tb_ready
        result = struct();
        result.tb_ready = false;
        result.pass = false;
        result.blocking_inputs = typeAudit.blocking_inputs;
        result.input_types = typeAudit.input_types;
        result.reason = ['Top-level TB simulation is not ready: unsupported root input datatypes for direct ' ...
            'workspace stimulus. Use typed fi/timeseries wiring in a dedicated wrapper TB or relax the root interface types.'];
        close_system(mdlName, 0);
        fprintf('Stage2 top KV IO TB smoke BLOCKED\n');
        fprintf('  Blocking inputs: %s\n', strjoin(result.blocking_inputs, ', '));
        return;
    end

    extInput = build_external_input(mdlName, inports);

    simIn = Simulink.SimulationInput(mdlName);
    simIn = simIn.setExternalInput(extInput);
    simIn = simIn.setModelParameter( ...
        'StopTime', '4', ...
        'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', ...
        'ReturnWorkspaceOutputs', 'on');

    simOut = sim(simIn);
    yout = simOut.get('yout');

    kvRdValid = extract_signal(yout, 'kv_mem_rd_valid');
    kvWrValid = extract_signal(yout, 'kv_mem_wr_valid');
    kvRdAddr = extract_signal(yout, 'kv_mem_rd_addr');
    kvWrAddr = extract_signal(yout, 'kv_mem_wr_addr');
    kvWrData = extract_signal(yout, 'kv_cache_wr_data');
    kvWrEn = extract_signal(yout, 'kv_cache_wr_en');

    expectedWrData = repmat(7, size(kvWrData));
    expectedWrEn = repmat(1, size(kvWrEn));

    result = struct();
    result.tb_ready = true;
    result.kv_rd_valid_seen = any(kvRdValid > 0.5);
    result.kv_wr_valid_seen = any(kvWrValid > 0.5);
    result.kv_rd_addr_nonzero = any(kvRdAddr > 0);
    result.kv_wr_addr_nonzero = any(kvWrAddr > 0);
    result.kv_cache_loopback_data_ok = all(abs(kvWrData - expectedWrData) < 1e-9);
    result.kv_cache_loopback_en_ok = all(abs(kvWrEn - expectedWrEn) < 1e-9);
    result.pass = result.kv_rd_valid_seen && result.kv_wr_valid_seen && ...
        result.kv_rd_addr_nonzero && result.kv_wr_addr_nonzero && ...
        result.kv_cache_loopback_data_ok && result.kv_cache_loopback_en_ok;

    close_system(mdlName, 0);

    if result.pass
        fprintf('Stage2 top KV IO TB smoke PASS\n');
    else
        fprintf('Stage2 top KV IO TB smoke FAIL\n');
        error('run_stage2_top_kv_io_tb_smoke:Failed', ...
            'Top-level KV IO TB-style smoke failed');
    end
end

function audit = audit_tb_input_types(mdlName, inports)
    audit = struct();
    audit.tb_ready = true;
    audit.blocking_inputs = {};
    audit.input_types = struct();
    knownFixedPointTbBlockers = ["in_hidden", "in_residual"];

    for i = 1:numel(inports)
        compiledType = get_inport_compiled_type(mdlName, inports(i).name);
        audit.input_types.(char(inports(i).name)) = compiledType;
        if any(strcmp(string(inports(i).name), knownFixedPointTbBlockers))
            audit.tb_ready = false;
            audit.blocking_inputs{end+1} = sprintf('%s:%s', inports(i).name, 'fixed-point-root-stimulus'); %#ok<AGROW>
        elseif ~is_supported_tb_type(compiledType, inports(i).name)
            audit.tb_ready = false;
            audit.blocking_inputs{end+1} = sprintf('%s:%s', inports(i).name, compiledType); %#ok<AGROW>
        end
    end
end

function yes = is_supported_tb_type(compiledType, name)
    if is_boolean_type(compiledType, name)
        yes = true;
        return;
    end

    if any(strcmpi(string(compiledType), ["double", "single", "uint8", "int8", "uint16", "int16", "uint32", "int32"]))
        yes = true;
        return;
    end

    yes = false;
end

function inports = get_root_inports(mdlName)
    hits = find_system(mdlName, 'SearchDepth', 1, 'BlockType', 'Inport');
    ports = zeros(numel(hits), 1);
    for i = 1:numel(hits)
        ports(i) = str2double(get_param(hits{i}, 'Port'));
    end
    [~, order] = sort(ports);
    hits = hits(order);

    inports = repmat(struct('name', '', 'port', 0), numel(hits), 1);
    for i = 1:numel(hits)
        inports(i).name = string(get_param(hits{i}, 'Name'));
        inports(i).port = str2double(get_param(hits{i}, 'Port'));
    end
end

function extInput = build_external_input(mdlName, inports)
    t = (0:4)';
    extInput = Simulink.SimulationData.Dataset;
    for i = 1:numel(inports)
        compiledType = get_inport_compiled_type(mdlName, inports(i).name);
        values = default_signal_for_name(inports(i).name, numel(t), compiledType);
        ts = timeseries(values, t);
        ts.Name = char(inports(i).name);
        extInput = extInput.addElement(ts, char(inports(i).name));
    end
end

function sig = default_signal_for_name(name, n, compiledType)
    if is_boolean_type(compiledType, name)
        sig = false(n, 1);
    else
        sig = zeros(n, 1);
    end
    switch char(name)
        case 'mode_decode'
            sig(:) = 1;
        case 'start'
            sig = [1; 0; 0; 0; 0];
        case {'in_valid', 'out_ready', 'kv_cache_rd_valid', 'kv_mem_rd_ready', 'kv_mem_wr_ready'}
            sig(:) = 1;
        case 'in_hidden'
            sig(:) = 2;
        case 'in_residual'
            sig(:) = 1;
        case 'kv_cache_rd_data'
            sig(:) = 7;
        case 'cfg_seq_len'
            sig(:) = 4;
        case 'cfg_token_pos'
            sig = [1; 1; 2; 2; 2];
        case 'cfg_eps'
            sig(:) = 1e-5;
        case 'cfg_weight_num_heads'
            sig(:) = 2;
        case {'cfg_weight_page_base', 'cfg_weight_page_stride'}
            sig(:) = 1;
        case {'cfg_rope_theta_scale', 'cfg_rope_sin_mix_scale'}
            sig(:) = 1;
        otherwise
            sig(:) = 0;
    end

    sig = cast_to_compiled_type(sig, compiledType, name);
end

function yes = is_boolean_type(compiledType, name)
    yes = strcmpi(compiledType, 'boolean') || strcmpi(compiledType, 'logical');
    if yes
        return;
    end

    if strlength(string(compiledType)) == 0
        yes = strcmp(string(name), "start");
    end
end

function compiledType = get_inport_compiled_type(mdlName, name)
    blk = [mdlName '/' char(name)];
    ph = get_param(blk, 'PortHandles');
    compiledType = char(get_param(ph.Outport(1), 'CompiledPortDataType'));
    if strlength(string(compiledType)) == 0 && ~is_boolean_type('', name)
        compiledType = 'double';
    end
end

function sig = cast_to_compiled_type(sig, compiledType, name)
    if is_boolean_type(compiledType, name)
        sig = logical(sig);
        return;
    end

    token = regexp(compiledType, '^(s|u)fix(\d+)_En(\d+)$', 'tokens', 'once');
    if ~isempty(token)
        isSigned = strcmp(token{1}, 's');
        wordLength = str2double(token{2});
        fracLength = str2double(token{3});
        sig = fi(sig, isSigned, wordLength, fracLength);
        return;
    end

    sig = cast(sig, compiledType);
end

function values = extract_signal(yout, name)
    if isa(yout, 'Simulink.SimulationData.Dataset')
        try
            elem = yout.getElement(name);
            values = extract_values(elem.Values);
            return;
        catch
        end
        for i = 1:yout.numElements
            elem = yout.getElement(i);
            if strcmp(string(elem.Name), string(name))
                values = extract_values(elem.Values);
                return;
            end
        end
    end
    error('run_stage2_top_kv_io_tb_smoke:MissingOutput', 'Missing output signal: %s', name);
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