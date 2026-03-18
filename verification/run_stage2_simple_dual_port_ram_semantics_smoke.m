function result = run_stage2_simple_dual_port_ram_semantics_smoke(~)
%RUN_STAGE2_SIMPLE_DUAL_PORT_RAM_SEMANTICS_SMOKE Characterize Simple Dual Port RAM timing.
%   Builds a temporary discrete model around hdlsllib/HDL RAMs/Simple Dual
%   Port RAM and measures when a written value becomes visible on dout while
%   rd_addr is held constant at the written address.

    mdlName = 'tmp_stage2_simple_dual_port_ram_semantics';
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end

    new_system(mdlName);
    cleanup = onCleanup(@()safe_close_model(mdlName)); %#ok<NASGU>
    open_system(mdlName);
    set_param(mdlName, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
        'FixedStep', '1', 'StopTime', '6');

    add_block('simulink/Sources/From Workspace', [mdlName '/wr_addr_src'], 'Position', [30, 40, 120, 60]);
    add_block('simulink/Sources/From Workspace', [mdlName '/din_src'], 'Position', [30, 90, 120, 110]);
    add_block('simulink/Sources/From Workspace', [mdlName '/we_src'], 'Position', [30, 140, 120, 160]);
    add_block('simulink/Sources/From Workspace', [mdlName '/rd_addr_src'], 'Position', [30, 190, 120, 210]);

    add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/wr_addr_u8'], ...
        'OutDataTypeStr', 'uint8', 'Position', [160, 40, 220, 60]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/din_u8'], ...
        'OutDataTypeStr', 'uint8', 'Position', [160, 90, 220, 110]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/we_bool'], ...
        'OutDataTypeStr', 'boolean', 'Position', [160, 140, 220, 160]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/rd_addr_u8'], ...
        'OutDataTypeStr', 'uint8', 'Position', [160, 190, 220, 210]);

    add_block('hdlsllib/HDL RAMs/Simple Dual Port RAM', [mdlName '/ram'], ...
        'ram_size', '8', 'Position', [270, 65, 360, 205]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/dout_double'], ...
        'OutDataTypeStr', 'double', 'Position', [400, 120, 470, 140]);
    add_block('simulink/Sinks/Out1', [mdlName '/dout_out'], 'Position', [510, 120, 540, 134]);

    add_line(mdlName, 'wr_addr_src/1', 'wr_addr_u8/1', 'autorouting', 'on');
    add_line(mdlName, 'din_src/1', 'din_u8/1', 'autorouting', 'on');
    add_line(mdlName, 'we_src/1', 'we_bool/1', 'autorouting', 'on');
    add_line(mdlName, 'rd_addr_src/1', 'rd_addr_u8/1', 'autorouting', 'on');
    add_line(mdlName, 'wr_addr_u8/1', 'ram/1', 'autorouting', 'on');
    add_line(mdlName, 'din_u8/1', 'ram/2', 'autorouting', 'on');
    add_line(mdlName, 'we_bool/1', 'ram/3', 'autorouting', 'on');
    add_line(mdlName, 'rd_addr_u8/1', 'ram/4', 'autorouting', 'on');
    add_line(mdlName, 'ram/1', 'dout_double/1', 'autorouting', 'on');
    add_line(mdlName, 'dout_double/1', 'dout_out/1', 'autorouting', 'on');

    assignin('base', 'wr_addr_trace', [0 3; 1 3; 2 3; 3 3; 4 3; 5 3; 6 3]);
    assignin('base', 'din_trace', [0 0; 1 55; 2 0; 3 0; 4 0; 5 0; 6 0]);
    assignin('base', 'we_trace', [0 0; 1 1; 2 0; 3 0; 4 0; 5 0; 6 0]);
    assignin('base', 'rd_addr_trace', [0 3; 1 3; 2 3; 3 3; 4 3; 5 3; 6 3]);
    set_param([mdlName '/wr_addr_src'], 'VariableName', 'wr_addr_trace');
    set_param([mdlName '/din_src'], 'VariableName', 'din_trace');
    set_param([mdlName '/we_src'], 'VariableName', 'we_trace');
    set_param([mdlName '/rd_addr_src'], 'VariableName', 'rd_addr_trace');

    simOut = sim(mdlName, 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
    yout = simOut.get('yout');
    doutValues = double(extract_signal(yout, 'dout_out'));
    hitIndex = find(abs(doutValues - 55) < 1e-9, 1, 'first');

    permutationResults = brute_force_input_mappings();

    result = struct();
    result.dout_values = doutValues(:).';
    result.visible_index = hitIndex;
    if isempty(hitIndex)
        result.visible_lag = NaN;
    else
        result.visible_lag = hitIndex - 2;
    end
    result.same_cycle_visible = ~isempty(hitIndex) && hitIndex == 2;
    result.one_cycle_visible = ~isempty(hitIndex) && hitIndex == 3;
    result.permutation_results = permutationResults;
    result.valid_mapping_count = sum([permutationResults.valid_mapping]);
    result.visible_mapping_count = sum([permutationResults.visible]);

    fprintf('Stage2 Simple Dual Port RAM semantics smoke PASS\n');
    fprintf('  dout_values=%s\n', mat2str(result.dout_values));
    fprintf('  visible_index=%s visible_lag=%s same_cycle=%d one_cycle=%d\n', ...
        stringify_numeric(result.visible_index), stringify_numeric(result.visible_lag), ...
        result.same_cycle_visible, result.one_cycle_visible);
    print_permutation_results(permutationResults);
end

    function permutationResults = brute_force_input_mappings()
        inputNames = {'wr_addr_u8', 'din_u8', 'we_bool', 'rd_addr_u8'};
        orderings = perms(1:4);
        permutationResults = repmat(struct( ...
            'mapping', '', ...
            'valid_mapping', false, ...
            'visible', false, ...
            'visible_index', NaN, ...
            'error_message', '', ...
            'dout_values', []), [1, size(orderings, 1)]);

        for i = 1:size(orderings, 1)
            perm = orderings(i, :);
            mdlName = ['tmp_stage2_simple_dual_port_ram_perm_' num2str(i)];
            if bdIsLoaded(mdlName)
                close_system(mdlName, 0);
            end
            new_system(mdlName);
            cleanup = onCleanup(@()safe_close_model(mdlName)); %#ok<NASGU>
            set_param(mdlName, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
                'FixedStep', '1', 'StopTime', '6');

            add_block('simulink/Sources/From Workspace', [mdlName '/wr_addr_src'], 'Position', [30, 40, 120, 60]);
            add_block('simulink/Sources/From Workspace', [mdlName '/din_src'], 'Position', [30, 90, 120, 110]);
            add_block('simulink/Sources/From Workspace', [mdlName '/we_src'], 'Position', [30, 140, 120, 160]);
            add_block('simulink/Sources/From Workspace', [mdlName '/rd_addr_src'], 'Position', [30, 190, 120, 210]);
            add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/wr_addr_u8'], ...
                'OutDataTypeStr', 'uint8', 'Position', [160, 40, 220, 60]);
            add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/din_u8'], ...
                'OutDataTypeStr', 'uint8', 'Position', [160, 90, 220, 110]);
            add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/we_bool'], ...
                'OutDataTypeStr', 'boolean', 'Position', [160, 140, 220, 160]);
            add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/rd_addr_u8'], ...
                'OutDataTypeStr', 'uint8', 'Position', [160, 190, 220, 210]);
            add_block('hdlsllib/HDL RAMs/Simple Dual Port RAM', [mdlName '/ram'], ...
                'ram_size', '8', 'Position', [270, 65, 360, 205]);
            add_block('simulink/Signal Attributes/Data Type Conversion', [mdlName '/dout_double'], ...
                'OutDataTypeStr', 'double', 'Position', [400, 120, 470, 140]);
            add_block('simulink/Sinks/Out1', [mdlName '/dout_out'], 'Position', [510, 120, 540, 134]);

            add_line(mdlName, 'wr_addr_src/1', 'wr_addr_u8/1', 'autorouting', 'on');
            add_line(mdlName, 'din_src/1', 'din_u8/1', 'autorouting', 'on');
            add_line(mdlName, 'we_src/1', 'we_bool/1', 'autorouting', 'on');
            add_line(mdlName, 'rd_addr_src/1', 'rd_addr_u8/1', 'autorouting', 'on');
            for p = 1:4
                add_line(mdlName, [inputNames{perm(p)} '/1'], ['ram/' num2str(p)], 'autorouting', 'on');
            end
            add_line(mdlName, 'ram/1', 'dout_double/1', 'autorouting', 'on');
            add_line(mdlName, 'dout_double/1', 'dout_out/1', 'autorouting', 'on');

            assignin('base', 'wr_addr_trace', [0 3; 1 3; 2 3; 3 3; 4 3; 5 3; 6 3]);
            assignin('base', 'din_trace', [0 0; 1 55; 2 0; 3 0; 4 0; 5 0; 6 0]);
            assignin('base', 'we_trace', [0 0; 1 1; 2 0; 3 0; 4 0; 5 0; 6 0]);
            assignin('base', 'rd_addr_trace', [0 3; 1 3; 2 3; 3 3; 4 3; 5 3; 6 3]);
            set_param([mdlName '/wr_addr_src'], 'VariableName', 'wr_addr_trace');
            set_param([mdlName '/din_src'], 'VariableName', 'din_trace');
            set_param([mdlName '/we_src'], 'VariableName', 'we_trace');
            set_param([mdlName '/rd_addr_src'], 'VariableName', 'rd_addr_trace');

            permutationResults(i).mapping = strjoin(inputNames(perm), ' -> ');
            try
                simOut = sim(mdlName, 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
                    'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
                yout = simOut.get('yout');
                doutValues = double(extract_signal(yout, 'dout_out'));
                hitIndex = find(abs(doutValues - 55) < 1e-9, 1, 'first');
                permutationResults(i).valid_mapping = true;
                permutationResults(i).visible = ~isempty(hitIndex);
                permutationResults(i).visible_index = iff_empty_nan(hitIndex);
                permutationResults(i).dout_values = doutValues(:).';
            catch ME
                permutationResults(i).valid_mapping = false;
                permutationResults(i).visible = false;
                permutationResults(i).visible_index = NaN;
                permutationResults(i).error_message = ME.message;
                permutationResults(i).dout_values = [];
            end
        end
    end

    function print_permutation_results(permutationResults)
        validMappings = permutationResults([permutationResults.valid_mapping]);
        fprintf('  valid_mappings=%d total_mappings=%d\n', numel(validMappings), numel(permutationResults));
        visibleHits = validMappings([validMappings.visible]);
        if isempty(visibleHits)
            fprintf('  visible_mappings=none\n');
            return;
        end
        for i = 1:numel(visibleHits)
            fprintf('  mapping=%s visible_index=%s dout_values=%s\n', ...
                visibleHits(i).mapping, stringify_numeric(visibleHits(i).visible_index), ...
                mat2str(visibleHits(i).dout_values));
        end
    end

    function value = iff_empty_nan(value)
        if isempty(value)
            value = NaN;
        end
    end

function text = stringify_numeric(value)
    if isempty(value) || any(isnan(value))
        text = 'NaN';
    else
        text = num2str(value);
    end
end

function safe_close_model(mdlName)
    if bdIsLoaded(mdlName)
        close_system(mdlName, 0);
    end
end

function values = extract_signal(yout, name)
    for i = 1:yout.numElements
        sig = yout.get(i);
        sigName = string(sig.Name);
        blockPath = string('');
        try
            blockPath = string(sig.BlockPath.getBlock(1));
        catch
        end
        if sigName == string(name) || endsWith(blockPath, "/" + string(name))
            values = sig.Values.Data;
            return;
        end
    end
    error('run_stage2_simple_dual_port_ram_semantics_smoke:MissingSignal', ...
        'Signal not found in Dataset: %s', name);
end