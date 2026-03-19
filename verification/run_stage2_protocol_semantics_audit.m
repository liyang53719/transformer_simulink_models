function result = run_stage2_protocol_semantics_audit(rootDir, options)
%RUN_STAGE2_PROTOCOL_SEMANTICS_AUDIT Reject fake stage2 protocol passthroughs.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    buildModel = getFieldOr(options, 'BuildModel', false);
    assert_stage2_manual_model_policy(buildModel, mfilename);

    mdlPath = fullfile(rootDir, 'simulink', 'models', 'qwen2_block_top.slx');
    [~, mdlName] = fileparts(mdlPath);

    load_system(mdlPath);
    set_param(mdlName, 'SimulationCommand', 'update');

    forbid_direct_edge(mdlName, 'in_valid/1', 'out_valid/1');
    forbid_direct_edge(mdlName, 'out_ready/1', 'in_ready/1');
    forbid_direct_edge(mdlName, 'eos_in/1', 'eos_out/1');

    require_edge(mdlName, 'ctrl_fsm_u/1', 'out_valid/1');
    require_edge(mdlName, 'stage2_in_ready_not/1', 'in_ready/1');
    require_edge(mdlName, 'stage2_eos_gate/1', 'eos_out/1');
    require_edge(mdlName, 'axi_master_wr_u/1', 'kv_cache_wr_data/1');
    require_edge(mdlName, 'axi_master_wr_u/5', 'ddr_model_if_u/5');

    require_no_block([mdlName '/axi_master_wr_u_unused_out1_term']);
    require_no_block([mdlName '/axi_master_wr_u_unused_out5_term']);

    close_system(mdlName, 0);
    result = struct('pass', true);
    fprintf('Stage2 protocol semantics audit PASS\n');
end

function forbid_direct_edge(sys, src, dst)
    if has_direct_edge(sys, src, dst)
        error('run_stage2_protocol_semantics_audit:FakePassthrough', ...
            'Forbidden direct passthrough still present: %s -> %s', src, dst);
    end
end

function tf = has_direct_edge(sys, src, dst)
    dstParts = split(string(dst), '/');
    dstBlk = [sys '/' char(dstParts(1))];
    dstPort = str2double(dstParts(2));
    phDst = get_param(dstBlk, 'PortHandles');
    if dstPort > numel(phDst.Inport)
        error('run_stage2_protocol_semantics_audit:BadDst', 'Invalid dst port %s', dst);
    end
    ln = get_param(phDst.Inport(dstPort), 'Line');
    if ln == -1
        tf = false;
        return;
    end
    srcH = get_param(ln, 'SrcPortHandle');
    srcFull = string(getfullname(srcH));
    srcParts = split(string(src), '/');
    want = string([sys '/' char(srcParts(1)) '/']);
    tf = startsWith(srcFull, want);
end

function require_edge(sys, src, dst)
    if ~has_direct_edge(sys, src, dst)
        error('run_stage2_protocol_semantics_audit:MissingEdge', ...
            'Missing required edge %s -> %s', src, dst);
    end
end

function require_no_block(path)
    if getSimulinkBlockHandle(path) ~= -1
        error('run_stage2_protocol_semantics_audit:UnexpectedBlock', ...
            'Unexpected block still present: %s', path);
    end
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
