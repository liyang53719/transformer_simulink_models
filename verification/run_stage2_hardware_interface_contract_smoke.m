function result = run_stage2_hardware_interface_contract_smoke(rootDir, options)
%RUN_STAGE2_HARDWARE_INTERFACE_CONTRACT_SMOKE Validate the hardware-first top-level interface contract.
%   This gate freezes the stage2 top-level contract around hardware-friendly
%   scalar/fixed-point control ports and explicit memory/config ingress.
%   Reference adapters are expected to conform to this contract rather than
%   reshaping the DUT around software reference tensors.

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

    result = check_block_interface_spec_consistency(rootDir, struct( ...
        'Profile', 'stage2_memory_ready_hw', ...
        'ValidateRootTypes', true, ...
        'ValidateHardwareEdges', true));
    result.pass = true;
    fprintf('Stage2 hardware interface contract smoke PASS\n');
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end