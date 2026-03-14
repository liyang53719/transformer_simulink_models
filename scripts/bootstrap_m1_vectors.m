function bootstrap_m1_vectors(rootDir, options)
%BOOTSTRAP_M1_VECTORS Create deterministic M1 placeholder test vectors.

    if nargin < 1 || strlength(string(rootDir)) == 0
        rootDir = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || ~isstruct(options)
        options = struct();
    end

    tvDir = fullfile(rootDir, 'verification', 'testvectors');
    if ~exist(tvDir, 'dir')
        mkdir(tvDir);
    end

    rng(42);

    hiddenSize = double(getFieldOr(options, 'HiddenSize', 16));
    tokensPrefill = double(getFieldOr(options, 'TokensPrefill', 4));
    decodeKvLen = double(getFieldOr(options, 'DecodeKvLen', 3));

    cfg = struct('ResidualScale', single(0.1), 'KvMixScale', single(0.05));

    % Prefill case
    inHidden = single(randn(hiddenSize, tokensPrefill));
    inResidual = single(randn(hiddenSize, tokensPrefill));
    [outHidden, outKV] = qwen2_block_ref_placeholder(inHidden, inResidual, single([]), cfg);

    meta = struct('case_id', 'block_prefill_case01', ...
                  'seed', 42, ...
                  'mode', 'prefill', ...
                  'seq_len', tokensPrefill, ...
                  'token_pos', 1);
    input = struct('hidden', inHidden, 'residual', inResidual, 'kv_cache', single([]));
    golden = struct('output_hidden', outHidden, 'output_kv', outKV);
    save(fullfile(tvDir, 'block_prefill_case01.mat'), 'meta', 'input', 'golden');

    % Decode case
    decodePos = 5;
    inHidden = single(randn(hiddenSize, 1));
    inResidual = single(randn(hiddenSize, 1));
    kvRead = single(randn(hiddenSize, decodeKvLen));
    [outHidden, outKV] = qwen2_block_ref_placeholder(inHidden, inResidual, kvRead, cfg);

    meta = struct('case_id', 'block_decode_case01', ...
                  'seed', 42, ...
                  'mode', 'decode', ...
                  'seq_len', 1, ...
                  'token_pos', decodePos);
    input = struct('hidden', inHidden, 'residual', inResidual, 'kv_cache', kvRead);
    golden = struct('output_hidden', outHidden, 'output_kv', outKV);
    save(fullfile(tvDir, 'block_decode_case01.mat'), 'meta', 'input', 'golden');

    fprintf('Generated vectors in %s\n', tvDir);
end

function out = getFieldOr(s, name, defaultValue)
    if isfield(s, name)
        out = s.(name);
    else
        out = defaultValue;
    end
end
