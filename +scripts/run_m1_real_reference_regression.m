function run_m1_real_reference_regression(varargin)
% Package wrapper for scripts.run_m1_real_reference_regression(...)

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(rootDir, 'scripts'));
    run_m1_real_reference_regression(varargin{:});
end
