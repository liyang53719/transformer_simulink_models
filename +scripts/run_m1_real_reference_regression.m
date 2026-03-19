function varargout = run_m1_real_reference_regression(varargin)
% Package wrapper for scripts.run_m1_real_reference_regression(...)

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(rootDir, 'scripts'));
    [varargout{1:nargout}] = run_m1_real_reference_regression(varargin{:});
end
