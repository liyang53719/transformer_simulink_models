function assert_stage2_manual_model_policy(buildModel, callerName)
%ASSERT_STAGE2_MANUAL_MODEL_POLICY Enforce manual-only DUT generation for stage2 verification.

    if nargin < 1
        buildModel = false;
    end
    if nargin < 2 || strlength(string(callerName)) == 0
        callerName = "stage2 verification";
    end

    if logical(buildModel)
        error('assert_stage2_manual_model_policy:BuildForbidden', ...
            ['%s forbids BuildModel=true. Manually regenerate simulink/models/qwen2_block_top.slx ' ...
             'before validation, then rerun verification with BuildModel=false.'], ...
            char(string(callerName)));
    end
end