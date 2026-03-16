function [done_out, busy_out, state_next] = ctrl_fsm_step(mode_decode, start, stop_req, state_in)
%CTRL_FSM_STEP External-maintained FSM step for ctrl_fsm_u MATLAB Function block.
%   state_in/state_next: 0=IDLE, 1=BUSY (double-typed for MATLAB Function compatibility)

%#codegen

    state = double(state_in);
    done_out = false;

    if stop_req
        state = 0;
        done_out = true;
    else
        switch state
            case 0
                if start
                    state = 1;
                end
            otherwise
                if ~mode_decode
                    % Prefill path modeled as single-step completion in current stage.
                    done_out = true;
                    state = 0;
                end
        end
    end

    busy_out = (state == 1);
    state_next = state;
end
