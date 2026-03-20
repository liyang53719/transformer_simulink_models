function [done_out, busy_out, state_next, count_next] = ctrl_fsm_step(mode_decode, start, stop_req, seq_len, out_fire, state_in, count_in)
%CTRL_FSM_STEP External-maintained FSM step for ctrl_fsm_u MATLAB Function block.
%   state_in/state_next: 0=IDLE, 1=BUSY (double-typed for MATLAB Function compatibility)

%#codegen

    state = double(state_in);
    count = double(count_in);
    done_out = false;

    if logical(mode_decode)
        target_count = 1;
    else
        target_count = double(seq_len);
    end
    if target_count < 1
        target_count = 1;
    end

    if stop_req
        state = 0;
        count = 0;
        done_out = true;
    else
        switch state
            case 0
                if start
                    state = 1;
                    count = 0;
                end
            otherwise
                if out_fire
                    count = count + 1;
                end
                if count >= target_count
                    done_out = true;
                    state = 0;
                    count = 0;
                end
        end
    end

    busy_out = (state == 1);
    state_next = state;
    count_next = count;
end
