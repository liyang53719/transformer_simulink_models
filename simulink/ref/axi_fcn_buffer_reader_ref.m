function [data_out, dvalid_out, rd_addr] = axi_fcn_buffer_reader_ref(rd_data, wr_ready, start, imageWidth)
%AXI_FCN_BUFFER_READER_REF Extracted from soc_image_rotation_fpga chart_43.xml

persistent burst_count rstate dvalid_d;
if isempty(burst_count), burst_count = uint16(0); end
if isempty(dvalid_d), dvalid_d = false; end

IDLE = uint32(0);
WAIT_WREADY = uint32(1);
DATA_COUNT = uint32(2);
if isempty(rstate), rstate = IDLE; end

data_out = rd_data;
dvalid_out = dvalid_d;
rd_addr = burst_count;

switch rstate
    case IDLE
        dvalid = false;
        burst_count = uint16(0);
        if start
            rstate = WAIT_WREADY;
        end
    case WAIT_WREADY
        dvalid = false;
        if wr_ready
            rstate = DATA_COUNT;
        end
    case DATA_COUNT
        if wr_ready
            dvalid = true;
            if burst_count == uint16(imageWidth - 1)
                rstate = IDLE;
            else
                burst_count = burst_count + uint16(1);
            end
        else
            dvalid = false;
        end
    otherwise
        dvalid = false;
        rstate = IDLE;
end

dvalid_d = dvalid;
end
