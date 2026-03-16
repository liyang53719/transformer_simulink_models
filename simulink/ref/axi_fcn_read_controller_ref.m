function [rd_data_out, rd_dvalid_out, rd_addr, rd_len, rd_avalid, rd_dready] = axi_fcn_read_controller_ref(rd_data, rd_aready, rd_dvalid, start, imageWidth, imageHeight, bytesPerPixel)
%AXI_FCN_READ_CONTROLLER_REF Extracted from soc_image_rotation_fpga chart_55.xml

persistent burst_count line_count rstate;
if isempty(burst_count), burst_count = uint32(0); end
if isempty(line_count), line_count = uint32(0); end

IDLE = uint32(0);
READ_BURST_START = uint32(1);
DATA_COUNT = uint32(2);
if isempty(rstate), rstate = IDLE; end

rd_addr = uint32(imageWidth * bytesPerPixel * line_count);
rd_len = uint32(imageWidth);
rd_dready = true;
rd_data_out = rd_data;
rd_dvalid_out = rd_dvalid;

switch rstate
    case IDLE
        rd_avalid = false;
        burst_count = uint32(0);
        if start
            rstate = READ_BURST_START;
        end
    case READ_BURST_START
        if rd_aready
            rd_avalid = true;
            rstate = DATA_COUNT;
        else
            rd_avalid = false;
        end
    case DATA_COUNT
        rd_avalid = false;
        burst_count = burst_count + uint32(rd_dvalid);
        if burst_count == rd_len
            rstate = IDLE;
            if line_count == uint32(imageHeight - 1)
                line_count = uint32(0);
            else
                line_count = line_count + uint32(1);
            end
        end
    otherwise
        rd_avalid = false;
        rstate = IDLE;
end
end
