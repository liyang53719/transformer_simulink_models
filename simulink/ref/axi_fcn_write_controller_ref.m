function [wr_data_out, wr_addr, wr_len, wr_valid, request_next_line] = axi_fcn_write_controller_ref(wr_data, wr_dvalid, wr_complete, imageWidth, imageHeight, bytesPerPixel)
%AXI_FCN_WRITE_CONTROLLER_REF Extracted from soc_image_rotation_fpga chart_27.xml

persistent burst_count line_count wstate;
if isempty(burst_count), burst_count = uint32(0); end
if isempty(line_count), line_count = uint32(0); end

IDLE = uint32(0);
DATA_COUNT = uint32(1);
if isempty(wstate), wstate = IDLE; end

wr_addr = uint32(imageWidth * bytesPerPixel * (imageHeight - 1) - line_count * imageWidth * bytesPerPixel);
wr_len = uint32(imageWidth);
wr_valid = wr_dvalid;
wr_data_out = wr_data;

switch wstate
    case IDLE
        if wr_dvalid
            wstate = DATA_COUNT;
            burst_count = burst_count + uint32(1);
        end
    case DATA_COUNT
        if burst_count == uint32(imageWidth)
            wstate = IDLE;
            burst_count = uint32(0);
            if line_count == uint32(imageHeight - 1)
                line_count = uint32(0);
            else
                line_count = line_count + uint32(1);
            end
        else
            burst_count = burst_count + uint32(wr_dvalid);
        end
    otherwise
        wstate = IDLE;
end

request_next_line = wr_complete && (line_count ~= 0);
end
