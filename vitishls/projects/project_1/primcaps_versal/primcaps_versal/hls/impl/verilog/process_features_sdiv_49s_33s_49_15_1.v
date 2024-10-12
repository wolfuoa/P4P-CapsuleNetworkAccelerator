// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
`timescale 1ns/1ps

//This is Divider IP VLOG RTL
module process_features_sdiv_49s_33s_49_15_1
#(parameter
ID=1,
NUM_STAGE=2,
din0_WIDTH=16,
din1_WIDTH=16,
dout_WIDTH=16
)(
clk,reset,ce,din0,din1,dout
);
//--- input/output ---

input clk,reset,ce;

input  [din0_WIDTH-1:0] din0;
input  [din1_WIDTH-1:0]  din1;
output [dout_WIDTH-1:0]  dout;
//--- local signal ---
function integer calcDataWidth;
   input integer x;
   integer y;
begin
   y = 8;
   while (y < x) y = y + 8;
   calcDataWidth = y;
end
endfunction

localparam dividendUpper = calcDataWidth(din0_WIDTH);
localparam divisorUpper  = calcDataWidth(din1_WIDTH);
localparam IPOutput      = dividendUpper + divisorUpper;
wire [IPOutput-1:0] IPOut;
wire [dividendUpper-1:0] IPDividend;
wire [divisorUpper-1:0] IPDivisor;
wire dividend_valid;
wire divisor_valid;
wire output_valid;
wire [din0_WIDTH-1:0] realQuotient;
wire [din1_WIDTH-1:0] realRemainder;

//--- instantiation ---
process_features_sdiv_49s_33s_49_15_1_ip process_features_sdiv_49s_33s_49_15_1_ip_u (

.aclk(clk),
.aclken(ce),

.s_axis_dividend_tvalid(dividend_valid),
.s_axis_dividend_tdata(IPDividend),
.s_axis_divisor_tvalid(divisor_valid),
.s_axis_divisor_tdata(IPDivisor),
.m_axis_dout_tvalid(output_valid),
.m_axis_dout_tdata(IPOut)
);
//--- assignment ---
assign dividend_valid = 1'b1;
assign divisor_valid = 1'b1;
assign IPDividend = din0;
assign IPDivisor = din1;
assign realQuotient  = IPOut[divisorUpper+din0_WIDTH-1:divisorUpper];
assign realRemainder = IPOut[din1_WIDTH-1:0];
assign dout = realQuotient;

endmodule
