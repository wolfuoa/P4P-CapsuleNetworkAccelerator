// ==============================================================
// Generated by Vitis HLS v2024.1
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module process_features_squash_Pipeline_6 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem3_AWVALID,
        m_axi_gmem3_AWREADY,
        m_axi_gmem3_AWADDR,
        m_axi_gmem3_AWID,
        m_axi_gmem3_AWLEN,
        m_axi_gmem3_AWSIZE,
        m_axi_gmem3_AWBURST,
        m_axi_gmem3_AWLOCK,
        m_axi_gmem3_AWCACHE,
        m_axi_gmem3_AWPROT,
        m_axi_gmem3_AWQOS,
        m_axi_gmem3_AWREGION,
        m_axi_gmem3_AWUSER,
        m_axi_gmem3_WVALID,
        m_axi_gmem3_WREADY,
        m_axi_gmem3_WDATA,
        m_axi_gmem3_WSTRB,
        m_axi_gmem3_WLAST,
        m_axi_gmem3_WID,
        m_axi_gmem3_WUSER,
        m_axi_gmem3_ARVALID,
        m_axi_gmem3_ARREADY,
        m_axi_gmem3_ARADDR,
        m_axi_gmem3_ARID,
        m_axi_gmem3_ARLEN,
        m_axi_gmem3_ARSIZE,
        m_axi_gmem3_ARBURST,
        m_axi_gmem3_ARLOCK,
        m_axi_gmem3_ARCACHE,
        m_axi_gmem3_ARPROT,
        m_axi_gmem3_ARQOS,
        m_axi_gmem3_ARREGION,
        m_axi_gmem3_ARUSER,
        m_axi_gmem3_RVALID,
        m_axi_gmem3_RREADY,
        m_axi_gmem3_RDATA,
        m_axi_gmem3_RLAST,
        m_axi_gmem3_RID,
        m_axi_gmem3_RFIFONUM,
        m_axi_gmem3_RUSER,
        m_axi_gmem3_RRESP,
        m_axi_gmem3_BVALID,
        m_axi_gmem3_BREADY,
        m_axi_gmem3_BRESP,
        m_axi_gmem3_BID,
        m_axi_gmem3_BUSER,
        sext_ln305,
        output_buffer_address0,
        output_buffer_ce0,
        output_buffer_q0,
        output_buffer_1_address0,
        output_buffer_1_ce0,
        output_buffer_1_q0,
        output_buffer_2_address0,
        output_buffer_2_ce0,
        output_buffer_2_q0,
        output_buffer_3_address0,
        output_buffer_3_ce0,
        output_buffer_3_q0
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem3_AWVALID;
input   m_axi_gmem3_AWREADY;
output  [63:0] m_axi_gmem3_AWADDR;
output  [0:0] m_axi_gmem3_AWID;
output  [31:0] m_axi_gmem3_AWLEN;
output  [2:0] m_axi_gmem3_AWSIZE;
output  [1:0] m_axi_gmem3_AWBURST;
output  [1:0] m_axi_gmem3_AWLOCK;
output  [3:0] m_axi_gmem3_AWCACHE;
output  [2:0] m_axi_gmem3_AWPROT;
output  [3:0] m_axi_gmem3_AWQOS;
output  [3:0] m_axi_gmem3_AWREGION;
output  [0:0] m_axi_gmem3_AWUSER;
output   m_axi_gmem3_WVALID;
input   m_axi_gmem3_WREADY;
output  [511:0] m_axi_gmem3_WDATA;
output  [63:0] m_axi_gmem3_WSTRB;
output   m_axi_gmem3_WLAST;
output  [0:0] m_axi_gmem3_WID;
output  [0:0] m_axi_gmem3_WUSER;
output   m_axi_gmem3_ARVALID;
input   m_axi_gmem3_ARREADY;
output  [63:0] m_axi_gmem3_ARADDR;
output  [0:0] m_axi_gmem3_ARID;
output  [31:0] m_axi_gmem3_ARLEN;
output  [2:0] m_axi_gmem3_ARSIZE;
output  [1:0] m_axi_gmem3_ARBURST;
output  [1:0] m_axi_gmem3_ARLOCK;
output  [3:0] m_axi_gmem3_ARCACHE;
output  [2:0] m_axi_gmem3_ARPROT;
output  [3:0] m_axi_gmem3_ARQOS;
output  [3:0] m_axi_gmem3_ARREGION;
output  [0:0] m_axi_gmem3_ARUSER;
input   m_axi_gmem3_RVALID;
output   m_axi_gmem3_RREADY;
input  [511:0] m_axi_gmem3_RDATA;
input   m_axi_gmem3_RLAST;
input  [0:0] m_axi_gmem3_RID;
input  [12:0] m_axi_gmem3_RFIFONUM;
input  [0:0] m_axi_gmem3_RUSER;
input  [1:0] m_axi_gmem3_RRESP;
input   m_axi_gmem3_BVALID;
output   m_axi_gmem3_BREADY;
input  [1:0] m_axi_gmem3_BRESP;
input  [0:0] m_axi_gmem3_BID;
input  [0:0] m_axi_gmem3_BUSER;
input  [57:0] sext_ln305;
output  [11:0] output_buffer_address0;
output   output_buffer_ce0;
input  [31:0] output_buffer_q0;
output  [11:0] output_buffer_1_address0;
output   output_buffer_1_ce0;
input  [31:0] output_buffer_1_q0;
output  [11:0] output_buffer_2_address0;
output   output_buffer_2_ce0;
input  [31:0] output_buffer_2_q0;
output  [11:0] output_buffer_3_address0;
output   output_buffer_3_ce0;
input  [31:0] output_buffer_3_q0;

reg ap_idle;
reg m_axi_gmem3_WVALID;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_idle_pp0;
reg   [0:0] empty_64_reg_347;
reg   [0:0] empty_64_reg_347_pp0_iter1_reg;
reg    ap_block_state3_io;
reg    ap_block_pp0_stage0_subdone;
wire   [0:0] exitcond1_fu_177_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem3_blk_n_W;
wire    ap_block_pp0_stage0;
reg    ap_block_pp0_stage0_11001;
wire   [1:0] empty_63_fu_193_p1;
reg   [1:0] empty_63_reg_322;
wire   [0:0] empty_64_fu_215_p2;
wire   [31:0] tmp_6_fu_226_p11;
reg   [31:0] tmp_6_reg_352;
wire   [63:0] p_cast29_fu_207_p1;
wire    ap_block_pp0_stage0_01001;
reg   [479:0] shiftreg_fu_86;
wire   [479:0] empty_61_fu_286_p3;
wire    ap_loop_init;
reg   [13:0] loop_index_fu_90;
wire   [13:0] empty_fu_183_p2;
reg   [13:0] ap_sig_allocacmp_loop_index_load;
reg    output_buffer_ce0_local;
reg    output_buffer_1_ce0_local;
reg    output_buffer_2_ce0_local;
reg    output_buffer_3_ce0_local;
wire   [11:0] tmp_s_fu_197_p4;
wire   [3:0] empty_62_fu_189_p1;
wire   [31:0] tmp_6_fu_226_p9;
wire   [447:0] tmp_13_fu_269_p4;
wire   [479:0] tmp_14_fu_279_p3;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_fsm;
wire    ap_enable_pp0;
wire    ap_start_int;
wire    ap_ready_sig;
wire    ap_done_sig;
wire   [1:0] tmp_6_fu_226_p1;
wire   [1:0] tmp_6_fu_226_p3;
wire  signed [1:0] tmp_6_fu_226_p5;
wire  signed [1:0] tmp_6_fu_226_p7;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 shiftreg_fu_86 = 480'd0;
#0 loop_index_fu_90 = 14'd0;
#0 ap_done_reg = 1'b0;
end

(* dissolve_hierarchy = "yes" *) process_features_sparsemux_9_2_32_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 2'h0 ),
    .din0_WIDTH( 32 ),
    .CASE1( 2'h1 ),
    .din1_WIDTH( 32 ),
    .CASE2( 2'h2 ),
    .din2_WIDTH( 32 ),
    .CASE3( 2'h3 ),
    .din3_WIDTH( 32 ),
    .def_WIDTH( 32 ),
    .sel_WIDTH( 2 ),
    .dout_WIDTH( 32 ))
sparsemux_9_2_32_1_1_U690(
    .din0(output_buffer_q0),
    .din1(output_buffer_1_q0),
    .din2(output_buffer_2_q0),
    .din3(output_buffer_3_q0),
    .def(tmp_6_fu_226_p9),
    .sel(empty_63_reg_322),
    .dout(tmp_6_fu_226_p11)
);

process_features_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready_sig),
    .ap_done(ap_done_sig),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter1 <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((ap_enable_reg_pp0_iter0 == 1'b1) & (exitcond1_fu_177_p2 == 1'd0))) begin
            loop_index_fu_90 <= empty_fu_183_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            loop_index_fu_90 <= 14'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp0_stage0_11001)) begin
        if (((1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
            shiftreg_fu_86 <= 480'd0;
        end else if ((ap_enable_reg_pp0_iter2 == 1'b1)) begin
            shiftreg_fu_86 <= empty_61_fu_286_p3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        empty_63_reg_322 <= empty_63_fu_193_p1;
        empty_64_reg_347 <= empty_64_fu_215_p2;
        empty_64_reg_347_pp0_iter1_reg <= empty_64_reg_347;
        tmp_6_reg_352 <= tmp_6_fu_226_p11;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (exitcond1_fu_177_p2 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (ap_idle_pp0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_loop_index_load = 14'd0;
    end else begin
        ap_sig_allocacmp_loop_index_load = loop_index_fu_90;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (empty_64_reg_347_pp0_iter1_reg == 1'd1) & (ap_enable_reg_pp0_iter2 == 1'b1))) begin
        gmem3_blk_n_W = m_axi_gmem3_WREADY;
    end else begin
        gmem3_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (empty_64_reg_347_pp0_iter1_reg == 1'd1) & (ap_enable_reg_pp0_iter2 == 1'b1))) begin
        m_axi_gmem3_WVALID = 1'b1;
    end else begin
        m_axi_gmem3_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        output_buffer_1_ce0_local = 1'b1;
    end else begin
        output_buffer_1_ce0_local = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        output_buffer_2_ce0_local = 1'b1;
    end else begin
        output_buffer_2_ce0_local = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        output_buffer_3_ce0_local = 1'b1;
    end else begin
        output_buffer_3_ce0_local = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        output_buffer_ce0_local = 1'b1;
    end else begin
        output_buffer_ce0_local = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_01001 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_enable_reg_pp0_iter2 == 1'b1) & (1'b1 == ap_block_state3_io));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((ap_enable_reg_pp0_iter2 == 1'b1) & (1'b1 == ap_block_state3_io));
end

always @ (*) begin
    ap_block_state3_io = ((empty_64_reg_347_pp0_iter1_reg == 1'd1) & (m_axi_gmem3_WREADY == 1'b0));
end

assign ap_done = ap_done_sig;

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_ready = ap_ready_sig;

assign empty_61_fu_286_p3 = ((empty_64_reg_347_pp0_iter1_reg[0:0] == 1'b1) ? 480'd0 : tmp_14_fu_279_p3);

assign empty_62_fu_189_p1 = ap_sig_allocacmp_loop_index_load[3:0];

assign empty_63_fu_193_p1 = ap_sig_allocacmp_loop_index_load[1:0];

assign empty_64_fu_215_p2 = ((empty_62_fu_189_p1 == 4'd15) ? 1'b1 : 1'b0);

assign empty_fu_183_p2 = (ap_sig_allocacmp_loop_index_load + 14'd1);

assign exitcond1_fu_177_p2 = ((ap_sig_allocacmp_loop_index_load == 14'd9216) ? 1'b1 : 1'b0);

assign m_axi_gmem3_ARADDR = 64'd0;

assign m_axi_gmem3_ARBURST = 2'd0;

assign m_axi_gmem3_ARCACHE = 4'd0;

assign m_axi_gmem3_ARID = 1'd0;

assign m_axi_gmem3_ARLEN = 32'd0;

assign m_axi_gmem3_ARLOCK = 2'd0;

assign m_axi_gmem3_ARPROT = 3'd0;

assign m_axi_gmem3_ARQOS = 4'd0;

assign m_axi_gmem3_ARREGION = 4'd0;

assign m_axi_gmem3_ARSIZE = 3'd0;

assign m_axi_gmem3_ARUSER = 1'd0;

assign m_axi_gmem3_ARVALID = 1'b0;

assign m_axi_gmem3_AWADDR = 64'd0;

assign m_axi_gmem3_AWBURST = 2'd0;

assign m_axi_gmem3_AWCACHE = 4'd0;

assign m_axi_gmem3_AWID = 1'd0;

assign m_axi_gmem3_AWLEN = 32'd0;

assign m_axi_gmem3_AWLOCK = 2'd0;

assign m_axi_gmem3_AWPROT = 3'd0;

assign m_axi_gmem3_AWQOS = 4'd0;

assign m_axi_gmem3_AWREGION = 4'd0;

assign m_axi_gmem3_AWSIZE = 3'd0;

assign m_axi_gmem3_AWUSER = 1'd0;

assign m_axi_gmem3_AWVALID = 1'b0;

assign m_axi_gmem3_BREADY = 1'b0;

assign m_axi_gmem3_RREADY = 1'b0;

assign m_axi_gmem3_WDATA = {{tmp_6_reg_352}, {shiftreg_fu_86}};

assign m_axi_gmem3_WID = 1'd0;

assign m_axi_gmem3_WLAST = 1'b0;

assign m_axi_gmem3_WSTRB = 64'd18446744073709551615;

assign m_axi_gmem3_WUSER = 1'd0;

assign output_buffer_1_address0 = p_cast29_fu_207_p1;

assign output_buffer_1_ce0 = output_buffer_1_ce0_local;

assign output_buffer_2_address0 = p_cast29_fu_207_p1;

assign output_buffer_2_ce0 = output_buffer_2_ce0_local;

assign output_buffer_3_address0 = p_cast29_fu_207_p1;

assign output_buffer_3_ce0 = output_buffer_3_ce0_local;

assign output_buffer_address0 = p_cast29_fu_207_p1;

assign output_buffer_ce0 = output_buffer_ce0_local;

assign p_cast29_fu_207_p1 = tmp_s_fu_197_p4;

assign tmp_13_fu_269_p4 = {{shiftreg_fu_86[479:32]}};

assign tmp_14_fu_279_p3 = {{tmp_6_reg_352}, {tmp_13_fu_269_p4}};

assign tmp_6_fu_226_p9 = 'bx;

assign tmp_s_fu_197_p4 = {{ap_sig_allocacmp_loop_index_load[13:2]}};

endmodule //process_features_squash_Pipeline_6
