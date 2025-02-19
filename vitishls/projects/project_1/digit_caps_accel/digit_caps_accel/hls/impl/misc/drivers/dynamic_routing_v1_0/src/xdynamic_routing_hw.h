// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 4  - ap_continue (Read/Write/SC)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of input_r
//        bit 31~0 - input_r[31:0] (Read/Write)
// 0x14 : Data signal of input_r
//        bit 31~0 - input_r[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of weights
//        bit 31~0 - weights[31:0] (Read/Write)
// 0x20 : Data signal of weights
//        bit 31~0 - weights[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of prediction
//        bit 31~0 - prediction[31:0] (Read/Write)
// 0x2c : Data signal of prediction
//        bit 31~0 - prediction[63:32] (Read/Write)
// 0x30 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL         0x00
#define XDYNAMIC_ROUTING_CONTROL_ADDR_GIE             0x04
#define XDYNAMIC_ROUTING_CONTROL_ADDR_IER             0x08
#define XDYNAMIC_ROUTING_CONTROL_ADDR_ISR             0x0c
#define XDYNAMIC_ROUTING_CONTROL_ADDR_INPUT_R_DATA    0x10
#define XDYNAMIC_ROUTING_CONTROL_BITS_INPUT_R_DATA    64
#define XDYNAMIC_ROUTING_CONTROL_ADDR_WEIGHTS_DATA    0x1c
#define XDYNAMIC_ROUTING_CONTROL_BITS_WEIGHTS_DATA    64
#define XDYNAMIC_ROUTING_CONTROL_ADDR_PREDICTION_DATA 0x28
#define XDYNAMIC_ROUTING_CONTROL_BITS_PREDICTION_DATA 64

