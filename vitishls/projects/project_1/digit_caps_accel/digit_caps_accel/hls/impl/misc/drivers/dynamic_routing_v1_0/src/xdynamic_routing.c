// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xdynamic_routing.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XDynamic_routing_CfgInitialize(XDynamic_routing *InstancePtr, XDynamic_routing_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XDynamic_routing_Start(XDynamic_routing *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL) & 0x80;
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XDynamic_routing_IsDone(XDynamic_routing *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XDynamic_routing_IsIdle(XDynamic_routing *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XDynamic_routing_IsReady(XDynamic_routing *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XDynamic_routing_Continue(XDynamic_routing *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL) & 0x80;
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XDynamic_routing_EnableAutoRestart(XDynamic_routing *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XDynamic_routing_DisableAutoRestart(XDynamic_routing *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_AP_CTRL, 0);
}

void XDynamic_routing_Set_input_r(XDynamic_routing *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XDynamic_routing_Get_input_r(XDynamic_routing *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XDynamic_routing_Set_weights(XDynamic_routing *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XDynamic_routing_Get_weights(XDynamic_routing *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XDynamic_routing_Set_prediction(XDynamic_routing *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_PREDICTION_DATA, (u32)(Data));
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_PREDICTION_DATA + 4, (u32)(Data >> 32));
}

u64 XDynamic_routing_Get_prediction(XDynamic_routing *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_PREDICTION_DATA);
    Data += (u64)XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_PREDICTION_DATA + 4) << 32;
    return Data;
}

void XDynamic_routing_InterruptGlobalEnable(XDynamic_routing *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_GIE, 1);
}

void XDynamic_routing_InterruptGlobalDisable(XDynamic_routing *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_GIE, 0);
}

void XDynamic_routing_InterruptEnable(XDynamic_routing *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_IER);
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_IER, Register | Mask);
}

void XDynamic_routing_InterruptDisable(XDynamic_routing *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_IER);
    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_IER, Register & (~Mask));
}

void XDynamic_routing_InterruptClear(XDynamic_routing *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDynamic_routing_WriteReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_ISR, Mask);
}

u32 XDynamic_routing_InterruptGetEnabled(XDynamic_routing *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_IER);
}

u32 XDynamic_routing_InterruptGetStatus(XDynamic_routing *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDynamic_routing_ReadReg(InstancePtr->Control_BaseAddress, XDYNAMIC_ROUTING_CONTROL_ADDR_ISR);
}

