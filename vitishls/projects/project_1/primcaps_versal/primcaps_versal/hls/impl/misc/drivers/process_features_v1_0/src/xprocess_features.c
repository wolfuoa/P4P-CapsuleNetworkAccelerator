// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xprocess_features.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XProcess_features_CfgInitialize(XProcess_features *InstancePtr, XProcess_features_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XProcess_features_Start(XProcess_features *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL) & 0x80;
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XProcess_features_IsDone(XProcess_features *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XProcess_features_IsIdle(XProcess_features *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XProcess_features_IsReady(XProcess_features *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XProcess_features_Continue(XProcess_features *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL) & 0x80;
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XProcess_features_EnableAutoRestart(XProcess_features *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XProcess_features_DisableAutoRestart(XProcess_features *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_AP_CTRL, 0);
}

void XProcess_features_Set_input_r(XProcess_features *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XProcess_features_Get_input_r(XProcess_features *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XProcess_features_Set_weights(XProcess_features *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XProcess_features_Get_weights(XProcess_features *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XProcess_features_Set_biases(XProcess_features *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_BIASES_DATA, (u32)(Data));
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_BIASES_DATA + 4, (u32)(Data >> 32));
}

u64 XProcess_features_Get_biases(XProcess_features *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_BIASES_DATA);
    Data += (u64)XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_BIASES_DATA + 4) << 32;
    return Data;
}

void XProcess_features_Set_output_r(XProcess_features *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_OUTPUT_R_DATA, (u32)(Data));
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_OUTPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XProcess_features_Get_output_r(XProcess_features *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_OUTPUT_R_DATA);
    Data += (u64)XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_OUTPUT_R_DATA + 4) << 32;
    return Data;
}

void XProcess_features_InterruptGlobalEnable(XProcess_features *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_GIE, 1);
}

void XProcess_features_InterruptGlobalDisable(XProcess_features *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_GIE, 0);
}

void XProcess_features_InterruptEnable(XProcess_features *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_IER);
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_IER, Register | Mask);
}

void XProcess_features_InterruptDisable(XProcess_features *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_IER);
    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_IER, Register & (~Mask));
}

void XProcess_features_InterruptClear(XProcess_features *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_features_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_ISR, Mask);
}

u32 XProcess_features_InterruptGetEnabled(XProcess_features *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_IER);
}

u32 XProcess_features_InterruptGetStatus(XProcess_features *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XProcess_features_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_FEATURES_CONTROL_ADDR_ISR);
}

