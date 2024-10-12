// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xrelu_conv_2d.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRelu_conv_2d_CfgInitialize(XRelu_conv_2d *InstancePtr, XRelu_conv_2d_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRelu_conv_2d_Start(XRelu_conv_2d *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRelu_conv_2d_IsDone(XRelu_conv_2d *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRelu_conv_2d_IsIdle(XRelu_conv_2d *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRelu_conv_2d_IsReady(XRelu_conv_2d *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRelu_conv_2d_Continue(XRelu_conv_2d *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XRelu_conv_2d_EnableAutoRestart(XRelu_conv_2d *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRelu_conv_2d_DisableAutoRestart(XRelu_conv_2d *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_AP_CTRL, 0);
}

void XRelu_conv_2d_Set_image_r(XRelu_conv_2d *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IMAGE_R_DATA, (u32)(Data));
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IMAGE_R_DATA + 4, (u32)(Data >> 32));
}

u64 XRelu_conv_2d_Get_image_r(XRelu_conv_2d *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IMAGE_R_DATA);
    Data += (u64)XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IMAGE_R_DATA + 4) << 32;
    return Data;
}

void XRelu_conv_2d_Set_weights(XRelu_conv_2d *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XRelu_conv_2d_Get_weights(XRelu_conv_2d *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XRelu_conv_2d_Set_biases(XRelu_conv_2d *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_BIASES_DATA, (u32)(Data));
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_BIASES_DATA + 4, (u32)(Data >> 32));
}

u64 XRelu_conv_2d_Get_biases(XRelu_conv_2d *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_BIASES_DATA);
    Data += (u64)XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_BIASES_DATA + 4) << 32;
    return Data;
}

void XRelu_conv_2d_Set_output_r(XRelu_conv_2d *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_OUTPUT_R_DATA, (u32)(Data));
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_OUTPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XRelu_conv_2d_Get_output_r(XRelu_conv_2d *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_OUTPUT_R_DATA);
    Data += (u64)XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_OUTPUT_R_DATA + 4) << 32;
    return Data;
}

void XRelu_conv_2d_InterruptGlobalEnable(XRelu_conv_2d *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_GIE, 1);
}

void XRelu_conv_2d_InterruptGlobalDisable(XRelu_conv_2d *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_GIE, 0);
}

void XRelu_conv_2d_InterruptEnable(XRelu_conv_2d *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IER);
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IER, Register | Mask);
}

void XRelu_conv_2d_InterruptDisable(XRelu_conv_2d *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IER);
    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRelu_conv_2d_InterruptClear(XRelu_conv_2d *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRelu_conv_2d_WriteReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_ISR, Mask);
}

u32 XRelu_conv_2d_InterruptGetEnabled(XRelu_conv_2d *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_IER);
}

u32 XRelu_conv_2d_InterruptGetStatus(XRelu_conv_2d *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRelu_conv_2d_ReadReg(InstancePtr->Control_BaseAddress, XRELU_CONV_2D_CONTROL_ADDR_ISR);
}

