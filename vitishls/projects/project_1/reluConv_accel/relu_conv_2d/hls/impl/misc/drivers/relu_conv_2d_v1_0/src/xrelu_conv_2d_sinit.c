// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xrelu_conv_2d.h"

extern XRelu_conv_2d_Config XRelu_conv_2d_ConfigTable[];

#ifdef SDT
XRelu_conv_2d_Config *XRelu_conv_2d_LookupConfig(UINTPTR BaseAddress) {
	XRelu_conv_2d_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XRelu_conv_2d_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XRelu_conv_2d_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XRelu_conv_2d_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRelu_conv_2d_Initialize(XRelu_conv_2d *InstancePtr, UINTPTR BaseAddress) {
	XRelu_conv_2d_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRelu_conv_2d_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRelu_conv_2d_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XRelu_conv_2d_Config *XRelu_conv_2d_LookupConfig(u16 DeviceId) {
	XRelu_conv_2d_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XRELU_CONV_2D_NUM_INSTANCES; Index++) {
		if (XRelu_conv_2d_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRelu_conv_2d_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRelu_conv_2d_Initialize(XRelu_conv_2d *InstancePtr, u16 DeviceId) {
	XRelu_conv_2d_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRelu_conv_2d_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRelu_conv_2d_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

