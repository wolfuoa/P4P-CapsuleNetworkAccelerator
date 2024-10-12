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
#include "xprocess_features.h"

extern XProcess_features_Config XProcess_features_ConfigTable[];

#ifdef SDT
XProcess_features_Config *XProcess_features_LookupConfig(UINTPTR BaseAddress) {
	XProcess_features_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XProcess_features_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XProcess_features_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XProcess_features_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XProcess_features_Initialize(XProcess_features *InstancePtr, UINTPTR BaseAddress) {
	XProcess_features_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XProcess_features_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XProcess_features_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XProcess_features_Config *XProcess_features_LookupConfig(u16 DeviceId) {
	XProcess_features_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XPROCESS_FEATURES_NUM_INSTANCES; Index++) {
		if (XProcess_features_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XProcess_features_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XProcess_features_Initialize(XProcess_features *InstancePtr, u16 DeviceId) {
	XProcess_features_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XProcess_features_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XProcess_features_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

