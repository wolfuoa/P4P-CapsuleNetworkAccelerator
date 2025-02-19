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
#include "xdynamic_routing.h"

extern XDynamic_routing_Config XDynamic_routing_ConfigTable[];

#ifdef SDT
XDynamic_routing_Config *XDynamic_routing_LookupConfig(UINTPTR BaseAddress) {
	XDynamic_routing_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XDynamic_routing_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XDynamic_routing_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XDynamic_routing_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDynamic_routing_Initialize(XDynamic_routing *InstancePtr, UINTPTR BaseAddress) {
	XDynamic_routing_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDynamic_routing_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDynamic_routing_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XDynamic_routing_Config *XDynamic_routing_LookupConfig(u16 DeviceId) {
	XDynamic_routing_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XDYNAMIC_ROUTING_NUM_INSTANCES; Index++) {
		if (XDynamic_routing_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XDynamic_routing_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDynamic_routing_Initialize(XDynamic_routing *InstancePtr, u16 DeviceId) {
	XDynamic_routing_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDynamic_routing_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDynamic_routing_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

