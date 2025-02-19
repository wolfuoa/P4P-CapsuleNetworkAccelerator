// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XDYNAMIC_ROUTING_H
#define XDYNAMIC_ROUTING_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xdynamic_routing_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XDynamic_routing_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XDynamic_routing;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XDynamic_routing_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XDynamic_routing_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XDynamic_routing_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XDynamic_routing_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XDynamic_routing_Initialize(XDynamic_routing *InstancePtr, UINTPTR BaseAddress);
XDynamic_routing_Config* XDynamic_routing_LookupConfig(UINTPTR BaseAddress);
#else
int XDynamic_routing_Initialize(XDynamic_routing *InstancePtr, u16 DeviceId);
XDynamic_routing_Config* XDynamic_routing_LookupConfig(u16 DeviceId);
#endif
int XDynamic_routing_CfgInitialize(XDynamic_routing *InstancePtr, XDynamic_routing_Config *ConfigPtr);
#else
int XDynamic_routing_Initialize(XDynamic_routing *InstancePtr, const char* InstanceName);
int XDynamic_routing_Release(XDynamic_routing *InstancePtr);
#endif

void XDynamic_routing_Start(XDynamic_routing *InstancePtr);
u32 XDynamic_routing_IsDone(XDynamic_routing *InstancePtr);
u32 XDynamic_routing_IsIdle(XDynamic_routing *InstancePtr);
u32 XDynamic_routing_IsReady(XDynamic_routing *InstancePtr);
void XDynamic_routing_Continue(XDynamic_routing *InstancePtr);
void XDynamic_routing_EnableAutoRestart(XDynamic_routing *InstancePtr);
void XDynamic_routing_DisableAutoRestart(XDynamic_routing *InstancePtr);

void XDynamic_routing_Set_input_r(XDynamic_routing *InstancePtr, u64 Data);
u64 XDynamic_routing_Get_input_r(XDynamic_routing *InstancePtr);
void XDynamic_routing_Set_weights(XDynamic_routing *InstancePtr, u64 Data);
u64 XDynamic_routing_Get_weights(XDynamic_routing *InstancePtr);
void XDynamic_routing_Set_prediction(XDynamic_routing *InstancePtr, u64 Data);
u64 XDynamic_routing_Get_prediction(XDynamic_routing *InstancePtr);

void XDynamic_routing_InterruptGlobalEnable(XDynamic_routing *InstancePtr);
void XDynamic_routing_InterruptGlobalDisable(XDynamic_routing *InstancePtr);
void XDynamic_routing_InterruptEnable(XDynamic_routing *InstancePtr, u32 Mask);
void XDynamic_routing_InterruptDisable(XDynamic_routing *InstancePtr, u32 Mask);
void XDynamic_routing_InterruptClear(XDynamic_routing *InstancePtr, u32 Mask);
u32 XDynamic_routing_InterruptGetEnabled(XDynamic_routing *InstancePtr);
u32 XDynamic_routing_InterruptGetStatus(XDynamic_routing *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
