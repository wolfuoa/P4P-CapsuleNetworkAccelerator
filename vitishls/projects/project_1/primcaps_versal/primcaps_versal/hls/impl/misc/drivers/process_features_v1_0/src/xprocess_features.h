// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XPROCESS_FEATURES_H
#define XPROCESS_FEATURES_H

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
#include "xprocess_features_hw.h"

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
} XProcess_features_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XProcess_features;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XProcess_features_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XProcess_features_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XProcess_features_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XProcess_features_ReadReg(BaseAddress, RegOffset) \
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
int XProcess_features_Initialize(XProcess_features *InstancePtr, UINTPTR BaseAddress);
XProcess_features_Config* XProcess_features_LookupConfig(UINTPTR BaseAddress);
#else
int XProcess_features_Initialize(XProcess_features *InstancePtr, u16 DeviceId);
XProcess_features_Config* XProcess_features_LookupConfig(u16 DeviceId);
#endif
int XProcess_features_CfgInitialize(XProcess_features *InstancePtr, XProcess_features_Config *ConfigPtr);
#else
int XProcess_features_Initialize(XProcess_features *InstancePtr, const char* InstanceName);
int XProcess_features_Release(XProcess_features *InstancePtr);
#endif

void XProcess_features_Start(XProcess_features *InstancePtr);
u32 XProcess_features_IsDone(XProcess_features *InstancePtr);
u32 XProcess_features_IsIdle(XProcess_features *InstancePtr);
u32 XProcess_features_IsReady(XProcess_features *InstancePtr);
void XProcess_features_Continue(XProcess_features *InstancePtr);
void XProcess_features_EnableAutoRestart(XProcess_features *InstancePtr);
void XProcess_features_DisableAutoRestart(XProcess_features *InstancePtr);

void XProcess_features_Set_input_r(XProcess_features *InstancePtr, u64 Data);
u64 XProcess_features_Get_input_r(XProcess_features *InstancePtr);
void XProcess_features_Set_weights(XProcess_features *InstancePtr, u64 Data);
u64 XProcess_features_Get_weights(XProcess_features *InstancePtr);
void XProcess_features_Set_biases(XProcess_features *InstancePtr, u64 Data);
u64 XProcess_features_Get_biases(XProcess_features *InstancePtr);
void XProcess_features_Set_output_r(XProcess_features *InstancePtr, u64 Data);
u64 XProcess_features_Get_output_r(XProcess_features *InstancePtr);

void XProcess_features_InterruptGlobalEnable(XProcess_features *InstancePtr);
void XProcess_features_InterruptGlobalDisable(XProcess_features *InstancePtr);
void XProcess_features_InterruptEnable(XProcess_features *InstancePtr, u32 Mask);
void XProcess_features_InterruptDisable(XProcess_features *InstancePtr, u32 Mask);
void XProcess_features_InterruptClear(XProcess_features *InstancePtr, u32 Mask);
u32 XProcess_features_InterruptGetEnabled(XProcess_features *InstancePtr);
u32 XProcess_features_InterruptGetStatus(XProcess_features *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
