// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XRELU_CONV_2D_H
#define XRELU_CONV_2D_H

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
#include "xrelu_conv_2d_hw.h"

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
} XRelu_conv_2d_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XRelu_conv_2d;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRelu_conv_2d_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRelu_conv_2d_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRelu_conv_2d_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRelu_conv_2d_ReadReg(BaseAddress, RegOffset) \
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
int XRelu_conv_2d_Initialize(XRelu_conv_2d *InstancePtr, UINTPTR BaseAddress);
XRelu_conv_2d_Config* XRelu_conv_2d_LookupConfig(UINTPTR BaseAddress);
#else
int XRelu_conv_2d_Initialize(XRelu_conv_2d *InstancePtr, u16 DeviceId);
XRelu_conv_2d_Config* XRelu_conv_2d_LookupConfig(u16 DeviceId);
#endif
int XRelu_conv_2d_CfgInitialize(XRelu_conv_2d *InstancePtr, XRelu_conv_2d_Config *ConfigPtr);
#else
int XRelu_conv_2d_Initialize(XRelu_conv_2d *InstancePtr, const char* InstanceName);
int XRelu_conv_2d_Release(XRelu_conv_2d *InstancePtr);
#endif

void XRelu_conv_2d_Start(XRelu_conv_2d *InstancePtr);
u32 XRelu_conv_2d_IsDone(XRelu_conv_2d *InstancePtr);
u32 XRelu_conv_2d_IsIdle(XRelu_conv_2d *InstancePtr);
u32 XRelu_conv_2d_IsReady(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_Continue(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_EnableAutoRestart(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_DisableAutoRestart(XRelu_conv_2d *InstancePtr);

void XRelu_conv_2d_Set_image_r(XRelu_conv_2d *InstancePtr, u64 Data);
u64 XRelu_conv_2d_Get_image_r(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_Set_weights(XRelu_conv_2d *InstancePtr, u64 Data);
u64 XRelu_conv_2d_Get_weights(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_Set_biases(XRelu_conv_2d *InstancePtr, u64 Data);
u64 XRelu_conv_2d_Get_biases(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_Set_output_r(XRelu_conv_2d *InstancePtr, u64 Data);
u64 XRelu_conv_2d_Get_output_r(XRelu_conv_2d *InstancePtr);

void XRelu_conv_2d_InterruptGlobalEnable(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_InterruptGlobalDisable(XRelu_conv_2d *InstancePtr);
void XRelu_conv_2d_InterruptEnable(XRelu_conv_2d *InstancePtr, u32 Mask);
void XRelu_conv_2d_InterruptDisable(XRelu_conv_2d *InstancePtr, u32 Mask);
void XRelu_conv_2d_InterruptClear(XRelu_conv_2d *InstancePtr, u32 Mask);
u32 XRelu_conv_2d_InterruptGetEnabled(XRelu_conv_2d *InstancePtr);
u32 XRelu_conv_2d_InterruptGetStatus(XRelu_conv_2d *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
