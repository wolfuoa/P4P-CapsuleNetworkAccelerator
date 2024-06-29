#ifndef CONSTANTS_H
#define CONSTANTS_H

// Dataset dimension
#define IN_IMG_ROWS	 28
#define IN_IMG_COLS	 28
#define IN_IMG_DEPTH 1

// Conv1
#define CONV1_KERNEL_ROWS	9
#define CONV1_KERNEL_COLS	9
#define CONV1_FILTERS		256
#define CONV1_OUTPUT_WIDTH	20
#define CONV1_OUTPUT_LENGTH 20

// Primary caps
#define PRIMARY_CAPS_KERNEL_ROWS		9
#define PRIMARY_CAPS_KERNEL_COLS		9
#define PRIMARY_CAPS_KERNEL_DEPTH		256
#define PRIMARY_CAPS_CAPSULES			32
#define PRIMARY_CAPS_CAPSULE_DIM		8
#define PRIMARY_CAPS_STRIDE				2
#define PRIMARY_CAPS_FILTERS			CAPSULES *CAPSULE_DIM
#define PRIMARY_CAPS_CONV_WIDTH			6
#define PRIMARY_CAPS_CONV_LENGTH		6
#define PRIMARY_CAPS_CONV_DEPTH			256
#define PRIMARY_CAPS_NUM_CONV_KERNELS	8
#define PRIMARY_CAPS_CONV_STRIDE_WIDTH	CONV1_OUTPUT_WIDTH - PRIMARY_CAPS_KERNEL_ROWS + 1
#define PRIMARY_CAPS_CONV_STRIDE_LENGTH CONV1_OUTPUT_LENGTH - PRIMARY_CAPS_KERNEL_COLS + 1

// Digit Caps
// The MNIST dataset has 10 classifications
#define DIGIT_CAPS_NUM_DIGITS 10
// The final Layer (DigitCaps) has one 16D capsule per digit class
#define DIGIT_CAPS_DIM_CAPSULE 16
// Per "Dynamic Routing Between Capsules"
#define DIGIT_CAPS_ROUTING_ITERATIONS 3
// Each capsule takes as input a 6x6x8x32 tensor.
// You can think of it as 6x6x32 8-dimensional vectors,
// which is 1152 capsule outputs from primcaps in total
#define DIGIT_CAPS_INPUT_CAPSULES 1152
// Each input capsule provices a grid of 8D vectors.
#define DIGIT_CAPS_INPUT_DIM_CAPSULE PRIMARY_CAPS_CAPSULE_DIM

#define OUT_IMG_ROWS				 IN_IMG_ROWS - CONV1_KERNEL_ROWS + 1
#define OUT_IMG_COLS				 IN_IMG_COLS - CONV1_KERNEL_COLS + 1
#define OUT_IMG_DEPTH				 IN_IMG_DEPTH

#endif	// CONSTANTS_H