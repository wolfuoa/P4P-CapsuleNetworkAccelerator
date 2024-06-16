#ifndef CONSTANTS_H
#define CONSTANTS_H

// Dataset dimension
#define IN_IMG_ROWS	 28
#define IN_IMG_COLS	 28
#define IN_IMG_DEPTH 1

// Conv1
#define KERNEL_ROWS 9
#define KERNEL_COLS 9
#define FILTERS		256

// Primary caps
#define CAPSULES				   32
#define CAPSULE_DIM				   8
#define PRIMARY_CAPS_STRIDE		   2
#define PRIMARY_CAPS_FILTERS	   CAPSULES *CAPSULE_DIM
#define PRIM_CAPS_CONV_WIDTH	   6
#define PRIM_CAPS_CONV_LENGTH	   6
#define PRIM_CAPS_CONV_DEPTH	   256
#define PRIM_CAPS_NUM_CONV_KERNELS 8

#define OUT_IMG_ROWS			   IN_IMG_ROWS - KERNEL_ROWS + 1
#define OUT_IMG_COLS			   IN_IMG_COLS - KERNEL_COLS + 1
#define OUT_IMG_DEPTH			   IN_IMG_DEPTH

#endif	// CONSTANTS_H