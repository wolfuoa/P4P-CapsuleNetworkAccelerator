/*
* ReLUConv1.cpp
* author: nicholas wolf
*
* The first layer of the Capsule Network takes in a 2D image, best represented
* as a 28x28x1 box in which each pixel is a singular color channel value.
*
* In this layer, 256 9x9x1 convolutional kernels convolve over the input
* volume. Each kernel has a 9x9x1 matrix of weights and an extra bias term
*
* The convolution operation results in 256 20x20 entries into Primary Caps.
* ie. the output tensor is 20x20x256
*/


#include "ReLUConv1.h"


#include <cstdlib>
#include <stdint.h>
#include <string.h>


#include "constants.h"


#ifndef __SYNTHESIS__
#include <math.h>
#else
#include <hls_math.h>
#endif


// How does inlining ReLU improve/decrease efficiency?
// #define RELU(x) ((x) > 0.0 ? (x) : 0.0)

// ------------------- PRIVATE TYPEDEF -------------------
typedef ap_fixed<32, 16> fixed_t;
// ------------------- PRIVATE TYPEDEF -------------------


static fixed_t relu(fixed_t x);
static void conv_2d(fixed_t *image, fixed_t *weights, fixed_t *biases, fixed_t *output);


static fixed_t relu(fixed_t x)
{
    if (x > 0.0){
        return x;
    }
    else{
        return 0.0;
    }
   // return (x > 0.0) ? x : 0.0;
}


// Convolution function that processes a single filter
// Maybe need to change the order of kernels/image...
// Is it better to do calculation for all kernels
// Before moving along in the image?
static void conv_2d(fixed_t *image, fixed_t *weights, fixed_t *biases, fixed_t *output)
{
   fixed_t image_to_convolve[IN_IMG_ROWS * IN_IMG_COLS];
   #pragma HLS BIND_STORAGE variable=image_to_convolve impl=auto type=ram_1wnr
   #pragma HLS ARRAY_PARTITION variable=image_to_convolve dim=1 type=cyclic factor=28


   fixed_t output_buffer[OUT_IMG_ROWS * OUT_IMG_COLS * CONV1_FILTERS];
   #pragma HLS BIND_STORAGE variable=output_buffer type=ram_1wnr impl=auto


   fixed_t weight_buffer[CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS];
   #pragma HLS BIND_STORAGE variable=weight_buffer type=ram_1wnr impl=auto


   fixed_t biases_buffer[CONV1_FILTERS];
   #pragma HLS BIND_STORAGE variable=biases_buffer type=ram_1wnr impl=auto


   // Burst read entire image input (its small)
   memcpy(image_to_convolve, (const fixed_t *)image, IN_IMG_ROWS * IN_IMG_COLS * sizeof(fixed_t));


   // Get all the biases
   memcpy(biases_buffer, (const fixed_t *)biases, CONV1_FILTERS * sizeof(fixed_t));


   // For all 256 convolutonal kernels
    for (uint16_t current_kernel = 0; current_kernel < CONV1_FILTERS; ++current_kernel)
	{
		// Read in all weights required for this kernel
		memcpy(weight_buffer, (const fixed_t *)weights + (current_kernel * CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS), CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * sizeof(fixed_t));
		// For all input image rows
		for (uint32_t r_image = 0; r_image < OUT_IMG_ROWS; ++r_image)
		{
			// For all input image columns
			for (uint32_t c_image = 0; c_image < OUT_IMG_COLS; ++c_image)
			{
				fixed_t sum = 0.0;
				// For all current kernel rows
				for (uint32_t r_filter = 0; r_filter < CONV1_KERNEL_ROWS; ++r_filter)
				{
					// For all current kernel columns
					for (uint32_t c_filter = 0; c_filter < CONV1_KERNEL_COLS; ++c_filter)
					{
                        #pragma HLS UNROLL
						fixed_t weight = weight_buffer[r_filter * CONV1_KERNEL_COLS + c_filter];
						uint32_t current_row = r_image + r_filter;
						fixed_t pixel = image_to_convolve[current_row * IN_IMG_COLS + c_image + c_filter];
						sum += weight * pixel;
					}
				}
				// Apply ReLU activation, concatenate the result to the convolutional
				// output for that kernel
				output_buffer[current_kernel * OUT_IMG_ROWS * OUT_IMG_COLS + r_image * OUT_IMG_COLS + c_image] = relu(sum + biases_buffer[current_kernel]);
			}
		}
	}
   memcpy(output, (const fixed_t *)output_buffer, OUT_IMG_ROWS * OUT_IMG_COLS * CONV1_FILTERS * sizeof(fixed_t));
}


void relu_conv_2d(fixed_t *image, fixed_t *weights, fixed_t *biases, fixed_t *output)
{
   // Convolution is applied for each filter.
   // The result is stored in a 256 wide stream of 20x20 matrices.
   // The matrices represent the convolution of each filter about the input volume.
   conv_2d(image, weights, biases, output);
}

