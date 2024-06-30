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

#include <string>

#include "constants.h"

// How does inlining ReLU improve/decrease efficiency?
// #define RELU(x) ((x) > 0.0 ? (x) : 0.0)

static float relu(float x);
static void conv_2d(float *image, float *weights, float *biases, float *output);

static float relu(float x)
{
	return (x > 0.0) ? x : 0.0;
}

// Convolution function that processes a single filter
// Maybe need to change the order of kernels/image...
// Is it better to do calculation for all kernels
// Before moving along in the image?
static void conv_2d(float *image, float *weights, float *biases, float *output)
{
	float *image_to_convolve = (float *)malloc(IN_IMG_ROWS * IN_IMG_COLS * sizeof(float));
	float *output_buffer = (float *)malloc(OUT_IMG_ROWS * OUT_IMG_COLS * CONV1_FILTERS * sizeof(float));
	float *weight_buffer = (float *)malloc(CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * sizeof(float));
	float *biases_buffer = (float *)malloc(CONV1_FILTERS * sizeof(float));

	// Burst read entire image input (its small)
	memcpy(image_to_convolve, (const float *)image, IN_IMG_ROWS * IN_IMG_COLS * sizeof(float));

	// Get all the biases
	memcpy(biases_buffer, (const float *)biases, CONV1_FILTERS);

	// For all 256 convolutonal kernels
	for (uint16_t current_kernel; current_kernel < CONV1_FILTERS; ++current_kernel)
	{
		// Read in all weights required for this kernel
		memcpy(weight_buffer, (const float *)weights + (current_kernel * CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS), CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * sizeof(float));

		// For all input image rows
		for (int r_image = 0; r_image < OUT_IMG_ROWS; ++r_image)
		{
			// For all input image columns
			for (int c_image = 0; c_image < OUT_IMG_COLS; ++c_image)
			{
				float sum = 0.0;

				// For all current kernel rows
				for (int r_filter = 0; r_filter < CONV1_KERNEL_ROWS; ++r_filter)
				{
					// For all current kernel columns
					for (int c_filter = 0; c_filter < CONV1_KERNEL_COLS; ++c_filter)
					{
						float weight = weights[r_filter * CONV1_KERNEL_COLS + c_filter];
						uint32_t current_row = r_image + r_filter;
						float pixel = image_to_convolve[current_row * IN_IMG_COLS + c_image + c_filter];
						sum += weight * pixel;
					}
				}

				// Apply ReLU activation, concatenate the result to the convolutional
				// output for that kernel
				output_buffer[current_kernel * OUT_IMG_ROWS * OUT_IMG_COLS + r_image * OUT_IMG_COLS + c_image] = relu(sum + biases_buffer[current_kernel]);
			}
		}
	}

	memcpy(output, (const float *)output_buffer, OUT_IMG_ROWS * OUT_IMG_COLS * CONV1_FILTERS * sizeof(float));
	free(image_to_convolve);
	free(output_buffer);
	free(weight_buffer);
	free(biases_buffer);
}

void relu_conv_2d(float *image, float *weights, float *biases, float *output)
{
	// Convolution is applied for each filter.
	// The result is stored in a 256 wide stream of 20x20 matrices.
	// The matrices represent the convolution of each filter about the input volume.
	// Pipeline?
	conv_2d(image, weights, biases, output);
}
