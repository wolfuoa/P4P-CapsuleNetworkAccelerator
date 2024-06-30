/*
 * PrimaryCaps.cpp
 * author: nicholas wolf
 *
 * The second layer of the Capsule Network takes in a three-dimensional
 * tensor best represented as a 20x20x256 three-dimensional matrix
 *
 * To evaluate this matrix, the Primary Capsule layer has 32 primary
 * capsules whose job is to combine the basic features provided by the
 * previous layer into cohesive groups of features.
 *
 * To do this, each capsule applies 8 three-dimensional convolutional
 * kernels, each of which natrually has their own 9x9x256 weight matrix
 *
 * We need to organize this operation such that the correct input
 * is matched to multiply to the correct weight. The order that
 * Conv1 populated the 20x20x256 input tensor was:
 * For each kernel(256), for each row(20), for each col(20),
 * meaning every 20 reads, we increase the row of the input
 * volume, and every 20 rows we change the kernel that was used.
 * Hence, we need to read a complete kernel entry before incrementing
 * and moving along the 256 kernels. In that kernel entry, we need
 * to read columns first, then rows.
 *
 * As the operation is done using a stride of two, each kernel only
 * indexes over the length and width of the tensor 6 times,
 * producing an output of a 6x6 matrix
 *
 * There are 8 convolutional kernels in a capsule, and there are 32
 * capsules in Primary Caps. Hence, we get a total output of
 * 6x6x8x32
 *
 */

#include "PrimaryCaps.h"

#include <math.h>

#include <string>

static void conv_2d(float *input, float *weights, float *biases, float *output);
static void reshape(float *input, float *output);
static void squash(float *input, float *output);

static void conv_2d(float *input, float *weights, float *biases, float *output)
{
	float *results = (float *)malloc(PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * sizeof(float));
	float *input_buffer = (float *)malloc(CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH * CONV1_FILTERS * sizeof(float));
	float *output_buffer = (float *)malloc(PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	float *weight_buffer = (float *)malloc(PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH * sizeof(float));
	float *biases_buffer = (float *)malloc(PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	memcpy(input_buffer, (const float *)input, CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH * CONV1_FILTERS * sizeof(float));
	memcpy(biases_buffer, (const float *)biases, PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	uint32_t prim_caps_kernel_dim = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH;
	// For all 32 capsules
	for (uint8_t current_capsule = 0; current_capsule < PRIMARY_CAPS_CAPSULES; ++current_capsule)
	{
		// For all 8 kernels
		for (uint8_t current_kernel = 0; current_kernel < PRIMARY_CAPS_NUM_CONV_KERNELS; ++current_kernel)
		{
			memcpy(weight_buffer, (const float *)weights + current_capsule * prim_caps_kernel_dim * PRIMARY_CAPS_NUM_CONV_KERNELS + current_kernel * prim_caps_kernel_dim, prim_caps_kernel_dim * sizeof(float));
			// Clear temporary result array
			memset(results, 0.0, PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * sizeof(float));

			// For all 256 input tensor layers
			for (int d_tensor = 0; d_tensor < PRIMARY_CAPS_CONV_DEPTH; ++d_tensor)
			{
				// For all 20 input tensor rows
				for (int r_tensor = 0; r_tensor < PRIMARY_CAPS_CONV_STRIDE_WIDTH; r_tensor += PRIMARY_CAPS_STRIDE)
				{
					// For all 20 input tensor columns
					for (int c_tensor = 0; c_tensor < PRIMARY_CAPS_CONV_STRIDE_LENGTH; c_tensor += PRIMARY_CAPS_STRIDE)
					{
						float sum = 0.0;

						// For all filter rows
						for (int r_filter = 0; r_filter < PRIMARY_CAPS_KERNEL_ROWS; ++r_filter)
						{
							// For all filter columns
							for (int c_filter = 0; c_filter < PRIMARY_CAPS_KERNEL_COLS; ++c_filter)
							{
								float weight = weight_buffer[d_tensor * PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS + r_filter * PRIMARY_CAPS_KERNEL_COLS + c_filter];
								float operand = input_buffer[d_tensor * CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH + (r_tensor + r_filter) * CONV1_OUTPUT_LENGTH + c_tensor + c_filter];
								sum += weight * operand;
							}
						}
						results[r_tensor * PRIMARY_CAPS_CONV_LENGTH / 2 + c_tensor / 2] += sum;
					}
				}
			}

			// Write results to stream
			for (int g = 0; g < PRIMARY_CAPS_CONV_WIDTH; ++g)
			{
				for (int h = 0; h < PRIMARY_CAPS_CONV_LENGTH; ++h)
				{
					// TODO: Im certain we can skip the reshape by doing something different here

					// Each capsule goes kernel by kernel, iterating over the input volume. Each kernel
					// produces a 6x6 convolution matrix. This process happens 8(kernel) times and 32(capsule)
					// times, resulting in the 6x6 matrices flattened in the order of capsule[0..31] -> kernel[0..7]
					// Ie. If we read the first item from the stream, it is the "top left" index of the 6x6 matrix
					// produced by the first convolutional kernel of the first primary capsule.
					output_buffer[current_capsule * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_NUM_CONV_KERNELS + current_kernel * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH + g * PRIMARY_CAPS_CONV_LENGTH + h] = (results[g * PRIMARY_CAPS_CONV_LENGTH + h] + biases_buffer[current_capsule * PRIMARY_CAPS_NUM_CONV_KERNELS + current_kernel]);
				}
			}
		}
	}
	memcpy(output, (const float *)output_buffer, PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	free(results);
	free(input_buffer);
	free(output_buffer);
	free(weight_buffer);
	free(biases_buffer);
}

// @brief process_features Process the output 20x20x256 tensor from the convolutional layer
// @param[in] stream_conv_s the 256 wide stream of 20x20 convolutions
// @param[out] stream_primary_caps_s the 32 wide stream of grouped features
void process_features(float *input, float *weights, float *biases, float *output)
{
	// Apply Conv2d 32 times and concatenate capsules
	float *conv_output = (float *)malloc(PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	float *reshape_output = (float *)malloc(PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	// Conv2d <- 20x20x256
	conv_2d(input, weights, biases, conv_output);

	// -> 6x6x8 (x32)

	// At reshape we have attained 256 6x6 feature maps containing scalars
	// We need to transform this into 32 6x6 maps containing 8D vectors
	// Reshape <- 256x6x6
	reshape(conv_output, reshape_output);
	// -> 1152 x 8

	// Squash <- 1152 x 8
	squash(reshape_output, output);
	// -> 1152 x 8

	free(conv_output);
	free(reshape_output);
}

static void reshape(float *input, float *output)
{
	uint32_t dim = PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS;
	uint32_t output_index = 0;
	float *temporary_feature_collection = (float *)malloc(dim * sizeof(float));
	float *output_buffer = (float *)malloc(dim * PRIMARY_CAPS_CAPSULES * sizeof(float));
	// For each spatial position in the feature map
	// [i, j]
	// [k, l]
	// Pair the corresponding scalar value with the corresponding outputs from the other feature maps
	// v0 -> [i0, i1, i2, i3, i4, i5, i6, i7]
	// v1 -> [j0, j1, j2, j3, j4, j5, j6, j7]
	// ...
	// [v00, v01]
	// [v10, v11]
	// Hence, we obtain a 6x6 8D output for each primary capsule
	for (int current_capsule = 0; current_capsule < PRIMARY_CAPS_CAPSULES; ++current_capsule)
	{
		// Stage feature maps
		memcpy(temporary_feature_collection, (const float *)input + current_capsule * dim, dim * sizeof(float));

		// For each spatial position (grid_row, grid_col)
		for (int grid_row = 0; grid_row < PRIMARY_CAPS_CONV_WIDTH; ++grid_row)
		{
			for (int grid_col = 0; grid_col < PRIMARY_CAPS_CONV_LENGTH; ++grid_col)
			{
				// For each corresponding scalar of all feature maps within this capsule
				// Write 8 values representing an entry of an 8D vector
				for (int current_map = 0; current_map < PRIMARY_CAPS_NUM_CONV_KERNELS; ++current_map)
				{
					output_buffer[output_index++] = temporary_feature_collection[current_map * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH + grid_row * PRIMARY_CAPS_CONV_LENGTH + grid_col];
				}
			}
		}
	}

	memcpy(output, (const float *)output_buffer, dim * PRIMARY_CAPS_CAPSULES * sizeof(float));
	free(temporary_feature_collection);
	free(output_buffer);
}

static void squash(float *input, float *output)
{
	uint32_t dim = PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS;
	float *input_buffer = (float *)malloc(dim * sizeof(float));
	float *output_buffer = (float *)malloc(dim * PRIMARY_CAPS_CAPSULES * sizeof(float));
	double squared_norm = 0.0;
	float scale = 0.0;

	// For all 32 capsules
	for (int current_capsule = 0; current_capsule < PRIMARY_CAPS_CAPSULES; ++current_capsule)
	{
		memcpy(input_buffer, (const float *)input + current_capsule * dim, dim * sizeof(float));
		// For each 8D vector (there are 6x6 of them for each capsule)

		for (int grid_rows = 0; grid_rows < PRIMARY_CAPS_CONV_WIDTH; ++grid_rows)
		{
			for (int grid_cols = 0; grid_cols < PRIMARY_CAPS_CONV_LENGTH; ++grid_cols)
			{
				squared_norm = 0.0;

				// For each dimension of the vector
				for (int i = 0; i < PRIMARY_CAPS_CAPSULE_DIM; ++i)
				{
					float value = input_buffer[grid_rows * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM + grid_cols * PRIMARY_CAPS_CAPSULE_DIM + i];

					squared_norm += value * value;
					// printf("val: %f\n", value);
					// printf("norm: %f\n", squared_norm);
				}

				scale = squared_norm / (1.0 + squared_norm) / sqrt(squared_norm + 1e-7);

				// printf("scale: %f\n", scale);
				for (int i = 0; i < PRIMARY_CAPS_CAPSULE_DIM; ++i)
				{
					output_buffer[current_capsule * dim + grid_rows * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM + grid_cols * PRIMARY_CAPS_CAPSULE_DIM + i] = (input_buffer[grid_rows * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM + grid_cols * PRIMARY_CAPS_CAPSULE_DIM + i] * scale);
				}
			}
		}
	}
	// printf("kel: %f\n", output_buffer[4]);

	memcpy(output, (const float *)output_buffer, dim * PRIMARY_CAPS_CAPSULES * sizeof(float));
	free(input_buffer);
	free(output_buffer);
}
