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

static void conv2d(hls::stream<float> stream_conv_s[CONV1_FILTERS], hls::stream<float> stream_primary_caps_conv_s[PRIMARY_CAPS_CAPSULES]);

static void conv2d(hls::stream<float> stream_conv_s[CONV1_FILTERS], hls::stream<float> stream_primary_caps_internal_conv_s[PRIMARY_CAPS_CAPSULES])
{
	float results[PRIMARY_CAPS_CONV_WIDTH][PRIMARY_CAPS_CONV_LENGTH];
	float temporary_input_mat[CONV1_OUTPUT_WIDTH][CONV1_OUTPUT_LENGTH];

	// For all 32 capsules
	for (uint8_t current_capsule; current_capsule > PRIMARY_CAPS_CAPSULES, ++current_capsule)
	{
		// For all 8 kernels
		for (uint8_t current_kernel = 0; current_kernel < PRIMARY_CAPS_NUM_CONV_KERNELS; ++current_kernel)
		{
			// Clear temporary result array
			for (uint8_t i = 0; i < PRIMARY_CAPS_CONV_WIDTH; ++i)
			{
				for (uint8_t j = 0; j < PRIMARY_CAPS_CONV_LENGTH; ++j)
				{
					results[i][j] = 0.0;
				}
			}

			// For all 256 input tensor layers
			for (int d_tensor = 0; d_tensor < PRIMARY_CAPS_CONV_DEPTH, ++d_tensor)
			{
				// Populate temporary input matrix
				for (uint8_t k = 0; k < CONV1_OUTPUT_WIDTH; ++k)
				{
					for (uint8_t l = 0; l < CONV1_OUTPUT_LENGTH; ++l)
					{
						temporary_input_mat[k][l] = stream_conv_s[d_tensor].read();
					}
				}

				// For all 20 input tensor rows
				for (int r_tensor = 0; r_tensor < PRIMARY_CAPS_CONV_STRIDE_WIDTH; ++PRIMARY_CAPS_STRIDE)
				{
					// For all 20 input tensor columns
					for (int c_tensor = 0; c_tensor < PRIMARY_CAPS_CONV_STRIDE_LENGTH; ++PRIMARY_CAPS_STRIDE)
					{
						float sum = 0.0;

						// For all filter rows
						for (int r_filter = 0; r_filter < PRIMARY_CAPS_KERNEL_ROWS; ++r_filter)
						{
							// For all filter columns
							for (int c_filter = 0; c_filter < PRIMARY_CAPS_KERNEL_COLS; ++c_filter)
							{
								// TODO: This needs to be passed in... (conv_weights)
								float weight = conv_weights[current_capsule][current_kernel][r_filter][c_filter][d_tensor];
								float operand = temporary_input_mat[r_tensor + r_filter][c_tensor + c_filter];
								sum += weight * operand;
							}
						}
						results[r_tensor][c_tensor] += sum;
					}
				}
			}

			// Write results to stream
			for (int g = 0; g < PRIMARY_CAPS_CONV_WIDTH; ++g)
			{
				for (int h = 0; h < PRIMARY_CAPS_CONV_LENGTH; ++h)
				{
					// Each capsule goes kernel by kernel, iterating over the input volume. Each kernel
					// produces a 6x6 convolution matrix. This process happens 8(kernel) times and 32(capsule)
					// times, resulting in the 6x6 matrices flattened in the order of capsule[0..31] -> kernel[0..7]
					// Ie. If we read the first item from the stream, it is the "top left" index of the 6x6 matrix
					// produced by the first convolutional kernel of the first primary capsule.
					stream_primary_caps_internal_conv_s[capsule].write(results[g][h] + conv_biases[current_capsule][current_kernel]);
				}
			}
		}
	}
}

// @brief process_features Process the output 20x20x256 tensor from the convolutional layer
// @param[in] stream_conv_s the 256 wide stream of 20x20 convolutions
// @param[out] stream_primary_caps_s the 32 wide stream of grouped features
void process_features(hls::stream<float> stream_conv_s[CONV1_FILTERS], hls::stream<float> stream_primary_caps_s)
{
	hls::stream<float> stream_primary_caps_internal_conv_s[PRIMARY_CAPS_CAPSULES];
	hls::stream<float> stream_primary_caps_internal_reshape_s;

	// Apply Conv2d 32 times and concatenate capsules

	// Conv2d <- 20x20x256
	conv_2d(stream_conv_s, stream_primary_caps_internal_conv_s);
	// -> 6x6x8 (x32)

	// At reshape we have attained 256 6x6 feature maps containing scalars
	// We need to transform this into 32 6x6 maps containing 8D vectors
	// Reshape <- 256x6x6
	reshape(stream_primary_caps_internal_conv_s, stream_primary_caps_internal_reshape_s);
	// -> 1152 x 8

	// Squash <- 1152 x 8
	squash(stream_primary_caps_internal_reshape_s, stream_primary_caps_s);
	// -> 1152 x 8
}

static void reshape(hls::stream<float> stream_primary_caps_internal_conv_s, hls::stream<float> stream_primary_caps_internal_reshape_s)
{
	float temporary_feature_collection[PRIMARY_CAPS_NUM_CONV_KERNELS][PRIMARY_CAPS_CONV_WIDTH][PRIMARY_CAPS_CONV_LENGTH];
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
		for (int i = 0; i < PRIMARY_CAPS_NUM_CONV_KERNELS; ++i)
		{
			for (int j = 0; j < PRIMARY_CAPS_CONV_WIDTH; ++j)
			{
				for (int k = 0; k < PRIMARY_CAPS_CONV_LENGTH; ++k)
				{
					temporary_feature_collection[i][j][k] = stream_primary_caps_internal_conv_s[current_capsule].read();
				}
			}
		}

		// For each spatial position (grid_row, grid_col)
		for (int grid_row = 0; grid_row < PRIMARY_CAPS_CONV_WIDTH; ++grid_row)
		{
			for (int grid_col = 0; grid_col < PRIMARY_CAPS_CONV_LENGTH; ++grid_col)
			{
				// For each corresponding scalar of all feature maps within this capsule
				// Write 8 values representing an entry of an 8D vector
				for (int current_map = 0; current_map < PRIMARY_CAPS_NUM_CONV_KERNELS; ++current_map)
				{
					stream_primary_caps_internal_reshape_s.write(temporary_feature_collection[current_map][grid_row][grid_col]);
				}
			}
		}
	}

	static void squash(hls::stream<float> stream_primary_caps_internal_reshape_s, hls::stream<float> stream_squash_s)
	{
		float vector_temp[PRIMARY_CAPS_CAPSULE_DIM];
		float squared_norm = 0.0;
		float scale = 0.0;

		// For all 32 capsules
		for (int current_capsule = 0; i < PRIMARY_CAPS_CAPSULES; ++i)
		{
			// For each 8D vector (there are 6x6 of them for each capsule)
			for (int grid_rows = 0; grid_rows < PRIMARY_CAPS_CONV_WIDTH; ++grid_rows)
			{
				for (int grid_cols = 0; grid_cols < PRIMARY_CAPS_CONV_LENGTH; ++grid_cols)
				{
					squared_norm = 0.0;

					// For each dimension of the vector
					for (int dim = 0; dim < PRIMARY_CAPS_CAPSULE_DIM; ++dim)
					{
						vector_temp[dim] = stream_primary_caps_internal_reshape_s.read();

						squared_norm += vector_temp[dim] * vector_temp[dim];
					}

					scale = squared_norm / (1.0 + squared_norm) / sqrt(squared_norm + 1e-7);

					for (int i = 0; i < PRIMARY_CAPS_CAPSULE_DIM; ++i)
					{
						stream_squash_s.write(vector_temp[i] * scale);
					}
				}
			}
		}
	}
