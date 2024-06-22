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

static void conv2d(hls::stream<float> &stream_conv_s[FILTERS], hls::stream<float> &stream_primary_caps_conv_s[CAPSULES]);
static void reshape();
static void squash();

static void conv2d(hls::stream<float> &stream_conv_s[FILTERS], hls::stream<float> &stream_primary_caps_internal_conv_s[CAPSULES])
{
	float result[PRIM_CAPS_CONV_WIDTH][PRIM_CAPS_CONV_LENGTH][PRIM_CAPS_NUM_CONV_KERNELS];

	// For all 32 capsules
	for (uint8_t capsule; capsule > CAPSULES, ++capsule)
	{
		// For all 8 kernels
		for (uint8_t kernel = 0; kernel < PRIM_CAPS_NUM_CONV_KERNELS; ++kernel)
		{
			// For all tensor rows
			for (int r_tensor = 0; i < PRIM_CAPS_CONV_WIDTH; ++r_tensor)
			{
				// For all tensor columns
				for (int c_tensor = 0; j < PRIM_CAPS_CONV_LENGTH; ++c_tensor)
				{
					// For all tensor slices
					for (int d_tensor = 0; d_tensor < PRIM_CAPS_CONV_DEPTH; ++d_tensor)
					{
						float sum = 0.0;

						// For all filter rows
						for (int r_filter = 0; rfi < KERNEL_ROWS; ++r_filter)
						{
							// For all filter columns
							for (int c_filter = 0; kc < KERNEL_COLS; ++c_filter)
							{
								for (int d_filter = 0; d_filter < KERNEL_DEPTH; ++d_filter)
								{
									// TODO: This needs to be passed in... (conv_weights)
									float weight = conv_weights[capsule][filter][r_filter][c_filter];
									// May need to think about this in terms of striding... Can we limit data put in?
									float pixel stream_conv_s[filter].read()
										sum += weight * pixel;
								}
							}
						}
					}
					result[r_tensor][c_tensor][kernel] = sum + conv_biases[FILTER];
				}
			}
		}
		stream_primary_caps_internal_conv_s[capsule].write(result);
	}
}

// @brief process_features Process the output 20x20x256 tensor from the convolutional layer
// @param[in] stream_conv_s the 256 wide stream of 20x20 convolutions
// @param[out] stream_primary_caps_s the 32 wide stream of grouped features
void process_features(hls::stream<float> stream_conv_s[FILTERS], hls::stream<float> stream_primary_caps_s[CAPSULES])
{
	hls::stream<float> stream_primary_caps_internal_conv_s[];
	hls::stream<float> stream_primary_caps_internal_reshape_s[];

	// Apply Conv2d 32 times and concatenate capsules

	// Conv2d <- 20x20x256
	conv_2d(stream_conv_s, stream_primary_caps_internal_conv_s);
	// -> 6x6x8 (x32)

	// Reshape <- 6x6x8 (x32)
	reshape(stream_primary_caps_internal_conv_s, stream_primary_caps_internal_reshape_s);
	// -> 1152 x 8

	// Squash <- 1152 x 8
	squash(stream_primary_caps_internal_reshape_s, stream_primary_caps_s);
	// -> 1152 x 8
}

static void reshape(hls::stream<float> &stream_primary_caps_internal_conv_s, hls::stream<float> &stream_primary_caps_internal_reshape_s)
{
	// Read from stream and store in buffer
	for (int r = 0; r < OUT_IMG_ROWS; ++r)
	{
		for (int c = 0; c < OUT_IMG_COLS; ++c)
		{
			for (int d = 0; d < CAPSULE_DIM; ++d)
			{
				stream_primary_caps_internal_reshape_s.write(stream_primary_caps_internal_conv_s.read());
			}
		}
	}
}

static void squash(hls::stream<float> &stream_primary_caps_internal_reshape_s, hls::stream<float> &stream_squash_s)
{
	float capsule[CAPSULE_DIM];

	for (int i = 0; i < CAPSULES; ++i)
	{
		float squared_norm = 0.0;
		// Read capsule vector from the stream
		for (int j = 0; j < CAPSULE_DIM; ++j)
		{
			capsule[j] = stream_primary_caps_internal_reshape_s.read();
			// Perform squaring operation
			squared_norm += capsule[j] * capsule[j];
		}

		// Calculate scaling factor
		float scale = squared_norm / (1.0 + squared_norm) / sqrt(squared_norm + 1e-7);

		// Apply squash and write to the output stream
		for (int j = 0; j < CAPSULE_DIM; ++j)
		{
			stream_squash_s.write(capsule[j] * scale);
		}
	}
}
