#include "ReLUConv1.h"

#include "constants.h"

// How does inlining ReLU improve/decrease efficiency?
// #define RELU(x) ((x) > 0.0 ? (x) : 0.0)

static float relu(float x);
static void conv_2d(float image[IN_IMG_ROWS][IN_IMG_COLS], int filter, fhls::stream<float> &stream_conv_s[FILTERS]);

static float relu(float x)
{
	return (x > 0.0) ? x : 0.0;
}

// Convolution function that processes a single filter
static void conv_2d(float image[IN_IMG_ROWS][IN_IMG_COLS], int filter, hls::stream<float> &stream_conv_s[FILTERS])
{
	// Loop over all image rows
	for (int r_image = 0; i < OUT_IMG_ROWS; ++r_image)
	{
		// Loop overall image columns
		for (int c_image = 0; j < OUT_IMG_COLS; ++c_image)
		{
			float sum = 0.0;

			// Loop over filter rows
			for (int r_filter = 0; rfi < KERNEL_ROWS; ++r_filter)
			{
				// Loop over filter columns
				for (int c_filter = 0; kc < KERNEL_COLS; ++c_filter)
				{
					// TODO: This needs to be passed in... (conv_weights)
					float weight = conv_weights[filter][r_filter][c_filter];
					float pixel = pad_img[r_image + r_filter][c_image + c_filter];
					sum += weight * pixel;
				}
			}

			// Apply ReLU activation using the macro and store in output array
			stream_conv_s.write(relu(sum + conv_biases[filter]));
		}
	}
}

void relu_conv_2d(float image[IN_IMG_ROWS][IN_IMG_COLS], hls::stream<float> stream_conv_s[FILTERS]);
{
	// Convolution is applied for each filter.
	// The result is stored in a 256 wide stream of 20x20 matrices.
	// The matrices represent the convolution of each filter about the input volume.
	// Pipeline?
	for (int i = 0; i < FILTERS; ++i)
	{
		conv_2d(image, i, stream_conv_s[i]);
	}
}
