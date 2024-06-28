#ifndef RELU_CONV_1_H
#define RELU_CONV_1_H

#include "constants.h"
#include "hls_stream.h"

void relu_conv_2d(float image[IN_IMG_ROWS][IN_IMG_COLS], hls::stream<float> stream_conv_s[CONV1_FILTERS]);

#endif;	 // RELU_CONV_1_H