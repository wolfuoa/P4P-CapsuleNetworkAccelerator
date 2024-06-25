#include "CapsuleNetwork.h"

#include "CapsuleLayer.h"
#include "PrimaryCaps.h"
#include "ReLUConv1.h"
#include "hls_stream.h"

void get_prediction(float image[IN_IMG_ROWS][IN_IMG_COLS], float prediction[DIGIT_CAPS_NUM_DIGITS])
{
	// ---------------- ReLU Convolutional 2D Layer ----------------
	hls::stream<float> stream_conv_s[CONV1_FILTERS];

	relu_conv_2d(image, stream_conv_s);
	// ---------------- ReLU Convolutional 2D Layer ----------------

	// ------------------- Primary Capsule Layer -------------------
	hls::stream<float> stream_primary_caps_s[PRIMARY_CAPS_CAPSULES];

	process_features(stream_conv_s, stream_primary_caps_s);
	// ------------------- Primary Capsule Layer -------------------

	// -------------------- Digit Capsule Layer --------------------
	dynamic_routing(stream_primary_caps_s, prediction);
	// -------------------- Digit Capsule Layer --------------------
}