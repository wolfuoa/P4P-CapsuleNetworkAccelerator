// DigitCaps.h

#ifndef DIGIT_CAPS_H
#define DIGIT_CAPS_H

#include "constants.h"
#include "hls_stream.h"

void dynamic_routing(hls::stream<float> stream_primary_caps_s, hls::stream<float> stream_prediction_s[DIGIT_CAPS_NUM_DIGITS]);

#endif	// DIGIT_CAPS_H