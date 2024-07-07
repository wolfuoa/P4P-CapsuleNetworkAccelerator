/*
 * CapsuleNetwork.cpp
 * author: nicholas wolf
 *
 * The top level of the capsule network...
 * Routes individual layers together
 */

#include "CapsuleNetwork.h"

#include <cstdlib>
#include <stdint.h>
#include <string.h>

#include "constants.h"

#ifndef __SYNTHESIS__
#include <math.h>
#else
#include <hls_math.h>
#endif

#include "DigitCaps.h"
#include "PrimaryCaps.h"
#include "ReLUConv1.h"

void get_prediction(float *image, float *weights, float *biases, float *prediction)
{
    


	// ---------------- ReLU Convolutional 2D Layer ----------------
	float *output_conv = (float *)malloc(CONV1_OUTPUT_LENGTH * CONV1_OUTPUT_WIDTH * CONV1_FILTERS * sizeof(float));
	float *conv_weights = (float *)malloc(CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * CONV1_FILTERS * sizeof(float));
	float *conv_biases = (float *)malloc(CONV1_FILTERS * sizeof(float));

	memcpy(conv_weights, (const float *)weights, CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * CONV1_FILTERS * sizeof(float));
	memcpy(conv_biases, (const float *)biases, CONV1_FILTERS);

	weights += CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * CONV1_FILTERS;
	biases += CONV1_FILTERS;

	relu_conv_2d(image, conv_weights, conv_biases, output_conv);
	// ---------------- ReLU Convolutional 2D Layer ----------------

	// ------------------- Primary Capsule Layer -------------------
	uint32_t prim_dim = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES;
	float *output_prim = (float *)malloc(PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	float *prim_weights = (float *)malloc(prim_dim * sizeof(float));
	float *prim_biases = (float *)malloc(PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	memcpy(prim_weights, (const float *)weights, prim_dim * sizeof(float));
	memcpy(prim_biases, (const float *)biases, PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	weights += prim_dim;

	process_features(output_conv, prim_weights, prim_biases, output_prim);
	// ------------------- Primary Capsule Layer -------------------

	// -------------------- Digit Capsule Layer --------------------
	uint32_t digit_dim = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE;
	float *digit_weights = (float *)malloc(digit_dim * sizeof(float));

	memcpy(digit_weights, (const float *)weights, digit_dim * sizeof(float));

	dynamic_routing(output_prim, digit_weights, prediction);
	// -------------------- Digit Capsule Layer --------------------
}