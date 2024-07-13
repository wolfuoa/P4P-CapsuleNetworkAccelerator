/*
 * CapsuleNetwork.cpp
 * author: nicholas wolf
 *
 * The top level of the capsule network...
 * Routes individual layers together
 */

#include "CapsuleNetwork.h"

#include <stdint.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "DigitCaps.h"
#include "PrimaryCaps.h"
#include "ReLUConv1.h"
#include "testing_suite.h"

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

	clock_t t_1;
	clock_t t_2;

	t_1 = clock();
	relu_conv_2d(image, conv_weights, conv_biases, output_conv);
	t_2 = clock();
	// ---------------- ReLU Convolutional 2D Layer ----------------

	printf("Conv 1: %.4fms\n", ((float)(t_2 - t_1) / CLOCKS_PER_SEC) * 1000);

#if DUMP_LAYERS
	FILE *fp;
	fp = fopen("../dump/output_conv_1.txt", "w");

	for (int i = 0; i < CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH * CONV1_FILTERS; ++i)
	{
		fprintf(fp, "%.10f\n", output_conv[i]);
	}
	fclose(fp);
#endif

	// ------------------- Primary Capsule Layer -------------------
	uint32_t prim_dim = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES;
	float *output_prim = (float *)malloc(PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	float *prim_weights = (float *)malloc(prim_dim * sizeof(float));
	float *prim_biases = (float *)malloc(PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	memcpy(prim_weights, (const float *)weights, prim_dim * sizeof(float));
	memcpy(prim_biases, (const float *)biases, PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));

	weights += prim_dim;

	t_1 = clock();
	process_features(output_conv, prim_weights, prim_biases, output_prim);
	t_2 = clock();
	// ------------------- Primary Capsule Layer -------------------

	printf("Primary Capsules: %.4fms\n", ((float)(t_2 - t_1) / CLOCKS_PER_SEC) * 1000);

#if DUMP_LAYERS
	fp = fopen("../dump/output_prim_caps.txt", "w");

	for (int i = 0; i < PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH; ++i)
	{
		fprintf(fp, "%.10f\n", output_prim[i]);
	}
	fclose(fp);
#endif

	// -------------------- Digit Capsule Layer --------------------
	uint32_t digit_dim = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE;
	float *digit_weights = (float *)malloc(digit_dim * sizeof(float));

	memcpy(digit_weights, (const float *)weights, digit_dim * sizeof(float));

	t_1 = clock();
	dynamic_routing(output_prim, digit_weights, prediction);
	t_2 = clock();
	// -------------------- Digit Capsule Layer --------------------

	printf("Digit Capsules: %.4fms\n", ((float)(t_2 - t_1) / CLOCKS_PER_SEC) * 1000);

#if DUMP_LAYERS
	fp = fopen("../dump/output_digit_caps.txt", "w");

	for (int i = 0; i < DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE; ++i)
	{
		fprintf(fp, "%.10f\n", prediction[i]);
	}
	fclose(fp);
#endif
}