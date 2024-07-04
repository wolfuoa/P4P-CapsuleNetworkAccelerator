/*
 * CapsuleNetworkTestbench.cpp
 * author: nicholas wolf
 *
 * Tests the capsule network with MNIST images
 */

#include <math.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "CapsuleNetworkTest.h"
#include "constants.h"
#include "testing_suite.h"

static errno_t get_data(const std::string& file_name, uint32_t start_index, float* output);

static errno_t get_data_n(const char* file_name, uint32_t data_amount, float* output);

static uint16_t get_max_prediction(float* prediction);

static void convert_to_magnitude(float* vector, float* output);

int main(void)
{
	uint32_t conv1_num_weights = CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * CONV1_FILTERS;
	uint32_t primary_caps_num_weights = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES;
	uint32_t digitcaps_num_weights = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE;

	float* weights = (float*)malloc((conv1_num_weights + primary_caps_num_weights + digitcaps_num_weights) * sizeof(float));
	float biases[CONV1_FILTERS + PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
	// float images[IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * NUM_IMAGES_TO_TEST];
	float image[IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH];
	// float* biases= (float*)malloc((CONV1_FILTERS + PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES) * sizeof(float));
	// float* image= (float*)malloc(IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));

	float labels[NUM_IMAGES_TO_TEST];
	float prediction[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE];
	float magnitudes[DIGIT_CAPS_NUM_DIGITS];

	// float* biases = (float*)malloc(CONV1_FILTERS + PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	float* images = (float*)malloc(IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * NUM_IMAGES_TO_TEST * sizeof(float));
	// float* label = (float*)malloc(DIGIT_CAPS_NUM_DIGITS * sizeof(float));
	// float* prediction = (float*)malloc(DIGIT_CAPS_NUM_DIGITS * sizeof(float));

	get_data("../../models/conv1_weights.txt", 0, weights);
	get_data("../../models/primcaps_weights.txt", conv1_num_weights, weights);
	get_data("../../models/digitcaps_weights.txt", conv1_num_weights + primary_caps_num_weights, weights);

	get_data("../../models/conv1_biases.txt", 0, biases);
	get_data("../../models/primcaps_biases.txt", CONV1_FILTERS, biases);

	get_data_n("../../datasets/MNIST/testimg.txt", IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * NUM_IMAGES_TO_TEST, images);
	get_data_n("../../datasets/MNIST/labels.txt", NUM_IMAGES_TO_TEST, labels);

	for (uint8_t i = 0; i < NUM_IMAGES_TO_TEST; ++i)
	{
		// acquire next image to test
		memcpy(image, (const float*)images + i * IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH, IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));

		clock_t t_1 = clock();
		get_prediction(image, weights, biases, prediction);
		clock_t t_2 = clock();

		convert_to_magnitude(prediction, magnitudes);

		for (uint8_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
		{
			std::cout << "pred: " << magnitudes[i] << std::endl;
		}
		uint16_t max_prediction = get_max_prediction(magnitudes);
		std::cout << "Label: " << labels[i] << std::endl;
		std::cout << "CapsNet prediction: " << max_prediction << std::endl;
		printf("Time: %.4fms\n", ((float)(t_2 - t_1) / CLOCKS_PER_SEC) * 1000);
	}

	free(images);
	free(weights);
}

static errno_t get_data(const std::string& file_name, uint32_t start_index, float* output)
{
	std::ifstream file(file_name);
	float entry;
	uint32_t i = 0;
	while (file >> entry)
	{
		output[start_index + i++] = entry;
	}
	file.close();
	return 0;
}

static errno_t get_data_n(const char* file_name, uint32_t data_amount, float* output)
{
	FILE* fp;

	fp = fopen(file_name, "r");

	if (fp == NULL)
	{
		std::cout << "Error" << std::endl;
		return 1;
	}

	for (uint32_t i = 0; i < data_amount; ++i)
	{
		(void)fscanf(fp, "%f", &output[i]);
	}

	return fclose(fp);
}

static uint16_t get_max_prediction(float* prediction)
{
	// [0..9]
	// prediction[0] -> 0 = 0.23492
	uint16_t digit;
	float currentLargest = 0.0;
	for (uint16_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		if (prediction[i] > currentLargest)
		{
			digit = i;
			currentLargest = prediction[i];
		}
	}
	return digit;
}

static void convert_to_magnitude(float* vector, float* output)
{
	float sum = 0.0;

	for (uint8_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		sum = 0.0;
		for (uint8_t j = 0; j < DIGIT_CAPS_DIM_CAPSULE; ++j)
		{
			float value = vector[i * DIGIT_CAPS_DIM_CAPSULE + j];
			sum += value * value;
		}
		output[i] = sqrt(sum);
	}
}