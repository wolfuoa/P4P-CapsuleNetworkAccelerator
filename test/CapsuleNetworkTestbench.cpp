/*
 * CapsuleNetworkTestbench.cpp
 * author: nicholas wolf
 *
 * Tests the capsule network with MNIST images
 */

#define PRINT_DEBUG 0

#include <fstream>
#include <iostream>
#include <string>

#include "CapsuleNetwork.h"
#include "constants.h"

static errno_t get_data(const std::string& file_name, uint32_t start_index, float* output);

static errno_t get_data_n(const char* file_name, uint32_t data_amount, float* output);

static uint16_t get_max_prediction(float* prediction);

int main(void)
{
	uint32_t conv1_num_weights = CONV1_KERNEL_ROWS * CONV1_KERNEL_COLS * CONV1_FILTERS;
	uint32_t primary_caps_num_weights = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES;
	uint32_t digitcaps_num_weights = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE;

	float* weights = (float*)malloc((conv1_num_weights + primary_caps_num_weights + digitcaps_num_weights) * sizeof(float));
	float biases[CONV1_FILTERS + PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
	float image[IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH];
	float label[DIGIT_CAPS_NUM_DIGITS];

	// float* biases = (float*)malloc(CONV1_FILTERS + PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(float));
	// float* image = (float*)malloc(IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));
	// float* label = (float*)malloc(DIGIT_CAPS_NUM_DIGITS * sizeof(float));

	float prediction[DIGIT_CAPS_NUM_DIGITS];

	get_data("../../models/quant_conv1_kernel_float.txt", 0, weights);
	get_data("../../models/quant_primarycap_conv2d_kernel_float.txt", conv1_num_weights, weights);
	get_data("../../models/digitcaps_w_float.txt", conv1_num_weights + primary_caps_num_weights, weights);

	get_data("../../models/quant_conv1_bias_float.txt", 0, biases);
	get_data("../../models/quant_primarycap_conv2d_bias_float.txt", CONV1_FILTERS, biases);

	// TODO: Get successive images
	get_data_n("../../datasets/MNIST/t10k-images-idx3-ubyte", IN_IMG_ROWS * IN_IMG_COLS, image);
	get_data_n("../../datasets/MNIST/t10k-labels-idx1-ubyte", DIGIT_CAPS_NUM_DIGITS, label);

#if PRINT_DEBUG
	for (uint8_t spec; spec < 10; ++spec)
	{
		std::cout << weights[spec] << std::endl;
	}

	std::cout << std::endl;

	for (uint8_t spec; spec < 10; ++spec)
	{
		std::cout << biases[spec] << std::endl;
	}

	std::cout << std::endl;

	for (uint16_t spec; spec < IN_IMG_ROWS * IN_IMG_COLS; ++spec)
	{
		std::cout << image[spec] << std::endl;
	}

	std::cout << std::endl;

	for (uint8_t spec; spec < 10; ++spec)
	{
		std::cout << label[spec] << std::endl;
	}

	std::cout << get_max_prediction(label) << std::endl;

#endif	// PRINT_DEBUG

	get_prediction(image, weights, biases, prediction);

	for (uint8_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		std::cout << "label: " << label[i] << std::endl;
		std::cout << "pred: " << prediction[i] << std::endl;
	}
	uint16_t max_prediction = get_max_prediction(prediction);
	std::cout << "Most likely prediction: " << max_prediction << std::endl;

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