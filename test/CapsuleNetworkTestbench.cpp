/*
 * CapsuleNetworkTestbench.cpp
 * author: nicholas wolf
 *
 * Tests the capsule network with MNIST images
 */

#include <math.h>
#include <time.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CapsuleNetwork.h"
#include "constants.h"
#include "testing_suite.h"

static errno_t get_data(const std::string& file_name, uint32_t start_index, float* output);

static errno_t get_data_n(const char* file_name, uint32_t data_amount, float* output);

static errno_t load_mnist_images(const std::string& image_path, uint32_t batch_size, std::vector<std::vector<float>>* images);

static errno_t load_mnist_labels(const std::string& label_path, uint32_t batch_size, std::vector<uint8_t>* labels);

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
	float prediction[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE] = {0.0};
	float image[IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH];
	float magnitudes[DIGIT_CAPS_NUM_DIGITS];
	std::vector<std::vector<float>> mnist;
	std::vector<uint8_t> labels;

	// Load MNIST data
	load_mnist_images("../../datasets/MNIST/t10k-images-idx3-ubyte", NUM_IMAGES_TO_TEST, &mnist);
	load_mnist_labels("../../datasets/MNIST/t10k-labels-idx1-ubyte", NUM_IMAGES_TO_TEST, &labels);

	for (uint8_t i = 0; i < NUM_IMAGES_TO_TEST; ++i)
	{
		// acquire next image to test
		memcpy(image, (const float*)mnist[i].data(), IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));

		get_data("../../models/conv1_weights.txt", 0, weights);
		get_data("../../models/primcaps_weights.txt", conv1_num_weights, weights);
		get_data("../../models/digitcaps_weights.txt", conv1_num_weights + primary_caps_num_weights, weights);

		get_data("../../models/conv1_biases.txt", 0, biases);
		get_data("../../models/primcaps_biases.txt", CONV1_FILTERS, biases);

		clock_t t_1 = clock();
		get_prediction(image, weights, biases, prediction);
		clock_t t_2 = clock();

		convert_to_magnitude(prediction, magnitudes);

		for (uint8_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
		{
			std::cout << "pred: " << magnitudes[i] << std::endl;
		}
		uint16_t max_prediction = get_max_prediction(magnitudes);
		std::cout << "Label: " << (int)labels[i] << std::endl;
		std::cout << "CapsNet prediction: " << max_prediction << std::endl;
		printf("Time: %.4fms\n", ((float)(t_2 - t_1) / CLOCKS_PER_SEC) * 1000);
	}

	free(weights);
}

int32_t bytes_to_int(const unsigned char* bytes)
{
	return (int32_t)(((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]));
}

static errno_t load_mnist_labels(const std::string& label_path, uint32_t batch_size, std::vector<uint8_t>* labels)
{
	std::ifstream label_file(label_path, std::ios::binary);

	// // Read headers
	unsigned char header[8];
	label_file.read(reinterpret_cast<char*>(header), 8);

	int32_t num_labels = bytes_to_int(header + 4);

	if (batch_size > num_labels)
	{
		throw std::runtime_error("Too large of a batch " + label_path);
	}

	labels->resize(batch_size);

	for (int i = 0; i < batch_size; ++i)
	{
		uint8_t label_entry;
		label_file.read(reinterpret_cast<char*>(&label_entry), 1);
		(*labels)[i] = static_cast<uint8_t>(label_entry);
	}

	label_file.close();
	return 0;
}

static errno_t load_mnist_images(const std::string& image_path, uint32_t batch_size, std::vector<std::vector<float>>* images)
{
	std::ifstream img_file(image_path, std::ios::binary);

	// Read headers
	unsigned char header[16];
	img_file.read(reinterpret_cast<char*>(header), 16);

	// Read number of images, rows, and columns
	int32_t num_images = bytes_to_int(header + 4);
	int32_t num_rows = bytes_to_int(header + 8);
	int32_t num_cols = bytes_to_int(header + 12);

	if (batch_size > num_images)
	{
		throw std::runtime_error("Too large of a batch " + image_path);
	}

	images->resize(batch_size);

	for (int i = 0; i < batch_size; ++i)
	{
		std::vector<uint8_t> temp_image(num_rows * num_cols);
		img_file.read(reinterpret_cast<char*>(temp_image.data()), num_rows * num_cols);

		(*images)[i].resize(num_rows * num_cols);
		for (int j = 0; j < num_rows * num_cols; ++j)
		{
			(*images)[i][j] = static_cast<float>(temp_image[j]) / 255.0f;
		}
	}

	img_file.close();
	return 0;
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