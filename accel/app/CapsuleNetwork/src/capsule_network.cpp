/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "DigitCaps.h"
#include "common.h"
#include "experimental/xrt_xclbin.h"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

// C++ Header
#include "accel_wrapper.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

int correct_classification = 0;
int total_images = 0;
const string wordsPath = "./";

// ---------------- PRIVATE FUNCTION DECLARATIONS ----------------
void runCapsuleNetwork(vart::RunnerExt *runner, uint32_t batch_size, const xir::Subgraph *subgraph, int digitcaps_sw_imp, const string image_path, const string label_path, int verbose);
static void load_mnist_images(string const &image_path, uint32_t batch_size, vector<vector<float>> *images);
static void load_mnist_labels(string const &label_path, uint32_t batch_size, vector<uint8_t> *labels);
static void get_data(string const &file_name, uint32_t start_index, vector<float> *output);
static void convert_to_magnitude(float *vector, float *output);
static uint16_t get_max_prediction(float *prediction);
int32_t bytes_to_int(const unsigned char *bytes);
// ---------------------------------------------------------------

/**
 * @brief Load MNIST images
 *
 * @param image_path - const string to image ubyte file
 * @param batch_size - num images to extract
 * @param images - output image vector<vector> (2d)
 *
 * @return none
 */
static void load_mnist_images(string const &image_path, uint32_t batch_size, vector<vector<float>> *images)
{
	ifstream img_file(image_path, std::ios::binary);

	// Read headers
	unsigned char header[16];
	img_file.read(reinterpret_cast<char *>(header), 16);

	// Read number of images, rows, and columns
	int32_t num_images = bytes_to_int(header + 4);
	int32_t num_rows = bytes_to_int(header + 8);
	int32_t num_cols = bytes_to_int(header + 12);

	if (batch_size > num_images)
	{
		throw runtime_error("Too large of a batch " + image_path);
	}

	images->resize(batch_size);

	for (int i = 0; i < batch_size; ++i)
	{
		vector<uint8_t> temp_image(num_rows * num_cols);
		img_file.read(reinterpret_cast<char *>(temp_image.data()), num_rows * num_cols);

		(*images)[i].resize(num_rows * num_cols);
		for (int j = 0; j < num_rows * num_cols; ++j)
		{
			(*images)[i][j] = static_cast<float>(temp_image[j]) / 255.0f;
		}
	}

	img_file.close();
	return 0;
}

/**
 * @brief Load MNIST labels
 *
 * @param label_path - const string to ubyte label file
 * @param batch_size - num labels to extract
 * @param labels - output vector
 *
 * @return none
 */
static void load_mnist_labels(string const &label_path, uint32_t batch_size, vector<uint8_t> *labels)
{
	ifstream label_file(label_path, std::ios::binary);

	// // Read headers
	unsigned char header[8];
	label_file.read(reinterpret_cast<char *>(header), 8);

	int32_t num_labels = bytes_to_int(header + 4);

	if (batch_size > num_labels)
	{
		throw runtime_error("Too large of a batch " + label_path);
	}

	labels->resize(batch_size);

	for (int i = 0; i < batch_size; ++i)
	{
		uint8_t label_entry;
		label_file.read(reinterpret_cast<char *>(&label_entry), 1);
		(*labels)[i] = static_cast<uint8_t>(label_entry);
	}

	label_file.close();
}

/**
 * @brief Get general txt file data for weights
 *
 * @param file_name - path to the file
 * @param start_index - at what point in the array to start placing data
 * @param output - output weight vector
 *
 * @return none
 */
static void get_data(string const &file_name, uint32_t start_index, vector<float> *output)
{
	ifstream file(file_name);
	float entry;

	output->resize(DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE);

	uint32_t i = 0;
	while (file >> entry)
	{
		(*output)[start_index + i++] = entry;
	}
	file.close();
	return 0;
}

/**
 * @brief Get prediction magnitudes
 *
 * @param vector - pointer to prediction data (10x16 vector)
 * @param output - prediction vector magnitudes (0-9)
 *
 * @return none
 */
static void convert_to_magnitude(float *vector, float *output)
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

/**
 * @brief Returns the max prediction vector
 *
 * @param prediction - the output of digitcaps
 *
 * @return the max prediction
 */
static uint16_t get_max_prediction(float *prediction)
{
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

/**
 * @brief Convert bytes into integer form
 *
 * @param bytes - an array of separate bytes
 *
 * @return none
 */
int32_t bytes_to_int(const unsigned char *bytes)
{
	return (int32_t)(((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]));
}

/**
 * @brief Run DPU Task for CapsuleNetwork
 *
 * @param runner - pointer to partial capsule network task
 * @param batch_size - number of images to test
 * @param subgraph - dpu model ctx
 * @param digitcaps_sw_imp - if 0, run the hardware accelerator
 * @param image_path - path to the MNIST images
 * @param label_path - path to the MNIST labels
 * @param verbose - output predictions
 *
 * @return none
 */
void runCapsuleNetwork(vart::RunnerExt *runner, uint32_t batch_size, const xir::Subgraph *subgraph, int digitcaps_sw_imp, const string image_path, const string label_path, const string weight_path, int verbose)
{
	vector<vector<float>> images;
	vector<uint8_t> labels;
	vector<float> weights;

	// Load MNIST images and labels
	load_mnist_images(image_path, batch_size, &images);
	load_mnist_labels(label_path, batch_size, &labels);

	// cout << (int)labels[0] << endl;

	if (images.size() == 0)
	{
		cerr << "\nError: No images loaded" << endl;
		return;
	}

	auto input_tensor_buffers = runner->get_inputs();
	auto output_tensor_buffers = runner->get_outputs();
	CHECK_EQ(input_tensor_buffers.size(), 1u) << "only supports 1 input";

	auto input_tensor = input_tensor_buffers[0]->get_tensor();
	auto batch = input_tensor->get_shape().at(0);

	int height = input_tensor->get_shape().at(1);
	int width = input_tensor->get_shape().at(2);
	auto channels = input_tensor->get_shape().at(3);
	auto input_scale = vart::get_input_scale(input_tensor);
	auto inSize = height * width * channels;
	vector<Mat> imageList;

	auto output_tensor = output_tensor_buffers[1]->get_tensor();
	auto out_height = output_tensor->get_shape().at(1);
	auto out_width = output_tensor->get_shape().at(2);
	auto output_scale = vart::get_output_scale(output_tensor);

	auto osize = out_height * out_width;
	vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	uint64_t dpu_input_size = 0u;
	vector<int8_t *> inptr_v;
	auto in_dims = input_tensor->get_shape();

	vector<uint64_t> data_in_addr(batch, 0u);

	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	{
		std::tie(data_in_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
		std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
	}

	vector<uint64_t> dpu_output_phy_addr(batch, 0u);
	uint64_t dpu_output_size = 0u;
	vector<int8_t *> outptr_v;

	auto dims = output_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	{
		auto idx = std::vector<int32_t>(dims.size());
		idx[0] = batch_idx;
		auto data = output_tensor_buffers[1]->data(idx);
		int8_t *data_out = (int8_t *)data.first;
		outptr_v.push_back(data_out);
		std::tie(dpu_output_phy_addr[batch_idx], dpu_output_size) = output_tensor_buffers[1]->data_phy({batch_idx, 0, 0, 0});
	}

	get_data(weight_path, 0, weights);

	DigitcapsAcceleratorType *digitcaps_accelerator = nullptr;
	if (!digitcaps_sw_imp)
		digitcaps_accelerator = init_digitcaps_accelerator(weights.data(), no_zcpy);

	int count = images.size();

	auto start = std::chrono::system_clock::now();

	/*run with batch*/
	for (unsigned int n = 0; n < images.size(); n += batch)
	{
		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;

		for (unsigned int i = 0; i < runSize; i++)
		{
			auto t1 = std::chrono::system_clock::now();

			// Potential Future: Hardware Preprocessing (float multiplication)

			std::memcpy(data_in_addr[i], (const float *)images[i].data(), IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));

			imageList.push_back(images[i].data());
		}

		total_images += imageList.size();
		auto exec_t1 = std::chrono::system_clock::now();

		// Potential Future: Hardware Preprocessing (float multiplication)
		for (auto &input : input_tensor_buffers)
			input->sync_for_write(0, input->get_tensor()->get_data_size() /
										 input->get_tensor()->get_shape()[0]);

		// Run DPU
		auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
		runner->wait(job_id.first, -1);

		if (digitcaps_sw_imp)
			for (auto output : output_tensor_buffers)
				output->sync_for_read(0, output->get_tensor()->get_data_size() /
											 output->get_tensor()->get_shape()[0]);

		auto exec_t2 = std::chrono::system_clock::now();
		auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
		exec_time += execvalue_t1.count();

		float prediction_data[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE];
		float prediction_magnitude[DIGIT_CAPS_NUM_DIGITS];

		for (unsigned int i = 0; i < imageList.size(); i++)
		{
			// Software DigitCaps
			if (digitcaps_sw_imp)
			{
				auto *out_data_sw = (float *)outptr_v[i];
				dynamic_routing(out_data_sw, weights.data(), prediction_data);
			}
			// Hardware DigitCaps using zero copy
			else
			{
				run_digitcaps_accelerator(digitcaps_accelerator, dpu_output_phy_addr[i]);
				float *out_prediction = (float *)digitcaps_accelerator->prediction_m;
				std::memcpy(prediction_data, out_prediction, DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));
			}

			convert_to_magnitude(prediction_data, prediction_magnitude);
			uint16_t final_answer = get_max_prediction(prediction_magnitude);
			if (final_answer == labels[i])
				correct_classification++;
		}

		auto post_t2 = std::chrono::system_clock::now();
		auto postvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(post_t2 - exec_t2);
		post_time += postvalue_t1.count();

		imageList.clear();
	}

	auto end = std::chrono::system_clock::now();
	auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	long e2e_time = value_t1.count();

	if (!verbose)
	{
		if (digitcaps_sw_imp)
			cout << "Profiling result with software digit caps: " << endl;
		else
			cout << "Profiling result with hardware digit caps " << endl;
		std::cout << "   E2E Performance: " << 1000000.0 / ((float)((e2e_time) / count)) << " fps\n";
		std::cout << "   Pre-process Latency: " << (float)(pre_time / count) / 1000 << " ms\n";
		std::cout << "   DPU Latency: " << (float)(exec_time / count) / 1000 << " ms\n";
		std::cout << "   DigitCaps Latency: " << (float)(post_time / count) / 1000 << " ms\n";
	}

	if (labels.size() != 0)
	{
		if (total_images == 0)
		{
			cout << "There are no images to calculate accuracy" << endl;
		}
		else
		{
			cout << correct_classification << " out of " << total_images << " images have been classified correctly" << endl;
			float accuracy = float(correct_classification) / float(total_images) * 100;
			cout << "The accuracy of the network is " << accuracy << " %" << endl;
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc < 6 || argc > 7)
	{
		cout << "Usage: ./CapsuleNetwork.exe <model> <image directory> <sw/hw dynamic routing (1 for sw imp / 0 for hw imp)> <weight_path> <verbose> <label_file <path> (OPTIONAL)>" << endl;
		return -1;
	}

	auto graph = xir::Graph::deserialize(argv[1]);
	string image_path = argv[2];
	int digitcaps_sw_imp = atoi(argv[3]);
	string weight_path = argv[4];
	auto attrs = xir::Attrs::create();
	int verbose = atoi(argv[5]);

	string label_path = "";
	if (argv[6])
		label_path = argv[6];

	auto subgraph = get_dpu_subgraph(graph.get());

	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	/*create runner*/
	std::unique_ptr<vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());

	/*run with batch*/
	runCapsuleNetwork(runner.get(), subgraph[0], digitcaps_sw_imp, image_path, label_path, weight_path, verbose);
	return 0;
}