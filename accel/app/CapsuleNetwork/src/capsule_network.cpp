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

GraphInfo shapes;

int correct_classification = 0;
int total_images = 0;
const string wordsPath = "./";

// ---------------- PRIVATE FUNCTION DECLARATIONS ----------------
void runCapsuleNetwork(vart::RunnerExt *runner, uint32_t num_images, const xir::Subgraph *subgraph, int digitcaps_sw_imp, const string image_path, string label_path, const string weight_path, int verbose);
static void load_mnist_images(string const &image_path, uint32_t num_images, float* images);
static void load_mnist_labels(string const &label_path, uint32_t num_images, vector<uint8_t> *labels);
static void get_data(string const &file_name, uint32_t start_index, vector<float> *output);
static void convert_to_magnitude(float *vector, float *output);
static uint16_t get_max_prediction(float *prediction);
int32_t bytes_to_int(const unsigned char *bytes);
// ---------------------------------------------------------------

/**
 * @brief Load MNIST images
 *
 * @param image_path - const string to image ubyte file
 * @param num_images - num images to extract
 * @param images - output image vector<vector> (2d)
 *
 * @return none
 */
static void load_mnist_images(string const &image_path, uint32_t num_images, float* images)
{
	ifstream img_file(image_path, std::ios::binary);

	// Read headers
	unsigned char header[16];
	img_file.read(reinterpret_cast<char *>(header), 16);

	// Read number of images, rows, and columns
	int32_t total_images = bytes_to_int(header + 4);
	int32_t num_rows = bytes_to_int(header + 8);
	int32_t num_cols = bytes_to_int(header + 12);

	if (num_images > total_images)
	{
		throw runtime_error("Too large of a batch " + image_path);
	}

	// images->resize(num_images);

	for (int i = 0; i < num_images; ++i)
	{
		vector<uint8_t> temp_image(num_rows * num_cols);
		img_file.read(reinterpret_cast<char *>(temp_image.data()), num_rows * num_cols);

		// (*images)[i].resize(num_rows * num_cols);
		for (int j = 0; j < num_rows * num_cols; ++j)
		{
			images[i * num_cols * num_rows + j] = static_cast<float>(temp_image[j]) / 255.0f;
		}
	}

	img_file.close();
}

/**
 * @brief Load MNIST labels
 *
 * @param label_path - const string to ubyte label file
 * @param num_images - num labels to extract
 * @param labels - output vector
 *
 * @return none
 */
static void load_mnist_labels(string const &label_path, uint32_t num_images, vector<uint8_t> *labels)
{
	ifstream label_file(label_path, std::ios::binary);

	// // Read headers
	unsigned char header[8];
	label_file.read(reinterpret_cast<char *>(header), 8);

	int32_t num_labels = bytes_to_int(header + 4);

	if (num_images > num_labels)
	{
		throw runtime_error("Too large of a batch " + label_path);
	}

	labels->resize(num_images);

	for (int i = 0; i < num_images; ++i)
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
 * @param num_images - number of images to test
 * @param subgraph - dpu model ctx
 * @param digitcaps_sw_imp - if 0, run the hardware accelerator
 * @param image_path - path to the MNIST images
 * @param label_path - path to the MNIST labels
 * @param verbose - output predictions
 *
 * @return none
 */
void runCapsuleNetwork(vart::RunnerExt *runner, uint32_t num_images, const xir::Subgraph *subgraph, int digitcaps_sw_imp, const string image_path, string label_path, const string weight_path, int verbose)
{
	vector<vector<float>> images;
	vector<uint8_t> labels;
	vector<float> weights;

	long imread_time = 0, dpu_latency = 0, post_time = 0;

	// auto input_tensor_buffers = runner->get_inputs();
	// auto output_tensor_buffers = runner->get_outputs();
	// CHECK_EQ(input_tensor_buffers.size(), 1u) << "only supports 1 input";

	// auto input_tensor = input_tensor_buffers[0]->get_tensor();
	// auto batch = input_tensor->get_shape().at(0);

	// int height = input_tensor->get_shape().at(1);
	// int width = input_tensor->get_shape().at(2);
	// auto channels = input_tensor->get_shape().at(3);
	// auto input_scale = vart::get_input_scale(input_tensor);
	// auto inSize = height * width * channels;
	// vector<float *> imageList;

	// auto output_tensor = output_tensor_buffers[1]->get_tensor();
	// auto out_height = output_tensor->get_shape().at(1);
	// auto out_width = output_tensor->get_shape().at(2);
	// auto output_scale = vart::get_output_scale(output_tensor);

	// auto osize = out_height * out_width;
	// vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	// uint64_t dpu_input_size = 0u;
	// vector<float *> inptr_v;
	// auto in_dims = input_tensor->get_shape();

	// vector<uint64_t> data_in_addr(batch, 0u);

	// for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	// {
	// 	std::tie(data_in_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
	// 	std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
	// }

	// vector<uint64_t> dpu_output_phy_addr(batch, 0u);
	// uint64_t dpu_output_size = 0u;
	// vector<float *> outptr_v;

	// auto dims =  output_tensor->get_shape();
	// for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	// {
	// 	auto idx = std::vector<int32_t>(dims.size());
	// 	idx[0] = batch_idx;
	// 	auto data = output_tensor_buffers[1]->data(idx);
	// 	float *data_out = (float *)data.first;
	// 	outptr_v.push_back(data_out);
	// 	std::tie(dpu_output_phy_addr[batch_idx], dpu_output_size) = output_tensor_buffers[1]->data_phy({batch_idx, 0, 0, 0});
	// }

	 /* get in/out tensors and dims*/
	auto outputTensors = runner->get_output_tensors();
	auto inputTensors = runner->get_input_tensors();
	auto out_dims = outputTensors[0]->get_shape();
	auto in_dims = inputTensors[0]->get_shape();

	// auto input_scale = get_input_scale(inputTensors[0]);
	// auto output_scale = get_output_scale(outputTensors[0]);

	/*get shape info*/
	int outSize = shapes.outTensorList[0].size;
	int inSize = shapes.inTensorList[0].size;
	int inHeight = shapes.inTensorList[0].height;
	int inWidth = shapes.inTensorList[0].width;

	int batchSize = in_dims[0];

	std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

	vector<Mat> imageList;
	float* imageInputs = new float[inSize * batchSize];

	float* primcaps_output = new float[batchSize * outSize];
	std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
	std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

	auto imread_start = std::chrono::system_clock::now();

	// Load MNIST images and labels
	load_mnist_images(image_path, num_images, imageInputs);
	if (label_path != "")
		load_mnist_labels(label_path, num_images, &labels);

	auto imread_end = std::chrono::system_clock::now();
	auto imread_duration = std::chrono::duration_cast<std::chrono::microseconds>(imread_end - imread_start);
	imread_time += imread_duration.count();

	get_data(weight_path, 0, &weights);

	DigitcapsAcceleratorType *digitcaps_accelerator = nullptr;
	if (!digitcaps_sw_imp)
		digitcaps_accelerator = init_digitcaps_accelerator(weights.data());

	int count = num_images;

	auto start = std::chrono::system_clock::now();

	/*run with batch*/
	for (unsigned int n = 0; n < num_images; n += batchSize)
	{
		unsigned int runSize = (num_images < (n + batchSize)) ? (num_images - n) : batchSize;

		// for (unsigned int i = 0; i < runSize; i++)
		// {

		// 	// Potential Future: Hardware Preprocessing (float multiplication)

		// 	std::memcpy(data_in_addr[i], (const float *)images[i].data(), IN_IMG_ROWS * IN_IMG_COLS * IN_IMG_DEPTH * sizeof(float));

		// 	imageList.push_back(images[i].data());
		// }

		total_images += num_images;
		auto dpu_start = std::chrono::system_clock::now();

		// Potential Future: Hardware Preprocessing (float multiplication)
		// for (auto &input : input_tensor_buffers)
		// 	input->sync_for_write(0, input->get_tensor()->get_data_size() /
		// 								 input->get_tensor()->get_shape()[0]);

		/* in/out tensor refactory for batch inout/output */
		batchTensors.push_back(std::shared_ptr<xir::Tensor>(
			xir::Tensor::create(inputTensors[0]->get_name(),
			in_dims,
			xir::DataType{xir::DataType::FLOAT, 32u})));

		inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
			imageInputs,
			batchTensors.back().get()));

		batchTensors.push_back(std::shared_ptr<xir::Tensor>(
			xir::Tensor::create(outputTensors[0]->get_name(),
			out_dims,
			xir::DataType{xir::DataType::FLOAT, 32u})));

		outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
			primcaps_output,
			batchTensors.back().get()));

		/*tensor buffer input/output */
		inputsPtr.clear();
		outputsPtr.clear();
		inputsPtr.push_back(inputs[0].get());
		outputsPtr.push_back(outputs[0].get());

		// Run DPU
		auto job_id = runner->execute_async(inputsPtr, outputsPtr);
		runner->wait(job_id.first, -1);

		// if (digitcaps_sw_imp)
		// 	for (auto output : output_tensor_buffers)
		// 		output->sync_for_read(0, output->get_tensor()->get_data_size() /
		// 									 output->get_tensor()->get_shape()[0]);

		auto dpu_end = std::chrono::system_clock::now();
		auto dpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpu_end - dpu_start);
		dpu_latency += dpu_duration.count();

		auto post_start = std::chrono::system_clock::now();

		float prediction_data[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE];
		float prediction_magnitude[DIGIT_CAPS_NUM_DIGITS];

		for (unsigned int i = 0; i < num_images; i++)
		{
			// Software DigitCaps
			if (digitcaps_sw_imp)
			{
				dynamic_routing(&primcaps_output[i * outSize], weights.data(), prediction_data);
			}
			// Hardware DigitCaps using zero copy
			else
			{
				run_digitcaps_accelerator(digitcaps_accelerator, (uint64_t)primcaps_output);
				float *out_prediction = (float *)digitcaps_accelerator->prediction_m;
				std::memcpy(prediction_data, out_prediction, DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));
			}

			convert_to_magnitude(prediction_data, prediction_magnitude);
			uint16_t final_answer = get_max_prediction(prediction_magnitude);
			if (final_answer == (uint16_t)labels[i])
				correct_classification++;
		}

		auto post_end = std::chrono::system_clock::now();
		auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
		post_time += post_duration.count();

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
		std::cout << "   Performance: " << 1000000.0 / ((float)((e2e_time) / count)) << " fps\n";
		std::cout << "   Image Read Latency: " << (float)(imread_time / count) / 1000 << " ms\n";
		std::cout << "   DPU Latency: " << (float)(dpu_latency / count) / 1000 << " ms\n";
		std::cout << "   DigitCaps Latency: " << (float)(post_time / count) / 1000 << " ms\n";
	}

	if (labels.size() != 0)
	{
		if (num_images == 0)
		{
			cout << "There are no images to calculate accuracy" << endl;
		}
		else
		{
			cout << correct_classification << " out of " << num_images << " images have been classified correctly" << endl;
			float accuracy = float(correct_classification) / float(num_images) * 100;
			cout << "The accuracy of the network is " << accuracy << " %" << endl;
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc < 7 || argc > 8)
	{
		cout << "Usage: ./CapsuleNetwork.exe <model> <image directory> <sw/hw dynamic routing (1 for sw imp / 0 for hw imp)> <weight_path> <verbose> <num images> <label_file <path> (OPTIONAL)>" << endl;
		return -1;
	}

	auto graph = xir::Graph::deserialize(argv[1]);
	string image_path = argv[2];
	int digitcaps_sw_imp = atoi(argv[3]);
	string weight_path = argv[4];
	auto attrs = xir::Attrs::create();
	int verbose = atoi(argv[5]);
	uint32_t num_images = atoi(argv[6]);

	string label_path = "";
	if (argv[7])
		label_path = argv[7];

	auto subgraph = get_dpu_subgraph(graph.get());

	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	// create DPU task
	std::unique_ptr<vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());

	/*get in/out tensor*/
	auto inputTensors = runner->get_input_tensors();
	auto outputTensors = runner->get_output_tensors();

	/*get in/out tensor shape*/
	int inputCnt = inputTensors.size();
	int outputCnt = outputTensors.size();
	TensorShape inshapes[inputCnt];
	TensorShape outshapes[outputCnt];
	shapes.inTensorList = inshapes;
	shapes.outTensorList = outshapes;
	getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

	runCapsuleNetwork(runner.get(), num_images, subgraph[0], digitcaps_sw_imp, image_path, label_path, weight_path, verbose);
	return 0;
}