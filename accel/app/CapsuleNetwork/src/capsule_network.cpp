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
void runCapsuleNetwork(vart::RunnerExt *runner, const xir::Subgraph *subgraph, int digitcaps_sw_imp, int no_zcpy, const string baseImagePath, string label_path, int verbose);
static void load_mnist_images(string const &image_path, uint32_t batch_size, vector<vector<float>> *images);
static void load_mnist_labels(string const &label_path, uint32_t batch_size, vector<uint8_t> *labels);
static void convert_to_magnitude(float *vector, float *output);
int32_t bytes_to_int(const unsigned char *bytes);
// ---------------------------------------------------------------

static void load_mnist_images(string const &image_path, uint32_t batch_size, vector<vector<float>> *images)
{
	std::ifstream img_file(image_path, std::ios::binary);

	// Read headers
	unsigned char header[16];
	img_file.read(reinterpret_cast<char *>(header), 16);

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

static void load_mnist_labels(string const &label_path, uint32_t batch_size, vector<uint8_t> *labels)
{
	std::ifstream label_file(label_path, std::ios::binary);

	// // Read headers
	unsigned char header[8];
	label_file.read(reinterpret_cast<char *>(header), 8);

	int32_t num_labels = bytes_to_int(header + 4);

	if (batch_size > num_labels)
	{
		throw std::runtime_error("Too large of a batch " + label_path);
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
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
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
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
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
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runCapsuleNetwork(vart::RunnerExt *runner, uint32_t batch_size, const xir::Subgraph *subgraph, int digitcaps_sw_imp, int no_zcpy, const string baseImagePath, string label_path, int verbose)
{
	vector<vector<float>> images;
	vector<int> labels;

	// Load MNIST images and labels
	load_mnist(baseImagePath, label_path, batch_size, &images, &labels);

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

	// TODO: Implement weights grab function

	DigitcapsAcceleratorType *digitcaps_accelerator = nullptr;
	if (!digitcaps_sw_imp)
		digitcaps_accelerator = init_digitcaps_accelerator(weights, no_zcpy);

	int count = images.size();

	auto start = std::chrono::system_clock::now();

	/*run with batch*/
	for (unsigned int n = 0; n < images.size(); n += batch)
	{
		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;

		for (unsigned int i = 0; i < runSize; i++)
		{
			auto t1 = std::chrono::system_clock::now();

			if (labels.size() != 0 && !filesystem::exists(baseImagePath + "/" + images[n + i]))
			{
				cout << "The image file " << baseImagePath + "/" + images[n + i] << " doesnot exist in the image directory " << baseImagePath << " (SKIPPING)" << endl;
				continue;
			}

			// Mat image = imread(baseImagePath + "/" + images[n + i]);
			float image;
			get_data(baseImagePath + "/" + images[n + i], image);

			if (!no_zcpy)
			{
				std::memcpy(dpu_input_phy_addr[i], image, IN_IMG_ROWS * IN_IMG_COLS * sizeof(float));
			}
			else
			{
				std::memcpy(dpu_in_addr[i], image, IN_IMG_ROWS * IN_IMG_COLS * sizeof(float));
			}

			imageList.push_back(image);
		}

		total_images += imageList.size();
		auto exec_t1 = std::chrono::system_clock::now();

		if (no_zcpy)
			for (auto &input : input_tensor_buffers)
				input->sync_for_write(0, input->get_tensor()->get_data_size() /
											 input->get_tensor()->get_shape()[0]);

		/*run*/
		auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
		runner->wait(job_id.first, -1);

		for (auto output : output_tensor_buffers)
			output->sync_for_read(0, output->get_tensor()->get_data_size() /
										 output->get_tensor()->get_shape()[0]);

		auto exec_t2 = std::chrono::system_clock::now();
		auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
		exec_time += execvalue_t1.count();

		for (unsigned int i = 0; i < imageList.size(); i++)
		{
			/* Calculate softmax on CPU and display TOP-5 classification results */
			CPUCalcSoftmax(outptr_v[i], outSize, softmax, output_scale);

			TopK(softmax, outSize, 5, kinds, images, labels, n + i, verbose);

			/* Display the impage */
			// cv::imshow("Classification of ResNet50", imageList[i]);
			// cv::waitKey(10000);
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
			cout << "Profiling result with software preprocessing: " << endl;
		else if (no_zcpy)
			cout << "Profiling result with hardware preprocessing without zero copy: " << endl;
		else
			cout << "Profiling result with hardware preprocessing with zero copy: " << endl;
		std::cout << "   E2E Performance: " << 1000000.0 / ((float)((e2e_time - imread_time) / count)) << " fps\n";
		std::cout << "   Pre-process Latency: " << (float)(pre_time / count) / 1000 << " ms\n";
		std::cout << "   Execution Latency: " << (float)(exec_time / count) / 1000 << " ms\n";
		std::cout << "   Post-process Latency: " << (float)(post_time / count) / 1000 << " ms\n";
	}
#if EN_BRKP
	std::cout << "imread latency: " << (float)(imread_time / count) / 1000 << "ms\n";
#endif

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
	delete[] softmax;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char *argv[])
{
	if (argc < 6 || argc > 7)
	{
		cout << "Usage: ./CapsuleNetwork.exe <model (-t to just test hardware accelerator)> <image directory> <sw/hw dynamic routing (1 for sw imp / 0 for hw imp)> <no_zero_copy (1 for no zero copy / 1 for zero copy)> <verbose> <label_file <path> (OPTIONAL)>" << endl;
		return -1;
	}

	auto graph = xir::Graph::deserialize(argv[1]);
	string baseImagePath = argv[2];
	int digitcaps_sw_imp = atoi(argv[3]);
	int no_zcpy = atoi(argv[4]);
	auto attrs = xir::Attrs::create();
	int verbose = atoi(argv[5]);

	string label_path = "";
	if (argv[6])
		label_path = argv[6];

	if (digitcaps_sw_imp)
		no_zcpy = 1;

	auto subgraph = get_dpu_subgraph(graph.get());

	if (!no_zcpy)
		attrs->set_attr<bool>("zero_copy", true);

	CHECK_EQ(subgraph.size(), 1u)
		<< "resnet50 should have one and only one dpu subgraph.";
	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	/*create runner*/
	std::unique_ptr<vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());

	/*run with batch*/
	runCapsuleNetwork(runner.get(), subgraph[0], digitcaps_sw_imp, no_zcpy, baseImagePath, label_path, verbose);
	return 0;
}
