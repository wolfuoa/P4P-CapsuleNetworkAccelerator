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

#include <sys/time.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "constants.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

typedef struct DigitcapsAcceleratorType_t
{
	xrt::kernel kernel;
	xrt::device device;
	xrt::run runner;
	xrt::bo input;
	xrt::bo weights;
	xrt::bo prediction;
	void *input_m;
	void *weights_m;
	void *prediction_m;
} DigitcapsAcceleratorType;

static std::vector<std::string> get_xclbins_in_dir(std::string path)
{
	if (path.find(".xclbin") != std::string::npos)
		return {path};

	std::vector<std::string> xclbinPaths;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(path.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL)
		{
			std::string name(ent->d_name);
			if (name.find(".xclbin") != std::string::npos)
				xclbinPaths.push_back(path + "/" + name);
		}
		closedir(dir);
	}
	return xclbinPaths;
}

int run_digitcaps_accelerator(DigitcapsAcceleratorType *a, uint64_t dpu_output_phy_addr)
{
	// Input size to transfer
	// const int volume_size = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE;

	// Copy to input buffer
	// std::memcpy(a->input_m, input_volume, volume_size);

	// Send the input volume data to the device memory
	// a->input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE, volume_size, 0);

	// Invoke accelerator
	a->runner(dpu_output_phy_addr, a->weights, a->prediction_m);

	// Wait for accelerator to finish processing
	a->runner.wait();

	// Extraneous probably
	a->prediction.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);

	// Return success
	return 0;
}

DigitcapsAcceleratorType *init_digitcaps_accelerator(float *weights_array)
{
	// get xclbin dir path and acquire handle
	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");

	if (xclbinPath == nullptr)
		throw std::runtime_error("Error: xclbinPath is not set, please consider setting XLNX_VART_FIRMWARE.");

	// get available xclbins
	auto xclbins = get_xclbins_in_dir(xclbinPath);
	const char *xclbin = xclbins[0].c_str();

	// Device/Card Index on system
	unsigned device_index = 0;

	// Check for devices on the system
	if (device_index >= xclProbe())
	{
		throw std::runtime_error("Cannot find device index specified");
		return nullptr;
	}

	// Acquire Device by index
	auto device = xrt::device(device_index);
	// Load XCLBIN
	auto uuid = device.load_xclbin(xclbin);
	// Get DigitCaps Kernel CU
	auto digitcaps_accelerator = xrt::kernel(device, uuid.get(), "digitcaps_accel");

	// Get runner instance from xrt
	auto runner = xrt::run(digitcaps_accelerator);
	// Create BO for input/output/params

	auto input_mem_grp = digitcaps_accelerator.group_id(0);
	auto weights_mem_grp = digitcaps_accelerator.group_id(1);
	auto prediction_mem_grp = digitcaps_accelerator.group_id(2);

	// Create memory for input volume
	const int input_size = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(float);

	auto input = xrt::bo(device, input_size, input_mem_grp);

	void *input_m = input.map();
	if (input_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");

	// Create memory for weights
	const int weights_size = DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(float);

	auto weights = xrt::bo(device, weights_size, weights_mem_grp);
	void *weights_m = weights.map();
	if (weights_m == nullptr)
		throw std::runtime_error("[ERRR] Weights pointer is invalid\n");

	// Copy weights into device
	std::memcpy(weights_m, weights_array, weights_size);

	// Send the weight data to device memory
	weights.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);

	auto a = new DigitcapsAcceleratorType();
	// 	// Create memory for output vector
	const int prediction_size = DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(float);

	auto prediction = xrt::bo(device, prediction_size, prediction_mem_grp);
	void *prediction_m = prediction.map();
	if (prediction_m == nullptr)
		throw std::runtime_error("[ERRR] Prediction pointer is invalid\n");

	a->kernel = std::move(digitcaps_accelerator);
	a->device = std::move(device);
	a->runner = std::move(runner);
	a->input = std::move(input);
	a->weights = std::move(weights);
	a->prediction = std::move(prediction);
	a->input_m = input_m;
	a->weights_m = weights_m;
	a->prediction_m = prediction_m;

	// Return accelerator
	return a;
}
