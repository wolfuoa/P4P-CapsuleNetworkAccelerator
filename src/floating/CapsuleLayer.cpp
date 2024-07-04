#include "CapsuleLayer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
using namespace std;

#ifndef __SYNTHESIS__
#include <math.h>
#else
#include <hls_math.h>
#endif

// Recommended: Avoid use of global variables for loop index variables, as this can inhibit some optimizations.
uint32_t g_weightSize = NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE * INPUT_DIM_CAPSULE;
uint32_t g_inputSize = INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE;
uint32_t g_inputHatSize = NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE;
uint32_t g_bcSize = NUM_CAPSULE * INPUT_NUM_CAPSULE;
uint32_t g_outputsMultiplySize = NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE;
uint32_t g_outputsSumSize = NUM_CAPSULE * DIM_CAPSULE;
uint32_t g_outputsSquashSize = g_outputsSumSize;
uint32_t g_outputsSquashSumSize = NUM_CAPSULE;
uint32_t g_outputsAgreementSize = NUM_CAPSULE * INPUT_NUM_CAPSULE;

// not synthesized code
#ifndef __SYNTHESIS__
float weights[NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE * INPUT_DIM_CAPSULE];
float input[INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE];
float outputsSum[NUM_CAPSULE * DIM_CAPSULE];

typedef std::numeric_limits<float> dbl;

// DEBUGGING: run dynamic routing from within the file (don't pass in inputs)
void runInFile()
{
	initialiseInputAndWeightArray();

	run(input, weights, outputsSum);

	cout.precision(dbl::max_digits10);
	for (uint32_t i = 0; i < g_outputsSumSize; i++)
	{
		cout << outputsSum[i] << endl;
	}
}

#endif

// capsule layer implementation
void run(float *input, float *weights, float *output)
{
// input/output axi buses
// automatically bundled into control_r s_axilite port
#pragma HLS INTERFACE mode = m_axi port = input offset = slave bundle = gmem0 max_read_burst_length = 256 max_write_burst_length = 256 depth = 9216
#pragma HLS INTERFACE mode = m_axi port = weights offset = slave bundle = gmem1 max_read_burst_length = 256 max_write_burst_length = 256 depth = 1474560
#pragma HLS INTERFACE mode = m_axi port = output offset = slave bundle = gmem2 max_read_burst_length = 256 max_write_burst_length = 256

	// partition input array, set cyclic array partitioning
	float *inputC = (float *)malloc(INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE * sizeof(float));
#pragma HLS bind_storage variable = inputC impl = auto
#pragma HLS array_partition type = cyclic factor = 8 dim = 1 variable = inputC

	// local array to store output
	float *outputC = (float *)malloc(NUM_CAPSULE * DIM_CAPSULE * sizeof(float));
#pragma HLS bind_storage variable = outputC impl = auto

	// burst read input into local array
	memcpy(inputC, (const float *)input, INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE * sizeof(float));

	// create arrays for all internal operations and bind them to RAM components
	static float inputHat[NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE];
#pragma HLS bind_storage variable = inputHat impl = auto

	static float b[NUM_CAPSULE * INPUT_NUM_CAPSULE];
#pragma HLS bind_storage variable = b type = RAM_1WNR impl = auto

	static float c[NUM_CAPSULE * INPUT_NUM_CAPSULE];
#pragma HLS bind_storage variable = c type = RAM_1WNR impl = auto

	static float outputsMultiply[NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE];
#pragma HLS bind_storage variable = outputsMultiply type = RAM_1WNR impl = auto

	static float outputsSquashSum[NUM_CAPSULE];
#pragma HLS bind_storage variable = outputsSquashSum type = RAM_1WNR impl = auto

	static float outputsAgreement[NUM_CAPSULE * INPUT_NUM_CAPSULE];
#pragma HLS bind_storage variable = outputsAgreement type = RAM_1WNR impl = auto

	getInputHat(inputC, weights, inputHat);

	// number of dynamic routing iterations (as per original CapsNet implementation)
	int numIter = 3;
dynamicRoutingLoop:
	for (int i = 0; i < numIter; i++)
	{
		getSoftmax(b, c);
		getMultiply(inputHat, c, outputsMultiply);
		getSum(outputsMultiply, outputC);
		getSquash(outputC, outputsSquashSum);
		if (i < numIter - 1)
		{
			getAgreement(inputHat, outputC, outputsAgreement);
			getAdd(outputsAgreement, b);
		}
	}

	// burst send output
	memcpy(output, (const float *)outputC, NUM_CAPSULE * DIM_CAPSULE * sizeof(float));
}

// add agreement to b and store in b matrix
void getAdd(float *A, float *B)
{
addLoop:
	for (uint32_t i = 0; i < NUM_CAPSULE * INPUT_NUM_CAPSULE; i++)
	{
#pragma HLS BIND_OP variable = B op = fadd impl = fulldsp
#pragma HLS PIPELINE
		B[i] = A[i] + B[i];
	}
}

// matrix multiplication to compute agreement
void getAgreement(float *inputHat, float *outputsSquash, float *agreement)
{
	uint32_t startIndexA = 0;
	uint32_t startIndexB = 0;
	uint32_t startIndexC = 0;
agreementOuterLoop:
	for (int i = 0; i < NUM_CAPSULE; i++)
	{
	agreementInnerLoop:
		for (int j = 0; j < INPUT_NUM_CAPSULE; j++)
		{
			// identify starting indexes for inputHat, squash, and agreement
			startIndexA = i * INPUT_NUM_CAPSULE * DIM_CAPSULE + j * DIM_CAPSULE;
			startIndexB = i * DIM_CAPSULE;
			startIndexC = i * INPUT_NUM_CAPSULE + j;

			// matrix multiplication
			matmulAgreement(inputHat, outputsSquash, agreement, startIndexA, startIndexB, startIndexC);
		}
	}
}

// matrix multiplication to compute inputHat
void getInputHat(float *input, float *weights, float *output)
{
	uint32_t startIndexA = 0;
	uint32_t startIndexB = 0;
	uint32_t startIndexC = 0;

	// index dimensions size for weight array
	uint32_t dimA1 = INPUT_NUM_CAPSULE * DIM_CAPSULE * INPUT_DIM_CAPSULE;  // 3rd dim
	uint32_t dimA2 = DIM_CAPSULE * INPUT_DIM_CAPSULE;					   // 2nd dim

	// index dimension size for inputHat array
	uint32_t dimC1 = DIM_CAPSULE * INPUT_NUM_CAPSULE;  // 2nd dim

inputHatOuterLoop:
	for (uint32_t i = 0; i < NUM_CAPSULE; i++)
	{
	inputHatInnerLoop:
		for (uint32_t j = 0; j < INPUT_NUM_CAPSULE; j++)
		{
			// burst read weight array in small chunks
			float weightsC[DIM_CAPSULE * INPUT_DIM_CAPSULE];
#pragma HLS bind_storage variable = weightsC type = RAM_1WNR impl = bram
#pragma HLS array_partition type = cyclic factor = 8 dim = 1 variable = weightsC

			memcpy(weightsC, (const float *)weights + dimA1 * i + dimA2 * j, DIM_CAPSULE * INPUT_DIM_CAPSULE * sizeof(float));

			// identify starting indexes
			startIndexB = INPUT_DIM_CAPSULE * j;
			startIndexC = dimC1 * i + DIM_CAPSULE * j;

			// matrix multiplication
			matmulInputHat(weightsC, input, output, 0, startIndexB, startIndexC);
		}
	}
}

// matrix multiplication (specifying starting index of input and output matrices) for input hat
void matmulInputHat(float *matA, float *matB, float *matC, uint32_t startIndexA, uint32_t startIndexB, uint32_t startIndexC)
{
#pragma HLS INLINE OFF
matmulOuterLoopInputHat:
	for (uint32_t i = 0; i < DIM_CAPSULE; i += 2)
	{
// dot product between rows of matA and cols of matB
#pragma HLS PIPELINE
		float sumA = 0.0;
		float sumB = 0.0;

#pragma HLS BIND_OP variable = sumA op = fmul impl = maxdsp

		uint32_t currIndexA = INPUT_DIM_CAPSULE * i;
		uint32_t currIndexB = INPUT_DIM_CAPSULE * (i + 1);

	// manually unrolled (two done in same iteration)
	matmulInnerLoopInputHatA:
		for (uint32_t m = 0; m < INPUT_DIM_CAPSULE; ++m)
		{
			float mult_val = matA[startIndexA + currIndexA + m] * matB[startIndexB + m];
#pragma HLS BIND_OP variable = mult_val op = fmul impl = maxdsp

			sumA += mult_val;
		}

	matmulInnerLoopInputHatB:
		for (uint32_t k = 0; k < INPUT_DIM_CAPSULE; ++k)
		{
			float mult_val = matA[startIndexA + currIndexB + k] * matB[startIndexB + k];
#pragma HLS BIND_OP variable = mult_val op = fmul impl = maxdsp
			sumB += mult_val;
		}

		matC[startIndexC + i] = sumA;
		matC[startIndexC + i + 1] = sumB;
	}
}

// matrix multiplication (specifying starting index of input and output matrices) for agreement
void matmulAgreement(float *matA, float *matB, float *matC, uint32_t startIndexA, uint32_t startIndexB, uint32_t startIndexC)
{
	// dot product between rows of matA and cols of matB
	float sum = 0.0;
matmulInnerLoopAgreement:
	for (uint32_t k = 0; k < DIM_CAPSULE; ++k)
	{  // colA = 8/16
#pragma HLS PIPELINE
		sum += matA[startIndexA + k] * matB[startIndexB + k];
	}
	matC[startIndexC] = sum;
}

// calculate softmax along columns
void getSoftmax(float *matA, float *matB)
{
softmaxOuterLoop:
	for (uint32_t j = 0; j < INPUT_NUM_CAPSULE; j++)
	{
		float sum = 0.0;

		// find max of columns
		float max = getMaxColumn(matA, j);

	// calculate softmax
	softmaxFindSumLoop:
		for (uint32_t k = 0; k < NUM_CAPSULE; k++)
		{
#pragma HLS PIPELINE

			// subtract max to prevent overflow before calculating exp()
			float value = exp(matA[k * INPUT_NUM_CAPSULE + j] - max);
			matB[k * INPUT_NUM_CAPSULE + j] = value;
			sum += value;
		}

	softmaxDivideLoop:
		for (uint32_t k = 0; k < NUM_CAPSULE; k++)
		{
#pragma HLS PIPELINE
			float div = matB[k * INPUT_NUM_CAPSULE + j] / (sum + 1e-7);
			matB[k * INPUT_NUM_CAPSULE + j] = div;
		}
	}
}

// find the max of the column
float getMaxColumn(float *mat, uint32_t starting_index)
{
	float max = mat[starting_index];

// go through column and store max
maxLoop:
	for (uint32_t i = 0; i < NUM_CAPSULE; i++)
	{
#pragma HLS PIPELINE
		float val = mat[starting_index + i * INPUT_NUM_CAPSULE];
		max = val > max ? val : max;
	}

	return max;
}

// multiply provided matrix with input multiplication value
void multiply(float *matA, float *matB, float *matC, uint32_t k, uint32_t j, float mult_value)
{
	uint32_t dimA1 = DIM_CAPSULE * INPUT_NUM_CAPSULE;
	uint32_t startIndex = j * dimA1 + k * DIM_CAPSULE;

multiplyInnerLoop:
	for (uint32_t l = 0; l < DIM_CAPSULE; l++)
	{
#pragma HLS PIPELINE
		float val = mult_value * matA[startIndex + l];
		matC[startIndex + l] = val;
	}
}

// multiply two matrices of different sizes
void getMultiply(float *matA, float *matB, float *matC)
{
multiplyOuterLoop:
	for (uint32_t j = 0; j < NUM_CAPSULE; j++)
	{
	multiplyMidLoop:
		for (uint32_t k = 0; k < INPUT_NUM_CAPSULE; k++)
		{
			// extract multiplication value
			float mult_value = matB[j * INPUT_NUM_CAPSULE + k];
			multiply(matA, matB, matC, k, j, mult_value);
		}
	}
}

// calculate sum along columns
void getSum(float *matA, float *matB)
{
sumOuterLoop:
	for (uint32_t j = 0; j < NUM_CAPSULE; j++)
	{
		// find start value for per capsule
		uint32_t indexSize = j * INPUT_NUM_CAPSULE * DIM_CAPSULE;
	sumMidLoop:
		for (uint32_t k = 0; k < DIM_CAPSULE; k += 2)
		{
			float sumA = 0.0;
			float sumB = 0.0;

			// index start per column
			uint32_t indexA = indexSize + k;
			uint32_t indexB = indexSize + k + 1;

		// sum along columns
		sumInnerLoop:
			for (uint32_t l = 0; l < INPUT_NUM_CAPSULE; l++)
			{
// multiple summations happening in one iteration
#pragma HLS PIPELINE
				uint32_t dim = l * DIM_CAPSULE;
				sumA += matA[indexA + dim];
				sumB += matA[indexB + dim];
			}

			// store calculated sum
			matB[j * DIM_CAPSULE + k] = sumA;
			matB[j * DIM_CAPSULE + k + 1] = sumB;
		}
	}
}

// calculate squash (vector magnitude converted between 0 - 1)
void getSquash(float *matA, float *matB)
{
	// squared norm (square values and sum them (per row))
	getSquaredNorm(matA, matB);

	// number to multiply each value in a particular row with
	getScale(matB);

squashOuterLoop:
	for (int i = 0; i < NUM_CAPSULE; i++)
	{
		float multVal = matB[i];
		uint32_t indexSize = i * DIM_CAPSULE;
	squashInnerLoop:
		for (int j = 0; j < DIM_CAPSULE; j += 2)
		{
// multiply with scale value
// manually unrolled, two multiplications per iteration
#pragma HLS PIPELINE
#pragma HLS dependence variable = matA type = intra true
			matA[indexSize + j] = matA[indexSize + j] * multVal;
			matA[indexSize + j + 1] = matA[indexSize + j + 1] * multVal;
		}
	}
}

// calculate squared norm
void getSquaredNorm(float *matA, float *matB)
{
// squared norm per row
squaredNormOuterLoop:
	for (int i = 0; i < NUM_CAPSULE; i++)
	{
#pragma HLS PIPELINE
		float val = 0.0;
		float sum = matB[i];
	squaredNormInnerLoop:
		for (int j = 0; j < DIM_CAPSULE; j++)
		{
			// access each value in matrix
			val = matA[i * DIM_CAPSULE + j];
			// square and add to sum
			sum += val * val;
		}
		matB[i] = sum;
	}
}

// get scale value for squash calculation
void getScale(float *matB)
{
scaleLoop:
	for (uint32_t i = 0; i < NUM_CAPSULE; i++)
	{
#pragma HLS PIPELINE
#pragma HLS dependence variable = matB type = intra true
		float currVal = matB[i];
		matB[i] = currVal / (1 + currVal) / sqrt(currVal + 1e-7);
	}
}

// DEBUGGING, manually put in weight/input values
#ifndef __SYNTHESIS__
void initialiseInputAndWeightArray()
{
	for (uint32_t i = 0; i < g_weightSize; i++)
	{
		weights[i] = i % 100;
	}
	for (uint32_t i = 0; i < g_inputSize; i++)
	{
		input[i] = i % 100;
	}
}

void printSize()
{
	cout << "Weight array size (float): " << (g_weightSize * (float)32) / 8000000 << endl;
	cout << "Input array size (float): " << (g_inputSize * (float)32) / 8000000 << endl;
	cout << "Input hat size (float): " << (g_inputHatSize * (float)32) / 8000000 << endl;
	cout << "B array size (float): " << (g_bcSize * (float)32) / 8000000 << endl;
	cout << "C array size (float): " << (g_bcSize * (float)32) / 8000000 << endl;
	cout << "Multiply array size (float): " << (g_outputsMultiplySize * (float)32) / 8000000 << endl;
	cout << "Sum array size (float): " << (g_outputsSumSize * (float)32) / 8000000 << endl;
	cout << "Squash sum array size (float): " << (g_outputsSquashSumSize * (float)32) / 8000000 << endl;
	cout << "Agreement array size (float): " << (g_outputsAgreementSize * (float)32) / 8000000 << endl;
}
#endif
