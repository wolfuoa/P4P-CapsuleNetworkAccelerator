/*
 * DigitCaps.cpp
 * author: nicholas wolf
 *
 * Performs dynamic routing
 */

#include "DigitCaps.h"

#include <math.h>

#include <cstdint>
#include <string>

// ---------------- FUNCTION DECLARATIONS ----------------
static void apply_weights(float *input_mat, float *weights, float *weighted_input);
static void softmax(float *mat_b, float *mat_c);
static void sum_of_products(float *input_mat, float *coupling_terms, float *output_mat);
static void squash(float *input_mat, float *squash_mat);
static void agreement(float *input_mat, float *squashed_mat, float *output_mat);
static void add(float *input_mat, float *coupling_terms);
// ---------------- FUNCTION DECLARATIONS ----------------

void dynamic_routing(float *input, float *weights, float *prediction)
{
	float *primary_caps = (float *)malloc(DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(float));
	float *squashed_v = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));

	// burst read input into local array
	memcpy(primary_caps, (const float *)input, DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(float));

	float *weighted_input_u = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));
	float *coupling_b = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * sizeof(float));
	float *coupling_c = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * sizeof(float));
	float *sum_of_products_s = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));
	float *output_agreement = (float *)malloc(DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * sizeof(float));

	apply_weights(primary_caps, weights, weighted_input_u);

	//  routing(uˆj|i, r, l)
	//  for all capsule i in layer l and capsule j in layer (l + 1): bij <- 0
	//  for r iterations do
	//  for all capsule i in layer l: ci <- softmax(bi)
	//  for all capsule j in layer (l + 1): sj <- sum(i++)( cij * uˆj|i )
	//  for all capsule j in layer (l + 1): vj <- squash(sj)
	//  for all capsule i in layer l and capsule j in layer (l + 1): bij <- bij + u^j|i * vj
	//  return vj
	for (uint32_t i = 0; i < DIGIT_CAPS_ROUTING_ITERATIONS; ++i)
	{
		// The coupling coefficients ci,j between capsule i and all the capsules in the layer
		// above sum to 1 and are determined by a “routing softmax” whose initial logits
		// bij are the log prior probabilities that capsule should be coupled to capsule j.
		softmax(coupling_b, coupling_c);
		// the total input to a capsule sj is a weighted sum over all “prediction vectors”
		// ui,j from the capsules in the layer below and is produced by multiplying the
		// output ui of a capsule in the layer below by a weight matrix Wi,j
		sum_of_products(weighted_input_u, coupling_c, sum_of_products_s);
		// non-linear "squashing" function to ensure that short vectors get shrunk to almost
		// zero length and long vectors get shrunk to a length slightly below 1
		squash(sum_of_products_s, squashed_v);
		if (i < DIGIT_CAPS_ROUTING_ITERATIONS - 1)
		{
			// The initial coupling coefficients are then iteratively refined by measuring the
			// agreement between the current output vj of each capsule, j, in the layer above
			// and the prediction uˆj|i made by capsule i. The agreement is simply the scalar
			// product aij = vj.uˆj|i.
			agreement(weighted_input_u, squashed_v, output_agreement);
			// This agreement is treated as if it was a log likelihood
			// and is added to the initial logit, bij before computing the new values for all
			// the coupling coefficients linking capsule i to higher level capsules.
			add(output_agreement, coupling_b);
		}
	}
	memcpy(prediction, (const float *)squashed_v, DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(float));
	free(primary_caps);
	free(output_agreement);
	free(squashed_v);
	free(coupling_b);
	free(coupling_c);
	free(sum_of_products_s);
	free(weighted_input_u);
}

static void apply_weights(float *input_mat, float *weights, float *weighted_input)
{
	uint32_t iterator_a = 0;
	uint32_t iterator_b = 0;

	uint32_t dimA1 = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE;	 // 3rd dim
	uint32_t dimA2 = DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE;								 // 2nd dim

	uint32_t dimC1 = DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_CAPSULES;  // 2nd dim

	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; i++)
	{
		for (uint32_t j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; j++)
		{
			// burst read weight array in small chunks
			float weightsC[DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE];

			memcpy(weightsC, (const float *)weights + dimA1 * i + dimA2 * j, DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(float));

			iterator_a = DIGIT_CAPS_INPUT_DIM_CAPSULE * j;
			iterator_b = dimC1 * i + DIGIT_CAPS_DIM_CAPSULE * j;

			for (uint32_t k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
			{
				// dot product between rows of matA and cols of matB

				float sum = 0.0;

				uint32_t capsule_index = DIGIT_CAPS_INPUT_DIM_CAPSULE * k;

				for (uint32_t l = 0; l < DIGIT_CAPS_INPUT_DIM_CAPSULE; ++l)
				{
					float product = weightsC[capsule_index + l] * input_mat[iterator_a + l];

					sum += product;
				}

				weighted_input[iterator_b + k] = sum;
			}
		}
	}
}

static void softmax(float *mat_b, float *mat_c)
{
	// For all input capsules i in layer l
	for (uint32_t i = 0; i < DIGIT_CAPS_INPUT_CAPSULES; ++i)
	{
		// Compute the exponential sum of log prior probability logits
		// $sum_{k}^{DIGITS} exp(b(i,k))
		float sum = 0.0;
		// For all possible routings
		for (uint32_t j = 0; j < DIGIT_CAPS_NUM_DIGITS; ++j)
		{
			float entry = exp(mat_b[i + j * DIGIT_CAPS_INPUT_CAPSULES]);
			// c (i,j) = sum(exp(probability that capsule goes to each digit))
			// Store the numerator temporarily
			mat_c[i + j * DIGIT_CAPS_INPUT_CAPSULES] = entry;
			// Calculate the denominator
			sum += entry;
		}

		for (uint32_t j = 0; j < DIGIT_CAPS_NUM_DIGITS; ++j)
		{
			// Divide the numerator by the denominator
			mat_c[i + j * DIGIT_CAPS_INPUT_CAPSULES] /= (sum + 1e-7);
		}
	}
}

static void sum_of_products(float *input_mat, float *coupling_terms, float *output_mat)
{
	// For all capsules j in layer (l + 1)
	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		// For all capsules i in layer l
		for (uint32_t j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; ++j)
		{
			float operand = coupling_terms[i * DIGIT_CAPS_INPUT_CAPSULES + j];
			uint32_t lin_index = (i * DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_CAPSULES) + (j * DIGIT_CAPS_DIM_CAPSULE);
			// For all capsule output dimensions
			for (uint32_t k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
			{
				output_mat[lin_index + k] = input_mat[lin_index + k] * operand;
			}
		}
	}

	// Combine into one loop?
	for (uint32_t sum_i = 0; sum_i < DIGIT_CAPS_NUM_DIGITS; ++sum_i)
	{
		for (uint32_t sum_j = 0; sum_j < DIGIT_CAPS_DIM_CAPSULE; ++sum_j)
		{
			float sum = 0.0;
			for (uint32_t sum_k = 0; sum_k < DIGIT_CAPS_INPUT_CAPSULES; ++sum_k)
			{
				sum += output_mat[sum_i * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE + sum_j + sum_k * DIGIT_CAPS_DIM_CAPSULE];
			}
			output_mat[sum_i * DIGIT_CAPS_DIM_CAPSULE + sum_j] = sum;
		}
	}
}

static void squash(float *input_mat, float *squash_mat)
{
	float squared_norm = 0.0;
	float scale = 0.0;

	// For all capsule j in layer l + 1
	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		squared_norm = 0.0;
		// For each corresponding output dimension
		for (uint32_t dim = 0; dim < DIGIT_CAPS_DIM_CAPSULE; ++dim)
		{
			squared_norm += input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] * input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim];
		}

		scale = squared_norm / (1.0 + squared_norm) / sqrt(squared_norm + 1e-7);

		for (uint32_t dim = 0; dim < DIGIT_CAPS_DIM_CAPSULE; ++dim)
		{
			squash_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] = (input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] * scale);
		}
	}
}

static void agreement(float *input_mat, float *squashed_mat, float *output_mat)
{
	for (int i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		for (int j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; ++j)
		{
			float sum = 0.0;
			for (int k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
			{
				sum += input_mat[i * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE + j * DIGIT_CAPS_DIM_CAPSULE + k] * squashed_mat[i * DIGIT_CAPS_DIM_CAPSULE + k];
			}
			output_mat[i * DIGIT_CAPS_INPUT_CAPSULES + j] = sum;
		}
	}
}

static void add(float *input_mat, float *coupling_terms)
{
	for (int i = 0; i < DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES; ++i)
	{
		// Update coupling terms
		coupling_terms[i] += input_mat[i];
	}
}