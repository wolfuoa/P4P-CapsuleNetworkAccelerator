/*
 * DigitCaps.cpp
 * author: nicholas wolf
 *
 * Performs dynamic routing
 */

#include "DigitCaps.h"

#include <cstdlib>
#include <stdint.h>
#include <string.h>

#include "constants.h"
#include <ap_fixed.h>

#ifndef __SYNTHESIS__
#include <math.h>
#else
#include <hls_math.h>
#endif
// ------------------- PRIVATE TYPEDEF -------------------
typedef ap_fixed<32, 16> fixed_t;
// ------------------- PRIVATE TYPEDEF -------------------

// ---------------- FUNCTION DECLARATIONS ----------------
static void apply_weights(fixed_t *input_mat, fixed_t *weights, fixed_t *weighted_input);
static void softmax(fixed_t *mat_b, fixed_t *mat_c);
static void sum_of_products(fixed_t *input_mat, fixed_t *coupling_terms, fixed_t *output_mat);
static void squash(fixed_t *input_mat, fixed_t *squash_mat);
static void agreement(fixed_t *input_mat, fixed_t *squashed_mat, fixed_t *output_mat);
static void add(fixed_t *input_mat, fixed_t *coupling_terms);
// ---------------- FUNCTION DECLARATIONS ----------------

void dynamic_routing(fixed_t *input, fixed_t *weights, fixed_t *prediction)
{
	fixed_t primary_caps[DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE];
	fixed_t squashed_v[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE];

	// burst read input into local array
	memcpy(primary_caps, (const fixed_t *)input, DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(fixed_t));

	fixed_t weighted_input_u[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE];
	fixed_t coupling_b[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES];
	fixed_t coupling_c[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES];
	fixed_t sum_of_products_s[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE];
	fixed_t output_agreement[DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES];

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
			// agreement between the current outHLS stream capsule i to higher level capsules.
			add(output_agreement, coupling_b);
		}
	}
	memcpy(prediction, (const fixed_t *)squashed_v, DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_DIM_CAPSULE * sizeof(fixed_t));
}

static void apply_weights(fixed_t *input_mat, fixed_t *weights, fixed_t *weighted_input)
{
	uint32_t iterator_a = 0;
	uint32_t iterator_b = 0;

	uint32_t weights_per_class = DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE;
	uint32_t num_outputs = DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE;

	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; i++)
	{
		for (uint32_t j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; j++)
		{
            #pragma HLS PIPELINE II=8
			// burst read weight array in small chunks
			fixed_t weight_buffer[DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE];
            #pragma HLS ARRAY_PARTITION variable=weight_buffer dim=1 type=complete

			memcpy(weight_buffer, (const fixed_t *)weights + weights_per_class * i + num_outputs * j, DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(fixed_t));

			iterator_a = DIGIT_CAPS_INPUT_DIM_CAPSULE * j;
			iterator_b = DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_CAPSULES * i + DIGIT_CAPS_DIM_CAPSULE * j;

			for (uint32_t k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
			{
                #pragma HLS PIPELINE
				// dot product between rows of matA and cols of matB

                fixed_t product = 0.0;
                fixed_t sum = 0.0;

                #pragma HLS BIND_OP variable=sum op=add
                #pragma HLS BIND_OP variable=product op=mul

				uint32_t capsule_index = DIGIT_CAPS_INPUT_DIM_CAPSULE * k;

				dot_product:for (uint32_t l = 0; l < DIGIT_CAPS_INPUT_DIM_CAPSULE; ++l)
				{
                    #pragma HLS UNROLL
					product = weight_buffer[capsule_index + l] * input_mat[iterator_a + l];

					sum += product;
				}

				weighted_input[iterator_b + k] = sum;
			}
		}
	}

// for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; i++)
// 	{
// 		for (uint32_t j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; j++)
// 		{
// 			// burst read weight array in small chunks
// 			fixed_t weight_buffer[DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE];
//             #pragma HLS ARRAY_PARTITION variable=weight_buffer factor=8 dim=1 type=cyclic

// 			memcpy(weight_buffer, (const fixed_t *)weights + weights_per_class * i + num_outputs * j, DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE * sizeof(fixed_t));

// 			iterator_a = DIGIT_CAPS_INPUT_DIM_CAPSULE * j;
// 			iterator_b = DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_CAPSULES * i + DIGIT_CAPS_DIM_CAPSULE * j;

// 			for (uint32_t k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
// 			{
// 				// dot product between rows of matA and cols of matB
//                 fixed_t a[DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE];
//                 #pragma HLS ARRAY_PARTITION variable=a dim=1 type=complete

//                 fixed_t product = 0.0;
//                 #pragma HLS BIND_OP variable=product op=mul

//                 for(uint8_t i = 0; i < DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE; ++i)
//                 {
//                     #pragma HLS PIPELINE
//                     a[i] = input_mat[iterator_a + i];
//                 }

//                 for(uint8_t i = 0; i < DIGIT_CAPS_DIM_CAPSULE * DIGIT_CAPS_INPUT_DIM_CAPSULE; ++i)
//                 {
//                     #pragma HLS UNROLL
//                     product += weight_buffer[i] * a[i];
//                 }

// 				weighted_input[iterator_b + k] = product;
// 			}
// 		}
// 	}

}

static void softmax(fixed_t *mat_b, fixed_t *mat_c)
{
	// For all input capsules i in layer l
	for (uint32_t i = 0; i < DIGIT_CAPS_INPUT_CAPSULES; ++i)
	{
		// Compute the exponential sum of log prior probability logits
		// $sum_{k}^{DIGITS} exp(b(i,k))
		fixed_t sum = 0.0;
		// For all possible routings
		for (uint32_t j = 0; j < DIGIT_CAPS_NUM_DIGITS; ++j)
		{
			fixed_t entry = exp(mat_b[i + j * DIGIT_CAPS_INPUT_CAPSULES]);
			// c (i,j) = sum(exp(probability that capsule goes to each digit))
			// Store the numerator temporarily
			mat_c[i + j * DIGIT_CAPS_INPUT_CAPSULES] = entry;
			// Calculate the denominator
			sum += entry;
		}

		for (uint32_t j = 0; j < DIGIT_CAPS_NUM_DIGITS; ++j)
		{
			// Divide the numerator by the denominator
			mat_c[i + j * DIGIT_CAPS_INPUT_CAPSULES] /= (sum + (fixed_t)1e-7);
		}
	}
}

static void sum_of_products(fixed_t *input_mat, fixed_t *coupling_terms, fixed_t *output_mat)
{
	// For all capsules j in layer (l + 1)
	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		// For all capsules i in layer l
		for (uint32_t j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; ++j)
		{
			fixed_t operand = coupling_terms[i * DIGIT_CAPS_INPUT_CAPSULES + j];
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
        //#pragma HLS LOOP_MERGE force
		for (uint32_t sum_j = 0; sum_j < DIGIT_CAPS_DIM_CAPSULE; ++sum_j)
		{
			fixed_t sum = 0.0;
			for (uint32_t sum_k = 0; sum_k < DIGIT_CAPS_INPUT_CAPSULES; sum_k = sum_k)
			{
                #pragma HLS PIPELINE II=3
				sum += output_mat[sum_i * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE + sum_j + sum_k * DIGIT_CAPS_DIM_CAPSULE];
			}
            output_mat[sum_i * DIGIT_CAPS_DIM_CAPSULE + sum_j] = sum;
		}
	}
}

static void squash(fixed_t *input_mat, fixed_t *squash_mat)
{
	fixed_t squared_norm = 0.0;
	fixed_t scale = 0.0;

	// For all capsule j in layer l + 1
    #pragma HLS loop_merge force
	for (uint32_t i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		squared_norm = 0.0;
		// For each corresponding output dimension
		for (uint32_t dim = 0; dim < DIGIT_CAPS_DIM_CAPSULE; ++dim)
		{
			squared_norm += input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] * input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim];
		}

		scale = squared_norm / ((fixed_t)1.0 + squared_norm) / (fixed_t)sqrt(squared_norm + (fixed_t)1e-7);

		for (uint32_t dim = 0; dim < DIGIT_CAPS_DIM_CAPSULE; ++dim)
		{
			squash_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] = (input_mat[i * DIGIT_CAPS_DIM_CAPSULE + dim] * scale);
		}
	}
}

static void agreement(fixed_t *input_mat, fixed_t *squashed_mat, fixed_t *output_mat)
{
	for (int i = 0; i < DIGIT_CAPS_NUM_DIGITS; ++i)
	{
		for (int j = 0; j < DIGIT_CAPS_INPUT_CAPSULES; ++j)
		{
			fixed_t sum = 0.0;
			for (int k = 0; k < DIGIT_CAPS_DIM_CAPSULE; ++k)
			{
				sum += input_mat[i * DIGIT_CAPS_INPUT_CAPSULES * DIGIT_CAPS_DIM_CAPSULE + j * DIGIT_CAPS_DIM_CAPSULE + k] * squashed_mat[i * DIGIT_CAPS_DIM_CAPSULE + k];
			}
			output_mat[i * DIGIT_CAPS_INPUT_CAPSULES + j] = sum;
		}
	}
}

static void add(fixed_t *input_mat, fixed_t *coupling_terms)
{
	for (int i = 0; i < DIGIT_CAPS_NUM_DIGITS * DIGIT_CAPS_INPUT_CAPSULES; ++i)
	{
        #pragma HLS PIPELINE II=2
		// Update coupling terms
		coupling_terms[i] += input_mat[i];
	}
}
