/*
* PrimaryCaps.cpp
* author: nicholas wolf
*
* The second layer of the Capsule Network takes in a three-dimensional
* tensor best represented as a 20x20x256 three-dimensional matrix
*
* To evaluate this matrix, the Primary Capsule layer has 32 primary
* capsules whose job is to combine the basic features provided by the
* previous layer into cohesive groups of features.
*
* To do this, each capsule applies 8 three-dimensional convolutional
* kernels, each of which natrually has their own 9x9x256 weight matrix
*
* We need to organize this operation such that the correct input
* is matched to multiply to the correct weight. The order that
* Conv1 populated the 20x20x256 input tensor was:
* For each kernel(256), for each row(20), for each col(20),
* meaning every 20 reads, we increase the row of the input
* volume, and every 20 rows we change the kernel that was used.
* Hence, we need to read a complete kernel entry before incrementing
* and moving along the 256 kernels. In that kernel entry, we need
* to read columns first, then rows.
*
* As the operation is done using a stride of two, each kernel only
* indexes over the length and width of the tensor 6 times,
* producing an output of a 6x6 matrix
*
* There are 8 convolutional kernels in a capsule, and there are 32
* capsules in Primary Caps. Hence, we get a total output of
* 6x6x8x32
*
*/


#include "PrimaryCaps.h"


#include <cstdint>
#include <cstdlib>
#include <stdint.h>
#include <string.h>


#include "constants.h"

#include <hls_stream.h>
#include <ap_fixed.h>


#ifndef __SYNTHESIS__
#include <math.h>
#else
#include <hls_math.h>
#endif

// ------------------- PRIVATE TYPEDEF -------------------
typedef ap_fixed<32, 16> fixed_t;
// ------------------- PRIVATE TYPEDEF -------------------

static void conv_2d(fixed_t *input, hls::stream<fixed_t> &weights, fixed_t *biases, fixed_t *output);
static void reshape(fixed_t *input, fixed_t *output);
static void squash(fixed_t *input, fixed_t *output);
static void coalesce_partial_sums(fixed_t *input, fixed_t *output);
static inline void calculate_single_value(fixed_t *input_a, fixed_t *input_b, uint32_t output_length, uint32_t output_width, fixed_t *output);

static void conv_2d(fixed_t *input, hls::stream<fixed_t> &weights, fixed_t *biases, fixed_t *output)
{
    fixed_t input_buffer[CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH * CONV1_FILTERS];
    #pragma HLS BIND_STORAGE variable=input_buffer impl=auto type=ram_1wnr
    #pragma HLS ARRAY_PARTITION variable=input_buffer type=cyclic factor=64

    fixed_t output_buffer[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
    #pragma HLS BIND_STORAGE variable=output_buffer impl=auto type=ram_1wnr

        // PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH]
    fixed_t weight_buffer[5184];
    #pragma HLS BIND_STORAGE variable=weight_buffer type=ram_1wnr impl=auto
    #pragma HLS ARRAY_RESHAPE variable=weight_buffer type=cyclic factor=81
    //    #pragma HLS ARRAY_PARTITION variable=weight_buffer factor=64 dim=1 type=block

    fixed_t biases_buffer[PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
    #pragma HLS BIND_STORAGE variable=biases_buffer type=ram_1wnr impl=auto

    memcpy(input_buffer, (const fixed_t *)input, CONV1_OUTPUT_WIDTH * CONV1_OUTPUT_LENGTH * CONV1_FILTERS * sizeof(fixed_t));
    memcpy(biases_buffer, (const fixed_t *)biases, PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(fixed_t));


    uint32_t prim_caps_kernel_dim = PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS * PRIMARY_CAPS_KERNEL_DEPTH;


    // Each capsule goes kernel by kernel, iterating over the input volume. Each kernel
    // produces a 6x6 convolution matrix. This process happens 8(kernel) times and 32(capsule)
    // times, resulting in the 6x6 matrices flattened in the order of capsule[0..31] -> kernel[0..7]
    // Ie. If we read the first item from the stream, it is the "top left" index of the 6x6 matrix
    // produced by the first convolutional kernel of the first primary capsule.

    // FFT of convolution
    for (uint32_t output_depth = 0; output_depth < PRIMARY_CAPS_CONV_DEPTH; ++output_depth)
    {
    //    memcpy(weight_buffer, (const fixed_t *)weights + output_depth * prim_caps_kernel_dim, prim_caps_kernel_dim * sizeof(fixed_t));
    // prim_caps_kernel_dim
        // Initially, 20736 - 210ms, 10368 - 183ms, 5184 - 170ms
        conv_weights:for (uint32_t i = 0; i < 5184; ++i)
        {
            #pragma HLS PIPELINE
            weight_buffer[i] = weights.read();
        }

        conv_kernel_length_6:for (uint32_t output_length = 0; output_length < PRIMARY_CAPS_CONV_LENGTH; ++output_length)
        {
            conv_kernel_width_6:for (uint32_t output_width = 0; output_width < PRIMARY_CAPS_CONV_WIDTH; ++output_width)
            {
                fixed_t sum = 0.0;
                calculate_single_value(input_buffer, weight_buffer, output_length, output_width, &sum);
                output_buffer[(output_depth * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH) + (output_length * PRIMARY_CAPS_CONV_WIDTH) + output_width] = biases_buffer[output_depth] + sum;
            }
        }
    }
    memcpy(output, (const fixed_t *)output_buffer, PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES * sizeof(fixed_t));
}

static inline void calculate_single_value(fixed_t *input_a, fixed_t *input_b, uint32_t output_length, uint32_t output_width, fixed_t *output)
{
    fixed_t partial_sums[PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS] = {0.0};
    conv_kernel_depth_256:for (uint32_t kernel_depth = 0; kernel_depth < PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CAPSULE_DIM; ++kernel_depth)
    {
        #pragma HLS PIPELINE II=3
        uint32_t stride_index_lengthwise = output_length * PRIMARY_CAPS_STRIDE;
        uint32_t stride_index_widthwise = output_width * PRIMARY_CAPS_STRIDE;
        conv_kernel_row_9:for (uint32_t kernel_row = 0; kernel_row < PRIMARY_CAPS_KERNEL_ROWS; ++kernel_row)
        {
            // #pragma HLS UNROLL
            conv_kernel_col_9:for (uint32_t kernel_col = 0; kernel_col < PRIMARY_CAPS_KERNEL_COLS; ++kernel_col)
            {
                // #pragma HLS UNROLL
                fixed_t operand = input_a[(kernel_depth * CONV1_OUTPUT_LENGTH * CONV1_OUTPUT_WIDTH) + ((stride_index_lengthwise + kernel_row) * CONV1_OUTPUT_WIDTH) + (stride_index_widthwise + kernel_col)];
                fixed_t weight = input_b[(kernel_depth * PRIMARY_CAPS_KERNEL_ROWS * PRIMARY_CAPS_KERNEL_COLS) + (kernel_row * PRIMARY_CAPS_KERNEL_COLS) + kernel_col];
                partial_sums[kernel_row * PRIMARY_CAPS_KERNEL_COLS + kernel_col] += operand * weight;
            }
        }
    }
    fixed_t sum = 0.0;
    coalesce_partial_sums(partial_sums, &sum);
    *output = sum;
}

static void coalesce_partial_sums(fixed_t *input, fixed_t *output)
{
    fixed_t result;
	for (int i = 0; i < PRIMARY_CAPS_KERNEL_ROWS; ++i)
	{
        #pragma HLS DEPENDENCE variable=result type=intra direction=RAW true
		for (int j = 0; j < PRIMARY_CAPS_KERNEL_COLS; ++j)
		{
            #pragma HLS PIPELINE II=3
            // #pragma HLS DEPENDENCE variable=output type=inter true
			result += input[i * PRIMARY_CAPS_KERNEL_COLS + j];
            #pragma HLS BIND_OP variable=result op=add impl=dsp latency=2
		}
	}
    *output = result;
}


// @brief process_features Process the output 20x20x256 tensor from the convolutional layer
// @param[in] stream_conv_s the 256 wide stream of 20x20 convolutions
// @param[out] stream_primary_caps_s the 32 wide stream of grouped features
void process_features(fixed_t *input, fixed_t *weights, fixed_t *biases, fixed_t *output)
{
   #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256 depth=784
   #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=2 depth=5308416
   #pragma HLS INTERFACE mode=m_axi port=biases offset=slave bundle=gmem2 max_read_burst_length=256 max_write_burst_length=256
   #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem3 max_read_burst_length=256 max_write_burst_length=256

   // Apply Conv2d 32 times and concatenate capsules


   fixed_t conv_output[PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
   #pragma HLS BIND_STORAGE variable=conv_output type=ram_1wnr impl=auto

   fixed_t reshape_output[PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CAPSULE_DIM * PRIMARY_CAPS_CAPSULES];
   #pragma HLS BIND_STORAGE variable=reshape_output type=ram_1wnr impl=auto

    hls::stream<fixed_t> weights_stream;
    for(uint32_t i = 0; i < 5308416; ++i)
    {
        #pragma HLS PIPELINE
        weights_stream.write(weights[i]);
    }

   // Conv2d <- 20x20x256
   conv_2d(input, weights_stream, biases, conv_output);
   // -> 6x6x8 (x32)


   // At reshape we have attained 256 6x6 feature maps containing scalars
   // We need to transform this into 32 6x6 maps containing 8D vectors


   // Reshape <- 256x6x6
   reshape(conv_output, reshape_output);
   // -> 1152 x 8


   // Squash <- 1152 x 8
   squash(reshape_output, output);
   // -> 1152 x 8
}


static void reshape(fixed_t *input, fixed_t *output)
{
   uint32_t dim = PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES;
   fixed_t feature_collection[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES];
   fixed_t output_buffer[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES];


   memcpy(feature_collection, (const fixed_t *)input, dim * sizeof(fixed_t));
   // For each spatial position in the feature map
   // [i, j]
   // [k, l]
   // Pair the corresponding scalar value with the corresponding outputs from the other feature maps
   // v0 -> [i0, i1, i2, i3, i4, i5, i6, i7]
   // v1 -> [j0, j1, j2, j3, j4, j5, j6, j7]
   // ...
   // [v00, v01]
   // [v10, v11]
   // Hence, we obtain a 6x6 8D output for each primary capsule
   uint32_t out_vector = 0;
   for (uint32_t grid_rows = 0; grid_rows < PRIMARY_CAPS_CONV_LENGTH; ++grid_rows)
   {
       for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CONV_WIDTH; ++grid_cols)
       {
           for (uint32_t current_kernel = 0; current_kernel < PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CAPSULE_DIM; current_kernel += PRIMARY_CAPS_CAPSULE_DIM)
           {
               for (uint32_t current_dim = 0; current_dim < PRIMARY_CAPS_CAPSULE_DIM; ++current_dim)
               {
                   output_buffer[out_vector * PRIMARY_CAPS_CAPSULE_DIM + current_dim] = feature_collection[(current_kernel + current_dim) * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH + grid_rows * PRIMARY_CAPS_CONV_WIDTH + grid_cols];
                   if (current_dim == PRIMARY_CAPS_CAPSULE_DIM - 1){
                       out_vector++;
                   }
               }
           }
       }
   }


   memcpy(output, (const fixed_t *)output_buffer, dim * sizeof(fixed_t));
}


static void squash(fixed_t *input, fixed_t *output)
{
   uint32_t dim_1 = PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES;
   uint32_t dim_2 = PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH;
   fixed_t input_buffer[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES];
   fixed_t output_buffer[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES];
   fixed_t squared_input[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES];
   fixed_t squared_norm[PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH] = {0};
   fixed_t scale[PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH];


   memcpy(input_buffer, (const fixed_t *)input, dim_1 * sizeof(fixed_t));


   // Get squared input
   for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
   {
       for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
       {
           squared_input[grid_rows * 8 + grid_cols] = input_buffer[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] * input_buffer[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols];
       }
   }


   // Get squared norm and find scale for each index
   for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
   {
       for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
       {
           squared_norm[grid_rows] += squared_input[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols];
       }
       // WARNING: [SYNCHK 200-23] variable-indexed range selection may cause suboptimal QoR
       scale[grid_rows] = (squared_norm[grid_rows] / (1 + squared_norm[grid_rows])) / (fixed_t)(sqrtf(squared_norm[grid_rows] + (fixed_t)1e-07));
   }


   // Multiply value by scale
   for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
   {
       for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
       {
           output_buffer[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] = input_buffer[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] * scale[grid_rows];
       }
   }


   memcpy(output, (const fixed_t *)output_buffer, dim_1 * sizeof(fixed_t));
}

