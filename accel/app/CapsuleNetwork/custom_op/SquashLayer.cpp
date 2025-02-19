/* SquashLayer.cpp
 * author: nicholas wolf
 *
 * 09 28 2024
 * 
 * custom op registration for
 * the squash layer in capsnet
 *  
 */

#include <cstdint>
#include <cmath>
#include <vart/op_imp.h>

#define PRIMARY_CAPS_CAPSULE_DIM		8
#define PRIMARY_CAPS_NUM_CONV_KERNELS	8
#define PRIMARY_CAPS_CONV_LENGTH		6
#define PRIMARY_CAPS_CONV_WIDTH			6
#define PRIMARY_CAPS_CAPSULES			32

class SquashLayer{
    public:
        SquashLayer(const xir::Op* op1, xir::Attrs* attrs) : op{op1} 
        {
            // op and attributes are not in use
        }

        int calculate(vart::simple_tensor_buffer_t<float> input, vart::simple_tensor_buffer_t<float> output)
        {
            uint32_t dim_2 = PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH;
            float squared_input[PRIMARY_CAPS_CONV_WIDTH * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_NUM_CONV_KERNELS * PRIMARY_CAPS_CAPSULES] = {0.0};
            float squared_norm[PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH] = {0.0};
            float scale[PRIMARY_CAPS_CAPSULES * PRIMARY_CAPS_CONV_LENGTH * PRIMARY_CAPS_CONV_WIDTH] = {0.0};

            // Get squared input
            for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
            {
                for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
                {
                    squared_input[grid_rows * 8 + grid_cols] = input.data[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] * input.data[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols];
                }
            }

            // Get squared norm and find scale for each index
            for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
            {
                for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
                {
                    squared_norm[grid_rows] += squared_input[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols];
                }
                scale[grid_rows] = (squared_norm[grid_rows] / (1 + squared_norm[grid_rows])) / sqrtf(squared_norm[grid_rows] + 1e-07);
            }

            // Multiply value by scale
            for (uint32_t grid_rows = 0; grid_rows < dim_2; ++grid_rows)
            {
                for (uint32_t grid_cols = 0; grid_cols < PRIMARY_CAPS_CAPSULE_DIM; ++grid_cols)
                {
                    output.data[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] = input.data[grid_rows * PRIMARY_CAPS_CAPSULE_DIM + grid_cols] * scale[grid_rows];
                }
            }
            return 0;
        }

    public:
        const xir::Op * const op;
};
