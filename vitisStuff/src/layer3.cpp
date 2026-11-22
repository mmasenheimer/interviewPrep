#include "layer3.h"
#include <adf.h>
#include "adf/io_buffer/io_buffer.h"
#include <aie_api/aie.hpp>
#include <cmath>

// int32_t apply_sigmoid(int32_t x) {
//     // 1. Convert fixed-point (2^6) to actual float
//     //    e.g. if x=64, x_float becomes 1.0
//     float x_float = static_cast<float>(x) / 256.0f;
    
//     // 2. Apply Sigmoid
//     float sigmoid_val = 1.0f / (1.0f + std::exp(-x_float));
    
//     // 3. Scale output to 0-4096 (12-bit range for output)
//     int32_t result = static_cast<int32_t>(sigmoid_val * 4096.0f);
    
//     // 4. Clamp
//     if (result > 4096) result = 4096;
//     if (result < 0) result = 0;
    
//     return result;
// }



/*
 * Fixed-point Hard Sigmoid: y = clamp(0.2 * x + 0.5, 0, 1)
 * Scaled to 12-bit range (0 to 4096).
 * - 2048: Represents the 0.5 y-intercept (offset).
 * - (205 >> 7): Approx 1.6 multiplier to achieve a 0.2 slope 
 * relative to the 4096 (1.0) ceiling.
 */
inline int32_t apply_hard_sigmoid(int32_t x) {

    int32_t linear_part = ((x * 205) >> 7) + 2048;

    if (linear_part < 0) return 0;
    if (linear_part > 4096) return 4096;
    return linear_part;
}

MULT41::MULT41(const std::array<int32, NUM_COEFFS>& coefficients, const int32 biases) 
    : coeffs(coefficients), bias(biases) {
}

void MULT41::multiply41(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out)
{

    auto in_t = in.data();
    auto out_t = out.data();
        
    // Load all 4 coefficients as a vector
    aie::vector<int32, 4> coeffs_vec = aie::load_v<4>(coeffs.data());

    #pragma unroll
    for (int i = 0; i < 2; i++) {

        aie::vector<int32, 4> input_vec = aie::load_v<4>(in_t + (i * 4));
        
        // Multiply coefficients with inputs
        aie::accum<acc64, 4> acc = aie::mul(coeffs_vec, input_vec);
        
        // Convert accumulator to int32 vector for reduction
        aie::vector<int32, 4> prod = acc.to_vector<int32>();
        
        // Horizontal reduction: sum all 4 products
        int32 result = 0;

        result += prod[0];
        result += prod[1];
        result += prod[2];
        result += prod[3];

        result += bias;

        result = result >> 8;
        
        out_t[i] = apply_hard_sigmoid(result);

    }
}