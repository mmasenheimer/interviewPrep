#include "layer1.h"
#include <adf.h>
#include "adf/io_buffer/io_buffer.h"
#include <aie_api/aie.hpp>

MULT28::MULT28(const std::array<int32, NUM_COEFFS>& coefficients, std::array<int32, NUM_OUTPUTS> bias) 
    : coeffs(coefficients), biases(bias) {
}

void MULT28::multiply28(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out)
{
    auto in_t = in.data();
    auto out_t = out.data();
    
    // Load coefficients/biases from memory
    aie::vector<int32, 8> coeffs_vec1 = aie::load_v<8>(coeffs.data());
    aie::vector<int32, 8> coeffs_vec2 = aie::load_v<8>(coeffs.data() + 8);
    aie::vector<int32, 8> bias_vec = aie::load_v<8>(biases.data());
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {

        int32_t packed = in_t[i];

        int16_t input0 = (int16_t)(packed >> 16);
        int16_t input1 = (int16_t)(packed & 0xFFFF);

        aie::vector<int32, 8> input_vec1 = aie::broadcast<int32, 8>(input0);
        aie::vector<int32, 8> input_vec2 = aie::broadcast<int32, 8>(input1);
        
        // Multiply first input with first set of coefficients
        aie::accum<acc64, 8> acc = aie::mul(coeffs_vec1, input_vec1);
        
        // Multiply-accumulate second input with second set of coefficients
        acc = aie::mac(acc, coeffs_vec2, input_vec2);
        
        // Add biases
        acc = aie::add(acc, bias_vec);

        // Actual accumulator and shift
        aie::vector<int32, 8> result = acc.to_vector<int32>(8);

        // NEW -- RELU->SHIFT->CLIP
        result = aie::max(result, 0);

        // for (int i = 0; i < 8; i++) {
        //     result[i] = result[i] >> 8;
        // }

        result = aie::min(result, 8192);
        
        aie::store_v(out_t + (i * 8), result);

    }
}