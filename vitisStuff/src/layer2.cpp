#include "layer2.h"
#include <adf.h>
#include "adf/io_buffer/io_buffer.h"
#include <aie_api/aie.hpp>

MULT84::MULT84(const std::array<int32, NUM_COEFFS>& coefficients, const std::array<int32, NUM_OUTPUTS> biases) 
    : coeffs(coefficients), bias(biases) {
}

void MULT84::multiply84(adf::input_buffer<int32> &in, adf::output_buffer<int32> &out)
{

    auto in_t = in.data();
    auto out_t = out.data();
        
    // Load biases (4 outputs)
    aie::vector<int32, 4> bias_vec = aie::load_v<4>(bias.data());

    #pragma unroll
    for (int i = 0; i < 2; i++) {

        // Load coefficients for each input (8 sets of 4 coefficients)
        aie::vector<int32, 4> coeffs_vec0 = aie::load_v<4>(coeffs.data() + 0);
        aie::vector<int32, 4> coeffs_vec1 = aie::load_v<4>(coeffs.data() + 4);
        aie::vector<int32, 4> coeffs_vec2 = aie::load_v<4>(coeffs.data() + 8);
        aie::vector<int32, 4> coeffs_vec3 = aie::load_v<4>(coeffs.data() + 12);
        aie::vector<int32, 4> coeffs_vec4 = aie::load_v<4>(coeffs.data() + 16);
        aie::vector<int32, 4> coeffs_vec5 = aie::load_v<4>(coeffs.data() + 20);
        aie::vector<int32, 4> coeffs_vec6 = aie::load_v<4>(coeffs.data() + 24);
        aie::vector<int32, 4> coeffs_vec7 = aie::load_v<4>(coeffs.data() + 28);
        aie::vector<int32, 8> input_vec = aie::load_v<8>(in_t + (i * 8));
        
        
        aie::vector<int32, 4> input_vec0 = aie::broadcast<int32, 4>(input_vec.get(0));
        aie::vector<int32, 4> input_vec1 = aie::broadcast<int32, 4>(input_vec.get(1));
        aie::vector<int32, 4> input_vec2 = aie::broadcast<int32, 4>(input_vec.get(2));
        aie::vector<int32, 4> input_vec3 = aie::broadcast<int32, 4>(input_vec.get(3));
        aie::vector<int32, 4> input_vec4 = aie::broadcast<int32, 4>(input_vec.get(4));
        aie::vector<int32, 4> input_vec5 = aie::broadcast<int32, 4>(input_vec.get(5));
        aie::vector<int32, 4> input_vec6 = aie::broadcast<int32, 4>(input_vec.get(6));
        aie::vector<int32, 4> input_vec7 = aie::broadcast<int32, 4>(input_vec.get(7));
        
        // Multiply
        aie::accum<acc64, 4> acc = aie::mul(coeffs_vec0, input_vec0);
        
        // Multiply-accumulate remaining inputs
        acc = aie::mac(acc, coeffs_vec1, input_vec1);
        acc = aie::mac(acc, coeffs_vec2, input_vec2);
        acc = aie::mac(acc, coeffs_vec3, input_vec3);
        acc = aie::mac(acc, coeffs_vec4, input_vec4);
        acc = aie::mac(acc, coeffs_vec5, input_vec5);
        acc = aie::mac(acc, coeffs_vec6, input_vec6);
        acc = aie::mac(acc, coeffs_vec7, input_vec7);
        
        // Add biases
        acc = aie::add(acc, bias_vec);
        
        // Convert accumulator to int32 vector
        //aie::vector<int32, 4> result = acc.to_vector<int32>();

        aie::vector<int32, 4> result = acc.to_vector<int32>(8);

        // NEW -- RELU->SHIFT->CLIP
        result = aie::max(result, 0);

        // for (int i = 0; i < 4; i++) {
        //     result[i] = result[i] >> 8;
        // }

        //result = result >> 8;

        result = aie::min(result, 65280);

        //aie::store_v(out_t, result);
        aie::store_v(out_t + (i * 4), result);
        
    }   
}