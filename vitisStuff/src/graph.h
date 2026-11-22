#include "adf.h"
#include "adf/new_frontend/adf.h"
#include "adf/window/types.h"
#include "layer1.h"
#include "layer2.h"
#include "layer3.h"

class simpleGraph : public adf::graph {
private:
    adf::kernel first;
    adf::kernel second;
    adf::kernel third;
public:
    adf::input_plio in;
    adf::output_plio out;
    
    simpleGraph(){

        const int batch_size = 2;
        // Process batch_size number of inputs

        const std::array<int32, 16> mycoeffslayer1 = {247, 51, -75, 0, -275, 5, 32, -71, -37, 248, -154, 0, 14, -246, 21, 155};

        const std::array<int32, 8> mybiaslayer1 = {
            -418625, -397131, -386662, -12943, -436189, -437218, -299252, -479932
        };

        const std::array<int32, 32> mycoeffslayer2 = {
            -200, 297, 221, 210, -45, -136, -152, 213, -292, 142, -99, 91, 2, 87, 32, -14, -86, 201, -76, 215, -36, -119, -146, 253, -73, 141, -122, 5, -114, 99, 168, 149
        };
        const std::array<int32, 4> mybiaslayer2 = {
            213123, -445727, -324784, -518380
        };

        const std::array<int32, 4> mycoeffslayer3 = {
           -9, -137, -93, -391
        };
        const int32 mybiaslayer3 = 553002;

        first = adf::kernel::create_object<MULT28>(mycoeffslayer1, mybiaslayer1);
        second = adf::kernel::create_object<MULT84>(mycoeffslayer2, mybiaslayer2);
        third = adf::kernel::create_object<MULT41>(mycoeffslayer3, mybiaslayer3);
        
        in = adf::input_plio::create(adf::plio_32_bits, "data/batchinput.txt" , true);
        out = adf::output_plio::create(adf::plio_32_bits, "data/output.txt");

        // Connect the kernels
        adf::connect(in.out[0], first.in[0]);
        
        adf::connect(first.out[0], second.in[0]);

        adf::connect(second.out[0], third.in[0]);

        adf::connect(third.out[0], out.in[0]);
        
        // Set dimensions
        // Update these to use your batch_size
        adf::dimensions(first.in[0])   = {batch_size};  // 64
        adf::dimensions(first.out[0])  = {8 * batch_size};  // 256
        adf::dimensions(second.in[0])  = {8 * batch_size};  // 256
        adf::dimensions(second.out[0]) = {4 * batch_size};  // 128
        adf::dimensions(third.in[0])   = {4 * batch_size};  // 128
        adf::dimensions(third.out[0])  = {1 * batch_size};  // 32
        
        // Set source files
        adf::source(first) = "layer1.cpp";
        adf::source(second) = "layer2.cpp";
        adf::source(third) = "layer3.cpp";
        
        // Set kernel locations
        //adf::location<adf::kernel>(first)  = adf::tile(10, 1);
        //adf::location<adf::kernel>(second) = adf::tile(11, 1);
        //adf::location<adf::kernel>(third)  = adf::tile(12, 1);
        
        // Set runtime ratios (percentage of time kernel is active)
        adf::runtime<ratio>(first) = 0.6;
        adf::runtime<ratio>(second) = 0.6;
        adf::runtime<ratio>(third) = 0.6;

        // Enable single buffer mode for pipelining
        // adf::single_buffer(first.in[0]);
        // adf::single_buffer(first.out[0]);
        // adf::single_buffer(second.in[0]);
        // adf::single_buffer(second.out[0]);
        // adf::single_buffer(third.in[0]);
        // adf::single_buffer(third.out[0]);
    }
};