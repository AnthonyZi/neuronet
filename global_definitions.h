#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

#include <cstdlib>

#define SIGMOID
//#define SOFTSIGN


typedef struct trainig_data{ 
        std::vector<float> input; 
        std::vector<float> output;                   
} training_data_s;

enum return_states
{
        ok = 0,
        argument_error = 1,
        broken_config = 2, 
        inconsistent_bias_numbers = 3,
        wrong_input_size = 4
};

#endif
