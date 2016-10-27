#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

#define SIGMOID
//#define SOFTSIGN

enum return_states
{
        ok = 0,
        argument_error = 1,
        broken_config = 2, 
        inconsistent_bias_numbers = 3,
        wrong_input_size = 4
};

#endif
