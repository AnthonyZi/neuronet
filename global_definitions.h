#ifndef RETURN_STATES_H
#define RETURN_STATES_H

//#define VER2
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
