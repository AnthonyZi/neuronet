#ifndef NN_MATH_H
#define NN_MATH_H

#include <vector>
#include <cmath>
#include "global_definitions.h"

#include "neuralnet.h"


float dot_product(std::vector<float> pveca, std::vector<float> pvecb);
float activation_function(float pz);
float activation_function_derivative(float pactivation_function_value_z);
std::vector<float> cost_derivative_times_activation_derivative(
                std::vector<float> poutput_actual,
                std::vector<float> poutput_expected);

//neuronarray_layer_1_index_i = SUM_over_j[ neoronarray_layer_0[index_j]*weightsmatrix_layer_01[index_i][index_j]
std::vector<float> calculate_layer(std::vector<float> pvec, std::vector<std::vector<float> > pmat, neuronlayer pbias_neurons);

#endif
