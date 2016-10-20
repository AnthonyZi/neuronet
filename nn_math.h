#ifndef NN_MATH_H
#define NN_MATH_H

#include <vector>
#include <cmath>
#include "global_definitions.h"

#ifndef VER2
#include "neuralnet.h"
#else
#include "neuralnet2.h"
#endif


float dot_product(std::vector<float> pveca, std::vector<float> pvecb);
float activation_function(float z);

//neuronarray_layer_1_index_i = SUM_over_j[ neoronarray_layer_0[index_j]*weightsmatrix_layer_01[index_i][index_j]
std::vector<float> calculate_layer(std::vector<float> pvec, std::vector<std::vector<float> > pmat, neuronlayer pbias_neurons);

#endif
