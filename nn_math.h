#ifndef NN_MATH_H
#define NN_MATH_H

#include <vector>
#include <cmath>
#include "global_definitions.h"

#include "neuralnet.h"


float dot_product(std::vector<float> pveca, std::vector<float> pvecb);

std::vector<std::vector<float> > vector_multiplication_2d(
                std::vector<float> pveca,
                std::vector<float> pvecb);

std::vector<float> vector_matrix_multiplication_fast(
                std::vector<float>pvec,
                std::vector<std::vector<float> > pmat);

std::vector<std::vector<std::vector<float> > > transpose_vector_of_2d_matrices(std::vector<std::vector<std::vector<float> > > pmat_vec);
std::vector<std::vector<float> > transponse_2d_matrix(std::vector<std::vector<float> > pmat);

neuronlayer calculate_delta_pre_layer_fast(
                neuronlayer pdelta_of_next_layer,
                edgelayer pedges_mat);

neuronlayers_vec sum_up_values_each_neuron(neuronlayers_vec* pbiases_p, neuronlayers_vec* tmp_biases);                          
edgelayers_vec sum_up_values_each_edge(edgelayers_vec* pedges_p, edgelayers_vec* tmp_edges);

float activation_function(float pz);

float activation_function_derivative(float pactivation_function_value_z);

neuronlayer cost_derivative_times_activation_derivative(
                neuronlayer poutput_actual,
                neuronlayer poutput_expected);

//neuronarray_layer_1_index_i = SUM_over_j[ neoronarray_layer_0[index_j]*weightsmatrix_layer_01[index_i][index_j]
neuronlayer calculate_next_layer_fast(
                neuronlayer pvec,
                edgelayer pmat,
                neuronlayer pbias_neurons);

#endif
