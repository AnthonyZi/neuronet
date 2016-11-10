#ifndef NN_MATH_H
#define NN_MATH_H

#include <vector>
#include <cmath>
#include "global_definitions.h"

#include "neuralnet.h"

//a[1]*b[1]+a[2]*b[2]+ ... + a[i]*b[i]
float dot_product(std::vector<float> pveca, std::vector<float> pvecb);

// | va1 |                              | . . . . . |
// | va2 |                              | . . . . . |
// | va3 |   *  [vb1 vb2 vb3 vb4 vb5] = | . . . . . |
// | va4 |                              | . . . . . |
// | va5 |                              | . . . . . |
std::vector<std::vector<float> > vector_multiplication_2d(
                std::vector<float> pveca,
                std::vector<float> pvecb);

//           | . . . . . |
// [ ... ] * | . . . . . | = [ ..... ]
//           | . . . . . |
//
// this function needs the matrix transposed in order to compute faster
// with nearby memory addressing (see calculate_next_layer - the same
// applies there)
std::vector<float> vector_matrix_multiplication_fast(
                std::vector<float>pvec,
                std::vector<std::vector<float> > pmat);

//use of transpose_2d_matrix for every matrix in the vector of matrices (see below)
std::vector<std::vector<std::vector<float> > > transpose_2d_matrices_of_vector(std::vector<std::vector<std::vector<float> > > pmat_vec);
//pay attention to use this only for matrices and not for arrays of arrays with
//different length
std::vector<std::vector<float> > transponse_2d_matrix(std::vector<std::vector<float> > pmat);

neuronlayer calculate_delta_pre_layer_fast(
                neuronlayer pdelta_of_next_layer,
                edgelayer pedges_mat,
                neuronlayer pactivations_of_pre_layer);

//result will be saved in the first argument .. simple component-wise addition
void sum_up_values_each_neuron(neuronlayers_vec* pbiases_p, neuronlayers_vec* tmp_biases);                          
//result will be saved in the first argument .. simple component-wise addition
void sum_up_values_each_edge(edgelayers_vec* pedges_p, edgelayers_vec* tmp_edges);

float activation_function(float pz);

float activation_function_derivative(float pactivation_function_value_z);

//standard is quadratic cost function
neuronlayer cost_derivative_times_activation_derivative(
                neuronlayer poutput_actual,
                neuronlayer poutput_expected);

//neuronarray_layer_1_index_i = SUM_over_j[ neoronarray_layer_0[index_j]*weightsmatrix_layer_01[index_i][index_j]
//
//           e01:edgefrom n0(layer i) to n1(layer i+1)
//
// inlayer   | e00 e01 |   outlayer_z     out_z    biases   outlayer
// [ ... ] * | e10 e11 | = [ .. ]       ; [ .. ] + [ .. ] = [ .. ]
//           | d20 e21 |
//
// edgelayers are arranged in the transposed way for a faster computation
// because of nearby memory:
neuronlayer calculate_next_layer_fast(
                neuronlayer pvec,
                edgelayer pmat,
                neuronlayer pbias_neurons);

#endif
