#include "nn_math.h"

//NOTE_FOR_LATER_DEVELOPEMENT
//potential parallelisation
//a[1]*b[1]+a[2]*b[2]+ ... + a[i]*b[i]
float dot_product(std::vector<float> pveca, std::vector<float> pvecb)
{
        float sum = 0;
        for(int i = 0; i<pveca.size(); i++)
                sum += pveca[i]*pvecb[i];
        return sum;
}

//NOTE_FORLATER_DEVELOPEMENT
//potential parallelism
//
// | va1 |                              | . . . . . |
// | va2 |                              | . . . . . |
// | va3 |   *  [vb1 vb2 vb3 vb4 vb5] = | . . . . . |
// | va4 |                              | . . . . . |
// | va5 |                              | . . . . . |
//
std::vector<std::vector<float> > vector_multiplication_2d(std::vector<float> pveca, std::vector<float> pvecb)
{
        std::vector<std::vector<float> > resMat;


        for(int a = 0; a<pveca.size(); a++)
        {
                std::vector<float> row;
                for(int b = 0; b<pvecb.size(); b++)
                {
                        row.push_back(pveca[a]*pvecb[b]);
                }
                resMat.push_back(row);
        }

        return resMat;
}

//NOTE_FORLATER_DEVELOPEMENT
//potential parallelism
//
//
//           | . . . . . |
// [ ... ] * | . . . . . | = [ ..... ]
//           | . . . . . |
//
// this function needs the matrix transposed in order to compute faster
// with nearby memory addressing (see calculate_next_layer - the same
// applies there)
std::vector<float> vector_matrix_multiplication_fast(std::vector<float> pvec, std::vector<std::vector<float> > pmat)
{
        std::vector<float> resVec;
        for(int i = 0; i<pmat.size(); i++)
        {
                float sum = 0;
                for(int j = 0; j<pvec.size(); j++)
                {
                        sum += pvec[j]*pmat[i][j];
                }
                resVec.push_back(sum);
        }
        return resVec;
}

//pay attention to use this only for matrices and not for arrays of arrays with
//different length
std::vector<std::vector<float> > transpose_2d_matrix(std::vector<std::vector<float> > pmat)
{
        std::vector<std::vector<float> > resMat;
        for(int newcolindex = 0; newcolindex<pmat[0].size(); newcolindex++)
        {
                std::vector<float> newrow;
                for(int newrowindex = 0; newrowindex<pmat.size(); newrowindex++)
                {
                        //newrowindex=oldcolindex , newcolindex=oldrowindex
                        newrow.push_back(pmat[newrowindex][newcolindex]);
                }
                resMat.push_back(newrow);
        }
        return resMat;
}

std::vector<std::vector<std::vector<float> > > transpose_2d_matrices_of_vector(std::vector<std::vector<std::vector<float> > > pmat_vec)
{
        std::vector<std::vector<std::vector<float> > > resMatVec;
        for(int i = 0; i<pmat_vec.size(); i++)
        {
                resMatVec.push_back(transpose_2d_matrix(pmat_vec[i]));
        }
        return resMatVec;
}

//NOTE_FOR_LATER_DEVELOPEMENT
//actually the same code as "vector_matrix_multiplication_fast" but adapted to:
//("vector" * "matrix" ) *(componentwise) "derivatives_of_activations_of_one_layer_before"
//feed the edgelayer-matrix transposed (the memory can be accessed faster with
//a reorganised/transposed matrix
neuronlayer calculate_delta_pre_layer_fast(neuronlayer pdelta_of_next_layer, edgelayer pedges_mat, neuronlayer pactivations_of_pre_layer)
{
        std::vector<float> resVec;
        for(int i = 0; i<pedges_mat.size(); i++)
        {
                float sum = 0;
                for(int j = 0; j<pdelta_of_next_layer.size(); j++)
                {
                        sum += pdelta_of_next_layer[j]*pedges_mat[i][j];
                }
                sum *= activation_function_derivative(pactivations_of_pre_layer[i]);
                std::cout << "sum=" << sum << std::endl;
                resVec.push_back(sum);
        }
        return resVec;
}

void sum_up_values_each_neuron(neuronlayers_vec* pbiases_p, neuronlayers_vec* tmp_biases)
{
        for(int dim1 = 0; dim1<pbiases_p->size(); dim1++)
        {
                for(int dim2 = 0; dim2<(*pbiases_p)[0].size(); dim2++)
                {
                        (*pbiases_p)[dim1][dim2] += (*tmp_biases)[dim1][dim2];
                }
        }
}


void sum_up_values_each_edge(edgelayers_vec* pedges_p, edgelayers_vec* tmp_edges)
{
        for(int dim1 = 0; dim1<pedges_p->size(); dim1++)
        {
                for(int dim2 = 0; dim2<(*pedges_p)[dim1].size(); dim2++)
                {
                        for(int dim3 = 0; dim3<(*pedges_p)[dim1][dim2].size(); dim3++)
                        {
                                std::cout << "dim1:" << dim1 << "  dim2:" << dim2 << "  dim3:" << dim3 << std::endl;
                                std::cout << "dimensions: " << pedges_p->size() << "," << tmp_edges->size() << "   " << (*pedges_p)[dim1].size() << "," << (*tmp_edges)[dim1].size() << "   " << (*pedges_p)[dim1][dim2].size() << "," << (*tmp_edges)[dim1][dim2].size() << std::endl;
                                (*pedges_p)[dim1][dim2][dim3] += (*tmp_edges)[dim1][dim2][dim3];
                        }
                }
        }
}


//NOTE_FOR_LATER_DEVELOPEMENT
//this should be declared inline
float activation_function(float pz)
{
        #ifdef SOFTSIGN
        return ( pz/(1+abs(pz)) );
        #endif

        #ifdef SIGMOID
        return ( 1/(1+exp(-pz)) );
        #endif
}

neuronlayer cost_derivative_times_activation_derivative(neuronlayer poutput_actual, neuronlayer poutput_expected)
{
        #ifdef QUADRATICCOST
        neuronlayer result;
        for(int i = 0; i<poutput_expected.size(); i++)
                result.push_back( (poutput_actual[i]-poutput_expected[i])*activation_function_derivative(poutput_actual[i]) );
        return result;
        #endif
}

//NOTE_FOR_LATER_DEVELOPEMENT
//this should be declared inline
float activation_function_derivative(float pactivation_function_value_z)
{
        #ifdef SOFTSIGN
        return pactivation_function_value_z*pactivation_function_value_z;
        #endif

        #ifdef SIGMOID
        return pactivation_function_value_z*(1-pactivation_function_value_z);
        #endif
}

//NOTE_FOR_LATER_DEVELOPEMENT
//potential parallellisation
//
//           e01:edgefrom n0(layer i) to n1(layer i+1)
//
// inlayer   | e00 e01 |   outlayer_z     out_z    biases   outlayer
// [ ... ] * | e10 e11 | = [ .. ]       ; [ .. ] + [ .. ] = [ .. ]
//           | d20 e21 |
//
// edgelayers are arranged in the transposed way for a faster computation
// because of nearby memory:
neuronlayer calculate_next_layer_fast(neuronlayer pvec, edgelayer pmat, neuronlayer pbias_neurons)
{
        neuronlayer tmpvec;
        for(int i = 0; i<pmat.size(); i++)
        {
                tmpvec.push_back(activation_function(dot_product(pvec, pmat[i]) + pbias_neurons[i]));
        }
        return tmpvec;
}
