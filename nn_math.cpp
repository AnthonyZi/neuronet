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
std::vector<float> vector_matrix_multiplication(std::vector<float> pvec, std::vector<std::vector<float> > pmat)
{
        
}

//pay attention to use this only for matrices and not for arrays of arrays with
//different length
std::vector<std::vector<float> > transpose_2d_matrix(std::vector<std::vector<float> > pmat)
{
        std::vector<std::vector<float> > resMat;
        for(int col = 0; col<pmat.size(); col++)
        {
                std::vector<float> newrow;
                for(int row = 0; row<pmat.size(); row++)
                {
                        newrow.push_back(pmat[row][col]);
                }
                resMat.push_back(newrow);
        }
        return resMat;
}

//NOTE_FOR_LATER_DEVELOPEMENT
//this should be declared inline!
neuronlayer calculate_delta_pre_layer(neuronlayer pdelta_of_next_layer, edgelayer pedges_mat)
{
        return vector_matrix_multiplication(pdelta_of_next_layer, transpose_2d_matrix(pedges_mat));
}

neuronlayers_vec sum_up_values_each_neuron(neuronlayers_vec* pbiases_p, neuronlayers_vec* tmp_biases)
{
        for(int dim1 = 0; dim1<pbiases_p->size(); dim1++)
        {
                for(int dim2 = 0; dim2<(*pbiases_p)[0].size(); dim2++)
                {
                        (*pbiases_p)[dim1][dim2] += (*tmp_biases)[dim1][dim2];
                }
        }
}


edgelayers_vec sum_up_values_each_edge(edgelayers_vec* pedges_p, edgelayers_vec* tmp_edges)
{
        for(int dim1 = 0; dim1<pedges_p->size(); dim1++)
        {
                for(int dim2 = 0; dim2<(*pedges_p)[0].size(); dim2++)
                {
                        for(int dim3 = 0; dim3<(*pedges_p)[0][0].size(); dim3++)
                        {
                                (*pedges_p)[dim1][dim2][dim3] += (*tmp_edges)[dim1][dim2][dim3];
                        }
                }
        }
}


float activation_function(float pz)
{
        #ifdef SOFTSIGN
        return ( pz/(1+abs(pz)) );

        //SIGMOID is standard
        #else
        return ( 1/(1+exp(-pz)) );
        #endif
}

neuronlayer cost_derivative_times_activation_derivative(neuronlayer poutput_actual, neuronlayer poutput_expected)
{
        neuronlayer result;
        for(int i = 0; i<poutput_expected.size(); i++)
                result.push_back( (poutput_actual[i]-poutput_expected[i])*activation_function_derivative(poutput_actual[i]) );
        return result;
}


float activation_function_derivative(float pactivation_function_value_z)
{
        #ifdef SOFTSIGN
        return pactivation_function_value_z*pactivation_function_value_z;

        #else
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
neuronlayer calculate_next_layer(neuronlayer pvec, edgelayer pmat, neuronlayer pbias_neurons)
{
        neuronlayer tmpvec;
        for(int i = 0; i<pmat.size(); i++)
        {
                tmpvec.push_back(activation_function(dot_product(pvec, pmat[i]) + pbias_neurons[i]));
        }
        return tmpvec;
}
