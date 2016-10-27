#include "nn_math.h"

float dot_product(std::vector<float> pveca, std::vector<float> pvecb)
{
        float sum = 0;
        for(int i = 0; i<pveca.size(); i++)
                sum += pveca[i]*pvecb[i];
        return sum;
}

float activation_function(float z)
{
        #ifdef SOFTSIGN
        return ( z/(1+abs(z)) );

        //SIGMOID is standard
        #else
        return ( 1/(1+exp(-z)) );
        #endif
}

float activation_function_derivative(float pactivation_function_of_z)
{
        #ifdef SOFTSIGN
        return pactivation_function_of_z*pactivation_function_of_z;

        #else
        return pactivation_function_of_z*(1-pactivation_function_of_z);
        #endif
}

std::vector<float> calculate_layer(std::vector<float> pvec, std::vector<std::vector<float> > pmat, neuronlayer pbias_neurons)
{
        std::vector<float> tmpvec;
        for(int i = 0; i<pmat.size(); i++)
        {
                tmpvec.push_back(activation_function(dot_product(pvec, pmat[i]) + pbias_neurons[i]));
        }
        return tmpvec;
}
