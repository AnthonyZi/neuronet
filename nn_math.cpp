#include "nn_math.h"

//NOTE_FOR_LATER_DEVELOPEMENT
//potential parallellisation
float dot_product(std::vector<float> pveca, std::vector<float> pvecb)
{
        float sum = 0;
        for(int i = 0; i<pveca.size(); i++)
                sum += pveca[i]*pvecb[i];
        return sum;
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

std::vector<float> cost_derivative_times_activation_derivative(std::vector<float> poutput_actual, std::vector<float> poutput_expected)
{
        std::vector<float> result;
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
std::vector<float> calculate_layer(std::vector<float> pvec, std::vector<std::vector<float> > pmat, neuronlayer pbias_neurons)
{
        std::vector<float> tmpvec;
        for(int i = 0; i<pmat.size(); i++)
        {
                tmpvec.push_back(activation_function(dot_product(pvec, pmat[i]) + pbias_neurons[i]));
        }
        return tmpvec;
}
