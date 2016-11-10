#include "neuralnet.h"

int random_func(int i)
{
        return std::rand()%i;
}

NeuralNet::NeuralNet(std::vector<int> player_sizes)
{
        for(int i = 0; i<player_sizes.size(); i++)
        {
                neuronlayer tmp_layer;
                tmp_layer.resize(player_sizes[i]);
                biases.push_back(tmp_layer);
        }

        initialise_neural_net();

// PRINT BIASES
/*
        std::cout << "biases" << std::endl;
        for(int i = 0; i < biases.size(); i++) //input_layer + output_layer + hidden layers
                print_layer_biases(i);
        std::cout << std::endl;
*/

// PRINT EDGES
/*
        std::cout << "edges" << std::endl;
        for(int i = 1; i < biases.size(); i++) //output_layer + hidden layers
                print_edges(i);
        std::cout << std::endl;
*/


//        std::vector<float> input;
//        std::vector<float> result;
//
//        input.push_back(1.0);
//        input.push_back(0.3);
//        input.push_back(0.5);
//
//        result = feedforward(input);
//        print_output(result);
}


//private

float NeuralNet::rand_range(float pmin, float pmax)
{
        float range = pmax-pmin;
        float random_variable = (float)std::rand();
        random_variable = (range*random_variable/RAND_MAX)+pmin;
        return random_variable;
}

//public
//
//structure:
//
//              input
//              ------>
//
//        |    | e00 e10 e20 e30 |  all weights from layer i to layer i+1 so that z[i+1][0] can be calculated with this
// output |    | e01 e11 e21 e31 |
//        V    | e02 e12 e22 e32 |
edgelayer NeuralNet::generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out)
{
        edgelayer tmp_edgelayer;

        for(int out = 0; out<pnlayer_out.size(); out++)
        {
                std::vector<float> edge_set; //one set of edges referring to one of the output neurons
                for(int in = 0; in<pnlayer_in.size(); in++)
                {
                        edge_set.push_back(rand_range(-0.005, 0.005));
                } tmp_edgelayer.push_back(edge_set);
        }

        return tmp_edgelayer;
}

void NeuralNet::generate_initial_edges()
{
        edgelayer tmp_edgelayer;

        for(int i = 1; i<biases.size(); i++)
        {
                tmp_edgelayer = generate_edge_layer(biases[i-1], biases[i]);
                edges.push_back(tmp_edgelayer);
        }
}

// structure:
//
// memory ->
// | i1 i2 i3 i4 i5 |   inputlayer
// | h1 h2 h3 h4 |      hiddenlayer1
// | h1 h2 h3 |         hiddenlayer2
// | h1 h2 |            hiddenlayer3
// | o1 |               outputlayer
//
// hiddenlayer1 & 3 neuron : biases[1][2]
void NeuralNet::generate_initial_biases()
{
        for(int l = 1; l<biases.size(); l++) // l=0 is the input layer and biases do not need to be set
        {
                for(int i = 0; i<biases[l].size(); i++)
                {
                        biases[l][i] = rand_range(-0.005, 0.005);
                }
        }
}

void NeuralNet::initialise_neural_net()
{
        std::srand(std::time(0)); //one seed for the now following random function calls
        generate_initial_edges();
        generate_initial_biases();
}

std::vector<float> NeuralNet::feedforward(std::vector<float> pinput)
{
        if(pinput.size() != biases[0].size())
                exit(wrong_input_size);
        std::vector<float> tmp(pinput);
        for(int l = 1; l<biases.size(); l++)
                tmp = calculate_next_layer_fast(tmp, edges[l-1], biases[l]);
        return tmp;
}

void NeuralNet::update_through_backprop_over_mini_batch(
                const std::vector<training_data_s>* ptraining_data_s_vec_p,
                float pupdate_rate)
{
        //allocate a copy of biases and edges so that backpropagation
        //can be applied on the "original" weights for all the training datas
        //in the mini_batch
        neuronlayers_vec tmp_biases(biases.begin()+1, biases.end());
        edgelayers_vec tmp_edges_transposed = transpose_2d_matrices_of_vector(edges);
        for(int layer = 0; layer<tmp_biases.size(); layer++)
        {
                std::fill(tmp_biases[layer].begin(), tmp_biases[layer].end(), 0);
                for(int i = 0; i<tmp_edges_transposed[layer].size(); i++)
                {
                        std::fill(tmp_edges_transposed[layer][i].begin(), tmp_edges_transposed[layer][i].end(), 0);
                }
        }



        //NOTE_FOR_LATER_DEVELOPEMENT
        //perfect for parallelisation
        for(int i = 0; i<ptraining_data_s_vec_p->size(); i++)
        {
                //NOTE_FOR_LATER_DEVELOPEMENT
                //biases and edges are forwarded as addresses, but pay attention
                //that these values are changed within a mutal exclusion if you
                //make use of parallelisation
                backpropagation_fast((*ptraining_data_s_vec_p)[i], &tmp_biases, &tmp_edges_transposed, pupdate_rate);
                std::cout << "iteration " << i << " over mini-batch finished" << std::endl;
        }

        //finally overwrite the old weights and biases with the slightly
        //adjusted version
        tmp_biases.insert(tmp_biases.begin(), biases[0]);
//!!!!!
//the functionality here is wrong! 
//it should be:
//      "biases" = "biases" -(componentwise) "pupdate_rate" /(componentwise) "tmp_biases"
//      "weights" = "weights" -(componentwise) "pupdate_rate" /(componentwise) "tmp_weights"
////        biases = tmp_biases;
////        edges = transpose_2d_matrices_of_vector(tmp_edges_transposed);
}

void NeuralNet::backpropagation_fast(
                const training_data_s ptraining_data,
                neuronlayers_vec* pbiases_p,
                edgelayers_vec* pedges_transposed_p,
                float pupdate_rate)
{
        //allocation of the temporal memomry by copying the same structure by
        //hoping that the copy-implementation of the vector-class is very
        //efficient
        neuronlayers_vec tmp_biases = *pbiases_p;
        edgelayers_vec tmp_edges_transposed = *pedges_transposed_p;

        neuronlayers_vec activations = tmp_biases;

        if(ptraining_data.input.size() != biases[0].size())
                exit(wrong_input_size);


        activations[0] = calculate_next_layer_fast(ptraining_data.input, edges[0], biases[0]);
                
        for(int i = 1; i<activations.size(); i++)
        {
                activations[i] = calculate_next_layer_fast(activations[i-1], edges[i], biases[i]);
        }


        //just for an better overview - nnl : number_neuron_layers
        int nnl = tmp_biases.size();


        //after calculation of the first delta(tmp_biases[nnl-1]) we can
        //recursivly calculate all the weights and all the biases

        //the last neuron layer is calculated with cost function(derivative)
        //in dependence of the last! activation(output), expected output
        //and the derivative of the activation-function in dep of the last
        //activation
        tmp_biases[nnl-1] = cost_derivative_times_activation_derivative(activations[nnl-1], ptraining_data.output);
        //feed vectors here in reverse order to obtain the
        //transposed edges
        //(see comment of vector_multiplication_2d for the idea of that
        //function)
        tmp_edges_transposed[nnl-1] = vector_multiplication_2d(activations[nnl-2], tmp_biases[nnl-1]);



        //for loop for recursive backprop        

        //i==0 needs the input of the system to compute the difference of
        //the weights of the first layer so this will take place after
        //the for loop
        for(int i = nnl-2; i>0; i--)
        {
                std::cout << "loop-iteration: " << i << std::endl;
                tmp_biases[i] = calculate_delta_pre_layer_fast(tmp_biases[i+1], tmp_edges_transposed[i+1], activations[i]);
                //feed vectors here in reverse order to obtain the
                //transposed edges as above
//                tmp_edges_transposed[i] =  vector_multiplication_2d(activations[i-1], tmp_biases[i]);
                tmp_edges_transposed[i] =  vector_multiplication_2d(tmp_biases[i], activations[i-1]);
        }


        //iteration i==0:
        tmp_biases[0] = calculate_delta_pre_layer_fast(tmp_biases[1], tmp_edges_transposed[1], activations[0]);
        //feed vectors here in reverse order to obtain the
        //transposed edges as above
//        tmp_edges_transposed[0] = vector_multiplication_2d(activations[0], ptraining_data.input);
        tmp_edges_transposed[0] = vector_multiplication_2d(ptraining_data.input, activations[0]);


        //referral of the adjustments to the net-bias and net-weights
        sum_up_values_each_neuron(pbiases_p, &tmp_biases);
        sum_up_values_each_edge(pedges_transposed_p, &tmp_edges_transposed);
}

void NeuralNet::stochastic_gradient_descent(
                std::vector<training_data_s>* ptraining_data,
                int pnum_epochs,
                int pmini_batch_size,
                float ptraining_rate)
{
        for(int e = 0; e<pnum_epochs; e++)
        {
                //shuffles the trainig_data (this is important for the mini-batch method!
                std::random_shuffle(ptraining_data->begin(), ptraining_data->end(), random_func);
                
                std::vector<training_data_s>::const_iterator first_elem;
                std::vector<training_data_s>::const_iterator last_elem;
                const std::vector<training_data_s>* mini_training_data;

                float update_rate = ptraining_rate/pmini_batch_size;
                int num_of_batches = (ptraining_data->size()+pmini_batch_size-1)/pmini_batch_size;
                //first over for-loop all "full" mini-batches
                for(int batch = 0; batch<num_of_batches-1; batch++)
                {
                        first_elem = ptraining_data->begin() + (batch*pmini_batch_size);
                        last_elem = ptraining_data->begin() + ((batch+1)*pmini_batch_size)-1;
                        mini_training_data = new std::vector<training_data_s>(first_elem, last_elem);
                        update_through_backprop_over_mini_batch(mini_training_data, update_rate);
                }


                update_rate = ptraining_rate/(ptraining_data->size()%pmini_batch_size);
                //rest of the training-data
                first_elem = ptraining_data->begin() + ((num_of_batches-1)*pmini_batch_size);
                last_elem = ptraining_data->end();
                mini_training_data = new std::vector<training_data_s>(first_elem, last_elem);
                update_through_backprop_over_mini_batch(mini_training_data, update_rate);
        }
}


std::string NeuralNet::get_layer_name(int player)
{
        std::string tmp;
        if(player==0)
                tmp = 'i';
        else
        {
                if(player==biases.size()-1)
                        tmp = 'o';
                else
                        tmp = "h"+std::to_string(player)+"-";
        }
        return tmp;
}

void NeuralNet::print_layer_biases(int player)
{
        for(int i = 0; i<biases[player].size(); i++)
        {
                std::cout << get_layer_name(player) << i+1 << " : " << biases[player][i] << std::endl;
        }
}

void NeuralNet::print_edges(int poutput_layer)
{
        for(int i = 0; i<biases[poutput_layer].size(); i++)
        {
                for(int j = 0; j<biases[poutput_layer-1].size(); j++)
                {
                        std::cout << get_layer_name(poutput_layer-1) << j+1 << " -> " << get_layer_name(poutput_layer) << i+1 << " : " << edges[poutput_layer-1][i][j] << std::endl;
                }
        }
}

void NeuralNet::print_output(std::vector<float> presult)
{
        for(int i = 0; i<presult.size(); i++)
                std::cout << "o" << i+1 << " -> " << presult[i] << std::endl;
}
