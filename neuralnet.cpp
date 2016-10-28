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
edgelayer NeuralNet::generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out)
{
        edgelayer tmp_edgelayer;

        for(int out = 0; out<pnlayer_out.size(); out++)
        {
                std::vector<float> edge_set; //one set of edges referring to one of the output neurons
                for(int in = 0; in<pnlayer_in.size(); in++)
                {
                        edge_set.push_back(rand_range(-1.0, 1.0));
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

void NeuralNet::generate_initial_biases()
{
        for(int l = 1; l<biases.size(); l++) // l=0 is the input layer and biases do not need to be set
        {
                for(int i = 0; i<biases[l].size(); i++)
                {
                        biases[l][i] = rand_range(-1.0, 1.0);
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
                tmp = calculate_layer(tmp, edges[l-1], biases[l]);
        return tmp;
}

void NeuralNet::update_through_backprop_over_mini_batch(
                const std::vector<training_data_s>* ptraining_data_s_vec_p,
                float pupdate_rate)
{
        //allocate a copy of biases and edges so that backpropagation
        //can be applied on the "original" weights for all the training datas
        //in the mini_batch
        neuronlayers_vec tmp_biases = biases;
        edgelayers_vec tmp_edges = edges;



        //NOTE_FOR_LATER_DEVELOPEMENT
        //perfect for parallelisation
        for(int i = 0; i<ptraining_data_s_vec_p->size(); i++)
        {
                //NOTE_FOR_LATER_DEVELOPEMENT
                //biases and edges are forwarded as addresses, but pay attention
                //that these values are changed within a mutal exclusion if you
                //make use of parallelisation
                backpropagation((*ptraining_data_s_vec_p)[i], &tmp_biases, &tmp_edges, pupdate_rate);
        }

        //finally overwrite the old weights and biases with the slightly
        //adjusted version
        biases = tmp_biases;
        edges = tmp_edges;
}

void NeuralNet::backpropagation(
                const training_data_s ptraining_data,
                neuronlayers_vec* pbiases_p,
                edgelayers_vec* pedges_p,
                float pupdate_rate)
{
        neuronlayers_vec tmp_biases = *pbiases_p;
        edgelayers_vec tmp_edges = *pedges_p;

        neuronlayers_vec activations(tmp_biases.begin()+1, tmp_biases.end());

        if(ptraining_data.input.size() != biases[0].size())
                exit(wrong_input_size);
/*
        std::vector<float> tmp(pinput);
        for(int l = 1; l<biases.size(); l++)
                tmp = calculate_layer(tmp, edges[l-1], biases[l]);
        return tmp;
*/
        activations[0] = calculate_layer(ptraining_data.input, edges[0], biases[1]);
                
        for(int i = 1; i<activations.size(); i++)
        {
                activations[i] = calculate_layer(activations[i-1], edges[i], biases[i+1]);
        }


        //just for an better overview - nnl : number_neuron_layers
        int nnl = tmp_biases.size();

        tmp_biases[nnl-1] = cost_derivative_times_activation_derivative(
                                                                        activations[nnl-2],
                                                                        ptraining_data.output);
        
        //hier fehlt noch ganz viel
        //backprop - for loop startet hier!
        std::cout << std::endl << std::endl;
        print_output(tmp_biases[nnl-1]);
        std::cout << std::endl << std::endl;
        print_output(activations[nnl-2]);
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
