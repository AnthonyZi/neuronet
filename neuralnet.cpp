#include "neuralnet.h"

NeuralNet::NeuralNet(std::vector<int> player_sizes)
{
        for(int i = 0; i<player_sizes.size(); i++)
        {
                neuronlayer tmp_layer;
                tmp_layer.resize(player_sizes[i]);
                layers.push_back(tmp_layer);
        }

        initialise_neural_net();

        std::cout << "biases" << std::endl;
        for(int i = 0; i < layers.size(); i++) //input_layer + output_layer + hidden layers
                print_layer_biases(i);

        std::cout << std::endl << "edges" << std::endl;
        for(int i = 1; i < layers.size(); i++) //output_layer + hidden layers
                print_edges(i);
        std::cout << std::endl;


        std::vector<float> test;

        test.push_back(1.0);
        test.push_back(0.3);
        test.push_back(0.5);

        test = feedforward(test);

        print_output(test);
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

        for(int i = 1; i<layers.size(); i++)
        {
                tmp_edgelayer = generate_edge_layer(layers[i-1], layers[i]);
                edges.push_back(tmp_edgelayer);
        }
}

void NeuralNet::set_biases_layer(std::vector<float> pbiases, int player)
{
        if(pbiases.size() != layers[player].size())
        {
                exit(inconsistent_bias_numbers);
        }
        layers[player] = pbiases;
}

void NeuralNet::generate_initial_biases()
{
        for(int l = 1; l<layers.size(); l++) // l=0 is the input layer and biases do not need to be set
        {
                for(int i = 0; i<layers[l].size(); i++)
                {
                        layers[l][i] = rand_range(-1.0, 1.0);
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
        if(pinput.size() != layers[0].size())
                exit(wrong_input_size);
        std::vector<float> tmp(pinput);
        for(int l = 1; l<layers.size(); l++)
                tmp = calculate_layer(tmp, edges[l-1], layers[l]);
        return tmp;
}

char NeuralNet::get_layer_type(int player)
{
        char tmp;
        if(player==0)
                tmp = 'i';
        else
        {
                if(player==layers.size()-1)
                        tmp = 'o';
                else
                        tmp = 'h';
        }
        return tmp;
}

void NeuralNet::print_layer_biases(int player)
{
        for(int i = 0; i<layers[player].size(); i++)
        {
                std::cout << get_layer_type(player) << i+1 << " : " << layers[player][i] << std::endl;
        }
}

void NeuralNet::print_edges(int poutput_layer)
{
        for(int i = 0; i<layers[poutput_layer].size(); i++)
        {
                for(int j = 0; j<layers[poutput_layer-1].size(); j++)
                {
                        std::cout << get_layer_type(poutput_layer-1) << j+1 << "->" << get_layer_type(poutput_layer) << i+1 << " : " << edges[poutput_layer-1][i][j] << std::endl;
                }
        }
}

void NeuralNet::print_output(std::vector<float> presult)
{
        for(int i = 0; i<presult.size(); i++)
                std::cout << "o" << i+1 << " -> " << presult[i] << std::endl;
}
