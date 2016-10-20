#include "neuralnet.h"

NeuralNet::NeuralNet(fileitem_vector pfvector)
{
        //print the configuration out for user-check
        for(int i = 0; i<pfvector.size(); i++)
        {
                std::cout << pfvector[i].label << ":";
                for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                {
                        std::cout << " ." << pfvector[i].itemnames[j];
                }
                std::cout << std::endl;
        }
        std::cout << std::endl;
        
        //iterate through each line and set up the neural net
        int order_var = 0;
        for(int i = 0; i<pfvector.size(); i++)
        {
                neuronlayer tmp_layer;
                if(pfvector[i].label.compare("input") == 0)
                {
                        if(order_var != 0)
                        {
                                std::cout << "format error in config file - (input,hidden,output order)" << std::endl;
                                exit(broken_config);
                        }
                        order_var++;

                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                tmp_layer.push_back(newneuron);
                        }
                        layers.push_back(tmp_layer);
                }
                
                if(pfvector[i].label.compare("hidden") == 0)
                {
                        if(order_var != 1)
                        {
                                std::cout << "format error in config file - (input,hidden,output order)" << std::endl;
                                exit(broken_config);
                        }

                        //initialize new layer
                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                tmp_layer.push_back(newneuron);
                        }
                        //push new layer to the hiddenlayers_vec 'hidden'
                        layers.push_back(tmp_layer);
                }
                if(pfvector[i].label.compare("output") == 0)
                {
                        if(order_var != 1)
                        {
                                std::cout << "format error in config file - (input,hidden,output order)" << std::endl;
                                exit(broken_config);
                        }
                        order_var++;

                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                tmp_layer.push_back(newneuron);
                        }
                        layers.push_back(tmp_layer);

                }
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

        test = feedforward(test);

        print_output(test);

//        std::vector<float> param;
//        param.push_back(1.0);
//        param.push_back(0.3);
//        param.push_back(0.9);
//        param.push_back(0.1);
//        param.push_back(0.5);
//
//        set_input(param);
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
        for(int i = 0; i<pbiases.size(); i++)
        {
                layers[player][i].bias = pbiases[i];
        }
}

void NeuralNet::generate_initial_biases()
{
        for(int l = 1; l<layers.size(); l++) // l=0 is the input layer and biases do not need to be set
        {
                /*
                std::vector<float> tmpbiases;
                for(int i = 0; i<layers[l].size(); i++)
                {
                       tmpbiases.push_back(rand_range(-1.0, 1.0)); 
                }
                set_biases_layer(tmpbiases, l);
                */

                //faster implementation of the above
                //initial set up of all biases within a layer
                for(int i = 0; i<layers[l].size(); i++)
                {
                        layers[l][i].bias = rand_range(-1.0, 1.0);
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

void NeuralNet::print_layer_biases(int player)
{
        for(int i = 0; i<layers[player].size(); i++)
        {
                std::cout << layers[player][i].name << " : " << layers[player][i].bias << std::endl;
        }
}

void NeuralNet::print_edges(int poutput_layer)
{
        for(int i = 0; i<layers[poutput_layer].size(); i++)
        {
                for(int j = 0; j<layers[poutput_layer-1].size(); j++)
                {
                        std::cout << layers[poutput_layer-1][j].name << "->" << layers[poutput_layer][i].name << " : " << edges[poutput_layer-1][i][j] << std::endl;
                }
        }
}

void NeuralNet::print_output(std::vector<float> presult)
{
        for(int i = 0; i<presult.size(); i++)
                std::cout << layers[layers.size()-1][i].name << " -> " << presult[i] << std::endl;
}
