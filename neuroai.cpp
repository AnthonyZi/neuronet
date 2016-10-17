#include "neuroai.h"

NeuroAI::NeuroAI(fileitem_vector pfvector)
{
        
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
        generate_edges();

        for(int i = 1; i < layers.size(); i++) //output_layer + hidden layers
                print_edges(i);

        std::vector<float> param;
        param.push_back(1.0);
        param.push_back(0.3);
        param.push_back(0.9);
        param.push_back(0.1);
        param.push_back(0.5);

        set_input(param);

        for(int i = 0; i < layers.size(); i++) //input_layer + output_layer + hidden layers
                print_layer_biases(i);

        calculate_output();

        for(int i = 0; i < layers.size(); i++) //input_layer + output_layer + hidden layers
                print_layer_biases(i);
}


//private

float NeuroAI::randweight(float pmin, float pmax)
{
        float range = pmax-pmin;
        float random_variable = (float)std::rand();
        random_variable = (range*random_variable/RAND_MAX)+pmin;
        return random_variable;
}

//public
edgelayer NeuroAI::generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out)
{
        edgelayer tmp_edgelayer;

        for(int out = 0; out<pnlayer_out.size(); out++)
        {
                std::vector<float> edge_set; //one set of edges referring to one of the output neurons
                for(int in = 0; in<pnlayer_in.size(); in++)
                {
                        edge_set.push_back(randweight(-1.0, 1.0));
                }
                tmp_edgelayer.push_back(edge_set);
        }

        return tmp_edgelayer;
}

void NeuroAI::generate_edges()
{
        std::srand(std::time(0)); // one seed for the now following random function calls
        edgelayer tmp_edgelayer;

        for(int i = 1; i<layers.size(); i++)
        {
                tmp_edgelayer = generate_edge_layer(layers[i-1], layers[i]);
                edges.push_back(tmp_edgelayer);
        }
        /*
        if(hidden.size()>0)
        {
                tmp_edgelayer = generate_edge_layer(input, hidden[0]);
                edges.push_back(tmp_edgelayer);
                for(int i = 1; i<hidden.size(); i++)
                {
                        tmp_edgelayer = generate_edge_layer(hidden[i-1], hidden[i]);
                        edges.push_back(tmp_edgelayer);
                }
                tmp_edgelayer = generate_edge_layer(hidden[hidden.size()-1], output);
                edges.push_back(tmp_edgelayer);
        }
        else
                tmp_edgelayer = generate_edge_layer(input, output);
                edges.push_back(tmp_edgelayer);
        */
}

void NeuroAI::set_input(std::vector<float> pbiases)
{
        if(pbiases.size() != layers[0].size())
                return;
        for(int i = 0; i<pbiases.size(); i++)
        {
                layers[0][i].bias = pbiases[i];
        }
}

void NeuroAI::calculate_output()
{
        for(int l = 1; l<layers.size(); l++)
        {
                for(int outp = 0; outp<layers[l].size(); outp++)
                {
                        for(int inp = 0; inp<layers[l-1].size(); inp++)
                        {
                                layers[l][outp].bias += layers[l-1][inp].bias * edges[l-1][outp][inp];
                        }
                }
        }
}

void NeuroAI::print_layer_biases(int player)
{
        for(int i = 0; i<layers[player].size(); i++)
        {
                std::cout << layers[player][i].name << " : " << layers[player][i].bias << std::endl;
        }
}

void NeuroAI::print_edges(int poutput_layer)
{
        for(int i = 0; i<layers[poutput_layer].size(); i++)
        {
                for(int j = 0; j<layers[poutput_layer-1].size(); j++)
                {
                        std::cout << layers[poutput_layer-1][j].name << "->" << layers[poutput_layer][i].name << " : " << edges[poutput_layer-1][i][j] << std::endl;
                }
        }

        

/*
        if(hidden.size()>0)
        {
                for(int i = 0; i<hidden[0].size(); i++)
                {
                        for(int j = 0; j<input.size(); j++)
                        {
                                std::cout << input[j].name << "->" << hidden[0][i].name << " : " << edges[0][i][j] << std::endl;
                        }
                }
                for(int h = 1; h<hidden.size()-1; h++)
                {
                        for(int i = 0; i<hidden[h].size(); i++)
                        {
                                for(int j = 0; j<hidden[h-1].size(); j++)
                                {
                                        std::cout << hidden[h-1][j].name << "->" << hidden[h][i].name << " : " << edges[h][i][j] << std::endl;
                                }
                        }
                }
                for(int i = 0; i<output.size(); i++)
                {
                        for(int j = 0; j<hidden[hidden.size()-1].size(); j++)
                        {
                                std::cout << hidden[hidden.size()-1][j].name << "->" << output[i].name << " : " << edges[hidden.size()-1][i][j] << std::endl;
                        }
                }
        }
        else
        {
                for(int i = 0; i<output.size(); i++)
                {
                        for(int j = 0; j<input.size(); j++)
                        {
                                std::cout << input[j].name << "->" << output[i].name << " : " << edges[0][i][j] << std::endl;
                        }
                }

        }
*/
}
