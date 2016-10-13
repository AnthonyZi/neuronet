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
        

        for(int i = 0; i<pfvector.size(); i++)
        {
                if(pfvector[i].label.compare("input") == 0)
                {
                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                input.push_back(newneuron);
                        }
                }
                if(pfvector[i].label.compare("output") == 0)
                {
                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                output.push_back(newneuron);
                        }

                }
                if(pfvector[i].label.compare("hidden") == 0)
                {
                        neuronlayer hlayer;
                        //initialize new layer
                        for(int j = 0; j<pfvector[i].itemnames.size(); j++)
                        {
                                neuron_st newneuron;
                                //initialize new neuron
                                newneuron.name = pfvector[i].itemnames[j];
                                newneuron.bias = 0;

                                hlayer.push_back(newneuron);
                        }
                        //push new layer to the hiddenlayers_vec 'hidden'
                        hidden.push_back(hlayer);
                }
        }
        generate_edges();

        for(int i = 1; i <= (1 + hidden.size()); i++) //output_layer + hidden layers
                print_edges(i);

        for(int i = 0; i <= (1 + hidden.size()); i++) //input_layer + output_layer + hidden layers
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
}

void NeuroAI::set_input(std::vector<float> pbiases)
{
}

void NeuroAI::calculate_output()
{
}

void NeuroAI::print_layer_biases(int player)
{
        neuronlayers_vec all_layers;
        all_layers.push_back(input);
        for(int i = 0; i<hidden.size(); i++)
                all_layers.push_back(hidden[i]);
        all_layers.push_back(output);

        for(int i = 0; i<all_layers[player].size(); i++)
        {
                std::cout << all_layers[player][i].name << " : " << all_layers[player][i].bias << std::endl;
        }
}

void NeuroAI::print_edges(int poutput_layer)
{
        
        neuronlayers_vec all_layers;
        all_layers.push_back(input);
        for(int i = 0; i<hidden.size(); i++)
                all_layers.push_back(hidden[i]);
        all_layers.push_back(output);

        for(int i = 0; i<all_layers[poutput_layer].size(); i++)
        {
                for(int j = 0; j<all_layers[poutput_layer-1].size(); j++)
                {
                        std::cout << all_layers[poutput_layer-1][j].name << "->" << all_layers[poutput_layer][i].name << " : " << edges[poutput_layer-1][i][j] << std::endl;
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
