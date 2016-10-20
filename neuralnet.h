#ifndef NEUROAI
#define NEUROAI

#include <vector>
#include <string>
#include <iostream>
#include "fileinput.h"
#include <cstdlib>
#include <ctime>
#include "global_definitions.h"

typedef struct neuron
{
        std::string name;
        float bias;
} neuron_st;

typedef std::vector<neuron_st> neuronlayer;
typedef std::vector<std::vector<float> > edgelayer; //syn[indexout][indexin] -> for(i)[..for(j)[.syn[i][j].]..]
typedef std::vector<neuronlayer> neuronlayers_vec; 
typedef std::vector<edgelayer> edgelayers_vec;

#include "nn_math.h"


class NeuralNet
{
private:
        neuronlayers_vec layers;
        edgelayers_vec edges;

        float rand_range(float pmin, float pmax);

public:
        NeuralNet(fileitem_vector pfvector);
        edgelayer generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out);
        void generate_initial_edges();

        void set_biases_layer(std::vector<float> pbiases, int player);
        void generate_initial_biases();
        void initialise_neural_net();
        std::vector<float> feedforward(std::vector<float> pinput);
        void print_layer_biases(int player); // arg-range:0 until #hidden + 1
        void print_edges(int poutput_layer); // arg-range:1 until #hidden + 1
        void print_output(std::vector<float> presult);
};

#endif