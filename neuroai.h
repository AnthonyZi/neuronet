#ifndef NEUROAI
#define NEUROAI

#include <vector>
#include <string>
#include <iostream>
#include "fileinput.h"
#include <cstdlib>
#include <ctime>

typedef struct neuron
{
        std::string name;
        float bias;
} neuron_st;

typedef std::vector<neuron_st> neuronlayer;
typedef std::vector<std::vector<float> > edgelayer; //syn[indexout][indexin] -> for(i)[..for(j)[.syn[i][j].]..]
typedef std::vector<neuronlayer> neuronlayers_vec; 
typedef std::vector<edgelayer> edgelayers_vec;



class NeuroAI
{
private:
        neuronlayer input, output;
        neuronlayers_vec hidden;
        edgelayers_vec edges;

        float randweight(float pmin, float pmax);

public:
        NeuroAI(fileitem_vector pfvector);
        edgelayer generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out);
        void generate_edges();

        void set_input(std::vector<float> pbiases);
        void calculate_output();
        void print_layer_biases(int player); // arg-range:0 until #hidden + 1
        void print_edges(int poutput_layer); // arg-range:1 until #hidden + 1
};

#endif
