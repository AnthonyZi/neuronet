#ifndef NEUROAI
#define NEUROAI

#include <vector>

class NeuroAI
{

typedef std::vector<float> layer_t;
typedef std::vector<std::vector<float>> edge_t; //syn[indexout][indexin] -> for(i)[..for(j)[.syn[i][j].]..]
typedef std::vector<layer_t> hiddenlayers_t; 
typedef std::vector<edge_t> edgelayers_t;

private:
        layer_t input;
        hiddenlayers_t hidden;
        layer_t output;
        edgelayers_t edges;

public:
        void add_hidden_layer(layer_t player);
        void add_edges(edge_t pedges);
}
