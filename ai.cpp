#include <iostream>
#include <string>
#include <vector>
#include "global_definitions.h"
#include "neuralnet.h"

int main(int argc, char* argv[])
{

        //configure neural net
        std::vector<int> layers_vec;
        layers_vec.push_back(3);
        layers_vec.push_back(2);
        layers_vec.push_back(5);
        layers_vec.push_back(6);
        layers_vec.push_back(5);
        layers_vec.push_back(3);
        layers_vec.push_back(1);
        layers_vec.push_back(2);
        layers_vec.push_back(5);
        layers_vec.push_back(6);
        layers_vec.push_back(5);
        layers_vec.push_back(3);
        layers_vec.push_back(1);
        layers_vec.push_back(2);
        layers_vec.push_back(5);
        layers_vec.push_back(6);
        layers_vec.push_back(5);
        layers_vec.push_back(3);
        layers_vec.push_back(1);
        layers_vec.push_back(2);
        layers_vec.push_back(5);
        layers_vec.push_back(6);
        layers_vec.push_back(5);
        layers_vec.push_back(3);
        layers_vec.push_back(1);

        NeuralNet net = NeuralNet(layers_vec);

        return ok;
}
