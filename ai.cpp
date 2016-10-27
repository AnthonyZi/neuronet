#include <iostream>
#include <string>
#include <vector>
#include "global_definitions.h"
#include "neuralnet.h"

int main(int argc, char* argv[])
{

        //configure neural net
        std::vector<int> input_vec;
        input_vec.push_back(3);
        input_vec.push_back(2);
        input_vec.push_back(5);
        input_vec.push_back(6);
        input_vec.push_back(5);
        input_vec.push_back(3);
        input_vec.push_back(1);
        input_vec.push_back(2);
        input_vec.push_back(5);
        input_vec.push_back(6);
        input_vec.push_back(5);
        input_vec.push_back(3);
        input_vec.push_back(1);
        input_vec.push_back(2);
        input_vec.push_back(5);
        input_vec.push_back(6);
        input_vec.push_back(5);
        input_vec.push_back(3);
        input_vec.push_back(1);
        input_vec.push_back(2);
        input_vec.push_back(5);
        input_vec.push_back(6);
        input_vec.push_back(5);
        input_vec.push_back(3);
        input_vec.push_back(1);

        NeuralNet net = NeuralNet(input_vec);

        return ok;
}
