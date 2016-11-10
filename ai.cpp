#include <iostream>
#include <string>
#include <vector>
#include "global_definitions.h"
#include "neuralnet.h"

int main(int argc, char* argv[])
{

        //configure neural net
        std::vector<int> layers_vec;
//        layers_vec.push_back(60);
//        layers_vec.push_back(40);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(400);
//        layers_vec.push_back(20);
        layers_vec.push_back(3);
        layers_vec.push_back(2);
        layers_vec.push_back(1);

        NeuralNet net = NeuralNet(layers_vec);
        
        std::vector<float> input;
        std::vector<float> result;
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
//        input.push_back(0);
//        input.push_back(1);
        input.push_back(0);
        input.push_back(0);
        input.push_back(1);


        std::cout << "feedforward-start" << std::endl;
        result = net.feedforward(input);
        std::cout << "feedforward-end" << std::endl;

        net.print_output(result);


        std::vector<float> expected;
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(0);
//        expected.push_back(0);
//        expected.push_back(1);
//        expected.push_back(0);
        expected.push_back(1);

        training_data_s tds;
        tds.input=input;
        tds.output=expected;

        std::vector<training_data_s> tdsv;
        tdsv.push_back(tds);
        float rate = 3.0;

        net.update_through_backprop_over_mini_batch(&tdsv, rate);

        return ok;
}
