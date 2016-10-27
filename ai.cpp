#include <iostream>
#include <string>
#include <vector>
#include "global_definitions.h"

#ifndef VER2
#include "fileinput.h"
#include "neuralnet.h"
#else
#include "neuralnet2.h"
#endif

int main(int argc, char* argv[])
{

//configure neural net


        #ifndef VER2
        if(argc <= 1)
        {
                std::cout << "please name a config-file as an argument" << std::endl;
                return argument_error;
        }

        fileitem_vector config_vec = readneuroconfig(argv[1]);

        NeuralNet net = NeuralNet(config_vec); 

        #else
        std::vector<int> configuration;
        configuration.push_back(3);
        configuration.push_back(2);
        configuration.push_back(5);
        configuration.push_back(6);
        configuration.push_back(5);
        configuration.push_back(3);
        configuration.push_back(1);
        configuration.push_back(2);
        configuration.push_back(5);
        configuration.push_back(6);
        configuration.push_back(5);
        configuration.push_back(3);
        configuration.push_back(1);
        configuration.push_back(2);
        configuration.push_back(5);
        configuration.push_back(6);
        configuration.push_back(5);
        configuration.push_back(3);
        configuration.push_back(1);
        configuration.push_back(2);
        configuration.push_back(5);
        configuration.push_back(6);
        configuration.push_back(5);
        configuration.push_back(3);
        configuration.push_back(1);

        NeuralNet net = NeuralNet(configuration);
        #endif
        /*
        for(int i = 0; i<config_vec.size(); i++)
        {
                std::cout << config_vec[i].label << std::endl;
        }
        */

        return ok;
}
