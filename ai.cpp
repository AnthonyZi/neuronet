#include <iostream>
#include <string>
#include <vector>
#include "fileinput.h"
#include "neuroai.h"
#include "return_states.h"


int main(int argc, char* argv[])
{
        if(argc <= 1)
        {
                std::cout << "please name a config-file as an argument" << std::endl;
                return argument_error;
        }


//configure neural net


        fileitem_vector config_vec = readneuroconfig(argv[1]);

        NeuroAI nai = NeuroAI(config_vec); 
        /*
        for(int i = 0; i<config_vec.size(); i++)
        {
                std::cout << config_vec[i].label << std::endl;
        }
        */

        return ok;
}
