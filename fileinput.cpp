#include "fileinput.h"

fileitem_vector readneuroconfig(std::string pfile)
{
        fileitem_vector itemlist;

        std::ifstream infile(pfile.c_str());

        std::string line;
        while(std::getline(infile, line))
        {
                fileitem_st item;
                std::istringstream iss(line);
                iss >> item.label;
                float value;
                std::string name;
                while(1)
                {
                        //get name of neuron
                        if(!(iss >> name))
                                break;
                        item.itemnames.push_back(name);
                }
              
                itemlist.push_back(item);
        }
        return itemlist;
}
