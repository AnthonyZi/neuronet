#ifndef FILEINPUT_H
#define FILEINPUT_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <iostream> //temp

typedef struct fileitem
{
        std::string label;
        std::vector<std::string> itemnames;
} fileitem_st;


typedef std::vector<fileitem_st> fileitem_vector;

fileitem_vector readneuroconfig(std::string pfile);

#endif
