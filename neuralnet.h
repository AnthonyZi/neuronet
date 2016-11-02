#ifndef NEUROAI
#define NEUROAI

#include <vector>
#include <string>
#include <iostream>
//#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "global_definitions.h"

typedef std::vector<float> neuronlayer;
typedef std::vector<std::vector<float> > edgelayer; //syn[indexout][indexin] -> for(i)[..for(j)[.syn[i][j].]..]
typedef std::vector<neuronlayer> neuronlayers_vec; 
typedef std::vector<edgelayer> edgelayers_vec;

#include "nn_math.h"

int random_func(int i);


class NeuralNet
{
private:
        neuronlayers_vec biases;
        edgelayers_vec edges;

        //mini functions (these could later be implemented inline)
        float rand_range(float pmin, float pmax);
        std::string get_layer_name(int player);

        void initialise_neural_net();
//        void update_through_backprop_over_mini_batch(
//                        const std::vector<training_data_s>* ptraining_data_s_vec_p,
//                        float pupdate_rate);

        //backpropagation_fast makes use of memory with edges in standard and
        //transposed form - this is much fast because transposing is not anymore
        //an operation in this algorithm but needs more memory
        void backpropagation_fast(
                        training_data_s ptraining_data_s_vec_p,
                        neuronlayers_vec* pbiases_p,
                        edgelayers_vec* pedges_p,
                        edgelayers_vec* pedges_transposed_p,
                        float pupdate_rate);


        //following methods are called by initialise_neural_net()
        //start...
        void generate_initial_edges();
        edgelayer generate_edge_layer(neuronlayer pnlayer_in, neuronlayer pnlayer_out);

        void generate_initial_biases();
        //...end

public:
        //Constructor
        NeuralNet(std::vector<int>);

        //intended to be used by a user
        std::vector<float> feedforward(std::vector<float> pinput);
        void stochastic_gradient_descent(
                        std::vector<training_data_s>* ptraining_data,
//                        std::vector<training_data_s>* ptraining_data,
                        int pnum_epochs,
                        int pmini_batch_size,
                        float ptraining_rate);

        void update_through_backprop_over_mini_batch(
                        const std::vector<training_data_s>* ptraining_data_s_vec_p,       
                        float pupdate_rate);
        //printing
        void print_layer_biases(int player); // loop-range:0 until #hidden + 1
        void print_edges(int poutput_layer); // loop-range:1 until #hidden + 1
        void print_output(std::vector<float> presult);
};

#endif
