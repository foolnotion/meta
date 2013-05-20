#ifndef TRAIN_H
#define TRAIN_H

#include "../ann.h"

void backprop(neural_net *ann, double learning_rate, dataset *d, int row);
void rprop(neural_net *ann, double learning_rate, dataset *d, int row); // to be implemented
void ga_train(neural_net *ann, rnd *r, dataset *d, std::vector<int>& indices, int generations, int popsize);

#endif // TRAIN_H
