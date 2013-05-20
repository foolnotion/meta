#include "train.h"

/// classic backpropagation, http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
/// the ann must be updated first (call ann->update() before calling backprop())
void backprop(neural_net *ann, double learning_rate, dataset *d, int row) {
    double target_value = d->rows[row].back();
    auto & layers = ann->layers;
    auto & output_layer = layers.back();
    for (auto & n : output_layer) {
        n->delta = target_value - n->value;
//        std::cout << "Delta: " << n->delta << std::endl;
    }
    // first pass updating deltas of hidden neurons
    for (size_t i = layers.size() - 2; i >= 1; --i) {
        for (size_t j = 0; j != layers[i].size(); ++j) {
            auto & n = layers[i][j];
            n->delta = 0;
            for (auto conn : n->connections) {
                if (conn->target == n) continue; // skip input connections
                n->delta += conn->weight * conn->target->delta;
            }
        }
    }
    // second pass updating the connection weights
    for (size_t i = 0; i != layers.size()-1; ++i) {
        for (size_t j = 0; j != layers[i].size(); ++j) {
            auto & n = layers[i][j];
            // cycle through output connections
            for (auto conn : n->connections) {
                if (conn->target == n) continue; // skip input connections
                auto target = static_cast<neuron*>(conn->target);
                conn->weight += learning_rate * n->value * target->delta * target->deriv();
            }
        }
    }
}
