#ifndef NODE_H
#define NODE_H

#include <vector>
#include <cmath>
#include "../random/random.h"
#include "../dataset/dataset.h"
#include "../statistics/statistics.h"
#include <iostream>
#include <cassert>

class node; // forward declaration

struct connection {
	double weight;
	node* target;
	node* source;
};

class node {
	public:
		virtual ~node() {}
		double value; /// every node has a cached output value
		std::vector<connection*> connections;
		double delta; /// used by the backpropagation algorithm
        virtual void update() {}
		void update_value(double x) { value = x; } 
};

class input : public node {
	public:
		int index; // input index (which column from the dataset)
};

class neuron : public node {
	public:
		virtual double func(double x) = 0; // activation function

		double deriv() const { return d; }
		double deriv2() const { return d2; }
		double bias;

	protected:
		double d, d2; // first- and second-order derivatives
};

class perceptron : public neuron {
	public:
		/// this activation function is recommended in http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
		/// f = 1.7159 * tanh(2/3 x)
		double func(double x) {
			return b * std::tanh(c * x);
		}

		void update() {
			double s = 0.0; // weighted sum of inputs and connection weights
			for (auto c : connections) {
				if (this == c->source) continue;
				s += c->source->value * c->weight;
			}
			/// update value and derivatives
			value = func(s);
			d = b * c - c / b * value * value;
			d2 = -2 * c / b * value * d;
		}

	private:
		const double b = 1.7159, c = 2.0 / 3.0;
};

class output : public neuron {
	public:
		double func(double x) { return x; } // the output neuron passes the input unchanged
		void update() {
			double s = 0.0; // weighted sum of inputs and connection weights
			for (auto c : connections) {
				if (this == c->source) continue;
				s += c->source->value * c->weight;
			}
			/// update value and derivatives
			value = func(s);
			d = 1;
			d2 = 0;
		}
};

typedef std::vector<node*> layer;

class neural_net {
public:
	std::vector<layer> layers;
	std::vector<connection*> connections;

	~neural_net() {
		for(size_t i = 0; i != connections.size(); ++i)
			delete connections[i];
		connections.clear();
		for(auto & l : layers) {
			for (auto & n : l) {
				if (n != nullptr) delete n;
			}
		}
	}

	void initialize(std::vector<int> layer_dimensions, rnd* rnd) {
		layers.resize(layer_dimensions.size());
		// populate input layer 
		auto & input_layer = layers[0];
		for (int j = 0; j != layer_dimensions[0]; ++j) {
			auto n = new input;
			n->index = j;
			input_layer.push_back(n);
		}

		// populate hidden and output layers
		for (size_t i = 1; i != layers.size()-1; ++i) {
			for (int j = 0; j != layer_dimensions[i]; ++j) {
				auto n = new perceptron;
				n->bias = rnd->next_double();
				layers[i].push_back(n);
			}
		}
		size_t size = layer_dimensions.size();
		for (int j = 0; j != layer_dimensions[size-1]; ++j) {
                auto n = new perceptron;
				n->bias = rnd->next_double();
				layers[size-1].push_back(n);
		}

		// add connections between the neurons
		if (layers.size() <= 1) return; // throw exception, complain, crash the program, etc.
		for (size_t i = 0; i != layers.size() - 1; ++i) {
			auto curr = layers[i]; // current layer
			auto next = layers[i+1]; // next layer
			for (size_t j = 0; j != next.size(); ++j) { 
				for (size_t k = 0; k != curr.size(); ++k) { 
					auto conn = new connection;
					conn->weight = rnd->next_double();
					conn->source = curr[k];
					conn->target = next[j];
					curr[k]->connections.push_back(conn);
					next[j]->connections.push_back(conn);
					connections.push_back(conn);
				}
			}
		}
	}

	void update(dataset *d, int row) {
		if (layers.size() == 0) return;
		// process inputs
		for (auto & n : layers[0]) {
			auto in = static_cast<input*>(n);
			in->update_value(d->rows[row][in->index]);
		}
		// process hidden neurons
		for (size_t i = 1; i < layers.size(); ++i) {
			for (auto & n : layers[i]) {
				n->update();
			}
		}
	}

};
#endif
