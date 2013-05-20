#include "train.h"
#include "../../ga/ga.h"
#include <memory>

class ann_ind : public ga_individual {
	public:
		ann_ind(std::vector<double>& _real) : real(_real) {}

		ga_individual* clone() {
			return new ann_ind(this->real);
		}
		std::vector<double> real;
};

class ann_eval : public op_base<double, ga_individual*>{
	public:
		ann_eval() {
			r2calc = std::unique_ptr<rsquared_calculator>(new rsquared_calculator);
		}
		double operator()(ga_individual* ind) {
			auto individual = static_cast<ann_ind*>(ind);
			std::vector<double> weights;
			// save old connections weights
			for (auto conn : n->connections) {
				weights.push_back(conn->weight);
			}
			// update connection weights
			for (int i = 0; i != weights.size(); ++i) {
				n->connections[i]->weight = individual->real[i];
			}
			std::vector<double> targets, outputs;
			for(auto i : indices) {
				n->update(d, i);
				targets.push_back(d->rows[i].back());
				outputs.push_back(n->layers.back()[0]->value);
			}
			// restore connection weights
			for (int i = 0; i != weights.size(); ++i) {
				n->connections[i]->weight = weights[i];
			}
			r2calc->reset();
			return r2calc->calculate(targets, outputs);
		}

		std::unique_ptr<rsquared_calculator> r2calc;
		neural_net *n;
		dataset *d;
		std::vector<int> indices;
};

class ann_creator : public op_base<ann_ind*> {
	public:
		ann_ind* operator()() {
			std::vector<double> v;
			for (int i = 0; i != rsize; ++i)
				v.push_back(r->next_double(-5, 5));
			return new ann_ind(v);	
		}
		
		int rsize;
};

class ann_crossover : public op_base<void, ga_individual*, ga_individual*> {
	public:
		void operator()(ga_individual* a, ga_individual* b) {
			auto aa = static_cast<ann_ind*>(a);
			auto bb = static_cast<ann_ind*>(b);
			for (int i = 0; i != aa->real.size() / 2; ++i) {
				std::swap(aa->real[i], bb->real[i]);
			}
		}
};

class ann_mutation : public op_base<void, ga_individual*> {
	public:
		void operator()(ga_individual* a) {
			auto aa = static_cast<ann_ind*>(a);
			int i = r->next(aa->real.size()-1);
			aa->real[i] = r->next_double();
		}
};

void ga_train(neural_net *ann, rnd *r, dataset *d, std::vector<int>& indices, int generations, int popsize) {
	auto creator = new ann_creator;
	creator->rsize = ann->connections.size();
	auto evaluator = new ann_eval;
	evaluator->n = ann;
	evaluator->d = d;
	evaluator->indices = indices; 
	auto crossover = new ann_crossover;
	auto mutation = new ann_mutation;
	auto optimizer = new ga_optimizer<ann_eval, ann_creator, ann_crossover, ann_mutation>(popsize);
	optimizer->eval = evaluator;
	optimizer->create = creator;
	optimizer->crossoverOp = crossover;
	optimizer->mutateOp = mutation;
	optimizer->set_random(r);
	optimizer->mutation_probability = 0.25;
	optimizer->start(generations);
	auto best = static_cast<ann_ind*>(optimizer->best());
	for (int i = 0; i != best->real.size(); ++i) {
		ann->connections[i]->weight = best->real[i];
	}
}
