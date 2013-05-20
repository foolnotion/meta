#ifndef GA_H
#define GA_H

#include "../random/random.h"
#include <vector>
#include <algorithm>
#include <iostream>

/**
 * @brief Base class for all genetic operators
 */
template<typename ReturnType, typename... Args>
class op_base {
public:
    virtual ReturnType operator()(Args... args) = 0;
    rnd *r;
};

/**
 * @brief The ga_individual class
 *
 * Interface for classes representing a GA individual.
 */

class ga_individual {
public:
    virtual ~ga_individual() {}
    virtual ga_individual* clone() = 0;
    double fitness = 0;
};

template<bool desc = false>
struct compare {
    bool operator()(const ga_individual *a, const ga_individual *b) {
        return desc ? b->fitness < a->fitness : b->fitness > a->fitness;
    }
};

template<class Evaluator, class Creator, class CrossoverOp, class MutationOp>
class ga_optimizer {
public:
    ga_optimizer(int pop_size) :
        initialized(false),
        population_size(pop_size),
        elites(1),
        create(nullptr),
        eval(nullptr),
        crossoverOp(nullptr),
        mutateOp(nullptr){}

    void start(int generations) {
        std::cout << "--- Start" << std::endl;
        initialize(); // initialize population
        for (int i = 0; i != generations; ++i) {
//            std::cout << "--- Generation " << i << ", Best fitness: " << pop[0]->fitness << std::endl;
            do_select();
            do_crossover();
            do_mutation();
            do_evaluate(sel);
            do_reinsert();
        }
        const bool descending = true;
        std::sort(begin(pop), end(pop), compare<descending>()); // sort descending by fitness
    }

    ga_individual* best() {
        return pop.front();
    }

    void set_random(rnd *r) {
        this->r = r;
        create->r = r;
        eval->r = r;
        crossoverOp->r = r;
        mutateOp->r = r;
    }

    double mutation_probability;

    Creator *create;
    Evaluator *eval;
    CrossoverOp *crossoverOp;
    MutationOp *mutateOp;

private:
    void initialize() {
		pop.clear();
        pop.resize(population_size);
        for(int i = 0; i != pop.size(); ++i)  {
            pop[i] = (*create)();
        }
        sel.resize(pop.size());
        do_evaluate(pop);
        initialized = true;
    }

    void do_select() {
        sel.clear();
        sel.resize(pop.size());
        std::vector<double> partials;
        double sum = 0;
        for (auto ind : pop) {
            sum += ind->fitness;
            partials.push_back(sum);
        }
        for (int i = 0; i != pop.size(); ++i) {
            double d = r->next_double(sum);
            int j = 0;
            while (d > partials[j]) ++j;
            if (j == pop.size()) {
                std::cerr << "--- Warning: index exceeded. Adjusting." << std::endl;
                --j;
            }
            sel[i] = pop[j]->clone();
        }
    }

    void do_reinsert() {
        const bool descending = true;
        std::sort(begin(sel), end(sel), compare<descending>());
        std::merge(begin(pop), begin(pop)+elites, begin(sel)+elites, end(sel), begin(pop), compare<descending>());
    }

    void do_crossover() {
        for (auto ind : sel) {
            auto mate = sel[r->next(pop.size()-1)];
            (*crossoverOp)(ind, mate);
        }
    }

    void do_mutation() {
        for (auto ind : sel) {
            if (r->next_double() < mutation_probability)
                (*mutateOp)(ind);
        }
    }

    void do_evaluate(std::vector<ga_individual*>& individuals) {
        for (auto & ind : individuals) {
            ind->fitness = (*eval)(ind);
        }
    }

    std::vector<ga_individual*> pop;
    std::vector<ga_individual*> sel;

    rnd *r;
    bool initialized;
    int population_size;
    int elites;

};

#endif // GA_H
