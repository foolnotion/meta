#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <functional>

typedef std::mt19937 engine_type;

// implementation
class rnd {
public:
	rnd() {
		std::random_device rd;
		twister = engine_type(rd());
	}
    int next(int end) {
        if (end == 0) return 0;
        std::uniform_int_distribution<int> dist(0, end);
        return dist(twister);
    }

    int next(int begin, int end) {
        if (begin == end) return begin;
        std::uniform_int_distribution<int> dist(begin, end);
        return dist(twister);
    }

    double next_double() {
        std::uniform_real_distribution<double> real(0,1);
        return real(twister);
    }

    double next_double(double end) {
        if (end == 0) return 0;
        std::uniform_real_distribution<double> real(0, end);
        return real(twister);
    }

    double next_double(double begin, double end) {
        if (begin == end) return begin;
        std::uniform_real_distribution<double> real(begin, end);
        return real(twister);
    }
    void seed(engine_type::result_type s) { twister.seed(s); }

private:
    engine_type twister;
};

#endif // RANDOM_H
