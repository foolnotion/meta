#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>

const double eps = std::numeric_limits<double>::epsilon();

class mv_calculator {
	public:
		mv_calculator() { reset(); }
		void add(double x) {
			++n;
			if (n == 1) {
				old_mean = new_mean = x;
				old_var = 0;
			}
			else {
				new_mean = old_mean + (x - old_mean) / n;
				new_var = old_var + (x - old_mean) * (x - new_mean);
				// set up for next iteration
				old_mean = new_mean; 
				old_var = new_var; 
			}
		}
		void reset() { n = 0; }
		double mean() {
			return n > 0 ? new_mean : 0.0;
		}

		double variance() {
			return  n > 1 ? new_var / (n - 1) : 0.0;
		}
		double stddev() {
			return std::sqrt(variance());
		}
	private:
		double new_mean, old_mean, new_var, old_var;
		int n; // number of elements
};

class covariance_calculator {
	public:
		covariance_calculator() { reset(); }

		double covariance() { return n > 0 ? cn / n : 0.0; }
		void reset() {
			n = 0; x_mean = 0; y_mean = 0; cn = 0;
		}
		void add(double x, double y) {
			++n;
			x_mean = x_mean + (x - x_mean) / n;
			double delta = y - y_mean;
			y_mean = y_mean + delta / n;
			cn = cn + delta * (x - x_mean);
		}
	private:
		double x_mean, y_mean, cn;
		int n;
};

class rsquared_calculator {
	public:
		rsquared_calculator() {
			sx_calculator = std::unique_ptr<mv_calculator>(new mv_calculator);
			sy_calculator = std::unique_ptr<mv_calculator>(new mv_calculator);
			cov_calculator = std::unique_ptr<covariance_calculator>(new covariance_calculator);
		}
		void add(double x, double y) {
			sx_calculator->add(x);
			sy_calculator->add(y);
			cov_calculator->add(x,y);
		}
		void reset() {
			sx_calculator->reset();
			sy_calculator->reset();
			cov_calculator->reset();
		}
		double calculate(std::vector<double>& original_values, std::vector<double>& estimated_values) {
			// the two vectors should have the same size
			for(size_t i = 0; i != original_values.size(); ++i) {
				add(original_values[i], estimated_values[i]);
			}
			double xvar = sx_calculator->variance();
	        double yvar = sy_calculator->variance();
			if( xvar < eps || yvar < eps)  { return 0.0;	}
			double r = cov_calculator->covariance() / (std::sqrt(xvar) * std::sqrt(yvar));
			return r * r;
		}
	private:
		std::unique_ptr<mv_calculator> sx_calculator;
		std::unique_ptr<mv_calculator> sy_calculator;
		std::unique_ptr<covariance_calculator> cov_calculator;
};

// linear scaling parameter calculator
// the reasons for scaling are explained in: http://www2.cs.uidaho.edu/~cs472_572/f11/scaledsymbolicRegression.pdf
class lsp_calculator {
	public: 
		lsp_calculator() {
			target_mean_calculator = std::unique_ptr<mv_calculator>(new mv_calculator);
			ov_calculator = std::unique_ptr<mv_calculator>(new mv_calculator);
			ot_calculator = std::unique_ptr<covariance_calculator>(new covariance_calculator);
			reset();
		}
		void reset() {
			target_mean_calculator->reset();
			ov_calculator->reset();
			ot_calculator->reset();
		}
		void add(double original, double target) {
			target_mean_calculator->add(target);
			ov_calculator->add(original);
			ot_calculator->add(original, target);

			if (ov_calculator->variance() < eps) beta = 1;
			else beta = ot_calculator->covariance() / ov_calculator->variance();
			alpha = target_mean_calculator->mean() - beta * ov_calculator->mean(); 
		}
		double Beta() const { return beta; }
		double Alpha() const { return alpha; }
	private:
		double alpha; // additive constant 
		double beta; // multiplicative factor

		std::unique_ptr<mv_calculator> target_mean_calculator;
		std::unique_ptr<mv_calculator> ov_calculator; // original values 
		std::unique_ptr<covariance_calculator> ot_calculator; // original - target covariance calculator
};

#endif
