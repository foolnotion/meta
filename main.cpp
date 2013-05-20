#include "ann/ann.h"
#include "ann/train/train.h"
#include "statistics/statistics.h"
#include "random/random.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <memory>

using namespace std;

int main(void) {
	auto rand = unique_ptr<rnd>(new rnd); // initialize a random number generator
	//rand->seed(12345); // in case we need predictability and reproducibility when testing, we fix the seed 
	unique_ptr<dataset> data;
	try {
		auto d = new dataset("ev_an.txt");
		data.reset(d);
	} catch(...) {
		cout << "Error opening data file." << endl;
		return 1;
	}

	cout << "Loaded " << data->rows.size() << " rows of data" << endl;
	data->normalize();
	auto nn = unique_ptr<neural_net>(new neural_net);
	// by convention, first n-1 columns in the dataset are the inputs while the last column is the target output (can easily be changed)
	const int inputs = data->rows[0].size()-1;
	const int hidden_neurons = 5;
	const int outputs = 1;

	auto layer_dimensions = std::vector<int> { inputs, hidden_neurons, outputs }; 
	nn->initialize(layer_dimensions, rand.get());

	// split the dataset into training and test data by taking half-half
	size_t training_rows = data->rows.size() / 2;
//	size_t training_rows = 1;

	vector<double> output_values(training_rows);
	vector<double> target_values(training_rows);

	// set learning parameters
	double learning_rate = 0.1;
  	int training_iterations = 2000;

	int popsize = 100;
	int generations = 500;
	vector<int> indices;
	for (int i = 0; i != training_rows; ++i)
		indices.push_back(i);
	ga_train(nn.get(), rand.get(), data.get(), indices, generations, popsize);

	// commence training
//	for (int i = 0; i != training_iterations; ++i) {
//		for(size_t row = 0; row != training_rows; ++row) {
//			nn->update(data.get(), row);
//            backprop(nn.get(), learning_rate, data.get(), row); // do the actual training
//		}
//	}
	// get values after training
	for(size_t row = 0; row != training_rows; ++row) {
		nn->update(data.get(), row);
		double v = nn->layers.back()[0]->value; // in this case we know we have one single output
		target_values[row] = data->rows[row].back();
		output_values[row] = v;
	}
	
	// apply linear scaling on the network's output (weird, but this works)
	auto scaling_calculator = unique_ptr<lsp_calculator>(new lsp_calculator);
	for(size_t i = 0; i != output_values.size(); ++i)
		scaling_calculator->add(output_values[i], target_values[i]);
	double alpha = scaling_calculator->Alpha();
	double beta = scaling_calculator->Beta();

	for(auto & v : output_values)
		v = alpha + v * beta;
	
	auto r2calc = unique_ptr<rsquared_calculator>(new rsquared_calculator);
	double r2training = r2calc->calculate(output_values, target_values);
	cout << "Pearson's R2 (training): " << r2training << endl;

	// write training values to file
	ofstream f("training.out");
	for(size_t i = 0; i != output_values.size(); ++i)  {
		f << output_values[i] << " " << target_values[i] << endl;
	}
	f.close();
	// reinitialize and reuse these variables
	output_values = vector<double>(data->rows.size()-training_rows);
	target_values = vector<double>(data->rows.size()-training_rows);

	for (size_t row = training_rows; row != data->rows.size(); ++row) {
		nn->update(data.get(), row); 
		double v = nn->layers.back()[0]->value; // in this case we know we have one single output
		target_values[row-training_rows] = data->rows[row].back();
		output_values[row-training_rows] = v;
	}
	// scaling
	scaling_calculator->reset();
	for(size_t i = 0; i != output_values.size(); ++i)
		scaling_calculator->add(output_values[i], target_values[i]);
	alpha = scaling_calculator->Alpha();
	beta = scaling_calculator->Beta();

	for(auto & v : output_values)
		v = alpha + v * beta;

	double r2test = r2calc->calculate(output_values, target_values);
	cout << "Pearson's R2 (test): " << r2test << endl;

	// write test values to file
	ofstream f1("test.out");
	for(size_t i = 0; i != output_values.size(); ++i) 
		f1 << output_values[i] << " " << target_values[i] << endl;
	f1.close();

	return 0;
}
