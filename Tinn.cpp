#include "Tinn.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <functional>

using std::vector;
using std::string;

Tinn::Tinn(int n_inputs, int n_hidden, int n_outputs) :
	n_inputs{n_inputs},
	n_hidden{n_hidden},
	n_outputs{n_outputs} {
	input_weights.reserve(n_inputs * n_hidden);
	hidden_weights.reserve(n_hidden * n_outputs);
	biases.reserve(2);

	std::random_device rd;
	generator = std::default_random_engine(rd());
	distribution = std::uniform_real_distribution<double>(-0.5, 0.5);

	randomize_weights_biases();
}

Tinn::Tinn(std::string path) {
	std::ifstream file{path};
	file >> n_inputs >> n_hidden >> n_outputs;
	
	input_weights.reserve(n_inputs * n_hidden);
	hidden_weights.reserve(n_hidden * n_outputs);
	biases.reserve(2);

	std::random_device rd;
	generator = std::default_random_engine(rd());
	distribution = std::uniform_real_distribution<double>(-0.5, 0.5);

	for(int i = 0; i < n_biases; i++) {
		double temp;
		file >> temp;
		biases.push_back(temp);
	}
	for(int i = 0; i < n_inputs * n_hidden; i++) {
		double temp;
		file >> temp;
		input_weights.push_back(temp);
	}
	for(int i = 0; i < n_hidden * n_outputs; i++) {
		double temp;
		file >> temp;
		hidden_weights.push_back(temp);
	}
}

double Tinn::train(const vector<double> &input, const vector<double> &target, double rate) {
	TinnState state = forward_propogate(input);
	back_propogate(state, input, target, rate);
	return total_error(target, state.get_outputs());
}

vector<double> Tinn::predict(const vector<double> &input) {
	return forward_propogate(input).get_outputs();
}

void Tinn::save(std::string path) {
	std::ofstream file{path};

	file << n_inputs << " " << n_hidden << " " << n_outputs << std::endl;

	for(int i = 0; i < n_biases; i++) file << biases[i] << std::endl;
	for(int i = 0; i < n_inputs * n_hidden; i++) file << input_weights[i] << std::endl;
	for(int i = 0; i < n_hidden * n_outputs; i++) file << hidden_weights[i] << std::endl;
}

void Tinn::randomize_weights_biases() {
	for(double& w : input_weights) w = distribution(generator);
	for(double& w : hidden_weights) w = distribution(generator);
	for(double& b : biases) b = distribution(generator);
}

TinnState Tinn::forward_propogate(const vector<double> &input) {
	vector<double> hidden(n_hidden);
	vector<double> output(n_outputs);

	for(int i = 0; i < n_hidden; i++) {
		double sum{0};

		for(int j = 0; j < n_inputs; j++) {
			sum += input[j] * input_weights[input_weight_index(j, i)];
		}
		
		hidden[i] = activation(sum + biases[0]);
	}

	for(int i = 0; i < n_outputs; i++) {
		double sum{0};

		for(int j = 0; j < n_hidden; j++) {
			sum+= hidden[j] * hidden_weights[hidden_weight_index(j, i)];
		}

		output[i] = activation(sum + biases[1]);
	}

	TinnState state{hidden, output};
	return state;
}

void Tinn::back_propogate(const TinnState &state, const vector<double> &input, const vector<double> &target, double rate) {
	for(int i = 0; i < n_hidden; i++) {
		double sum{0};

		for(int j = 0; j < n_outputs; j++) {
			double a{partial_error(state.get_output(j), target[j])};
			double b{partial_activation(state.get_output(j))};

			sum += a * b * hidden_weights[hidden_weight_index(i,j)];
			hidden_weights[hidden_weight_index(i, j)] -= rate * a * b * state.get_hidden(i);
		}

		for(int j = 0; j < n_inputs; j++) {
			input_weights[input_weight_index(j, i)] -= rate * sum * partial_activation(state.get_hidden(i)) * input[j];
		}
	}
}

double activation(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double partial_activation(double x) {
	return x * (1.0 - x);
}

double error(double expected, double actual) {
	return 0.5 * (expected - actual) * (expected-actual);
}

double partial_error(double expected, double actual) {
	return expected - actual;
}

double total_error(const vector<double> &expected, const vector<double> &actual) {
	if(expected.size() != actual.size())
		throw std::range_error("Expected and actual different sizes");

	return std::inner_product(expected.begin(), expected.end(), actual.begin(), 0.0,
			std::plus<>(), [](double ex, double ac) { return error(ex, ac); });
}

