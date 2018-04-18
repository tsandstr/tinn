#pragma once

#include <vector>
#include <string>
#include <random>

struct TinnState;

class Tinn {
public:
	Tinn(int n_inputs, int n_hidden, int n_outputs);
	Tinn(std::string path);

	double train(const std::vector<double> &input, const std::vector<double> &target, double rate);
	std::vector<double> predict(const std::vector<double> &input);

	void save(std::string path);
private:
	TinnState forward_propogate(const std::vector<double> &input);
	void back_propogate(const TinnState &state, const std::vector<double> &input, const std::vector<double> &target, double rate);

	std::uniform_real_distribution<double> distribution;
	std::default_random_engine generator;
	void randomize_weights_biases();

	int input_weight_index(int input, int hidden) { return hidden * n_inputs + input; }
	int hidden_weight_index(int hidden, int output) { return output * n_hidden + hidden; }

	int n_inputs, n_hidden, n_outputs;
	const static int n_biases = 2;

	std::vector<double> input_weights, hidden_weights, biases;
};

struct TinnState {
public:
	TinnState(const std::vector<double> &hidden, const std::vector<double> &output) : hidden{hidden}, output{output} {}
	double get_hidden(int n) const { return hidden[n]; }
	double get_output(int n) const { return output[n]; }
	std::vector<double> get_outputs() const { return output; }
private:
	std::vector<double> hidden, output;
};

double activation(double x);
double partial_activation(double x);
double error(double expected, double actual);
double partial_error(double expected, double actual);
double total_error(const std::vector<double> &expected, const std::vector<double> &actual);
