#pragma once
#ifndef MACHINE_LEARNING_GRP5_LIBRARY_LIBRARY_H
#define MACHINE_LEARNING_GRP5_LIBRARY_LIBRARY_H

#include <Eigen>
#include <cmath>
#include <iostream>
#include <cassert>
#include <ctime>


using namespace Eigen;
extern "C" {
	__declspec(dllexport) int perceptron_classification_classify(const double *model, const double *inputs, int inputsNumber);

	__declspec(dllexport) void perceptron_classification_train(double *const model, const double *inputs, const int *outputs, const int samplesNumber,
		const int inputsNumber, double rate);

	__declspec(dllexport) double perceptron_regression_predict(const double *model, const double *inputs, int inputsNumber);

	__declspec(dllexport) void perceptron_regression_train(double *const model, const double *inputs, const double *outputs, const int samplesNumber,
		const int inputsNumber, const int outputsNumber);

	__declspec(dllexport) double *init_perceptron_regression_model(const int inputsNumber);

	__declspec(dllexport) double *init_perceptron_classification_model(const int inputsNumber);

	__declspec(dllexport) void delete_perceptron_regression_model(double *model);

	__declspec(dllexport) void delete_perceptron_classification_model(double *model);


	////////////////////////////////////////////////////////////////////////////////////////////////////////

	__declspec(dllexport) void init();

	__declspec(dllexport) double activation(double &x);

	__declspec(dllexport) double get_random_double(double min, double max);

	__declspec(dllexport) double ***get_random_model(int *modelStruct, int modelStructSize, int inputsSize);

	__declspec(dllexport) double **mlp_regression_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize);

	__declspec(dllexport) double **mlp_classification_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize);

	__declspec(dllexport) void mlp_update_weight(double &weight, double &learningRate, double &output, double &error);

	__declspec(dllexport) void mlp_regression_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize, double **outputs, double *desiredOutputs, double &learningRate);

	__declspec(dllexport) void mlp_classification_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize, double **outputs, int *desiredOutputs, double &learningRate);

	__declspec(dllexport) void mlp_classification_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
		int *desiredOutput, double learningRate);

	__declspec(dllexport) void mlp_regression_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
		double *desiredOutput, double learningRate);

	__declspec(dllexport) void
		mlp_classification_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
			int inputsSize,
			int **desiredOutputs, double learningRate, int epochs);

	__declspec(dllexport) void
		mlp_regression_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
			int inputsSize,
			double **desiredOutputs, double learningRate, int epochs);

	__declspec(dllexport) double ***mlp_classification_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
		int inputsSize,
		int **desiredOutputs, double learningRate, int epochs);

	__declspec(dllexport) double ***mlp_regression_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
		int inputsSize,
		double **desiredOutputs, double learningRate, int epochs);

}
template<typename T>
int get_random_example_pos(T *examples, int examplesSize, int inputSize);

template<typename T>
T two_dim_get(T *&array, int &width, int &x, int &y);


template<typename T>
T *get_example_at(T *examples, int inputSize, int pos);

#endif