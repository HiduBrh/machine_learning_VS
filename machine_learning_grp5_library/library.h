#pragma once
#ifndef MACHINE_LEARNING_GRP5_LIBRARY_LIBRARY_H
#define MACHINE_LEARNING_GRP5_LIBRARY_LIBRARY_H

#include <Eigen>

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
}
#endif