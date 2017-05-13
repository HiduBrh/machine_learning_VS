// machine_learning_grp5_library.cpp : définit les fonctions exportées pour l'application DLL.
//

#include "stdafx.h"
#include "library.h"

using namespace Eigen;

extern "C"
{
	__declspec(dllexport) double *init_perceptron_regression_model(const int inputsNumber) {
		double *model = new double[inputsNumber + 1];
		for (int i = 0; i <= inputsNumber; i++) {
			model[i] = 0.;
		}
		return model;
	}


	__declspec(dllexport) void perceptron_regression_train(double *const model, const double *inputs, const double *outputs, const int samplesNumber,
		const int inputsNumber, const int outputsNumber) {

		Eigen::MatrixXd inputs_Mat(samplesNumber, inputsNumber + 1);
		Eigen::MatrixXd outputs_Mat(samplesNumber, outputsNumber);
		for (int i = 0; i < samplesNumber * inputsNumber; i++) {
			inputs_Mat(i / inputsNumber, i % inputsNumber) = *(inputs + i);
		}

		for (int i = 0; i < samplesNumber; i++) {
			inputs_Mat(i, inputs_Mat.cols() - 1) = 1.0;
		}

		for (int i = 0; i < samplesNumber * outputsNumber; i++) {
			outputs_Mat(i / outputsNumber, i % outputsNumber) = *(outputs + i);
		}

		Eigen::MatrixXd w = (((inputs_Mat.transpose() * inputs_Mat).inverse()) * inputs_Mat.transpose()) * outputs_Mat;
		int k = 0;
		for (int j = 0; j < w.cols(); j++) {
			for (int i = 0; i < w.rows(); i++) {
				*(model + k) = w(i, j);
				k++;
			}

		}
	}

	__declspec(dllexport) double perceptron_regression_predict(const double *model, const double *inputs, int inputsNumber) {
		double x = 0.;
		for (int j = 0; j < inputsNumber; j++) {
			x = x + (inputs[j] * model[j]);
		}
		x += model[inputsNumber];
		return x;
	}


	__declspec(dllexport) double *init_perceptron_classification_model(const int inputsNumber) {
		double *model = new double[inputsNumber + 1];
		for (int i = 0; i <= inputsNumber; i++) {
			model[i] = 0.;
		}
		return model;
	}

	__declspec(dllexport) void perceptron_classification_train(double * const model, const double *inputs, const int *outputs, const int samplesNumber,
		const int inputsNumber, double rate) {
		int y;
		bool error = true;
		double x = 0.;
		while (error)
		{
			error = false;

			for (int i = 0; i < samplesNumber; i++)
			{
				for (int j = 0; j < inputsNumber; j++) {
					x += (inputs[(i*inputsNumber) + j] * model[j]);
				}
				x = x + model[inputsNumber];
				if (x < 0.) { y = -1; }
				else { y = 1; }
				if (y != outputs[i])
				{
					error = true;
					for (int k = 0; k<inputsNumber; k++) {
						model[k] += (rate*outputs[i] * inputs[(i*inputsNumber) + k]);
					}
					model[inputsNumber] += (rate*outputs[i]);
				}
				x = 0.;
			}
		}

	}


	__declspec(dllexport) int perceptron_classification_classify(const double *model, const double *inputs, int inputsNumber) {
		double x = 0.;
		for (int j = 0; j < inputsNumber; j++) {
			x = x + (inputs[j] * model[j]);
		}
		x += model[inputsNumber];
		if (x < 0)
			return -1;
		return 1;
	}

	__declspec(dllexport) void delete_perceptron_regression_model(double *model) {
		delete[]model;
	}

	__declspec(dllexport) void delete_perceptron_classification_model(double *model) {
		delete[]model;
	}
}


