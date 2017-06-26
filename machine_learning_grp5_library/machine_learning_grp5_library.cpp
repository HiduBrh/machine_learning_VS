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

	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	__declspec(dllexport) void init() {
		srand(static_cast <unsigned> (time(0)));
		std::cout << std::boolalpha;
		std::cout.precision(25);
	}

	__declspec(dllexport) double activation(double &x) {
		return tanh(x);
	}

	__declspec(dllexport) double get_random_double(double min, double max) {
		return min + static_cast <double> (rand()) / ((RAND_MAX / (max - min)));
	}


	__declspec(dllexport) double ***get_random_model(int *modelStruct, int modelStructSize, int inputsSize) {
		assert(modelStruct != nullptr);
		assert(modelStructSize > 0);
		assert(inputsSize > 0);

		double ***model = new double **[modelStructSize];

		for (int i = 0; i < modelStructSize; ++i) {
			if (modelStruct[i] <= 0) {
				throw new std::out_of_range("Number of neurons must be greater than 0.");
			}
			model[i] = new double *[modelStruct[i]];
			for (int j = 0; j < modelStruct[i]; ++j) {
				int prevInputsSize = i == 0 ? inputsSize : modelStruct[i - 1];
				model[i][j] = new double[prevInputsSize];
				for (int k = 0; k < prevInputsSize; ++k) {
					model[i][j][k] = get_random_double(-1.f, 1.f);
				}
			}
		}
		return model;
	}


	__declspec(dllexport) double *add_bias_input(double *inputs, int inputsSize) {
		double *inputsWithBias = new double[inputsSize + 1];
		for (int i = 0; i < inputsSize; ++i) {
			inputsWithBias[i] = inputs[i];
		}
		inputsWithBias[inputsSize] = 1;
		return inputsWithBias;
	}

	__declspec(dllexport) double **mlp_regression_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize) {
		double **outputs = new double *[modelStructSize];
		double *inputsWithBias = add_bias_input(inputs, inputsSize);
		inputsSize += 1;
		for (int i = 0; i < modelStructSize - 1; ++i) {
			outputs[i] = new double[modelStruct[i]]{ 0 };
			for (int j = 0; j < modelStruct[i] - 1; ++j) {
				if (i == 0) {
					for (int k = 0; k < inputsSize; ++k) {
						outputs[i][j] += inputsWithBias[k] * model[i][j][k];
					}
				}
				else {
					for (int k = 0; k < modelStruct[i - 1]; ++k) {
						outputs[i][j] += outputs[i - 1][k] * model[i][j][k];
					}
				}
				outputs[i][j] = activation(outputs[i][j]);
			}
			outputs[i][modelStruct[i] - 1] = 1;
		}
		outputs[modelStructSize - 1] = new double[modelStruct[modelStructSize - 1]];
		outputs[modelStructSize - 1][0] = 0;
		for (int i = 0; i < modelStruct[modelStructSize - 2]; ++i) {
			outputs[modelStructSize - 1][0] += outputs[modelStructSize - 2][i] * model[modelStructSize - 1][0][i];
		}

		return outputs;
	}

	__declspec(dllexport) double **mlp_classification_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize) {
		double **outputs = new double *[modelStructSize];
		double *inputsWithBias = add_bias_input(inputs, inputsSize);
		inputsSize += 1;
		for (int i = 0; i < modelStructSize - 1; ++i) {
			outputs[i] = new double[modelStruct[i]]{ 0 };
			for (int j = 0; j < modelStruct[i] - 1; ++j) {
				if (i == 0) {
					for (int k = 0; k < inputsSize; ++k) {
						outputs[i][j] += inputsWithBias[k] * model[i][j][k];
					}
				}
				else {
					for (int k = 0; k < modelStruct[i - 1]; ++k) {
						outputs[i][j] += outputs[i - 1][k] * model[i][j][k];
					}
				}
				outputs[i][j] = activation(outputs[i][j]);
			}
			outputs[i][modelStruct[i] - 1] = 1;
		}

		outputs[modelStructSize - 1] = new double[modelStruct[modelStructSize - 1]];
		outputs[modelStructSize - 1][0] = 0;
		for (int i = 0; i < modelStruct[modelStructSize - 2]; ++i) {
			outputs[modelStructSize - 1][0] += outputs[modelStructSize - 2][i] * model[modelStructSize - 1][0][i];
		}

		outputs[modelStructSize - 1][0] = activation(outputs[modelStructSize - 1][0]);
		return outputs;
	}


	__declspec(dllexport) void mlp_update_weight(double &weight, double &learningRate, double &output, double &error) {
		weight = weight - learningRate * output * error;
	}


	__declspec(dllexport) void mlp_regression_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize, double **outputs, double *desiredOutputs, double &learningRate) {
		double **errors = new double *[modelStructSize];

		int lastLayer = modelStructSize - 1;
		errors[lastLayer] = new double[modelStruct[lastLayer]];
		for (int i = 0; i < modelStruct[lastLayer]; ++i) {
			errors[lastLayer][i] = outputs[lastLayer][i] - desiredOutputs[i];
		}

		for (int i = lastLayer - 1; i >= 0; --i) {
			errors[i] = new double[modelStruct[i]];
			for (int j = 0; j < modelStruct[i]; ++j) {
				int error = 0;
				for (int k = 0; k < modelStruct[i + 1]; ++k) {
					error += model[i + 1][k][j] * errors[i + 1][k];
				}
				errors[i][j] = (1 - pow(outputs[i][j], 2)) * error;
			}
		}

		for (int i = 0; i < modelStructSize; ++i) {
			for (int j = 0; j < modelStruct[i]; ++j) {
				if (i == 0) {
					for (int k = 0; k < inputsSize; ++k) {
						mlp_update_weight(model[i][j][k], learningRate, inputs[k], errors[i][j]);
					}
				}
				else {
					for (int k = 0; k < modelStruct[i - 1]; ++k) {
						mlp_update_weight(model[i][j][k], learningRate, outputs[i - 1][k], errors[i][j]);
					}
				}
			}
		}
	}

	__declspec(dllexport) void mlp_classification_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
		int inputsSize, double **outputs, int *desiredOutputs, double &learningRate) {
		double **errors = new double *[modelStructSize];

		int lastLayer = modelStructSize - 1;
		errors[lastLayer] = new double[modelStruct[lastLayer]];
		for (int i = 0; i < modelStruct[lastLayer]; ++i) {
			errors[lastLayer][i] = (1 - pow(outputs[lastLayer][i], 2)) * (outputs[lastLayer][i] - desiredOutputs[i]);
		}

		for (int i = lastLayer - 1; i >= 0; --i) {
			errors[i] = new double[modelStruct[i]];
			for (int j = 0; j < modelStruct[i]; ++j) {
				int error = 0;
				for (int k = 0; k < modelStruct[i + 1]; ++k) {
					error += model[i + 1][k][j] * errors[i + 1][k];
				}
				errors[i][j] = (1 - pow(outputs[i][j], 2)) * error;
			}
		}

		for (int i = 0; i < modelStructSize; ++i) {
			for (int j = 0; j < modelStruct[i]; ++j) {
				if (i == 0) {
					for (int k = 0; k < inputsSize; ++k) {
						mlp_update_weight(model[i][j][k], learningRate, inputs[k], errors[i][j]);
					}
				}
				else {
					for (int k = 0; k < modelStruct[i - 1]; ++k) {
						mlp_update_weight(model[i][j][k], learningRate, outputs[i - 1][k], errors[i][j]);
					}
				}
			}
		}
	}

	__declspec(dllexport) void mlp_classification_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
		int *desiredOutput, double learningRate) {
		double **outputs = mlp_classification_feed_forward(model, modelStruct, modelStructSize, inputs, inputsSize);
		mlp_classification_back_propagate(model, modelStruct, modelStructSize, inputs, inputsSize, outputs, desiredOutput,
			learningRate);
	}

	__declspec(dllexport) void mlp_regression_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
		double *desiredOutput, double learningRate) {
		double **outputs = mlp_regression_feed_forward(model, modelStruct, modelStructSize, inputs, inputsSize);
		mlp_regression_back_propagate(model, modelStruct, modelStructSize, inputs, inputsSize, outputs, desiredOutput,
			learningRate);
	}

	__declspec(dllexport) void
		mlp_classification_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
			int inputsSize,
			int **desiredOutputs, double learningRate, int epochs) {
		for (int i = 0; i < epochs; ++i) {
			int pos = get_random_example_pos(examples, examplesSize, inputsSize);
			double *example = get_example_at(examples, inputsSize, pos);
			mlp_classification_fit(model, modelStruct, modelStructSize, example, inputsSize, desiredOutputs[pos],
				learningRate);
		}
	}

	__declspec(dllexport) void
		mlp_regression_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
			int inputsSize,
			double **desiredOutputs, double learningRate, int epochs) {
		for (int i = 0; i < epochs; ++i) {
			int pos = get_random_example_pos(examples, examplesSize, inputsSize);
			double *example = get_example_at(examples, inputsSize, pos);
			mlp_regression_fit(model, modelStruct, modelStructSize, example, inputsSize, desiredOutputs[pos],
				learningRate);
		}
	}

	__declspec(dllexport) double ***mlp_classification_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
		int inputsSize,
		int **desiredOutputs, double learningRate, int epochs) {
		double ***model = get_random_model(modelStruct, modelStructSize, inputsSize);
		mlp_classification_train(model, modelStruct, modelStructSize, examples, examplesSize, inputsSize, desiredOutputs,
			learningRate, epochs);

		return model;
	}

	__declspec(dllexport) double ***mlp_regression_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
		int inputsSize,
		double **desiredOutputs, double learningRate, int epochs) {
		double ***model = get_random_model(modelStruct, modelStructSize, inputsSize);
		mlp_regression_train(model, modelStruct, modelStructSize, examples, examplesSize, inputsSize, desiredOutputs,
			learningRate, epochs);

		return model;
	}

}
template<typename T>
int get_random_example_pos(T *examples, int examplesSize, int inputSize) {
	assert(examples);
	assert(examplesSize > 0);
	assert(inputSize > 0);

	return rand() % examplesSize;
}

template<typename T>
T two_dim_get(T *&array, int &width, int &x, int &y) {
	return array[width * y + x];
}


template<typename T>
T *get_example_at(T *examples, int inputSize, int pos) {
	T *example = new T[inputSize];

	for (int i = 0; i < inputSize; ++i) {
		example[i] = two_dim_get(examples, inputSize, i, pos);
	}

	return example;
}


