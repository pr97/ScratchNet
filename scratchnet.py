import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle_wrapper as pw
from matplotlib import style

style.use("ggplot")

def mnist_loader(fname):
	dataset = pd.read_csv(fname, header = None)
	
	labels = np.array(dataset[0])
	one_hot_y = np.zeros((dataset.shape[0], 10))
	one_hot_y[np.arange(dataset.shape[0]), labels] = 1
	one_hot_y = one_hot_y.T
	
	X = dataset.drop([0], axis = 1)
	X = np.array(X).T

	return X, one_hot_y

def normalize(X):
	"""
	X - numpy array containing training examples stacked as columns {shape - (num_features, num_training examples)}
	------------------
	returns - Normalized X ((X - mu) / sigma))
	"""
	mu = np.mean(X, axis = 1, keepdims = True)
	sigma = np.var(X, axis = 1, keepdims = True)

	X_norm = np.divide((X - mu), 255)

	return X_norm

def initialize_parameters(layer_sizes, input_size, initializer = "xavier", multiplier = 0.01):
	"""
	layer_sizes - list containing layer sizes from layer 1 to the last layer i.e. layer L
	input_size - number of input features
	initializer - string specification of initialization method
	multiplier - parameter multiplier in case initializer != xavier
	-------------------------
	returns - parameters, dictionary of weights and biases
	"""
	L = len(layer_sizes)
	k = 0

	parameters = {}

	for l in range(1, L + 1):
		if initializer == "xavier":
			if l == 1:
				k = np.sqrt(2 / input_size)
			else:
				k = np.sqrt(2 / layer_sizes[l - 1])
		else:
			k = multiplier

		if l == 1:
			parameters["W" + str(l)] = np.random.randn(layer_sizes[l - 1], input_size) * k
		else:
			parameters["W" + str(l)] = np.random.randn(layer_sizes[l - 1], layer_sizes[l - 2]) * k
		parameters["b" + str(l)] = np.zeros((layer_sizes[l - 1], 1))
	
	return parameters

def relu(Z):
	return np.maximum(Z, 0.)

def relu_grad(Z):
	res = np.ones(Z.shape)
	res[Z < 0.] = 0.
	return res

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def sigmoid_grad(Z):
	return sigmoid(Z) * (1 - sigmoid(Z))

def forward_propagation(X, y, parameters):
	cache = {}
	Z = 0
	A = 0
	A_prev = X
	L = len(parameters) // 2
	
	for l in range(1, L + 1):
		
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		
		if l == 1:
			Z = np.dot(W, X) + b
		else:
			Z = np.dot(W, A_prev) + b
		
		if(l < L):
			A = relu(Z)
		else:
			A = sigmoid(Z)

		A_prev = A

		cache["Z" + str(l)] = Z
		cache["A" + str(l)] = A

	AL = A

	return AL, cache

def compute_cost_without_regularization(AL, y):
	m = y.shape[1]
	return (-1 / m) * np.squeeze(np.sum((y * np.log(AL) + (1 - y) * np.log(1 - AL))))

def compute_cost(AL, y, parameters, regularization = False, lambd = 0.):
	m = y.shape[1]
	L = len(parameters) // 2
	regularization_cache = 0
	if regularization:
		for l in range(1, L + 1):
			W = parameters["W" + str(l)]
			regularization_cache += np.power(np.linalg.norm(W), 2)

	return (-1 / m) * np.squeeze(np.sum((y * np.log(AL) + (1 - y) * np.log(1 - AL)))) + (lambd / (2 * m)) * regularization_cache

def back_propagation(X, y, parameters, cache, regularization = False, lambd = 0.):
	L = len(parameters) // 2
	m = y.shape[1]
	
	grads = {}
	dZ = 0
	dW = 0
	db = 0
	Z_prev = 0

	for l in range(L, 0, -1):
		
		if l > 1:
			A_prev = cache["A" + str(l - 1)]
			Z_prev = cache["Z" + str(l - 1)]
			W = parameters["W" + str(l)]

			if l == L:
				AL = cache["A" + str(l)]
				dZ = AL - y
			
			dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
			db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)

			dZ = np.dot(W.T, dZ) * relu_grad(Z_prev)
		
		else:
			W = parameters["W" + str(l)]
			dW = (1 / m) * np.dot(dZ, X.T) + (lambd / m) * W
			db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)

		grads["dW" + str(l)] = dW
		grads["db" + str(l)] = db

	return grads



def parameters_update(parameters, grads, optimizer = "gd", learning_rate = 0.01):
	
	L = len(parameters) // 2

	if(optimizer == "gd"):
		for l in range(1, L + 1):
			parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
			parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

def predict(X, y, parameters):
	
	AL, _ = forward_propagation(X, y, parameters)
	y_hat = np.float32(AL > 0.5)

	return y_hat


def model(X_train_orig, y_train_orig, X_test_orig, y_test_orig,
	layer_sizes, optimizer = "gd", num_iterations = 1500, learning_rate = 0.01,
	regularization = False, lambd = 0., print_cost = True, graph = False):
	
	X_train = normalize(X_train_orig)
	X_test = normalize(X_test_orig)
	y_train = y_train_orig
	y_test = y_test_orig

	input_size = X_train.shape[0]

	parameters = initialize_parameters(layer_sizes = layer_sizes, input_size = input_size, initializer = "xavier")

	cost_list = []

	print("Training Network...\n")

	for i in range(num_iterations):
		AL, cache = forward_propagation(X_train, y_train, parameters)
		cost = compute_cost(AL, y_train, parameters, regularization = regularization, lambd = lambd)
		grads = back_propagation(X_train, y_train, parameters, cache, regularization = regularization, lambd = lambd)
		parameters_update(parameters, grads, optimizer = optimizer, learning_rate = learning_rate)

		if graph and i % 100 == 0:
			cost_list.append(cost)

		if print_cost and i % 1 == 0:
			print("Cost after iteration " + str(i) + " = " + str(cost))

	print("Training Complete...\n")

	pw.pickle_it(parameters, "mnist_shallow_1000")

	y_hat_train = predict(X_train, y_train, parameters)
	y_hat_test = predict(X_test, y_test, parameters)

	train_error = 100 * (1 / (y_train.shape[0] * y_train.shape[1])) * np.squeeze(np.sum(np.float32(y_hat_train == y_train)))
	test_error = 100 * (1 / (y_test.shape[0] * y_test.shape[1])) * np.squeeze(np.sum(np.float32(y_hat_test == y_test)))

	train_accuracy = 100 - train_error
	test_accuracy = 100 - test_error

	print("Model Evaluation:\n")
	print("> Training Set Accuracy = " + str(train_accuracy))
	print("> Test Set Accuracy = " + str(test_accuracy))

def main():
	X_train_orig, y_train_orig = mnist_loader("mnist_train.csv")
	X_test_orig, y_test_orig = mnist_loader("mnist_test.csv")

	layer_sizes = [1000, 10]

	# model(X_train_orig, y_train_orig, X_test_orig, y_test_orig,
	# layer_sizes = layer_sizes, optimizer = "gd", num_iterations = 450, learning_rate = 0.01,
	# regularization = False, lambd = 0., print_cost = True, graph = False)

	parameters = pw.read_pickle("mnist_shallow_1000")

	X_train = normalize(X_train_orig)
	X_test = normalize(X_test_orig)
	y_train = y_train_orig
	y_test = y_test_orig

	y_hat_train = predict(X_train, y_train, parameters)
	y_hat_test = predict(X_test, y_test, parameters)

	train_accuracy = (100 / (y_train.shape[0] * y_train.shape[1])) * np.squeeze(np.sum(np.float32(y_hat_train == y_train)))
	test_accuracy = (100 / (y_test.shape[0] * y_test.shape[1])) * np.squeeze(np.sum(np.float32(y_hat_test == y_test)))

	train_error = (100 - train_accuracy) / 100
	test_error = (100 - test_accuracy) / 100

	print("Model Evaluation:\n")
	print("> Training Set Error = " + str(train_error))
	print("> Test Set Error = " + str(test_error))
	print("> Training Set Accuracy = " + str(train_accuracy))
	print("> Test Set Accuracy = " + str(test_accuracy))


if __name__ == '__main__':
	main()
