import numpy as np
import pandas as pd

def mnist_loader(fname):
	dataset = pd.read_csv(fname, header = None)
	
	labels = np.array(dataset[0])
	one_hot_y = np.zeros((60000, 10))
	one_hot_y[np.arange(60000), labels] = 1
	one_hot_y = one_hot_y.T
	
	X = dataset.drop([0], axis = 1)
	X = np.array(X).T

	return X, one_hot_y

# dataset = pd.read_csv("mnist_train.csv", header = None)
# print(dataset.head())
# print(dataset.shape)


# labels_train = np.array(dataset[0])
# one_hot_y = np.zeros((60000, 10))
# one_hot_y[np.arange(60000), labels_train] = 1
# one_hot_y = one_hot_y.T
# X_train = dataset.drop([0], axis = 1)
# X_train = np.array(X_train).T
# print(X_train)
# print(one_hot_y)
# print(X_train.shape)
# print(one_hot_y.shape)
