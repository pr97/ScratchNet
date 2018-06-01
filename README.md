# ScratchNet

**Language used**: Python (version=3.6)

>I follow the ideology that while learning new Machine Learning or Deep Learning (abbreviated as ML or DL from hereon) algorithms and techniques, it's always a good idea to implement any newly learnt algorithm from scratch, without using any existing ML/DL frameworks.

## Description
This repository holds one of my first Deep Learning projects. The project implements an MNIST classifying fully-connected neural network from scratch (in python) using only NumPy for numeric computations.

The fully-connected neural network has been programmed to use any architecture, which can be specified by the user in the form of a python list where the *i<sup>th</sup>* member of the list specifies the number of neurons in the *i<sup>th</sup>* layer. This means that the first item in list corresponds to the first layer, the second corresponds to the second layer and so on up to the output(last) layer. **To specify this list, just pass it as the 'layer_sizes' parameter in the 'initialize_parameters' function in scratchnet.py**

## Important Note
Because GitHub does not allow the upload of files with sizes larger than 100 MB, the dataset has been compressed into the dataset.zip folder. For the scratchnet.py script to work, the contents (mnist_train.csv and mnist_test.csv) of the dataset.zip folder need to be extracted into the working directory.
