# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Himanshu Himanshu -- hhimansh
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
#
# Reference for activation function: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

import numpy as np
import copy
import math

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # Calculating Eucledian distance for two points using numpy 
    # normalization method
    return np.linalg.norm(x1 - x2)
    

def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # Calculating mannhattan distance by iterating through each point taking absolute of difference and adding it
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i])
    return distance
    

def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    # if derivative is false returning x 
    if not derivative:
        return x
    
    # if derivative is true returning numpy array all with 1 of dimension of x
    return np.ones_like(x)
    

def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    # if dervative is false
    if not derivative:

        # making a deep copy of x so that no changes occur in x
        y = copy.deepcopy(x)

        # applying the sigmoid formula on whole array
        y = 1 /(1 + np.exp(-y))

        # returning the changed y
        return y

    # if derivative is true returning derivative formula of sigmoid which is sigmoid * (1-sigmoid)
    return sigmoid(x) * (1-sigmoid(x))

   

def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    # if derivative is false
    if not derivative:

        # returning tanh of array
        return np.tanh(x)

    # if derivative is true 1 - tanh^2
    return 1 - np.square(tanh(x))


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    # making a deep copy of x so that no changes occur in x 
    y = copy.deepcopy(x)

    # if derivative is false
    if not derivative:

        # changing y where value is less than 0 setting it 0 otherwise same
        y = np.where(y < 0, 0, y)

        # returning y
        return y
    else:

        # changing y where value is less than 0 setting it 0
        y = np.where(y < 0, 0, y)

        # changing y where value is greater than equal to 0 setting it 1
        y = np.where(y >= 0, 1, y)

        # returning y
        return y

    

def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    # setting loss variable as 0
    loss = 0

    # iterating through each cell of both array and adding negative of y*log(p) to loss
    for i in range(len(y)):
        for j in range(len(y[i])):
            if p[i][j] !=0 :
                loss -= y[i][j]*math.log2(p[i][j])

    # returning loss
    return loss
    

def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    # getting unique categories from the list
    y1 = set(y)

    # initialising i for every index
    i = 0

    # initialising empty categories dictionary to store index of every category
    categories = {}

    # iterating through each unique categoer and assigning it a index and increasing index value
    for value in y1:
        categories[value] = i
        i += 1

    # intialising a multidimensional numpy array of dimensions length of data * no of categories
    one_hot_encoded = np.zeros((len(y),i))

    # iterating through each input data
    for i in range(len(y)):

        # setting index for category 1 for the point
        one_hot_encoded[i][categories[y[i]]] = 1

    # returning the encoded array
    return one_hot_encoded
    